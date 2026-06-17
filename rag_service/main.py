import os
# O modelo de embeddings (e5-large) já fica em cache local. Sem isto, o
# sentence-transformers bate no HuggingFace Hub a CADA startup só pra checar
# metadados/versão — lento, dependente de internet e gera warning de HF_TOKEN.
# Modo offline = usa direto o cache (startup rápido, sem rede). Se algum dia
# trocar de modelo de embeddings, rode uma vez com HF_HUB_OFFLINE=0 pra baixar.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import hashlib
import pickle
import traceback
import httpx
import time
import queue
import threading
import uuid
import unicodedata
from dataclasses import dataclass, field
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from contextlib import asynccontextmanager
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from llama_index.core import (
    VectorStoreIndex,
    Settings,
    PromptTemplate,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.llms.openai_like import OpenAILike as _OpenAILikeBase
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PDFReader
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from cachetools import TTLCache

# ---------------------------------------------------------------------------
# Configurações gerais
# ---------------------------------------------------------------------------
CAMINHO_DOCS   = "../documentos"
CAMINHO_INDICE = "./storage"
CAMINHO_CHROMA = "./chroma_db"
CAMINHO_NODES  = "./storage/nodes.pkl"

# ---------------------------------------------------------------------------
# Configurações de paralelismo — conservadoras para não travar o PC
#
# THREAD_WORKERS = 2: o ONNX Runtime já usa todos os núcleos INTERNAMENTE
# por thread. Com threads=4 no modelo + 2 workers externos = 8 operações
# paralelas — suficiente sem saturar a CPU.
#
# TAMANHO_LOTE = 32: lotes menores = menos RAM por vez (seguro com 16GB
# e um PDF de 889 páginas que gera ~2000+ chunks).
# ---------------------------------------------------------------------------
THREAD_WORKERS = 4    # threads para leitura/chunking (CPU) — embedding vai pra GPU
TAMANHO_LOTE   = 128  # GPU aguenta lotes maiores — muito mais rápido
PIPELINE_FILA  = 64   # fila maior para alimentar a GPU sem pausa

# ---------------------------------------------------------------------------
# Reranking — precisão do retrieval
#
# O retriever híbrido (vetorial+BM25/RRF) é bom de RECALL mas a ORDEM dele é
# pouco confiável. Recuperamos um conjunto maior (RECUPERAR_TOP_K) e um
# cross-encoder (bge-reranker-base) reordena por relevância real, ficando só
# com os RERANK_TOP_N melhores que vão pro LLM. O reranker roda na CPU de
# propósito: são poucos pares (~20), é rápido, e não disputa a VRAM de 4GB da
# RTX 3050 com o e5-large (~1.5GB já ocupados).
# ---------------------------------------------------------------------------
RERANKER_MODEL  = "BAAI/bge-reranker-base"
RECUPERAR_TOP_K = 20   # candidatos que o híbrido entrega ao reranker
RERANK_TOP_N    = 5    # melhores que sobram após o rerank → vão pro contexto do LLM

# ---------------------------------------------------------------------------
# 1. LLM — REMOVIDO deste serviço (a geração vive no ia_service, porta 8004).
#    Porém o QueryFusionRetriever do LlamaIndex EXIGE um LLM no construtor e,
#    sem nenhum definido, tenta usar a OpenAI (pede OPENAI_API_KEY → erro).
#    Solução: um MockLLM (LLM "de mentira") que NUNCA é chamado pra gerar texto
#    (usamos num_queries=1, sem geração de query) — só satisfaz a exigência interna.
# ---------------------------------------------------------------------------
from llama_index.core.llms import MockLLM
Settings.llm = MockLLM()

# ---------------------------------------------------------------------------
# 2. Embeddings — e5-large na GPU (RTX 3050 CUDA)
# ~1.5GB de VRAM — sobram ~2.5GB livres na RTX 3050
# GPU é ~10x mais rápida que CPU para embeddings em batch
# ---------------------------------------------------------------------------
print("📥 Carregando modelo de embeddings na GPU (CUDA)...")
Settings.embed_model = HuggingFaceEmbedding(
    model_name="intfloat/multilingual-e5-large",
    device="cuda",          # RTX 3050 — ~10x mais rápido que CPU
    embed_batch_size=32,    # RTX 3050 4GB: ~2.4GB nesse batch — mais throughput sem OOM
    query_instruction="query: ",
    text_instruction="passage: ",
)

# ---------------------------------------------------------------------------
# 3. Chunking
# ---------------------------------------------------------------------------
Settings.text_splitter = SentenceSplitter(
    chunk_size=768,
    chunk_overlap=150,
)

# ---------------------------------------------------------------------------
# 4. Templates de prompt — ARIA
# ---------------------------------------------------------------------------
TEXT_QA_TEMPLATE = PromptTemplate(
    "Você é o ARIA (Assistente de Revisão Inteligente Acadêmica), um tutor virtual "
    "especializado em ajudar estudantes a compreender o conteúdo dos seus materiais de aula.\n"
    "\n"
    "REGRA ABSOLUTA: Responda SEMPRE em português do Brasil. NUNCA em inglês.\n"
    "\n"
    "SUA PERSONALIDADE:\n"
    "- Seja didático e claro, como um bom professor explicaria\n"
    "- Use exemplos quando ajudar a entender\n"
    "- Organize a resposta com estrutura lógica (tópicos, listas, etapas)\n"
    "- Seja encorajador e paciente\n"
    "- Vá direto ao ponto, sem introduções desnecessárias\n"
    "\n"
    "REGRAS DE CONTEÚDO:\n"
    "- Baseie-se ESTRITAMENTE nas informações do contexto abaixo\n"
    "- Se a resposta não estiver no contexto, diga: 'Essa informação não está no material disponível.'\n"
    "- Nunca invente dados, fórmulas, autores ou conceitos fora do documento\n"
    "- Para resumos, cubra todos os conceitos importantes de forma detalhada\n"
    "- Para listas ou enumerações do PDF, preserve a estrutura original\n"
    "- Se a pergunta for sobre definição de um conceito, explique com suas próprias palavras baseado no contexto\n"
    "\n"
    "CONTEXTO DO MATERIAL DE AULA:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "\n"
    "PERGUNTA DO ESTUDANTE: {query_str}\n"
    "\n"
    "RESPOSTA DETALHADA EM PORTUGUÊS DO BRASIL:"
)

SUMMARY_TEMPLATE = PromptTemplate(
    "Você é o ARIA, um tutor virtual acadêmico.\n"
    "INSTRUÇÃO OBRIGATÓRIA: Responda SEMPRE em português do Brasil. NUNCA em inglês.\n"
    "Seja didático, claro e organizado. Vá direto ao conteúdo.\n"
    "\n"
    "Abaixo estão trechos do material de aula do estudante:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "\n"
    "Com base APENAS nos trechos acima, responda de forma completa e didática:\n"
    "{query_str}\n"
    "\n"
    "RESPOSTA EM PORTUGUÊS:"
)

# Template usado quando o contexto vem da BUSCA NA WEB (não dos PDFs de aula)
WEB_TEMPLATE = PromptTemplate(
    "Você é o ARIA (Assistente de Revisão Inteligente Acadêmica), um tutor virtual.\n"
    "\n"
    "REGRA ABSOLUTA: Responda SEMPRE em português do Brasil. NUNCA em inglês.\n"
    "\n"
    "O conteúdo abaixo NÃO veio do material de aula do estudante — veio de uma BUSCA NA "
    "INTERNET, porque a pergunta não foi encontrada nos PDFs disponíveis.\n"
    "\n"
    "INSTRUÇÕES:\n"
    "- Avise o estudante, de forma natural, que essa resposta vem de uma busca na web "
    "(não do material de aula).\n"
    "- Seja didático, claro e organizado.\n"
    "- Baseie-se nos resultados de busca abaixo. Não invente dados.\n"
    "\n"
    "RESULTADOS DA BUSCA NA WEB:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "\n"
    "PERGUNTA DO ESTUDANTE: {query_str}\n"
    "\n"
    "RESPOSTA EM PORTUGUÊS DO BRASIL:"
)

# ---------------------------------------------------------------------------
# Estado encapsulado
# ---------------------------------------------------------------------------
@dataclass
class ARIAState:
    query_engine:     Any      = None
    retriever_debug:  Any      = None
    vector_retriever: Any      = None   # retriever vetorial puro — score de cosseno calibrado (0-1)
    reranker:         Any      = None   # cross-encoder que reordena os candidatos por relevância real
    nodes_globais:    list     = field(default_factory=list)
    ficha:            dict     = field(default_factory=dict)  # título/autor do PDF atual (p/ ancorar a busca web no tema)
    cache_respostas: TTLCache = field(
        default_factory=lambda: TTLCache(maxsize=256, ttl=3600)
    )

aria = ARIAState()


# ---------------------------------------------------------------------------
# Modelos Pydantic
# ---------------------------------------------------------------------------
class Pergunta(BaseModel):
    texto: str

class MCPToolCall(BaseModel):
    tool: str
    arguments: dict

class SinteseRequest(BaseModel):
    pergunta:  str
    contexto:  str
    fonte:     str = "material"   # "material" (PDFs) ou "web" (busca na internet)
    historico: str = ""           # histórico recente da conversa (opcional, p/ memória)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _calcular_hash_documentos(caminho: str) -> str:
    """Hash por metadados — muito mais rápido que ler o conteúdo dos PDFs."""
    hasher = hashlib.md5()
    for nome in sorted(os.listdir(caminho)):
        filepath = os.path.join(caminho, nome)
        if os.path.isfile(filepath):
            stat = os.stat(filepath)
            hasher.update(f"{nome}:{stat.st_size}:{stat.st_mtime}".encode())
    return hasher.hexdigest()


def _preAquecer_modelo():
    """Chama o modelo uma vez para o ONNX Runtime inicializar antes da indexação."""
    print("🔥 Pré-aquecendo modelo de embeddings...")
    Settings.embed_model.get_text_embedding("aquecimento")
    print("✅ Modelo pronto.")


# ---------------------------------------------------------------------------
# Ficha do documento (título / autor) — funciona em QUALQUER PDF
#
# Os campos /Title e /Author embutidos no PDF são pouco confiáveis (muitas vezes
# vêm vazios, com lixo — ex: autor "1140" — ou com o nome do gerador do PDF).
# Estratégia: usa os metadados quando fazem sentido e cai para o texto da
# primeira página como fallback, deixando a IA deduzir título/autor a partir dele.
# ---------------------------------------------------------------------------
FICHA_NODE_ID = "ficha_documento"


def _autor_valido(autor: str) -> bool:
    """Rejeita autores vazios, curtos demais ou sem letras (lixo numérico tipo '1140')."""
    if not autor or len(autor.strip()) < 3:
        return False
    return any(c.isalpha() for c in autor)


def _extrair_ficha_documento(caminho_pdf: str) -> dict:
    """Extrai título, autor e o início do texto de um PDF, robusto a metadados ruins."""
    import io
    from pypdf import PdfReader

    nome_arquivo    = os.path.basename(caminho_pdf)
    titulo          = ""
    autor           = ""
    primeira_pagina = ""

    try:
        # Lê os bytes para a memória e FECHA o arquivo imediatamente. Passar o
        # caminho direto pro PdfReader deixa o pypdf com o handle aberto (leitura
        # preguiçosa) → no Windows isso TRAVA o arquivo e impede apagá-lo no upload
        # (WinError 32). Com BytesIO o handle do disco já está liberado aqui.
        with open(caminho_pdf, "rb") as fh:
            dados = fh.read()
        reader = PdfReader(io.BytesIO(dados))
        meta   = reader.metadata
        t = (meta.title  or "").strip() if (meta and meta.title)  else ""
        a = (meta.author or "").strip() if (meta and meta.author) else ""
        if t and len(t) >= 3:
            titulo = t
        if _autor_valido(a):
            autor = a
        if reader.pages:
            primeira_pagina = (reader.pages[0].extract_text() or "").strip()
    except Exception as exc:
        print(f"   (não consegui ler os metadados do PDF: {exc})")

    # Fallback do título: primeira linha "de cara" da página 1, senão o nome do arquivo
    if not titulo:
        for linha in primeira_pagina.splitlines():
            if len(linha.strip()) >= 5:
                titulo = linha.strip()
                break
    if not titulo:
        titulo = os.path.splitext(nome_arquivo)[0]

    return {
        "arquivo":         nome_arquivo,
        "titulo":          titulo,
        "autor":           autor or "não informado nos metadados do documento",
        "primeira_pagina": primeira_pagina[:1500],
    }


def _construir_node_ficha(ficha: dict) -> TextNode:
    """Monta um chunk sintético com a ficha do documento, rico em sinônimos para
    que perguntas sobre título/autor o recuperem com score alto (rota = material)."""
    texto = (
        "FICHA DO DOCUMENTO (informações sobre o próprio material de aula).\n"
        f"Título do documento / nome do PDF / nome do livro / título da obra / "
        f"nome do material: {ficha['titulo']}.\n"
        f"Autor / autoria / quem escreveu este documento: {ficha['autor']}.\n"
        f"Nome do arquivo: {ficha['arquivo']}.\n"
        "Use estas informações para responder perguntas como: qual é o título, "
        "qual o nome do documento ou do livro, quem é o autor, quem escreveu o material.\n"
        "Trecho inicial do documento (para confirmar título e autoria):\n"
        f"{ficha['primeira_pagina']}"
    )
    node     = TextNode(text=texto, metadata={"file_name": ficha["arquivo"], "tipo": "ficha_documento"})
    node.id_ = FICHA_NODE_ID
    return node


def _injetar_ficha(chroma_collection):
    """Calcula a ficha do PDF atual e garante (idempotente) que ela esteja no
    ChromaDB e em aria.nodes_globais (para o BM25). Chamado em todo startup."""
    pdfs = (
        [os.path.join(CAMINHO_DOCS, f)
         for f in sorted(os.listdir(CAMINHO_DOCS)) if f.lower().endswith(".pdf")]
        if os.path.exists(CAMINHO_DOCS) else []
    )
    if not pdfs:
        return

    ficha = _extrair_ficha_documento(pdfs[0])
    print(f"🪪 Ficha do documento → título='{ficha['titulo']}' | autor='{ficha['autor']}'")

    # Guarda a ficha estruturada para o Gateway ancorar a busca web no tema do PDF.
    aria.ficha = {
        "titulo":  ficha["titulo"],
        "autor":   ficha["autor"],
        "arquivo": ficha["arquivo"],
    }

    node = _construir_node_ficha(ficha)
    node.embedding = Settings.embed_model.get_text_embedding(node.get_content())

    # upsert = idempotente: não duplica a ficha a cada reinício do servidor
    chroma_collection.upsert(
        ids=[node.id_],
        documents=[node.get_content()],
        metadatas=[node.metadata],
        embeddings=[node.embedding],
    )

    # Mantém a ficha única no BM25 também
    aria.nodes_globais = [
        n for n in aria.nodes_globais if getattr(n, "id_", None) != FICHA_NODE_ID
    ]
    aria.nodes_globais.append(node)


# ---------------------------------------------------------------------------
# Pipeline em streaming: Leitura → Chunking → Embedding → Inserção no Chroma
#
# Estágio 1 — Leitura (ThreadPoolExecutor, I/O bound)
# Estágio 2 — Chunking (ThreadPoolExecutor — rápido o suficiente, sem MemoryError)
# Estágio 3 — Embedding em lotes (ThreadPoolExecutor + ONNX usa 8 threads internos)
# Estágio 4 — Inserção no Chroma em batch (thread dedicada)
#
# Os 4 estágios correm em paralelo via filas — enquanto o PDF 3 é lido,
# o PDF 2 já está sendo chunkado e o PDF 1 já está sendo embedado.
# ---------------------------------------------------------------------------
def _indexar_pipeline_completo(arquivos_pdf: list, chroma_collection) -> list:
    SENTINEL       = object()
    fila_leitura   = queue.Queue(maxsize=PIPELINE_FILA)
    fila_chunks    = queue.Queue(maxsize=PIPELINE_FILA)
    fila_embedding = queue.Queue(maxsize=PIPELINE_FILA)

    todos_nodes  = []
    lock_nodes   = threading.Lock()
    erros        = []
    tempo_inicio = time.time()

    # Tempo de trabalho ACUMULADO por estágio (os 4 rodam em paralelo, então o
    # relógio de parede se sobrepõe — esta soma revela qual estágio é o gargalo real).
    tempos      = {"leitura": 0.0, "chunking": 0.0, "embedding": 0.0, "insercao": 0.0}
    lock_tempos = threading.Lock()
    def _marcar(estagio, segundos):
        with lock_tempos:
            tempos[estagio] += segundos

    # ── Estágio 1: Leitura paralela dos PDFs (threads — I/O bound) ──────────
    def _estagio_leitura():
        def _ler(caminho):
            try:
                t0   = time.time()
                docs = PDFReader().load_data(file=caminho)
                resultado = [(d.get_content(), d.metadata) for d in docs]
                _marcar("leitura", time.time() - t0)
                print(f"   📄 Lido: {os.path.basename(caminho)} ({len(docs)} págs)")
                return resultado
            except Exception as exc:
                erros.append(f"Leitura {caminho}: {exc}")
                return []

        with ThreadPoolExecutor(max_workers=min(THREAD_WORKERS, len(arquivos_pdf))) as ex:
            for fut in as_completed({ex.submit(_ler, p): p for p in arquivos_pdf}):
                for item in fut.result():
                    fila_leitura.put(item)
        fila_leitura.put(SENTINEL)

    # ── Estágio 2: Chunking em threads (SentenceSplitter é rápido, sem GIL pesado)
    def _estagio_chunking():
        splitter = SentenceSplitter(chunk_size=768, chunk_overlap=150)

        def _chunkar(args):
            from llama_index.core.schema import Document
            t0 = time.time()
            texto, metadata = args
            doc   = Document(text=texto, metadata=metadata or {})
            nodes = splitter.get_nodes_from_documents([doc])
            resultado = [{"text": n.get_content(), "metadata": n.metadata} for n in nodes]
            _marcar("chunking", time.time() - t0)
            return resultado

        with ThreadPoolExecutor(max_workers=THREAD_WORKERS) as ex:
            futuros = []
            buffer  = []
            LOTE    = 16

            def _flush(docs):
                if docs:
                    futuros.extend([ex.submit(_chunkar, d) for d in docs])

            while True:
                item = fila_leitura.get()
                if item is SENTINEL:
                    _flush(buffer)
                    break
                buffer.append(item)
                if len(buffer) >= LOTE:
                    _flush(buffer)
                    buffer = []

            for fut in as_completed(futuros):
                for chunk in fut.result():
                    fila_chunks.put(chunk)

        fila_chunks.put(SENTINEL)

    # ── Estágio 3: Embedding em lotes (GPU/PyTorch — 1 thread só) ─
    # A GPU serializa o trabalho de qualquer forma; múltiplas threads no mesmo
    # modelo não aceleram e ainda multiplicam o uso de VRAM (risco de OOM nos 4GB).
    def _estagio_embedding():
        acumulador = []

        def _embedar(lote):
            textos  = [c["text"] for c in lote]
            t0      = time.time()
            vetores = Settings.embed_model.get_text_embedding_batch(
                textos, show_progress=False
            )
            _marcar("embedding", time.time() - t0)
            for chunk, vetor in zip(lote, vetores):
                fila_embedding.put((chunk, vetor))

        with ThreadPoolExecutor(max_workers=1) as ex:
            futuros = []
            while True:
                item = fila_chunks.get()
                if item is SENTINEL:
                    if acumulador:
                        futuros.append(ex.submit(_embedar, acumulador))
                    break
                acumulador.append(item)
                if len(acumulador) >= TAMANHO_LOTE:
                    futuros.append(ex.submit(_embedar, acumulador))
                    acumulador = []
            for fut in as_completed(futuros):
                fut.result()
        fila_embedding.put(SENTINEL)

    # ── Estágio 4: Inserção no Chroma em batch ────────────────────────────────
    def _estagio_insercao():
        BATCH = 128
        ids_b, docs_b, metas_b, vecs_b = [], [], [], []
        total = 0

        def _flush_chroma():
            nonlocal total
            if not ids_b:
                return
            t0 = time.time()
            chroma_collection.add(
                ids=ids_b, documents=docs_b,
                metadatas=metas_b, embeddings=vecs_b,
            )
            _marcar("insercao", time.time() - t0)
            total += len(ids_b)
            print(f"   💾 {total} chunks no ChromaDB...")
            ids_b.clear(); docs_b.clear(); metas_b.clear(); vecs_b.clear()

        while True:
            item = fila_embedding.get()
            if item is SENTINEL:
                _flush_chroma()
                break
            chunk, vetor = item

            node = TextNode(text=chunk["text"], metadata=chunk["metadata"])
            node.embedding = vetor
            with lock_nodes:
                todos_nodes.append(node)

            ids_b.append(str(uuid.uuid4()))
            docs_b.append(chunk["text"])
            metas_b.append(chunk["metadata"] or {})
            vecs_b.append(vetor)
            if len(ids_b) >= BATCH:
                _flush_chroma()

        print(f"   ✅ Total inserido: {total} chunks")

    # ── Inicia os 4 estágios em paralelo ─────────────────────────────────────
    threads = [
        threading.Thread(target=_estagio_leitura,   name="leitura",   daemon=True),
        threading.Thread(target=_estagio_chunking,  name="chunking",  daemon=True),
        threading.Thread(target=_estagio_embedding, name="embedding", daemon=True),
        threading.Thread(target=_estagio_insercao,  name="insercao",  daemon=True),
    ]

    print(f"🚀 Pipeline: {len(arquivos_pdf)} PDF(s) | lote={TAMANHO_LOTE} | {THREAD_WORKERS} threads")

    for t in threads: t.start()
    for t in threads: t.join()

    if erros:
        print(f"⚠️  {len(erros)} erro(s):")
        for e in erros: print(f"   - {e}")

    total_parede = time.time() - tempo_inicio
    print(f"⏱️  Pipeline em {total_parede:.1f}s — {len(todos_nodes)} chunks")
    print("─" * 50)
    print("📊 TEMPO DE TRABALHO POR ESTÁGIO (onde está o gargalo):")
    soma_trabalho = sum(tempos.values()) or 1.0
    rotulos = {
        "leitura":   "📄 Leitura PDF  (CPU)",
        "chunking":  "✂️  Chunking     (CPU)",
        "embedding": "🧠 Embeddings   (GPU)",
        "insercao":  "💾 Chroma insert(CPU)",
    }
    for chave in ("leitura", "chunking", "embedding", "insercao"):
        seg = tempos[chave]
        print(f"   {rotulos[chave]}: {seg:7.1f}s  ({seg/soma_trabalho*100:4.1f}% do trabalho)")
    print("─" * 50)
    return todos_nodes


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
def iniciar_sistema_rag():
    global aria

    print("⏳ Iniciando RAG — serviço de retrieval (busca no material)...")
    _preAquecer_modelo()

    try:
        chroma_client     = chromadb.PersistentClient(path=CAMINHO_CHROMA)
        chroma_collection = chroma_client.get_or_create_collection("documentos")
        vector_store      = ChromaVectorStore(chroma_collection=chroma_collection)

        # A pasta 'documentos' agora é apenas o DESTINO do upload — não exigimos
        # mais um PDF pré-colocado nela. O material entra exclusivamente via /upload.
        pdfs_na_pasta = (
            [f for f in sorted(os.listdir(CAMINHO_DOCS)) if f.lower().endswith(".pdf")]
            if os.path.exists(CAMINHO_DOCS) else []
        )

        hash_salvo   = ""
        caminho_hash = os.path.join(CAMINHO_INDICE, "docs.hash")
        if os.path.exists(caminho_hash):
            with open(caminho_hash, "r") as f:
                hash_salvo = f.read().strip()

        indice_persistido = (
            os.path.exists(CAMINHO_NODES)
            and os.path.exists(CAMINHO_INDICE)
            and os.listdir(CAMINHO_INDICE)
            and chroma_collection.count() > 0
        )

        # ── Caso 1: nenhum PDF na pasta ─────────────────────────────────────────
        if not pdfs_na_pasta:
            if indice_persistido:
                # Recarrega o ÚLTIMO material indexado — sobrevive ao restart mesmo
                # sem o PDF na pasta. NUNCA limpamos o índice aqui (pasta vazia não
                # significa "documento mudou": significa "ainda não houve novo upload").
                print("📦 Pasta sem PDF, mas há índice salvo — recarregando o último material...")
                with open(CAMINHO_NODES, "rb") as f:
                    aria.nodes_globais = pickle.load(f)
                storage_context = StorageContext.from_defaults(
                    persist_dir=CAMINHO_INDICE,
                    vector_store=vector_store,
                )
                index = load_index_from_storage(storage_context)
            else:
                # Sobe vazio e tranquilo, aguardando o upload de um PDF pelo frontend.
                print("📭 Nenhum material indexado ainda. RAG no ar — aguardando upload de PDF pelo frontend.")
                aria.nodes_globais    = []
                aria.retriever_debug  = None
                aria.vector_retriever = None
                aria.ficha            = {}
                return

        # ── Caso 2: há PDF na pasta (fluxo normal / logo após um upload) ────────
        else:
            hash_atual         = _calcular_hash_documentos(CAMINHO_DOCS)
            documentos_mudaram = hash_atual != hash_salvo
            indice_existe      = indice_persistido and not documentos_mudaram

            if documentos_mudaram:
                import shutil
                print("🔄 Documentos alterados! Limpando índice antigo e reindexando...")
                # Storage (índice do LlamaIndex)
                if os.path.exists(CAMINHO_INDICE):
                    shutil.rmtree(CAMINHO_INDICE, ignore_errors=True)
                # ChromaDB: apaga a coleção PELA API do Chroma — robusto no Windows.
                # (Apagar a PASTA falha quando o arquivo está travado e deixa vetores
                #  antigos vivos → o livro novo entra POR CIMA do velho = mistura de assuntos.)
                try:
                    chroma_client.delete_collection("documentos")
                    print("🧹 Coleção antiga do ChromaDB removida (limpeza total).")
                except Exception as exc:
                    print(f"   (coleção já estava limpa: {exc})")
                chroma_collection = chroma_client.get_or_create_collection("documentos")
                vector_store      = ChromaVectorStore(chroma_collection=chroma_collection)

            if indice_existe:
                print("📦 Carregando índice existente do disco (startup rápido)...")
                with open(CAMINHO_NODES, "rb") as f:
                    aria.nodes_globais = pickle.load(f)
                storage_context = StorageContext.from_defaults(
                    persist_dir=CAMINHO_INDICE,
                    vector_store=vector_store,
                )
                index = load_index_from_storage(storage_context)

            else:
                # Política "um livro por vez": evita misturar assuntos de livros diferentes.
                if len(pdfs_na_pasta) > 1:
                    nomes = ", ".join(pdfs_na_pasta)
                    raise RuntimeError(
                        f"Encontrei {len(pdfs_na_pasta)} PDFs na pasta 'documentos' ({nomes}). "
                        "A política atual é UM LIVRO POR VEZ — deixe apenas um PDF na pasta."
                    )

                arquivos_pdf = [os.path.join(CAMINHO_DOCS, pdfs_na_pasta[0])]
                print(f"📂 1 PDF encontrado. Iniciando pipeline...")

                aria.nodes_globais = _indexar_pipeline_completo(arquivos_pdf, chroma_collection)

                if not aria.nodes_globais:
                    raise RuntimeError("Nenhum chunk gerado. Verifique se o PDF tem texto legível.")

                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                index = VectorStoreIndex.from_vector_store(
                    vector_store=vector_store,
                    storage_context=storage_context,
                )

                os.makedirs(CAMINHO_INDICE, exist_ok=True)
                index.storage_context.persist(persist_dir=CAMINHO_INDICE)

                with open(CAMINHO_NODES, "wb") as f:
                    pickle.dump(aria.nodes_globais, f)

                with open(caminho_hash, "w") as f:
                    f.write(hash_atual)

                print(f"💾 Índice salvo — {len(aria.nodes_globais)} chunks persistidos")

        if not aria.nodes_globais:
            raise RuntimeError("Nenhum node encontrado. Verifique os PDFs em 'documentos'.")

        print(f"📄 {len(aria.nodes_globais)} chunks disponíveis.")

        # Ficha do documento (título/autor) — sempre presente, em qualquer PDF.
        try:
            _injetar_ficha(chroma_collection)
        except Exception as exc:
            print(f"⚠️  Não consegui montar a ficha do documento: {exc}")

        # Recuperamos MAIS candidatos (RECUPERAR_TOP_K) para dar material ao
        # reranker. Ele reduz para os RERANK_TOP_N melhores depois.
        vector_retriever = index.as_retriever(similarity_top_k=RECUPERAR_TOP_K)
        bm25_retriever   = BM25Retriever.from_defaults(
            nodes=aria.nodes_globais, similarity_top_k=RECUPERAR_TOP_K,
        )
        hybrid_retriever = QueryFusionRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            similarity_top_k=RECUPERAR_TOP_K,
            num_queries=1,
            mode="reciprocal_rerank",
            use_async=False,
        )

        aria.retriever_debug  = hybrid_retriever
        aria.vector_retriever = vector_retriever

        # Cross-encoder de reranking (na CPU — não disputa a VRAM da GPU). Se o
        # modelo não estiver em cache (modo offline), seguimos SEM rerank em vez
        # de derrubar o serviço — o retrieval híbrido continua funcionando.
        try:
            print(f"🎯 Carregando reranker ({RERANKER_MODEL}) na CPU...")
            aria.reranker = SentenceTransformerRerank(
                model=RERANKER_MODEL,
                top_n=RERANK_TOP_N,
                device="cpu",
            )
            print("✅ Reranker pronto.")
        except Exception as exc:
            aria.reranker = None
            print(f"⚠️  Reranker indisponível (seguindo sem rerank): {exc}")

        rerank_status = "com rerank" if aria.reranker is not None else "SEM rerank"
        print(f"✅ RAG pronto! Hybrid search (vetorial + BM25/RRF) + {rerank_status} ativo. (geração de texto = ia_service)")

    except Exception:
        print("\n" + "=" * 60)
        print("❌ ERRO AO INICIAR O RAG:")
        print(traceback.format_exc())
        print("=" * 60 + "\n")
        raise


# ---------------------------------------------------------------------------
# Lifespan + App
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        iniciar_sistema_rag()
    except Exception:
        print("⚠️  ARIA iniciou com erros — verifique os logs acima.")
    yield


app = FastAPI(
    title="ARIA — Assistente de Revisão Inteligente Acadêmica",
    description="Tutor virtual que lê PDFs de aula e responde perguntas dos estudantes.",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
# NOTA: os endpoints de geração de texto (/mcp/tools/call, /perguntar, /sintetizar)
# foram REMOVIDOS deste serviço. A geração agora vive no ia_service (porta 8004).
# Este serviço expõe apenas RETRIEVAL: /contexto e /debug.


@app.delete("/cache")
def limpar_cache():
    aria.cache_respostas.clear()
    return {"mensagem": "Cache limpo com sucesso."}


@app.post("/debug")
def debug_retrieval(pergunta: Pergunta):
    if aria.retriever_debug is None:
        raise HTTPException(status_code=500, detail="A ARIA não foi inicializada.")
    nodes = aria.retriever_debug.retrieve(pergunta.texto)
    return [
        {
            "rank":    i + 1,
            "score":   round(float(n.score), 4) if n.score is not None else 0.0,
            "arquivo": n.node.metadata.get("file_name", "?"),
            "pagina":  n.node.metadata.get("page_label", n.node.metadata.get("page", "?")),
            "trecho":  n.node.get_content()[:400],
        }
        for i, n in enumerate(nodes)
    ]


# ---------------------------------------------------------------------------
# Endpoints de orquestração (usados pelo API Gateway para o roteamento)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Resumo do documento inteiro
#
# A busca por similaridade é ótima pra perguntas pontuais, mas RUIM pra "faça um
# resumo do PDF": a query é genérica e acaba puxando a ficha (título/autor) em vez
# de conteúdo. Para esses casos, montamos o contexto com uma AMOSTRA AMPLA dos
# chunks de conteúdo, distribuída ao longo do documento (início→meio→fim).
# ---------------------------------------------------------------------------
def _sem_acentos(texto: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", texto or "")
        if unicodedata.category(c) != "Mn"
    ).lower()


def _eh_pedido_resumo(texto: str) -> bool:
    t = _sem_acentos(texto)
    chaves = (
        "resum", "sintese", "sintetiz", "visao geral", "panorama",
        "sobre o que", "do que trata", "do que se trata", "de que trata",
        "de que se trata", "ideia geral", "fala sobre o que", "aborda o que",
        "principais pontos", "principais conceitos", "do que fala",
        # "relatório/redigir/dissertação sobre o material" = resumo longo e
        # estruturado: precisa da MESMA amostra ampla do documento. Sem isto a
        # query meta casa mal no reranker e a rota cai (erroneamente) na web.
        "relatorio", "redig", "disserta", "trabalho sobre", "texto sobre",
    )
    return any(k in t for k in chaves)


def _eh_pedido_questoes(texto: str) -> bool:
    """Pedido para GERAR questões/exercícios/quiz a partir do material.

    Esses pedidos são 'meta' ("elabore questões sobre o conteúdo") — casam mal com
    o texto do PDF, então o reranker dá score baixo e a rota cairia errado na web.
    Como o resumo, precisam de uma AMOSTRA AMPLA do documento (não top-k de uma
    query genérica) — por isso reaproveitam o caminho de _contexto_para_resumo.
    """
    t = _sem_acentos(texto)
    # Substantivos fortes: por si só já indicam pedido de questões.
    nomes = (
        "questao", "questoes", "exercicio", "exercicios", "quiz",
        "simulado", "questionario", "multipla escolha", "verdadeiro ou falso",
    )
    if any(k in t for k in nomes):
        return True
    # "perguntas" é ambíguo ("tenho uma pergunta") — só conta com verbo de criação.
    verbos = ("elabor", "criar", "crie", "faca", "fazer", "monte", "montar",
              "gere", "gerar", "formul", "prepar")
    if "pergunta" in t and any(v in t for v in verbos):
        return True
    return False


# ---------------------------------------------------------------------------
# Formatação do trecho de contexto
#
# Os trechos vão para o LLM SEM rótulo de página. O grounding é garantido pelo
# reranker (top-N altamente relevante) + a regra "baseie-se estritamente no
# contexto" do prompt. Decidimos NÃO marcar a página de origem porque isso fazia
# o modelo poluir as respostas com citações ('(p. 29)', '[Página 114]') que o
# estudante não quer ver. O metadado de página continua disponível no /debug.
# ---------------------------------------------------------------------------
def _formatar_trecho(content: str, metadata: dict) -> str:
    return content


def _truncar(texto: str, limite: int) -> str:
    """Corta um trecho no limite de caracteres, sem partir palavra no meio."""
    texto = (texto or "").strip()
    if len(texto) <= limite:
        return texto
    return texto[:limite].rsplit(" ", 1)[0].rstrip() + " (...)"


def _contexto_para_resumo(max_chunks: int = 16, limite_chars: int = 8500,
                          cap_por_trecho: int = 500) -> dict:
    """Amostra ampla do CONTEÚDO (sem a ficha), distribuída por TODO o documento.

    Estratégia: amostrar MUITOS pontos (início → meio → fim) e TRUNCAR cada trecho,
    em vez de pegar poucos trechos inteiros. Antes (6 trechos inteiros), o teto de
    caracteres era atingido já nos primeiros — que, por estarem ordenados por
    página, eram as PRIMEIRAS páginas. O resto do livro nunca chegava ao LLM, e o
    resumo parecia 'ler só o começo'. Truncar garante que meio e fim também entrem.

    Dois objetivos, ambos limitados pela janela FIXA do Qwen 14B no T4 (num_ctx não
    pode crescer — OLLAMA_CONTEXT_LENGTH=8192 joga o modelo pra CPU):
      • COBERTURA: 16 trechos espalhados pelo livro inteiro (não só o começo).
      • RESPOSTA MAIS LONGA: contexto compacto (~8k) deixa mais espaço de SAÍDA
        dentro da janela → o resumo pode crescer. 16 × 500 = 8000 < 8500: todos entram.
    """
    conteudo = [
        n for n in aria.nodes_globais
        if n.metadata.get("tipo") != "ficha_documento"
    ]

    # Ordena por página quando o metadado existir (mantém a ordem do documento)
    def _pagina(n):
        p = n.metadata.get("page_label", n.metadata.get("page", 0))
        try:
            return int(p)
        except (TypeError, ValueError):
            return 0
    conteudo.sort(key=_pagina)

    # Amostragem uniforme: início, meio e fim — cobre o documento todo
    if len(conteudo) > max_chunks:
        passo       = len(conteudo) / max_chunks
        selecionados = [conteudo[int(i * passo)] for i in range(max_chunks)]
    else:
        selecionados = conteudo

    # Trunca cada trecho e respeita o teto total. Com max_chunks*cap < limite_chars,
    # TODAS as amostras (do começo ao fim do livro) entram — o break é só salvaguarda.
    trechos, total = [], 0
    for n in selecionados:
        c = _truncar(n.get_content(), cap_por_trecho)
        if trechos and total + len(c) > limite_chars:
            break
        trechos.append(c)
        total += len(c)

    # Anexa a ficha (título/autor) — assim, se o estudante pedir resumo E autor/título
    # na mesma mensagem, o modelo tem as duas informações.
    ficha = next(
        (n for n in aria.nodes_globais if n.metadata.get("tipo") == "ficha_documento"),
        None,
    )
    if ficha is not None:
        trechos.append(_truncar(ficha.get_content(), 700))

    print(f"📝 /contexto (RESUMO): {len(trechos)} trechos de {len(conteudo)} "
          f"(amostra ampla início→fim, ~{total} chars + ficha)")
    return {
        "score_max":    1.0,   # força a rota do MATERIAL (resumo vem do PDF)
        "rerank_score": 1.0,   # idem — resumo sempre vem do material
        "trechos":      trechos,
        "contexto":     "\n\n---\n\n".join(trechos),
    }


@app.post("/contexto")
def obter_contexto(pergunta: Pergunta):
    """
    Recupera o contexto dos PDFs e devolve um score de relevância CALIBRADO.

    - 'trechos'/'contexto': vêm do retriever híbrido (vetorial + BM25) — melhor recall.
    - 'score_max': similaridade de cosseno (0-1) do melhor match VETORIAL puro.
      É esse score que o Gateway usa pra decidir entre PDFs e busca na web.
      (Os scores do retriever híbrido são RRF e NÃO servem de limiar — ficam ~0.01-0.03.)

    Pedidos de RESUMO usam um caminho próprio (amostra ampla do conteúdo).
    """
    # Nenhum material indexado ainda (RAG subiu vazio, antes de qualquer upload):
    # devolve score 0 e contexto vazio para o Gateway cair na busca web — em vez
    # de derrubar a conversa com um erro 500.
    if aria.retriever_debug is None or aria.vector_retriever is None:
        print("📭 /contexto: sem material indexado — devolvendo score 0 (Gateway usará a web).")
        return {"score_max": 0.0, "rerank_score": 0.0, "trechos": [], "contexto": ""}

    # Caminho especial: resumo OU geração de questões/exercícios/quiz → amostra
    # ampla do conteúdo e rota FORÇADA no material (senão a query meta cai na web).
    if _eh_pedido_resumo(pergunta.texto) or _eh_pedido_questoes(pergunta.texto):
        return _contexto_para_resumo()

    # 1) Híbrido recupera MUITOS candidatos (RECUPERAR_TOP_K).
    nodes = aria.retriever_debug.retrieve(pergunta.texto)

    # 2) Reranker reordena por relevância real e fica só com os melhores
    #    (RERANK_TOP_N). Se o reranker não carregou, mantemos a ordem do híbrido
    #    e cortamos manualmente no top-N para não inundar o LLM de trechos.
    #
    #    rerank_score = relevância do MELHOR trecho após o rerank (0-1). É um
    #    sinal MUITO mais calibrado que o cosseno cru para o Gateway decidir entre
    #    material e web. Fica None se o reranker estiver indisponível — aí o
    #    Gateway cai no cosseno (score_max) como antes.
    if aria.reranker is not None:
        nodes        = aria.reranker.postprocess_nodes(nodes, query_str=pergunta.texto)
        # float() nativo: o cross-encoder devolve numpy.float32 e o FastAPI NÃO
        # serializa numpy em JSON (ValueError → 500 "Internal Server Error" em
        # TODA pergunta pontual). round() de np.float32 continua np.float32, então
        # o cast tem de vir ANTES.
        rerank_score = round(float(max((n.score or 0.0 for n in nodes), default=0.0)), 4)
    else:
        nodes        = nodes[:RERANK_TOP_N]
        rerank_score = None

    # 3) Cada trecho sai marcado com a página de origem (grounding verificável).
    trechos = [_formatar_trecho(n.node.get_content(), n.node.metadata) for n in nodes]

    vnodes    = aria.vector_retriever.retrieve(pergunta.texto)
    # float() nativo pelo mesmo motivo do rerank_score acima (scores podem vir numpy).
    score_max = float(max((n.score or 0.0 for n in vnodes), default=0.0))

    print(f"🧭 /contexto: score_max={score_max:.4f} | rerank_score={rerank_score} | {len(trechos)} trechos (após rerank)")
    return {
        "score_max":    round(score_max, 4),
        "rerank_score": rerank_score,
        "trechos":      trechos,
        "contexto":     "\n\n---\n\n".join(trechos),
    }


@app.post("/upload")
async def upload_pdf(arquivo: UploadFile = File(...)):
    """
    Recebe um PDF enviado pelo usuário (frontend → Gateway → aqui), aplica a
    política 'um livro por vez' (limpa a pasta antes), salva o novo arquivo e
    reindexa. A reindexação roda em threadpool para não travar o event loop —
    pode levar alguns minutos num PDF grande (embeddings na GPU).
    """
    nome = arquivo.filename or ""
    if not nome.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Envie um arquivo .pdf.")

    conteudo = await arquivo.read()
    if not conteudo:
        raise HTTPException(status_code=400, detail="O arquivo enviado está vazio.")

    os.makedirs(CAMINHO_DOCS, exist_ok=True)

    # Política "um livro por vez": remove qualquer PDF antigo da pasta antes de salvar.
    for f in os.listdir(CAMINHO_DOCS):
        caminho = os.path.join(CAMINHO_DOCS, f)
        if os.path.isfile(caminho):
            try:
                os.remove(caminho)
            except OSError as exc:
                raise HTTPException(
                    status_code=500,
                    detail=f"Não consegui remover o material anterior ('{f}'): {exc}",
                )

    destino = os.path.join(CAMINHO_DOCS, os.path.basename(nome))
    with open(destino, "wb") as out:
        out.write(conteudo)
    print(f"📤 Upload recebido: {nome} ({len(conteudo)} bytes). Reindexando...")

    # Reindexa: iniciar_sistema_rag() detecta o hash mudado, limpa o índice antigo
    # (storage + ChromaDB) e indexa o PDF novo. Roda fora do event loop.
    try:
        await run_in_threadpool(iniciar_sistema_rag)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Falha ao indexar o PDF: {exc}")

    aria.cache_respostas.clear()
    return {
        "mensagem": f"'{os.path.basename(nome)}' indexado com sucesso.",
        "chunks":   len(aria.nodes_globais),
    }


@app.post("/reindexar")
def reindexar():
    if not os.path.exists(CAMINHO_DOCS) or not os.listdir(CAMINHO_DOCS):
        raise HTTPException(status_code=400, detail="Nenhum PDF encontrado em 'documentos'.")
    import shutil
    if os.path.exists(CAMINHO_INDICE): shutil.rmtree(CAMINHO_INDICE)
    if os.path.exists(CAMINHO_CHROMA): shutil.rmtree(CAMINHO_CHROMA)
    aria.cache_respostas.clear()
    aria.nodes_globais = []
    iniciar_sistema_rag()
    return {"mensagem": "Reindexação concluída com sucesso."}


@app.get("/ficha")
def obter_ficha():
    """Título/autor do PDF atualmente indexado. O Gateway usa o título como
    ÂNCORA de tema ao cair na busca web (ex.: pedido de 'livros da área'),
    para que os resultados fiquem no assunto do material e não aleatórios.
    Devolve {} quando não há material indexado."""
    return aria.ficha or {}


@app.get("/health")
def health():
    return {
        "status":         "ok",
        "servico":        "RAG — retrieval (busca no material)",
        "rag_carregado":  aria.retriever_debug is not None,
        "chunks":         len(aria.nodes_globais),
        "workers_thread": THREAD_WORKERS,
        "tamanho_lote":   TAMANHO_LOTE,
    }


# ---------------------------------------------------------------------------
# Entrada principal
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    # Passa o objeto `app` direto (não o texto "main:app") para o uvicorn NÃO
    # reimportar o módulo — senão o modelo de embeddings carrega 2x e estoura
    # a VRAM da GPU de 4GB, caindo em "spillover" para a RAM (lentíssimo).
    uvicorn.run(app, host="127.0.0.1", port=8001, reload=False)