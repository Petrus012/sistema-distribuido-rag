import os
import hashlib
import pickle
import traceback
import httpx

from cachetools import TTLCache

from contextlib import asynccontextmanager
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    PromptTemplate,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from typing import List, Optional
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.llms.openai_like import OpenAILike as _OpenAILikeBase
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.readers.file import PDFReader
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# ---------------------------------------------------------------------------
# Configurações gerais
# ---------------------------------------------------------------------------
CAMINHO_DOCS   = "../documentos"
CAMINHO_INDICE = "./storage"
CAMINHO_CHROMA = "./chroma_db"
CAMINHO_NODES  = "./storage/nodes.pkl"
# Cache fixo do modelo de embeddings (dentro do projeto, não na pasta Temp do
# Windows). Garante que o modelo (~2GB) seja baixado UMA vez e reusado sempre,
# sem re-download a cada startup.
CAMINHO_MODELO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache")

# ---------------------------------------------------------------------------
# 1. LLM — conecta ao Colab via ngrok (URL definida no .env)
# ---------------------------------------------------------------------------
NGROK_URL = os.getenv("NGROK_URL", "http://localhost:11434")

# ---------------------------------------------------------------------------
# CORREÇÃO PRINCIPAL: header ngrok-skip-browser-warning
#
# O ngrok retorna 403 quando o header está ausente OU quando o túnel expirou.
# Esta classe injeta o header obrigatório em TODAS as requisições ao Ollama.
# Se ainda receber 403, o túnel expirou — atualize NGROK_URL no arquivo .env.
# ---------------------------------------------------------------------------
class OllamaViaNgrok(_OpenAILikeBase):
    """OpenAILike com header ngrok-skip-browser-warning injetado em todas as requisições."""

    # Timeout granular: conexão falha rápido (10s) se o Colab estiver inacessível;
    # leitura espera até 90s pela geração. SEM retries (max_retries=0) para não
    # multiplicar o tempo de espera quando o Colab trava (antes: 120s x 3 = 360s).
    _TIMEOUT = httpx.Timeout(connect=10.0, read=90.0, write=10.0, pool=10.0)

    def _get_client(self):
        import openai
        return openai.OpenAI(
            api_key=self.api_key or "ollama",
            base_url=self.api_base,
            max_retries=0,
            http_client=httpx.Client(
                headers={"ngrok-skip-browser-warning": "true"},
                timeout=self._TIMEOUT,
            ),
        )

    def _get_aclient(self):
        import openai
        return openai.AsyncOpenAI(
            api_key=self.api_key or "ollama",
            base_url=self.api_base,
            max_retries=0,
            http_client=httpx.AsyncClient(
                headers={"ngrok-skip-browser-warning": "true"},
                timeout=self._TIMEOUT,
            ),
        )


# ---------------------------------------------------------------------------
# 2./3. Carregamento dos modelos (LLM + Embeddings) + chunking
#
# IMPORTANTE: este carregamento NÃO fica no nível do módulo de propósito.
# Com reload=True, o uvicorn importa o main.py em DOIS processos (o vigia do
# WatchFiles e o worker). Se o FastEmbed estivesse no topo do arquivo, o modelo
# seria carregado na memória uma vez em cada processo (e a cada reload).
# Colocando aqui dentro, ele é carregado UMA única vez, no startup do worker.
# O guard `_modelos_configurados` evita recarregar em chamadas de /reindexar.
# ---------------------------------------------------------------------------
_modelos_configurados = False


def _configurar_modelos() -> None:
    """Carrega LLM e modelo de embeddings uma única vez (lazy)."""
    global _modelos_configurados
    if _modelos_configurados:
        return

    print(f"🔗 Conectando ao LLM em: {NGROK_URL}")
    Settings.llm = OllamaViaNgrok(
        model="qwen2.5:14b",
        api_base=f"{NGROK_URL}/v1",
        api_key="ollama",
        temperature=0.1,
        context_window=4096,
        max_tokens=2048,
        is_chat_model=True,
        is_function_calling_model=False,
    )

    print("📥 Carregando modelo de embeddings local (FastEmbed)...")
    Settings.embed_model = FastEmbedEmbedding(
        model_name="intfloat/multilingual-e5-large",
        cache_dir=CAMINHO_MODELO,
    )

    Settings.text_splitter = SentenceSplitter(
        chunk_size=1024,
        chunk_overlap=128,
    )

    _modelos_configurados = True

# ---------------------------------------------------------------------------
# 4. Templates de prompt
#
# Os templates abaixo contêm o placeholder {ficha_material}, que é preenchido
# DINAMICAMENTE no startup (via partial_format) com o autor e o título do PDF
# carregado. Assim, se o material for trocado, a ARIA passa a conhecer o novo
# autor/título automaticamente, sem editar o código.
# ---------------------------------------------------------------------------
TEXT_QA_TEMPLATE_BASE = PromptTemplate(
    "Você é a ARIA, uma assistente de estudos especializada em analisar e explicar materiais acadêmicos.\n"
    "REGRA ABSOLUTA: Responda SEMPRE em português do Brasil. NUNCA em inglês.\n"
    "Seu nome é ARIA. Se perguntarem quem você é, diga que é a ARIA, assistente de estudos.\n"
    "\n"
    "MATERIAL QUE VOCÊ ESTÁ LENDO (ficha técnica):\n"
    "{ficha_material}\n"
    "Se perguntarem o AUTOR ou o TÍTULO/nome do material, responda com base nesta ficha técnica.\n"
    "Se a ficha não trouxer o dado, procure no início do documento mostrado nela.\n"
    "\n"
    "REGRAS DE PRECISÃO:\n"
    "- Baseie-se ESTRITAMENTE no contexto e na ficha técnica abaixo. NÃO invente.\n"
    "- Seja preciso e específico: cite números, nomes, datas e termos EXATAMENTE como aparecem no material.\n"
    "- Se a resposta não estiver no contexto, diga: 'Não encontrei essa informação no documento.'\n"
    "- Vá direto ao conteúdo. NÃO use introduções ('Sim, posso...') nem conclusões genéricas.\n"
    "- Para resumos longos, escreva parágrafos detalhados cobrindo cada conceito do texto.\n"
    "\n"
    "CONTEXTO:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "\n"
    "PERGUNTA: {query_str}\n"
    "\n"
    "RESPOSTA DETALHADA E PRECISA EM PORTUGUÊS DO BRASIL:"
)

SUMMARY_TEMPLATE_BASE = PromptTemplate(
    "Você é a ARIA, assistente de estudos. Responda SEMPRE em português do Brasil. NUNCA em inglês.\n"
    "Vá direto ao conteúdo. NÃO use introduções ou conclusões genéricas.\n"
    "\n"
    "MATERIAL QUE VOCÊ ESTÁ LENDO (ficha técnica):\n"
    "{ficha_material}\n"
    "Se perguntarem o autor ou o título, responda com base nesta ficha técnica.\n"
    "\n"
    "Abaixo estão trechos do documento acadêmico:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "\n"
    "Com base APENAS nos trechos acima e na ficha técnica, responda em português do Brasil:\n"
    "{query_str}\n"
    "\n"
    "RESPOSTA EM PORTUGUÊS:"
)

# Template de refine em PT-BR — o modo "compact" recorre ao refine quando os
# trechos não cabem numa única chamada. O padrão do LlamaIndex é em inglês e
# quebraria a regra do português, por isso definimos o nosso.
REFINE_TEMPLATE = PromptTemplate(
    "Você é a ARIA, assistente de estudos. Responda SEMPRE em português do Brasil. NUNCA em inglês.\n"
    "Já existe uma resposta parcial para a pergunta:\n"
    "{existing_answer}\n"
    "\n"
    "Há mais contexto do documento disponível abaixo:\n"
    "---------------------\n"
    "{context_msg}\n"
    "---------------------\n"
    "Use o novo contexto para REFINAR a resposta. Se o novo contexto não ajudar, "
    "repita a resposta original sem alterá-la. NÃO invente informações fora do documento.\n"
    "\n"
    "PERGUNTA: {query_str}\n"
    "RESPOSTA REFINADA E PRECISA EM PORTUGUÊS DO BRASIL:"
)

# ---------------------------------------------------------------------------
# Variáveis globais
# ---------------------------------------------------------------------------
query_engine    = None
retriever_debug = None
# Cache com limite de tamanho e expiração — evita crescimento ilimitado da RAM.
# maxsize=500 respostas; ttl=3600s (1h). Ao estourar, descarta as mais antigas.
cache_respostas: TTLCache = TTLCache(maxsize=500, ttl=3600)
nodes_globais   = []
ficha_material  = "Nenhum material carregado."


# ---------------------------------------------------------------------------
# Modelos Pydantic
# ---------------------------------------------------------------------------
class Pergunta(BaseModel):
    texto: str

class MCPToolCall(BaseModel):
    tool: str
    arguments: dict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _calcular_hash_documentos(caminho: str) -> str:
    """Calcula um hash MD5 de todos os arquivos na pasta de documentos."""
    hasher = hashlib.md5()
    for nome in sorted(os.listdir(caminho)):
        filepath = os.path.join(caminho, nome)
        if os.path.isfile(filepath):
            with open(filepath, "rb") as f:
                hasher.update(f.read())
    return hasher.hexdigest()


def _extrair_info_material(caminho: str) -> str:
    """
    Extrai título e autor de cada PDF na pasta de documentos e monta uma
    "ficha técnica" textual para injetar no prompt da ARIA.

    Estratégia (independente de qual material esteja carregado):
      1. Lê os metadados embutidos no PDF (campos Title e Author).
      2. Se o título estiver vazio, usa o nome do arquivo como fallback.
      3. Se título OU autor estiverem ausentes, anexa um trecho do início do
         documento (onde normalmente aparecem título e autor) para a ARIA
         conseguir responder mesmo assim.
    """
    from pypdf import PdfReader

    if not os.path.exists(caminho) or not os.listdir(caminho):
        return "Nenhum material carregado."

    fichas = []
    for nome in sorted(os.listdir(caminho)):
        filepath = os.path.join(caminho, nome)
        if not os.path.isfile(filepath) or not nome.lower().endswith(".pdf"):
            continue

        titulo = None
        autor = None
        inicio = ""
        try:
            reader = PdfReader(filepath)
            meta = reader.metadata
            if meta:
                titulo = (meta.title or "").strip() or None
                autor = (meta.author or "").strip() or None
            if reader.pages:
                inicio = (reader.pages[0].extract_text() or "").strip()
        except Exception as exc:
            # PDF protegido/corrompido — segue com os fallbacks, mas avisa.
            print(f"⚠️  Não consegui ler metadados de '{nome}': {exc}")

        if not titulo:
            titulo = os.path.splitext(nome)[0]  # fallback: nome do arquivo

        ficha = (
            f"- Arquivo: {nome}\n"
            f"  Título: {titulo}\n"
            f"  Autor: {autor or 'não informado nos metadados'}"
        )
        # Trecho de apoio só quando o autor faltou nos metadados.
        # (titulo nunca é vazio aqui — já recebeu o nome do arquivo como fallback.)
        if autor is None and inicio:
            ficha += (
                "\n  Início do documento (use para identificar título/autor "
                f"caso acima estejam ausentes):\n  \"{inicio[:500]}\""
            )
        fichas.append(ficha)

    return "\n".join(fichas) if fichas else "Nenhum material carregado."


def _verificar_conexao_ngrok() -> bool:
    """
    Testa a conexão com o túnel ngrok.
    Retorna True se OK, False se o túnel expirou ou está inacessível.
    """
    print(f"📡 Verificando conexão com o Google Colab ({NGROK_URL})...")
    try:
        resposta = httpx.get(
            NGROK_URL,
            timeout=5.0,
            headers={"ngrok-skip-browser-warning": "true"},
            follow_redirects=True,
        )
        # Ollama responde com 200 e "Ollama is running" no corpo
        if resposta.status_code == 200 and "Ollama is running" in resposta.text:
            print("🚀 Conexão com o Google Colab estabelecida com sucesso!")
            return True

        # 403 = túnel expirou ou URL errada (não é problema de header)
        if resposta.status_code == 403:
            print("\n" + "!" * 60)
            print("❌ ERRO 403: O túnel ngrok expirou ou a URL está errada!")
            print("👉 O que fazer:")
            print("   1. Abra o notebook no Google Colab e execute novamente a célula do ngrok.")
            print("   2. Copie a nova URL gerada (ex: https://xxxx-xxxx.ngrok-free.app).")
            print("   3. Atualize o arquivo .env:  NGROK_URL=https://nova-url.ngrok-free.app")
            print("   4. Reinicie este servidor (Ctrl+C e python main.py novamente).")
            print("!" * 60 + "\n")
            return False

        print(f"⚠️  URL respondeu com status inesperado: {resposta.status_code}")
        return False

    except httpx.RequestError as exc:
        print("\n" + "!" * 60)
        print("❌ ERRO: Não foi possível conectar ao Google Colab!")
        print(f"   Detalhe: {exc}")
        print("!" * 60 + "\n")
        return False


# ---------------------------------------------------------------------------
# Startup (lifespan — substitui o on_event depreciado)
# ---------------------------------------------------------------------------
def iniciar_sistema_rag():
    global query_engine, retriever_debug, nodes_globais, ficha_material

    print("⏳ Iniciando Microserviço de RAG...")

    _configurar_modelos()
    _verificar_conexao_ngrok()

    try:
        # --- ChromaDB -------------------------------------------------------
        chroma_client     = chromadb.PersistentClient(path=CAMINHO_CHROMA)
        chroma_collection = chroma_client.get_or_create_collection("documentos")
        vector_store      = ChromaVectorStore(chroma_collection=chroma_collection)

        hash_atual = _calcular_hash_documentos(CAMINHO_DOCS) if (
            os.path.exists(CAMINHO_DOCS) and os.listdir(CAMINHO_DOCS)
        ) else ""

        hash_salvo = ""
        caminho_hash = os.path.join(CAMINHO_INDICE, "docs.hash")
        if os.path.exists(caminho_hash):
            with open(caminho_hash, "r") as f:
                hash_salvo = f.read().strip()

        documentos_mudaram = hash_atual != hash_salvo

        indice_existe = (
            os.path.exists(CAMINHO_NODES)
            and os.path.exists(CAMINHO_INDICE)
            and os.listdir(CAMINHO_INDICE)
            and chroma_collection.count() > 0
            and not documentos_mudaram
        )

        if documentos_mudaram and os.path.exists(CAMINHO_INDICE):
            import shutil
            print("🔄 Documentos alterados! Limpando índice antigo e reindexando...")
            shutil.rmtree(CAMINHO_INDICE)
            shutil.rmtree(CAMINHO_CHROMA, ignore_errors=True)
            # IMPORTANTE: recria o PersistentClient. O antigo apontava para a
            # pasta que acabamos de apagar — reutilizá-lo daria erro/comportamento
            # indefinido. Recriar garante um cliente apontando para o diretório novo.
            chroma_client     = chromadb.PersistentClient(path=CAMINHO_CHROMA)
            chroma_collection = chroma_client.get_or_create_collection("documentos")
            vector_store      = ChromaVectorStore(chroma_collection=chroma_collection)

        if indice_existe:
            print("📦 Carregando índice existente do disco (startup rápido)...")
            with open(CAMINHO_NODES, "rb") as f:
                nodes_globais = pickle.load(f)
            storage_context = StorageContext.from_defaults(
                persist_dir=CAMINHO_INDICE,
                vector_store=vector_store,
            )
            index = load_index_from_storage(storage_context)

        else:
            if not os.path.exists(CAMINHO_DOCS) or not os.listdir(CAMINHO_DOCS):
                print("❌ ERRO: Coloque um PDF na pasta 'documentos'!")
                return

            print("🔨 Criando índice pela primeira vez (pode demorar)...")
            leitores_customizados = {".pdf": PDFReader()}
            documentos = SimpleDirectoryReader(
                CAMINHO_DOCS,
                file_extractor=leitores_customizados,
            ).load_data()

            parser        = SentenceSplitter(chunk_size=1024, chunk_overlap=128)
            nodes_globais = parser.get_nodes_from_documents(documentos)

            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex(
                nodes_globais,
                storage_context=storage_context,
            )

            os.makedirs(CAMINHO_INDICE, exist_ok=True)
            index.storage_context.persist(persist_dir=CAMINHO_INDICE)

            with open(CAMINHO_NODES, "wb") as f:
                pickle.dump(nodes_globais, f)

            with open(os.path.join(CAMINHO_INDICE, "docs.hash"), "w") as f:
                f.write(hash_atual)

            print(f"💾 Índice salvo em '{CAMINHO_INDICE}' ({len(nodes_globais)} chunks)")

        if not nodes_globais:
            print("❌ ERRO: Nenhum node encontrado. Verifique os PDFs em 'documentos'.")
            return

        print(f"📄 {len(nodes_globais)} chunks disponíveis.")

        # --- Ficha técnica do material (autor/título dinâmicos) -------------
        ficha_material = _extrair_info_material(CAMINHO_DOCS)
        print(f"📑 Ficha do material carregada:\n{ficha_material}")

        # --- Hybrid Search: vetorial + BM25 ---------------------------------
        vector_retriever = index.as_retriever(similarity_top_k=8)
        bm25_retriever   = BM25Retriever.from_defaults(
            nodes=nodes_globais,
            similarity_top_k=8,
        )

        hybrid_retriever = QueryFusionRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            similarity_top_k=5,
            num_queries=1,
            mode="reciprocal_rerank",
            use_async=False,
        )

        retriever_debug = hybrid_retriever

        # --- Templates com a ficha do material preenchida dinamicamente -----
        text_qa_template = TEXT_QA_TEMPLATE_BASE.partial_format(ficha_material=ficha_material)
        summary_template = SUMMARY_TEMPLATE_BASE.partial_format(ficha_material=ficha_material)

        # --- Query engine ---------------------------------------------------
        # response_mode="compact": junta os trechos numa única chamada ao LLM
        # (mais rápido que "tree_summarize"); recorre ao REFINE_TEMPLATE só se
        # os trechos não couberem no contexto.
        query_engine = RetrieverQueryEngine.from_args(
            retriever=hybrid_retriever,
            node_postprocessors=[],
            response_mode="compact",
            text_qa_template=text_qa_template,
            summary_template=summary_template,
            refine_template=REFINE_TEMPLATE,
        )

        print("✅ RAG pronto! Hybrid search (vetorial + BM25/RRF) + índice persistido ativos.")

    except Exception:
        print("\n" + "=" * 60)
        print("❌ ERRO AO INICIAR O RAG:")
        print(traceback.format_exc())
        print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Lifespan (substitui o @app.on_event("startup") depreciado)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    iniciar_sistema_rag()
    yield   # servidor rodando
    # (limpeza ao desligar pode ser adicionada aqui se necessário)


app = FastAPI(
    title="Microserviço de RAG",
    description="Lê qualquer PDF e conversa com o Qwen 2.5",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/mcp/tools")
def listar_ferramentas_mcp():
    return {
        "tools": [
            {
                "name": "consultar_documentos_aula",
                "description": "Busca informações relevantes dentro dos PDFs de aula carregados no servidor.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "A pergunta ou termo de busca do aluno."}
                    },
                    "required": ["query"]
                }
            }
        ]
    }


@app.post("/mcp/tools/call")
def executar_ferramenta_mcp(call: MCPToolCall):
    if query_engine is None:
        raise HTTPException(
            status_code=500,
            detail="O servidor MCP não carregou os documentos. Verifique a pasta 'documentos'."
        )

    if call.tool != "consultar_documentos_aula":
        raise HTTPException(status_code=400, detail="Ferramenta MCP não encontrada.")

    pergunta_texto = call.arguments.get("query", "")

    chave = hashlib.md5(pergunta_texto.strip().lower().encode()).hexdigest()
    if chave in cache_respostas:
        print("⚡ Respondendo do cache (MCP)")
        resposta_final = cache_respostas[chave]
    else:
        print(f"🤔 Processando via MCP: '{pergunta_texto}'")
        try:
            resposta = query_engine.query(pergunta_texto)
            resposta_final = str(resposta)
        except Exception as exc:
            # Detecta erro 403 especificamente e dá mensagem clara
            msg = str(exc)
            if "403" in msg or "PermissionDenied" in msg or "Forbidden" in msg:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "❌ O túnel ngrok expirou (erro 403). "
                        "Gere uma nova URL no Colab, atualize o .env e reinicie o servidor."
                    ),
                )
            raise HTTPException(status_code=500, detail=f"Erro ao consultar o LLM: {msg}")

        cache_respostas[chave] = resposta_final

    return {
        "content": [
            {
                "type": "text",
                "text": resposta_final
            }
        ]
    }


@app.post("/perguntar/sem-cache")
def fazer_pergunta_sem_cache(pergunta: Pergunta):
    """Força uma nova consulta ao RAG, ignorando o cache."""
    if query_engine is None:
        raise HTTPException(status_code=500, detail="O RAG não foi inicializado.")

    print(f"🔄 Forçando consulta sem cache: '{pergunta.texto}'")
    try:
        resposta = query_engine.query(pergunta.texto)
    except Exception as exc:
        msg = str(exc)
        if "403" in msg or "PermissionDenied" in msg or "Forbidden" in msg:
            raise HTTPException(
                status_code=503,
                detail="❌ O túnel ngrok expirou (erro 403). Atualize NGROK_URL no .env e reinicie.",
            )
        raise HTTPException(status_code=500, detail=f"Erro ao consultar o LLM: {msg}")

    return {
        "pergunta":    pergunta.texto,
        "resposta_ia": str(resposta),
        "cache":       False,
    }


@app.delete("/cache")
def limpar_cache():
    """Limpa todo o cache de respostas em memória."""
    cache_respostas.clear()
    return {"mensagem": "Cache limpo com sucesso."}


@app.post("/debug")
def debug_retrieval(pergunta: Pergunta):
    """Retorna os chunks recuperados antes de passar pelo LLM."""
    if retriever_debug is None:
        raise HTTPException(status_code=500, detail="O RAG não foi inicializado.")

    nodes = retriever_debug.retrieve(pergunta.texto)
    return [
        {
            "rank":   i + 1,
            "score":  round(node.score, 4) if node.score is not None else None,
            "trecho": node.text[:400],
        }
        for i, node in enumerate(nodes)
    ]


@app.post("/reindexar")
def reindexar():
    """Recria o índice a partir dos PDFs na pasta 'documentos'."""
    global query_engine, retriever_debug, cache_respostas, nodes_globais

    if not os.path.exists(CAMINHO_DOCS) or not os.listdir(CAMINHO_DOCS):
        raise HTTPException(status_code=400, detail="Nenhum PDF encontrado em 'documentos'.")

    import shutil
    if os.path.exists(CAMINHO_INDICE):
        shutil.rmtree(CAMINHO_INDICE)
    if os.path.exists(CAMINHO_CHROMA):
        shutil.rmtree(CAMINHO_CHROMA)

    cache_respostas.clear()
    nodes_globais = []
    iniciar_sistema_rag()

    return {"mensagem": "Reindexação concluída com sucesso."}


@app.get("/health")
def health():
    return {
        "status":         "ok",
        "rag_carregado":  query_engine is not None,
        "chunks":         len(nodes_globais),
        "cache_entradas": len(cache_respostas),
        "ngrok_url":      NGROK_URL,
        "ficha_material": ficha_material,
    }


# ---------------------------------------------------------------------------
# Entrada principal
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    # reload=True ajuda no desenvolvimento, MAS no Windows trava o Ctrl+C quando
    # há uma chamada de rede pendurada (ex.: Colab lento). Por isso o padrão é
    # DESLIGADO — assim o Ctrl+C mata o servidor na hora. Para reativar o hot
    # reload em dev, defina DEV_RELOAD=true no .env.
    dev_reload = os.getenv("DEV_RELOAD", "false").lower() in ("1", "true", "yes")
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8001,
        reload=dev_reload,
        # Quando o reload está ligado, evita que gravar o índice/banco (dentro
        # desta pasta) dispare reloads em cascata e recarregue o modelo do zero.
        reload_excludes=["storage/*", "chroma_db/*", "model_cache/*", "*.pkl", "*.hash"],
    )