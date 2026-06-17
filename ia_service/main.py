"""
Microserviço de IA / Proxy — ARIA (fase 2)
------------------------------------------
Único responsável por falar com o LLM (Qwen 14B) hospedado no Google Colab
via túnel ngrok. Recebe um pedido estruturado (pergunta + contexto + histórico),
monta o prompt do ARIA e devolve a resposta gerada.

NÃO faz retrieval (isso é o RAG). NÃO acessa banco. Só prompt + LLM.

Endpoint:
  POST /gerar   → {pergunta, contexto, fonte, historico} -> {resposta}
  GET  /health  → checa a conexão com o Colab/ngrok
"""

import os
import re
import unicodedata
import httpx

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from llama_index.core import PromptTemplate
from llama_index.llms.openai_like import OpenAILike as _OpenAILikeBase

# ---------------------------------------------------------------------------
# Conexão com o LLM (Qwen no Colab via ngrok)
# ---------------------------------------------------------------------------
NGROK_URL = os.getenv("NGROK_URL", "http://localhost:11434")
print(f"🔗 Conectando ao LLM em: {NGROK_URL}")

# Timeout (segundos) das chamadas ao LLM. Respostas longas — ex.: "gere 20 questões"
# — num 14B rodando no T4 do Colab podem passar de 3 min. Por isso o default é alto.
# Ajuste por env (LLM_TIMEOUT) sem mexer no código.
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "600"))
print(f"⏱️  Timeout do LLM: {LLM_TIMEOUT}s")


class OllamaViaNgrok(_OpenAILikeBase):
    """OpenAILike com header ngrok-skip-browser-warning injetado em todas as requisições."""

    def _get_client(self):
        import openai
        return openai.OpenAI(
            api_key=self.api_key or "ollama",
            base_url=self.api_base,
            http_client=httpx.Client(
                headers={"ngrok-skip-browser-warning": "true"},
                timeout=httpx.Timeout(LLM_TIMEOUT),
            ),
        )

    def _get_aclient(self):
        import openai
        return openai.AsyncOpenAI(
            api_key=self.api_key or "ollama",
            base_url=self.api_base,
            http_client=httpx.AsyncClient(
                headers={"ngrok-skip-browser-warning": "true"},
                timeout=httpx.Timeout(LLM_TIMEOUT),
            ),
        )


llm = OllamaViaNgrok(
    model="qwen2.5:14b",
    api_base=f"{NGROK_URL}/v1",
    api_key="ollama",
    temperature=0.2,
    context_window=8192,
    max_tokens=4096,
    is_chat_model=True,
    is_function_calling_model=False,
)

# ---------------------------------------------------------------------------
# Identidade do ARIA
#
# O modelo (Qwen 2.5) tem o vício de se dizer "Claude"/"GPT" quando perguntam
# o nome — instrução solta no prompt não vence isso de forma confiável. Então:
#   • O nome é tratado de forma DETERMINÍSTICA (texto fixo, não vem do LLM).
#   • Reforçamos a identidade num SYSTEM PROMPT (peso bem maior que texto inline).
# ---------------------------------------------------------------------------
APRESENTACAO_ARIA = (
    "Olá! Meu nome é ARIA (Assistente de Revisão Inteligente Acadêmica), "
    "sua tutora virtual de estudos. 😊"
)


def _corrigir_portugues(texto: str) -> str:
    """Corrige deslizes de concordância recorrentes do modelo no texto gerado.
    Hoje cobre 'boa(s)/bom estudos' → 'bons estudos' (estudos é masculino)."""
    def _troca(m):
        return "Bons estudos" if m.group(0)[:1].isupper() else "bons estudos"
    return re.sub(r"\bbo(?:a|as|m)\s+estudos\b", _troca, texto, flags=re.IGNORECASE)


# Detecta saudação / auto-apresentação no começo da resposta (ex.: "Olá! Me chamo
# Claro...", "Oi! Eu sou o ARIA..."). Usado nas perguntas combinadas para remover
# o nome que o modelo inventa — a apresentação correta é colada por nós depois.
_SAUDACAO_RE = re.compile(
    r"(?i)\b(ol[áa]|oi|bom dia|boa tarde|boa noite|e a[íi]"
    r"|me chamo|meu nome (é|e)|eu sou|pode me chamar|sou (o|a) )\b"
)


def _remover_saudacao_inicial(texto: str) -> str:
    """Remove o primeiro parágrafo se ele for uma saudação/auto-apresentação curta
    (onde o modelo costuma inventar um nome errado). Conservador: só corta quando
    há quebra de parágrafo e o trecho inicial é curto e bate com a saudação."""
    partes = texto.split("\n\n", 1)
    if len(partes) == 2:
        primeira = partes[0].strip()
        if len(primeira) <= 200 and _SAUDACAO_RE.search(primeira):
            return partes[1].lstrip()
    return texto


def _sem_acentos(texto: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", texto or "")
        if unicodedata.category(c) != "Mn"
    ).lower()


def _limpar_markdown_questoes(texto: str) -> str:
    """Remove marcadores de negrito/itálico de respostas de QUESTÕES.

    O Qwen 14B, em listas longas de alternativas, emite '**' desbalanceados — um
    marcador que não fecha faz o renderizador 'vazar' o negrito para as próximas
    alternativas (A/B/C destacadas e D não). Como em questões NÃO queremos negrito
    nenhum, a forma à prova de falhas é tirar os asteriscos no código, em vez de
    depender de o modelo balanceá-los. Remove '**' e '*', e some com linhas que
    ficaram só de asteriscos/espaços."""
    texto = texto.replace("**", "")
    texto = re.sub(r"(?<!\w)\*(?!\w)", "", texto)        # asteriscos soltos restantes
    texto = re.sub(r"(?m)^[ \t*]+$", "", texto)          # linhas que sobraram só com lixo
    texto = re.sub(r"\n{3,}", "\n\n", texto)             # colapsa excesso de linhas em branco
    return texto.strip()


def _eh_pedido_resumo(texto: str) -> bool:
    """Pedido de RESUMO/síntese/visão geral do material. Espelha o rag_service —
    aqui serve para escolher um template ENXUTO (libera tokens para a resposta
    crescer) e específico de resumo (organização temática, não por 'Unidade')."""
    t = _sem_acentos(texto)
    chaves = (
        "resum", "sintese", "sintetiz", "visao geral", "panorama",
        "sobre o que", "do que trata", "do que se trata", "de que trata",
        "de que se trata", "ideia geral", "fala sobre o que", "aborda o que",
        "principais pontos", "principais conceitos", "do que fala",
        "relatorio", "redig", "disserta", "trabalho sobre", "texto sobre",
    )
    return any(k in t for k in chaves)


def _eh_pedido_questoes(texto: str) -> bool:
    """Pedido para GERAR questões/exercícios/quiz a partir do material.
    Espelha a detecção do rag_service — aqui serve para injetar instruções de
    formato (e anti-repetição), já que o modelo tende a clonar a mesma questão
    com outro número quando o estudante pede muitas (ex.: 'gere 20 questões')."""
    t = _sem_acentos(texto)
    nomes = (
        "questao", "questoes", "exercicio", "exercicios", "quiz",
        "simulado", "questionario", "multipla escolha", "verdadeiro ou falso",
    )
    if any(k in t for k in nomes):
        return True
    verbos = ("elabor", "criar", "crie", "faca", "fazer", "monte", "montar",
              "gere", "gerar", "formul", "prepar")
    if "pergunta" in t and any(v in t for v in verbos):
        return True
    return False


def _pediu_sem_gabarito(texto: str) -> bool:
    """O estudante pediu as questões SEM gabarito / sem as respostas?"""
    t = _sem_acentos(texto)
    return bool(re.search(
        r"sem (o |as |a )?(gabarito|respostas?|resolucao)"
        r"|nao (quero|coloque|inclua|ponha|bote)\w* (o |as |a )?(gabarito|respostas?)"
        r"|sem mostrar (a|as) respost",
        t,
    ))


def _remover_gabarito(texto: str) -> str:
    """Backstop determinístico: tira as linhas de resposta/gabarito do texto de
    questões (quando o estudante pediu sem gabarito). O modelo às vezes inclui
    'Resposta correta: c' mesmo instruído a não incluir."""
    texto = re.sub(
        r"(?im)^\s*(resposta\s+correta|resposta|gabarito)\s*[:\-].*$",
        "",
        texto,
    )
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip()


# Injetado quando o estudante pede questões. O modelo pequeno (Qwen 14B) tem dois
# vícios aqui: (1) repetir a mesma questão com outro número; (2) copiar QUALQUER
# exemplo de formato literalmente (por isso NÃO damos um esqueleto preenchível —
# ele saía "Enunciado da questão aqui?" / "a) primeira alternativa"). As regras
# abaixo descrevem o formato em palavras, exigem conteúdo real e respeitam o tipo
# (aberta/fechada) e o "sem gabarito" que o estudante pedir.
QUESTOES_INSTRUCAO = (
    "INSTRUÇÕES PARA GERAR QUESTÕES (SIGA À RISCA):\n"
    "- Gere EXATAMENTE a quantidade de questões que o estudante pediu — se pediu 20, "
    "entregue 20, nem mais nem menos. Numere em sequência (1, 2, 3, ...).\n"
    "- CONTEÚDO REAL: cada enunciado deve ser escrito de verdade, sobre um conceito "
    "CONCRETO presente no material. É TERMINANTEMENTE PROIBIDO usar textos de "
    "preenchimento como 'Enunciado da questão aqui', 'primeira alternativa', "
    "'segunda alternativa' etc. — escreva sempre o conteúdo de verdade.\n"
    "- Cada questão deve ser ÚNICA e cobrir um ponto DIFERENTE do material. NÃO repita "
    "enunciados nem alternativas, nem reaproveite a mesma questão com outro número. "
    "Distribua as questões por TODO o conteúdo, sem insistir nos mesmos 2 ou 3 temas.\n"
    "- TIPO DE QUESTÃO (respeite o que o estudante pediu):\n"
    "   • Questões FECHADAS / múltipla escolha: dê um enunciado e 4 alternativas reais "
    "(a, b, c, d), todas plausíveis e diferentes entre si, apenas uma correta.\n"
    "   • Questões ABERTAS / dissertativas: escreva só o enunciado, SEM alternativas.\n"
    "   • Se o estudante pediu 'abertas e fechadas' (ou 'mistas'), ALTERNE os dois tipos "
    "ao longo da lista.\n"
    "- GABARITO: por padrão, após cada questão FECHADA escreva a resposta numa linha "
    "própria no formato 'Resposta correta: c'. MAS, se o estudante pedir SEM gabarito "
    "(sem as respostas), NÃO inclua nenhuma resposta, gabarito ou indicação da correta.\n"
    "- NÃO use NENHUMA formatação Markdown (sem asteriscos, negrito ou itálico): escreva "
    "tudo em TEXTO NORMAL. NÃO cite números de página em momento algum.\n"
    "----------------------------------------\n\n"
)

# ---------------------------------------------------------------------------
# Templates de prompt — ARIA
# ---------------------------------------------------------------------------
TEXT_QA_TEMPLATE = PromptTemplate(
    "Você é a ARIA (Assistente de Revisão Inteligente Acadêmica), uma tutora virtual "
    "especializada em ajudar estudantes a compreender o conteúdo dos seus materiais de aula.\n"
    "\n"
    "REGRA ABSOLUTA: Responda SEMPRE em português do Brasil. NUNCA em inglês.\n"
    "\n"
    "SEU NOME E IDENTIDADE: Você se chama ARIA e se refere a si mesma no feminino. "
    "Se o estudante perguntar seu nome, quem você é, ou como deve te chamar, responda SEMPRE que você é a ARIA "
    "(Assistente de Revisão Inteligente Acadêmica). Isso faz parte de quem você é — "
    "você NÃO precisa (e NÃO deve) procurar essa informação no material/contexto. "
    "IGNORE qualquer outro nome que possa ter aparecido antes na conversa: seu ÚNICO nome é ARIA.\n"
    "\n"
    "SUA PERSONALIDADE:\n"
    "- Seja didático e claro, como um bom professor explicaria\n"
    "- Use exemplos quando ajudar a entender\n"
    "- Organize com estrutura E FORMATAÇÃO em Markdown: destaque os termos e conceitos-chave "
    "em **negrito** (ex.: **escassez**, **custo de oportunidade**), use listas com marcadores "
    "para enumerações e subtítulos quando ajudar a leitura. SEMPRE destaque em negrito os pontos "
    "mais importantes da resposta.\n"
    "- Seja encorajador e paciente\n"
    "- Sem enrolação no início, MAS DESENVOLVA BEM a resposta: explique com profundidade, "
    "detalhe os conceitos e traga exemplos do material — NÃO responda de forma curta ou telegráfica.\n"
    "- Por padrão, dê respostas COMPLETAS e bem desenvolvidas (vários parágrafos quando o assunto "
    "permitir), a menos que o estudante peça explicitamente algo curto/resumido.\n"
    "\n"
    "REGRAS DE CONTEÚDO (PRECISÃO É PRIORIDADE MÁXIMA):\n"
    "- Baseie-se ESTRITAMENTE nas informações do contexto abaixo\n"
    "- PRECISÃO ABSOLUTA: ao citar dados do material (números, datas, fórmulas, definições, "
    "nomes, classificações, etapas, listas), reproduza-os EXATAMENTE como estão no contexto — "
    "sem arredondar, trocar termos, inverter ordem ou alterar o sentido.\n"
    "- NÃO extrapole nem complete com conhecimento externo (de fora do material). "
    "Se algo não está no contexto, NÃO afirme.\n"
    "- Se a resposta não estiver no contexto, diga: 'Essa informação não está no material disponível.'\n"
    "- Se o contexto for ambíguo ou insuficiente, diga claramente o que o material cobre e o que NÃO cobre, "
    "em vez de chutar.\n"
    "- Nunca invente dados, fórmulas, autores ou conceitos fora do documento\n"
    "- NÃO cite números de página na resposta. NUNCA escreva referências como '(p. 29)', "
    "'(p. 12)', '[Página 114]', 'na página 57' ou 'conforme a página X'. O contexto pode conter "
    "marcações de página — IGNORE-as por completo e escreva o conteúdo de forma fluida, sem "
    "nenhuma menção a números de página.\n"
    "- NÃO estruture a resposta por 'Unidade 1', 'Unidade 2', 'Unidade 3'... nem use 'Unidade X' "
    "como título de seção. Organize por TEMAS/CONCEITOS (ex.: pensamento econômico, microeconomia, "
    "macroeconomia), escrevendo de forma fluida.\n"
    "- Para RESUMOS: percorra o material e cite os PRINCIPAIS conceitos com substância — "
    "mencione termos, definições e exemplos concretos que aparecem no contexto (ex.: escassez, "
    "custo de oportunidade, método científico, micro vs macroeconomia, fluxo circular da renda...). "
    "NÃO se limite a uma frase genérica do tipo 'o material aborda conceitos fundamentais': "
    "diga QUAIS são esses conceitos e o que o material fala sobre cada um.\n"
    "- RESPEITE a extensão pedida pelo estudante: se ele pedir 'resumo de 10 linhas', "
    "'em 3 parágrafos', 'detalhado', etc., desenvolva o texto até atingir esse tamanho — "
    "NÃO encurte. Se pediu 10 linhas, escreva de fato cerca de 10 linhas.\n"
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
    "INSTRUÇÃO FINAL DE TAMANHO: Escreva uma resposta LONGA, COMPLETA e aprofundada. "
    "Desenvolva CADA ponto relevante do contexto em parágrafos próprios, explicando o 'porquê' "
    "e o 'como', com exemplos e contexto. Esgote o assunto dentro do material — é melhor pecar "
    "por completo do que por curto. Use vários parágrafos e destaque os termos-chave em **negrito**.\n"
    "\n"
    "RESPOSTA DETALHADA E COMPLETA EM PORTUGUÊS DO BRASIL:"
)

# Template ENXUTO só para RESUMOS. O TEXT_QA_TEMPLATE é enorme (~1000 tokens de
# instruções) e, como a janela do Qwen 14B no T4 é fixa (não dá pra crescer o
# num_ctx), cada token de instrução é um token a menos para a RESPOSTA. Um prompt
# curto deixa muito mais espaço de saída → resumos mais longos e completos.
SUMMARY_TEMPLATE = PromptTemplate(
    "Você é a ARIA, tutora virtual acadêmica. Responda SEMPRE em português do Brasil.\n"
    "\n"
    "Faça um RESUMO do material de aula do estudante, com base APENAS nos trechos abaixo.\n"
    "\n"
    "REGRAS:\n"
    "- Escreva um resumo LONGO, COMPLETO e bem desenvolvido. Desenvolva cada tema em "
    "vários parágrafos, com definições e exemplos concretos do material. NÃO seja breve "
    "nem telegráfico. Se o estudante pediu um tamanho (ex.: '200 linhas', 'detalhado'), "
    "ATINJA esse tamanho.\n"
    "- Organize por TEMAS/CONCEITOS, de forma fluida. NÃO use 'Unidade 1', 'Unidade 2'... "
    "como títulos nem estruture por unidades — escreva pelos assuntos (ex.: pensamento "
    "econômico, microeconomia, macroeconomia, comércio internacional...).\n"
    "- Baseie-se ESTRITAMENTE nos trechos; não invente nem traga conhecimento externo. "
    "NÃO cite números de página.\n"
    "- Destaque os termos-chave em **negrito** e use subtítulos temáticos quando ajudar.\n"
    "\n"
    "TRECHOS DO MATERIAL:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "\n"
    "PEDIDO DO ESTUDANTE: {query_str}\n"
    "\n"
    "RESUMO COMPLETO, LONGO E DESENVOLVIDO EM PORTUGUÊS DO BRASIL:"
)

WEB_TEMPLATE = PromptTemplate(
    "Você é a ARIA (Assistente de Revisão Inteligente Acadêmica), uma tutora virtual.\n"
    "\n"
    "REGRA ABSOLUTA: Responda SEMPRE em português do Brasil. NUNCA em inglês.\n"
    "\n"
    "SEU NOME E IDENTIDADE: Você se chama ARIA e se refere a si mesma no feminino. "
    "Se o estudante perguntar seu nome, quem você é, ou como deve te chamar, responda SEMPRE que você é a ARIA "
    "(Assistente de Revisão Inteligente Acadêmica). Isso faz parte de quem você é — "
    "você NÃO precisa procurar essa informação no material/contexto. "
    "IGNORE qualquer outro nome que possa ter aparecido antes na conversa: seu ÚNICO nome é ARIA.\n"
    "\n"
    "O conteúdo abaixo NÃO veio do material de aula do estudante — veio de uma BUSCA NA "
    "INTERNET, porque a pergunta não foi encontrada nos PDFs disponíveis.\n"
    "\n"
    "INSTRUÇÕES:\n"
    "- Avise o estudante, de forma natural, que essa resposta vem de uma busca na web "
    "(não do material de aula).\n"
    "- Seja didático, claro e organizado.\n"
    "- Use formatação Markdown: destaque os termos-chave em **negrito** e use listas com marcadores.\n"
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
# Modelos Pydantic
# ---------------------------------------------------------------------------
class GerarRequest(BaseModel):
    pergunta:  str
    contexto:  str
    fonte:     str = "material"   # "material" (PDFs), "web" (busca na internet) ou "identidade" (sobre o próprio ARIA)
    historico: str = ""           # histórico recente da conversa (opcional, p/ memória)
    reforcar_identidade: bool = False  # pergunta combinada (nome + conteúdo): blinda o nome contra título/autor do PDF


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="ARIA — Microserviço de IA / Proxy",
    description="Fala com o Qwen 14B (Colab/ngrok) e gera as respostas do ARIA.",
)


@app.post("/gerar")
def gerar(req: GerarRequest):
    # Pergunta SÓ sobre a identidade do ARIA: resposta DETERMINÍSTICA, sem chamar
    # o LLM. Garante 100% que o nome é ARIA (o Qwen tende a se dizer "Claude").
    if req.fonte == "identidade":
        resposta = (
            f"{APRESENTACAO_ARIA} Estou aqui pra te ajudar a entender o seu "
            "material de aula — é só perguntar! 📚"
        )
        print(f"🪪 /gerar (identidade — resposta fixa): '{req.pergunta}'")
        return {"resposta": resposta, "fonte": req.fonte}

    # Escolhe o template: web > resumo (enxuto, p/ liberar espaço de resposta) > geral.
    if req.fonte == "web":
        template = WEB_TEMPLATE
    elif _eh_pedido_resumo(req.pergunta):
        template = SUMMARY_TEMPLATE
    else:
        template = TEXT_QA_TEMPLATE
    prompt   = template.format(context_str=req.contexto, query_str=req.pergunta)

    # Pedido de questões/quiz: injeta regras de formato e anti-repetição na frente
    # do prompt (o modelo lê as restrições antes do conteúdo). Sem isto, ao pedir
    # muitas questões o Qwen clona as mesmas com outro número.
    if _eh_pedido_questoes(req.pergunta):
        prompt = QUESTOES_INSTRUCAO + prompt

    # Pergunta combinada (nome + conteúdo): a apresentação como ARIA é colada por
    # nós (depois da geração), de forma determinística — porque o modelo inventa
    # o nome ("Claude", "Claro", etc.) mesmo instruído. Aqui só mandamos ele NÃO
    # se apresentar/cumprimentar, pra ir direto ao conteúdo.
    if req.reforcar_identidade:
        reforco = (
            "ATENÇÃO: IGNORE a parte da pergunta sobre o seu nome ou quem você é "
            "(isso já está sendo respondido automaticamente). NÃO se apresente, NÃO diga "
            "seu nome e NÃO cumprimente. Comece DIRETO pelo conteúdo que o estudante pediu "
            "(resumo, título, autor, conceitos, etc.), com a mesma riqueza e formatação de sempre.\n"
            "----------------------------------------\n\n"
        )
        prompt = reforco + prompt

    # Injeta o histórico recente da conversa (memória) antes do prompt principal
    if req.historico.strip():
        preambulo = (
            "HISTÓRICO RECENTE DA CONVERSA (use para manter o contexto, "
            "mas baseie a resposta no material/contexto abaixo):\n"
            f"{req.historico.strip()}\n"
            "----------------------------------------\n\n"
        )
        prompt = preambulo + prompt

    print(f"🧪 /gerar (fonte={req.fonte}, historico={'sim' if req.historico.strip() else 'não'}, "
          f"reforco_id={req.reforcar_identidade}): '{req.pergunta}'")

    # complete() (e não chat com system prompt) — dá respostas mais ricas e bem
    # formatadas. A identidade da ARIA já é garantida de forma determinística
    # (resposta fixa p/ pergunta de nome + saudação recolada na combinada), então
    # não dependemos do modelo para o nome e podemos priorizar a qualidade.
    try:
        resposta = str(llm.complete(prompt)).strip()
    except Exception as exc:
        msg  = str(exc)
        nome = type(exc).__name__
        if "403" in msg or "PermissionDenied" in msg or "Forbidden" in msg:
            raise HTTPException(
                status_code=503,
                detail="⚠️ O modelo de IA está indisponível: o túnel ngrok expirou (403). "
                       "Reinicie o ngrok no Colab.",
            )
        # Túnel/Colab fora do ar: conexão recusada, timeout ou DNS não resolve.
        # Mensagem clara (e acionável) em vez do erro genérico de orquestração.
        ml = msg.lower()
        if ("Connection" in nome or "Timeout" in nome or "APIConnect" in nome
                or "connect" in ml or "timed out" in ml or "max retries" in ml
                or "getaddrinfo" in ml or "name or service" in ml
                or "offline" in ml or "err_ngrok" in ml or "ngrok" in ml
                or "3200" in ml):
            raise HTTPException(
                status_code=503,
                detail="⚠️ O modelo de IA está offline (o servidor do Qwen no Colab/ngrok caiu). "
                       "Reabra o Colab, rode 'ollama serve' + ngrok e confira o NGROK_URL.",
            )
        raise HTTPException(status_code=500, detail=f"Erro ao consultar o LLM: {msg}")

    # Pergunta combinada: remove o nome que o modelo eventualmente inventou e cola
    # a apresentação correta (determinística) da ARIA na frente do conteúdo.
    if req.reforcar_identidade:
        resposta = _remover_saudacao_inicial(resposta)
        resposta = f"{APRESENTACAO_ARIA}\n\n{resposta}"

    # Questões: tira asteriscos de negrito (Qwen os desbalanceia e "vaza" o
    # destaque pelas alternativas). Determinístico — não depende do modelo.
    if _eh_pedido_questoes(req.pergunta):
        resposta = _limpar_markdown_questoes(resposta)
        # "sem gabarito": remove qualquer linha de resposta que o modelo tenha
        # incluído mesmo instruído a não incluir.
        if _pediu_sem_gabarito(req.pergunta):
            resposta = _remover_gabarito(resposta)

    resposta = _corrigir_portugues(resposta)
    return {"resposta": resposta, "fonte": req.fonte}


@app.get("/health")
def health():
    """Checa a conexão com o Colab/ngrok."""
    try:
        resp = httpx.get(
            NGROK_URL,
            timeout=5.0,
            headers={"ngrok-skip-browser-warning": "true"},
            follow_redirects=True,
        )
        online = resp.status_code == 200 and "Ollama is running" in resp.text
        return {
            "status":    "ok",
            "servico":   "IA / Proxy",
            "modelo":    "qwen2.5:14b",
            "ngrok_url": NGROK_URL,
            "llm_online": online,
        }
    except Exception as exc:
        return {"status": "ok", "servico": "IA / Proxy", "llm_online": False, "erro": str(exc)}


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8004, reload=False)
