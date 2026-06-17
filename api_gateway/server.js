const express = require('express');
const axios   = require('axios');
const cors    = require('cors');

const app = express();
app.use(cors());
app.use(express.json());

const PORT        = 8000;
const RAG_URL     = 'http://127.0.0.1:8001';   // RAG — só retrieval (contexto)
const MCP_URL     = 'http://127.0.0.1:8002';   // MCP — ferramentas externas (busca web)
const HIST_URL    = 'http://127.0.0.1:8003';   // Histórico (PostgreSQL)
const IA_URL      = 'http://127.0.0.1:8004';   // IA / Proxy — geração de texto (Qwen)

// Limiar de relevância para a rota MATERIAL × WEB.
//
// LIMIAR_RERANK (preferido): score do cross-encoder de reranking (0-1) — sinal
//   MUITO mais calibrado. É o usado quando o RAG devolve 'rerank_score'.
// LIMIAR_SCORE (fallback): similaridade de cosseno do retriever vetorial — só
//   entra em ação se o reranker estiver indisponível (rerank_score = null).
// Score >= limiar → responde com o material de aula (PDFs); abaixo → busca web.
const LIMIAR_RERANK = parseFloat(process.env.LIMIAR_RERANK || '0.3');
const LIMIAR_SCORE  = parseFloat(process.env.LIMIAR_SCORE  || '0.66');

// Quantas mensagens anteriores buscar para dar memória à conversa
const HIST_LIMITE = parseInt(process.env.HIST_LIMITE || '6', 10);

// Teto de caracteres POR mensagem do histórico injetado. Respostas da ARIA são
// longas (resumos/relatórios); sem corte, 6 delas viram milhares de tokens e
// engolem a janela do LLM — sobra pouco pra saída (resposta curta) e o modelo
// se perde (perde o contexto do material). Cortamos cada mensagem mantendo só
// o começo (suficiente pra memória de "sobre o que falávamos").
const HIST_MAX_CHARS = parseInt(process.env.HIST_MAX_CHARS || '600', 10);

// Timeout (ms) das chamadas de GERAÇÃO ao ia_service. Respostas longas (ex.: "gere
// 20 questões") num 14B no T4 podem levar minutos. Mantém-se MAIOR que o LLM_TIMEOUT
// do ia_service (default 600s) p/ que o ia_service seja quem corta, com erro claro.
const GERAR_TIMEOUT = parseInt(process.env.GERAR_TIMEOUT || '630000', 10);

// ---------------------------------------------------------------------------
// Detecção de intenção sobre a IDENTIDADE do próprio ARIA ("qual seu nome?",
// "quem é você?", "como se chama?"). A ficha do documento (título/autor) tem
// score alto pra essas perguntas e o modelo pequeno acaba dizendo que se chama
// o autor/título do PDF — ou inventa um nome. Tratamos isso em dois casos:
//   • Pergunta SÓ de identidade  → responde sem retrieval (zero contexto).
//   • Pergunta COMBINADA (nome + resumo/autor/etc.) → segue o fluxo normal,
//     mas com um reforço de identidade pro ia_service não confundir o nome.
// ---------------------------------------------------------------------------
function _normalizar(texto) {
    return (texto || '').toLowerCase().normalize('NFD').replace(/[̀-ͯ]/g, '');  // remove acentos
}

// O estudante está perguntando sobre o NOME / a identidade do próprio assistente?
function mencionaIdentidade(texto) {
    const t = _normalizar(texto);
    return (
        /\b(seu|teu)\s+nome\b/.test(t) ||                       // "qual seu nome", "teu nome"
        /\bvoce\s+tem\s+(algum\s+)?nome\b/.test(t) ||           // "você tem nome?"
        /\bcomo\s+(voce\s+)?se\s+chama\b/.test(t) ||            // "como você se chama"
        /\bcomo\s+(te|devo\s+te)\s+cham(ar|o)\b/.test(t) ||     // "como te chamo", "como devo te chamar"
        /\bquem\s+(e|es)\s+voce\b/.test(t) ||                   // "quem é você"
        /\bvoce\s+quem\s+e\b/.test(t) ||                        // "você quem é"
        /\bqual\s+(e\s+)?o?\s*teu\s+nome\b/.test(t)
    );
}

// O estudante também está pedindo algo sobre o CONTEÚDO (PDF/material/autor)?
function pedeConteudo(texto) {
    const t = _normalizar(texto);
    return /\b(autor|autora|livro|documento|material|pdf|obra|arquivo|texto|apostila|titulo|resumo|resum[ai]|explica|explique|fale|fala|o que e|conteudo|capitulo|pagina)\b/.test(t);
}

// O estudante pediu EXPLICITAMENTE pra buscar na web/internet? Nesse caso forçamos
// a busca web, ignorando o score do material (ele já avisou que quer fonte externa).
function pedeBuscaWeb(texto) {
    const t = _normalizar(texto);
    return /\b(na (web|internet|rede)|no google|online)\b/.test(t)
        || /\b(pesquis\w*|busc\w*|procur\w*|consult\w*|olh\w*|d[áa] uma olhada|ver)\b[^.?!]*\b(web|internet|google|online)\b/.test(t);
}

// A pergunta se refere ao PDF/material do estudante? (ex.: "livros sobre o material
// desse pdf", "autores da área deste documento"). Quando um pedido assim cai na
// busca web (porque não está literalmente no PDF), ancoramos a query no TEMA do
// material — senão o buscador "viaja" para assuntos aleatórios.
function referenciaMaterial(texto) {
    const t = _normalizar(texto);
    // Exige um substantivo que aponte claramente para o material do estudante —
    // evita disparar a âncora em perguntas genéricas que só por acaso caíram na web.
    return /\b(pdf|material|livro|livros|documento|apostila|obra|obras|conteudo|capitulo|arquivo)\b/.test(t);
}

// Assunto/tema do PDF indexado (título da ficha) — usado como ÂNCORA da busca web.
// Best-effort: se o RAG não responder, devolve '' e a query segue sem âncora.
async function buscarAssuntoPDF() {
    try {
        const r = await axios.get(`${RAG_URL}/ficha`, { timeout: 3000 });
        let titulo = (r.data && r.data.titulo) ? String(r.data.titulo).trim() : '';
        // Mantém só a parte principal do título (antes de ':' ou '-') p/ uma âncora
        // concisa — subtítulos longos diluem a busca.
        titulo = titulo.split(/[:\-–—]/)[0].trim();
        return titulo;
    } catch (err) {
        console.warn(`⚠️  Não consegui obter o tema do PDF (busca web sem âncora): ${err.message}`);
        return '';
    }
}

// ---------------------------------------------------------------------------
// Limpa a query ANTES de mandar pro RAG/busca: tira saudação, perguntas de
// identidade e gentilezas, que poluem o retrieval (puxam a ficha do documento
// em vez de trechos de conteúdo). O ia_service continua recebendo a pergunta
// COMPLETA — esta versão limpa é só para a BUSCA. Se sobrar pouco, usa a original.
// ---------------------------------------------------------------------------
const _RUIDO_BUSCA = [
    /\b(oi+|ol[áa]|e a[íi]|opa|fala)\b/gi,
    /\bchat\b/gi,
    /\btudo bem\b/gi,
    /\b(bom dia|boa tarde|boa noite)\b/gi,
    /\bpor favor\b/gi,
    /\bse poss[íi]vel\b/gi,
    /\bdesde j[áa]( agrade(ç|c)o)?\b/gi,
    /\b(muito )?obrigad[oa]\b/gi,
    /\bpode (me )?(fazer|informar|dizer|falar|ajudar)\b/gi,
    // sub-perguntas de identidade
    /\bqual (é |e )?(o |a )?(seu|teu) nome\b/gi,
    /\bquem (é|e) voc[êe]\b/gi,
    /\bcomo (voc[êe] )?se chama\b/gi,
    /\bcomo (te|devo te) cham(ar|o)\b/gi,
    /\bvoc[êe] tem (algum )?nome\b/gi,
];

function limparQueryBusca(texto) {
    let t = texto || '';
    for (const p of _RUIDO_BUSCA) t = t.replace(p, ' ');
    t = t.replace(/\s*[?!.,]+\s*/g, ' ').replace(/\s{2,}/g, ' ').trim();
    return t.length >= 8 ? t : (texto || '').trim();
}

// ---------------------------------------------------------------------------
// Helpers de histórico — todos "best-effort": se o serviço de histórico
// estiver fora do ar, o chat continua respondendo normalmente.
// ---------------------------------------------------------------------------
async function buscarHistorico(sessao) {
    try {
        const r = await axios.get(
            `${HIST_URL}/historico/${encodeURIComponent(sessao)}?limite=${HIST_LIMITE}`,
            { timeout: 4000 }
        );
        // Formata como "Estudante: ... / ARIA: ..." para o LLM entender.
        // Cada mensagem é truncada em HIST_MAX_CHARS pra a memória não estourar
        // a janela de contexto (ver comentário na constante).
        return (r.data || [])
            .map(m => {
                const quem = m.papel === 'usuario' ? 'Estudante' : 'ARIA';
                let txt = m.conteudo || '';
                if (txt.length > HIST_MAX_CHARS) {
                    txt = txt.slice(0, HIST_MAX_CHARS).trimEnd() + ' […]';
                }
                return `${quem}: ${txt}`;
            })
            .join('\n');
    } catch (err) {
        console.warn(`⚠️  Histórico indisponível (seguindo sem memória): ${err.message}`);
        return '';
    }
}

async function salvarMensagem(sessao, papel, conteudo) {
    try {
        await axios.post(`${HIST_URL}/mensagens`, { session_id: sessao, papel, conteudo }, { timeout: 4000 });
    } catch (err) {
        console.warn(`⚠️  Não foi possível salvar mensagem no histórico: ${err.message}`);
    }
}

// Rota padrão — saúde do Gateway
app.get('/', (req, res) => {
    res.json({
        status:        "API Gateway (Node.js/Express) Online",
        porta:         PORT,
        limiar_rerank: LIMIAR_RERANK,
        limiar_score:  LIMIAR_SCORE,
    });
});

// ---------------------------------------------------------------------------
// Rota principal — orquestra TODO o fluxo:
//   1. Busca o histórico da sessão (memória)
//   2. Salva a pergunta do usuário
//   3. Pede contexto + score ao RAG → roteia entre PDFs e busca web
//   4. Sintetiza a resposta (com o histórico injetado)
//   5. Salva a resposta do assistente
// ---------------------------------------------------------------------------
app.post('/perguntar', async (req, res) => {
    const { pergunta } = req.body;
    const sessao = req.body.session_id || 'default';

    if (!pergunta) {
        return res.status(422).json({ erro: "O campo 'pergunta' é obrigatório." });
    }

    console.log(`\n========================================`);
    console.log(`📥 Gateway recebeu (sessão=${sessao}): "${pergunta}"`);

    try {
        // ── 1. Histórico anterior (antes de salvar a pergunta atual) ────────
        const historico = await buscarHistorico(sessao);

        // ── 2. Salva a pergunta do usuário (best-effort) ────────────────────
        await salvarMensagem(sessao, 'usuario', pergunta);

        // ── 2b. Intenção de identidade do ARIA? ─────────────────────────────
        const querIdentidade = mencionaIdentidade(pergunta);
        const querConteudo   = pedeConteudo(pergunta);

        // Pergunta SÓ de identidade ("qual seu nome?") → responde sem RAG, pra
        // que nenhum nome de autor/título/histórico confunda o modelo.
        if (querIdentidade && !querConteudo) {
            console.log(`🪪 Pergunta de IDENTIDADE pura — respondendo sem retrieval.`);
            const r = await axios.post(`${IA_URL}/gerar`, {
                pergunta,
                contexto:  "",
                fonte:     "identidade",
                historico,
            }, { timeout: GERAR_TIMEOUT });
            const resposta = r.data.resposta;
            await salvarMensagem(sessao, 'assistente', resposta);
            return res.json({ resposta, origem: "identidade", score: null, session_id: sessao });
        }

        // Para perguntas COMBINADAS (nome + conteúdo), seguimos o fluxo normal
        // mas sinalizamos o reforço de identidade ao ia_service.
        if (querIdentidade) {
            console.log(`🪪 Pergunta COMBINADA (identidade + conteúdo) — reforço de identidade ativado.`);
        }

        // ── 3. Contexto + score calibrado dos PDFs ──────────────────────────
        // Busca com a query LIMPA (sem saudação/identidade/gentilezas) p/ recuperar
        // trechos de conteúdo relevantes — e não a ficha do documento.
        const queryBusca = limparQueryBusca(pergunta);
        if (queryBusca !== pergunta.trim()) {
            console.log(`🔎 Query de busca limpa: "${queryBusca}"`);
        }
        const ctx   = await axios.post(`${RAG_URL}/contexto`, { texto: queryBusca });
        const score       = ctx.data.score_max;
        const rerankScore = ctx.data.rerank_score;   // null se o reranker não estiver ativo
        const forcarWeb   = pedeBuscaWeb(pergunta);

        // Decide a rota: prefere o score do reranker (calibrado). Cai no cosseno
        // só quando o reranker está indisponível (rerank_score == null).
        let relevante;
        if (rerankScore !== null && rerankScore !== undefined) {
            relevante = rerankScore >= LIMIAR_RERANK;
            console.log(`🧭 rerank_score=${rerankScore} | limiar_rerank=${LIMIAR_RERANK} (cosseno=${score})${forcarWeb ? ' | 🌐 busca web pedida explicitamente' : ''}`);
        } else {
            relevante = score >= LIMIAR_SCORE;
            console.log(`🧭 score_max=${score} | limiar=${LIMIAR_SCORE} (fallback cosseno — reranker off)${forcarWeb ? ' | 🌐 busca web pedida explicitamente' : ''}`);
        }

        let resposta, origem;

        if (!forcarWeb && relevante) {
            // ── 4a. Resposta com o material de aula ─────────────────────────
            console.log(`📚 Roteando para o MATERIAL DE AULA (PDFs).`);
            const r = await axios.post(`${IA_URL}/gerar`, {
                pergunta,
                contexto:  ctx.data.contexto,
                fonte:     "material",
                historico,
                reforcar_identidade: querIdentidade,
            }, { timeout: GERAR_TIMEOUT });
            resposta = r.data.resposta;
            origem   = "material";
        } else {
            // ── 4b. Não está no material → busca na web via MCP ─────────────
            // Ancora a query no TEMA do PDF quando o pedido se refere ao material
            // (ex.: "livros sobre o material desse pdf") e o estudante NÃO pediu
            // explicitamente uma busca web livre. Sem isso, o buscador perde o
            // assunto e devolve resultados aleatórios (fora do tema do material).
            let queryWeb = queryBusca;
            if (!forcarWeb && referenciaMaterial(pergunta)) {
                const assunto = await buscarAssuntoPDF();
                if (assunto) {
                    queryWeb = `${queryBusca} ${assunto}`;
                    console.log(`🎯 Busca web ancorada no tema do PDF: "${assunto}"`);
                }
            }
            console.log(`🌐 Não encontrado no material. Roteando para BUSCA WEB (MCP). Query: "${queryWeb}"`);
            const web = await axios.post(`${MCP_URL}/mcp/tools/call`, {
                tool:      "buscar_web",
                arguments: { query: queryWeb },
            });
            const contextoWeb = web.data.content[0].text;

            const r = await axios.post(`${IA_URL}/gerar`, {
                pergunta,
                contexto:  contextoWeb,
                fonte:     "web",
                historico,
                reforcar_identidade: querIdentidade,
            }, { timeout: GERAR_TIMEOUT });
            resposta = r.data.resposta;
            origem   = "web";
        }

        // ── 5. Salva a resposta do assistente (best-effort) ─────────────────
        await salvarMensagem(sessao, 'assistente', resposta);

        return res.json({ resposta, origem, score, session_id: sessao });

    } catch (error) {
        const status  = error.response?.status || 500;
        const detalhe = error.response?.data?.detail || error.response?.data?.erro || error.message;
        console.error(`❌ Erro na orquestração (status ${status}):`, detalhe);
        // 503 = serviço de IA indisponível (LLM/Colab offline). Mostra a mensagem
        // acionável DIRETO no campo 'erro' (é o que o frontend exibe), em vez do
        // genérico "Erro no Gateway" — que mascarava a causa real (Colab caído).
        const erro = status === 503
            ? (detalhe || "⚠️ A IA está offline no momento. Reinicie o Colab/ngrok e tente novamente.")
            : "Erro no Gateway ao orquestrar a resposta.";
        return res.status(status).json({ erro, detalhes: detalhe });
    }
});

// ---------------------------------------------------------------------------
// Upload de PDF — o usuário envia o material pelo frontend.
// O Gateway NÃO parseia o multipart: repassa o stream bruto direto pro RAG,
// preservando o Content-Type (que carrega o boundary). Por isso não precisa de
// multer/busboy. express.json() ignora multipart, então o body chega intacto.
// A indexação pode levar minutos num PDF grande → sem timeout e sem limite de tamanho.
// ---------------------------------------------------------------------------
app.post('/upload', async (req, res) => {
    try {
        console.log(`\n📤 Gateway recebeu upload de PDF — repassando ao RAG...`);
        const ragResp = await axios.post(`${RAG_URL}/upload`, req, {
            headers: { 'content-type': req.headers['content-type'] },
            maxBodyLength:    Infinity,
            maxContentLength: Infinity,
            timeout:          0,   // indexação pode demorar alguns minutos
        });
        console.log(`✅ Upload indexado: ${JSON.stringify(ragResp.data)}`);
        return res.json(ragResp.data);
    } catch (error) {
        const detalhe = error.response?.data?.detail || error.response?.data?.erro || error.message;
        console.error(`❌ Erro no upload:`, detalhe);
        return res.status(error.response?.status || 500).json({
            erro:     "Erro no Gateway ao enviar o PDF para indexação.",
            detalhes: detalhe,
        });
    }
});

app.listen(PORT, () => {
    console.log(`🚀 API Gateway (orquestrador) rodando em http://127.0.0.1:${PORT}`);
    console.log(`🧭 Roteamento: rerank>=${LIMIAR_RERANK} (preferido) | cosseno>=${LIMIAR_SCORE} (fallback) → PDFs acima, web abaixo`);
    console.log(`📡 RAG: ${RAG_URL} | MCP: ${MCP_URL} | Histórico: ${HIST_URL} | IA: ${IA_URL}`);
});
