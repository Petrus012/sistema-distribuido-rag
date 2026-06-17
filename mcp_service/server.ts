import express, { Request, Response } from 'express';
import axios from 'axios';
import 'dotenv/config';

const app = express();
app.use(express.json());

const PORT            = 8002;
const TAVILY_API_KEY  = process.env.TAVILY_API_KEY ?? '';
const TAVILY_URL      = 'https://api.tavily.com/search';

// Formato padrão de resposta do protocolo MCP
interface MCPContent {
    content: { type: 'text'; text: string }[];
}

// ---------------------------------------------------------------------------
// Ferramenta: buscar_web — busca na internet via Tavily
// Devolve os resultados como texto, prontos para o LLM (ia_service) sintetizar.
// ---------------------------------------------------------------------------
interface TavilyResult {
    title:   string;
    url:     string;
    content: string;
}

async function buscarWeb(query: string, maxResultados = 5): Promise<string> {
    if (!TAVILY_API_KEY) {
        throw new Error(
            "TAVILY_API_KEY não configurada. Adicione a chave no arquivo .env do mcp_service."
        );
    }

    const resposta = await axios.post(TAVILY_URL, {
        api_key:        TAVILY_API_KEY,
        query,
        search_depth:   'basic',
        max_results:    maxResultados,
        include_answer: false,
    });

    const resultados: TavilyResult[] = resposta.data.results ?? [];
    if (resultados.length === 0) {
        return 'Nenhum resultado encontrado na busca web.';
    }

    return resultados
        .map((r, i) =>
            `[Resultado ${i + 1}] ${r.title}\nFonte: ${r.url}\n${r.content}`
        )
        .join('\n\n---\n\n');
}

// ---------------------------------------------------------------------------
// MCP — Status
// ---------------------------------------------------------------------------
app.get('/', (_req: Request, res: Response) => {
    res.json({
        status:    'Servidor MCP Online (TypeScript)',
        versao:    '2.1.0',
        porta:     PORT,
        protocolo: 'MCP over HTTP',
    });
});

// ---------------------------------------------------------------------------
// MCP — Descoberta de ferramentas (só ferramentas EXTERNAS, conforme a arquitetura)
// ---------------------------------------------------------------------------
app.get('/mcp/tools', (_req: Request, res: Response) => {
    res.json({
        tools: [
            {
                name:        'buscar_web',
                description:
                    'Busca informações atualizadas na internet. Use quando a resposta ' +
                    'não estiver no material de aula (PDFs).',
                input_schema: {
                    type:       'object',
                    properties: {
                        query: { type: 'string', description: 'O que buscar na web.' },
                    },
                    required: ['query'],
                },
            },
        ],
    });
});

// ---------------------------------------------------------------------------
// MCP — Execução de ferramenta
// Recebe: { tool: string, arguments: object }
// Devolve: { content: [{ type: "text", text: string }] }
// ---------------------------------------------------------------------------
app.post('/mcp/tools/call', async (req: Request, res: Response) => {
    const { tool, arguments: args } = req.body as { tool?: string; arguments?: Record<string, unknown> };

    if (!tool || !args) {
        return res.status(422).json({
            erro: 'Corpo inválido. Esperado: { tool: string, arguments: object }',
        });
    }

    console.log(`\n========================================`);
    console.log(`⚙️  MCP acionado | 🔧 Ferramenta: ${tool}`);
    console.log(`📥 Argumentos:`, args);
    console.log(`========================================`);

    try {
        switch (tool) {
            // ── Busca na web (Tavily) ──────────────────────────────────────
            case 'buscar_web': {
                const query = args.query as string | undefined;
                if (!query) {
                    return res.status(422).json({ erro: "O argumento 'query' é obrigatório." });
                }
                console.log(`🌐 Buscando na web: "${query}"`);
                const texto = await buscarWeb(query);
                console.log(`✅ Busca web concluída.`);
                const out: MCPContent = { content: [{ type: 'text', text: texto }] };
                return res.json(out);
            }

            // ── Ferramenta desconhecida ────────────────────────────────────
            default:
                return res.status(404).json({
                    erro: `Ferramenta '${tool}' não encontrada neste servidor MCP.`,
                });
        }
    } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        console.error(`❌ Erro ao executar '${tool}':`, msg);
        return res.status(502).json({ erro: `Erro ao executar a ferramenta '${tool}'.`, detalhes: msg });
    }
});

// ---------------------------------------------------------------------------
// MCP — Health check
// ---------------------------------------------------------------------------
app.get('/health', (_req: Request, res: Response) => {
    res.json({
        mcp_status: 'ok',
        tavily:     TAVILY_API_KEY ? 'configurada' : 'NÃO configurada',
    });
});

// ---------------------------------------------------------------------------
app.listen(PORT, () => {
    console.log(`🚀 Servidor MCP (TypeScript) rodando em http://127.0.0.1:${PORT}`);
    console.log(`🛠️  Ferramentas: buscar_web`);
    console.log(`🌐 Tavily: ${TAVILY_API_KEY ? 'configurada ✅' : 'NÃO configurada ⚠️ (defina TAVILY_API_KEY no .env)'}`);
});
