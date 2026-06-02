# 🧠 Assistente Acadêmico Inteligente (RAG + MCP)

Trabalho prático desenvolvido para a disciplina **GCC129 - Sistemas Distribuídos** da Universidade Federal de Lavras (UFLA), sob orientação do Prof. André de Lima Salgado.

> **Status:** Em desenvolvimento (Fase 2 - Implementação Parcial)

## Objetivo do Projeto
Este projeto implementa um sistema inteligente e distribuído para atuar como um **Assistente Acadêmico**. Ele utiliza a arquitetura RAG (*Retrieval-Augmented Generation*) integrada ao MCP (*Model Context Protocol*) para responder a dúvidas de alunos com base em documentos reais fornecidos pelos professores, contornando alucinações e garantindo respostas fundamentadas em arquivos institucionais.

## Arquitetura do Sistema
O sistema foi construído utilizando uma abordagem de microsserviços, distribuindo as responsabilidades de interface, roteamento, integração de ferramentas e processamento de IA.

1. **Frontend (React / TypeScript):** Interface de usuário (App do Estudante) onde as perguntas são feitas.
2. **API Gateway (Node.js / Express):** Ponto de entrada único (porta `8000`). Roteia as dúvidas dos alunos para o servidor MCP e as ações administrativas diretamente para o RAG.
3. **Servidor MCP (Node.js / Express):** Expõe as ferramentas do sistema (ex: `consultar_documentos_aula`) e orquestra as chamadas para recuperação de contexto.
4. **Microsserviço RAG (Python / FastAPI / LlamaIndex):** Responsável por ler os PDFs locais, processar os textos e armazenar os vetores no banco **ChromaDB**. Ele recebe o contexto e monta o prompt final.
5. **Infraestrutura LLM (Google Colab / Ngrok):** Para contornar limitações de hardware local, a LLM **Qwen 2.5 (14B)** é executada na nuvem via Ollama, recebendo o prompt do serviço RAG através de um túnel seguro gerado pelo Ngrok.

## Estrutura do Repositório (Monorepo)

```bash
/
  ├── /api-gateway       # Ponto de entrada e roteamento (Node.js)
  ├── /frontend          # Aplicação web do estudante (React)
  ├── /mcp-server        # Servidor de ferramentas via Model Context Protocol
  └── /rag-service       # Serviço de busca vetorial e IA (Python)
```

## Como Executar o Projeto

Como o sistema é composto por múltiplos microsserviços, é necessário iniciar cada componente individualmente. 

### Passo 1: Subir a LLM na Nuvem (Backend IA)
1. Acesse o Notebook do Google Colab fornecido pela equipe.
2. Execute as células para instalar o Ollama, baixar o modelo Qwen 2.5 e iniciar o túnel Ngrok.
3. Copie a URL pública gerada pelo Ngrok.

### Passo 2: Configurar o Microsserviço RAG
1. Navegue até a pasta `rag-service`.
2. Configure a URL do Ngrok no arquivo de ambiente (ex: `LLM_HOST=https://seu-link-ngrok.ngrok.io`).
3. Instale as dependências e inicie o servidor (porta 8001):
```bash
cd rag-service
pip install -r requirements.txt
uvicorn main:app --port 8001
```

### Passo 3: Iniciar o Servidor MCP
O servidor que gerencia as ferramentas para a IA (porta 8002):
```bash
cd mcp-server
npm install
npm start
```

### Passo 4: Iniciar o API Gateway
O ponto de entrada que orquestra as requisições (porta 8000):
```bash
cd api-gateway
npm install
npm start
```

### Passo 5: Iniciar o Frontend
A interface visual do usuário:
```bash
cd frontend
npm install
npm run dev
```
O sistema estará disponível para acesso no navegador em `http://localhost:3000` (ou a porta configurada no Frontend).

## 👥 Equipe
* **Pyêtro Augusto Malaquias** 
* **Lídio Júnior Pereira Batista** 
* **Helder Jose Avila** 
* **Gustavo Batista Bissoli** 
* **Miguel Chagas Figueiredo** 