"""
Microserviço de Histórico — ARIA (fase 2)
-----------------------------------------
Guarda e recupera o histórico de conversas dos estudantes em PostgreSQL.
Modelo "Opção A": uma linha por mensagem (normalizado).

Endpoints:
  POST   /mensagens            → salva uma mensagem
  GET    /historico/{sessao}   → últimas N mensagens da sessão (ordem cronológica)
  DELETE /historico/{sessao}   → limpa o histórico de uma sessão
  GET    /health               → saúde do serviço
"""

import os
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import (
    create_engine, String, Text, DateTime, Integer, select, delete,
)
from sqlalchemy.orm import (
    DeclarativeBase, Mapped, mapped_column, Session,
)

# ---------------------------------------------------------------------------
# Conexão com o PostgreSQL
# ---------------------------------------------------------------------------
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://postgres:postgres@localhost:5432/aria_historico",
)
engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True)


# ---------------------------------------------------------------------------
# Modelo ORM — tabela 'mensagens' (uma linha por mensagem)
# ---------------------------------------------------------------------------
class Base(DeclarativeBase):
    pass


class Mensagem(Base):
    __tablename__ = "mensagens"

    id:         Mapped[int]      = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str]      = mapped_column(String(128), index=True, nullable=False)
    papel:      Mapped[str]      = mapped_column(String(20),  nullable=False)  # 'usuario' | 'assistente'
    conteudo:   Mapped[str]      = mapped_column(Text,        nullable=False)
    criado_em:  Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


# ---------------------------------------------------------------------------
# Modelos Pydantic (entrada/saída da API)
# ---------------------------------------------------------------------------
class MensagemIn(BaseModel):
    session_id: str
    papel:      str   # 'usuario' ou 'assistente'
    conteudo:   str


class MensagemOut(BaseModel):
    id:         int
    session_id: str
    papel:      str
    conteudo:   str
    criado_em:  datetime


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="ARIA — Microserviço de Histórico",
    description="Guarda e recupera o histórico de conversas em PostgreSQL.",
)


@app.on_event("startup")
def criar_tabelas():
    """Cria a tabela 'mensagens' se ainda não existir."""
    Base.metadata.create_all(engine)
    print("🗄️  Tabela 'mensagens' pronta no PostgreSQL.")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/mensagens", response_model=MensagemOut)
def salvar_mensagem(msg: MensagemIn):
    if msg.papel not in ("usuario", "assistente"):
        raise HTTPException(status_code=422, detail="papel deve ser 'usuario' ou 'assistente'.")
    if not msg.conteudo.strip():
        raise HTTPException(status_code=422, detail="conteudo não pode estar vazio.")

    with Session(engine) as sessao:
        registro = Mensagem(
            session_id=msg.session_id,
            papel=msg.papel,
            conteudo=msg.conteudo,
        )
        sessao.add(registro)
        sessao.commit()
        sessao.refresh(registro)
        return registro


@app.get("/historico/{session_id}", response_model=list[MensagemOut])
def obter_historico(session_id: str, limite: int = 10):
    """Retorna as últimas `limite` mensagens da sessão, em ordem cronológica (mais antiga → mais nova)."""
    with Session(engine) as sessao:
        # Pega as N mais recentes (DESC) e depois reordena pra ordem cronológica
        stmt = (
            select(Mensagem)
            .where(Mensagem.session_id == session_id)
            .order_by(Mensagem.criado_em.desc(), Mensagem.id.desc())
            .limit(limite)
        )
        recentes = sessao.scalars(stmt).all()
        return list(reversed(recentes))


@app.delete("/historico/{session_id}")
def limpar_historico(session_id: str):
    with Session(engine) as sessao:
        resultado = sessao.execute(
            delete(Mensagem).where(Mensagem.session_id == session_id)
        )
        sessao.commit()
        return {"mensagem": f"Histórico da sessão '{session_id}' limpo.", "removidas": resultado.rowcount}


@app.get("/health")
def health():
    try:
        with Session(engine) as sessao:
            total = sessao.scalar(select(Mensagem.id).limit(1))
        return {"status": "ok", "banco": "conectado", "tem_dados": total is not None}
    except Exception as exc:
        return {"status": "ok", "banco": "ERRO", "detalhe": str(exc)}


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8003, reload=False)
