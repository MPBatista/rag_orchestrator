from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

from app.llm import generate_answer, _call_model  # _call_model é usado para bypass do fallback
from app.settings import (
    healthcheck_all,
    LLM_MODELS,
    FALLBACK_ORDER,
    get_model_url
)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "llama3"
    question: str
    context: str = ""
    temperature: float = 0.2
    max_tokens: int = 512

app = FastAPI(
    title="RAG Orchestrator",
    description="Orquestrador multi-modelo com fallback automático",
    version="1.0.0"
)

# ---------------------------------------------------------
# HEALTHCHECK DO ORQUESTRADOR + BACKENDS
# ---------------------------------------------------------
@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "orchestrator": "ok",
        "backends": healthcheck_all()
    }

# ---------------------------------------------------------
# LISTA DE MODELOS + STATUS
# ---------------------------------------------------------
@app.get("/models")
def models() -> Dict[str, Any]:
    backend_status = healthcheck_all()

    return {
        "available_models": [
            {
                "name": name,
                "url": LLM_MODELS[name],
                "healthy": backend_status.get(name, False)
            }
            for name in LLM_MODELS
        ],
        "fallback_order": FALLBACK_ORDER
    }


# ---------------------------------------------------------
# ENDPOINT PRINCIPAL DE CHAT (COM FALLBACK)
# ---------------------------------------------------------
@app.post("/chat")
def chat(body: ChatRequest) -> Dict[str, Any]:
    answer = generate_answer(
        model_name=body.model,
        question=body.question,
        context=body.context
    )

    return {
        "model_requested": body.model,
        "answer": answer
    }


# ---------------------------------------------------------
# ENDPOINT /direct/{model} — SEM FALLBACK
# ---------------------------------------------------------
@app.post("/direct/{model}")
def direct_chat(model: str, body: ChatRequest) -> Dict[str, Any]:
    """
    Chama um modelo específico SEM fallback.
    Se o modelo estiver offline, retorna erro imediatamente.
    """

    if model not in LLM_MODELS:
        raise HTTPException(status_code=404, detail=f"Modelo '{model}' não existe.")

    try:
        # Chamada direta ao backend, ignorando fallback
        answer = _call_model(
            model_name=model,
            question=body.question,
            context=body.context
        )

        return {
            "model_used": model,
            "answer": answer
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao chamar modelo '{model}': {str(e)}"
        )
