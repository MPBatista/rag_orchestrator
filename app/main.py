import logging
import os
import tempfile

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from typing import Dict, Any
from app.ingestion.pipeline import ingest_folder, ingest_text, ingest_text_content
from app.retrieval.retrieve import retrieve
from app.retrieval.rerank import rerank
from app.chat.chat import generate_answer
from app.chat.schemas import ChatRequest
from app.llm import generate_llm_answer, _call_model  # _call_model é usado para bypass do fallback
from app.settings import (
    healthcheck_all,
    LLM_MODELS,
    FALLBACK_ORDER,
    get_model_url
)
from app.ingestion.loader import load_file

logger = logging.getLogger("rag.main")

# ---------------------------------------------------------
# SCHEMAS (validação automática -> 422, nunca mais 500 de payload)
# ---------------------------------------------------------
class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    model: str = "llama_cpp"
    top_k: int = Field(default=5, ge=1, le=20)

class IngestTextRequest(BaseModel):
    text: str = Field(min_length=1)

app = FastAPI(
    title="RAG Orchestrator",
    description="Orquestrador multi-modelo com fallback automático",
    version="1.0.0"
)

# ---------------------------------------------------------
# INGEST DO ORQUESTRADOR + BACKENDS
# ---------------------------------------------------------
@app.post("/ingest")
def ingest(req: dict):
    if "folder" in req:
        return {"ingested_chunks": ingest_folder(req["folder"])}

    if "text" in req:
        return {"ingested_chunks": ingest_text(req["text"])}

    raise HTTPException(400, "Envoie 'folder' ou 'text'")

@app.post("/ingest_text")
def ingest_text_endpoint(req: IngestTextRequest):
    return ingest_text_content(req.text, source="text_input")

@app.post("/ingest_file")
async def ingest_file(file: UploadFile = File(...)):
    content = await file.read()
    if not content:
        raise HTTPException(400, "Arquivo vazio")

    # Salva o upload num arquivo temporário com a extensão correta
    # (load_file precisa de um CAMINHO real, não do nome do arquivo)
    suffix = os.path.splitext(file.filename or "")[1].lower()
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        text = load_file(tmp_path)
        if not text or not text.strip():
            raise HTTPException(400, f"Nenhum texto extraível de '{file.filename}'")
        result = ingest_text_content(text, source=file.filename)
        return {"filename": file.filename, **result}
    except HTTPException:
        raise
    except Exception:
        logger.exception("ingest_file falhou para %s", file.filename)
        raise HTTPException(500, f"Falha ao processar '{file.filename}'")
    finally:
        if tmp_path:
            os.unlink(tmp_path)

# ---------------------------------------------------------
# QUERY DO ORQUESTRADOR + BACKENDS
# ---------------------------------------------------------
@app.post("/query")
def query(req: QueryRequest):
    # Pydantic já garantiu question não-vazia e top_k inteiro entre 1 e 20.
    if req.model not in LLM_MODELS:
        raise HTTPException(
            404,
            f"Modelo '{req.model}' desconhecido. Disponíveis: {list(LLM_MODELS.keys())}",
        )
    try:
        # retrieve() retorna list[str] — NÃO embrulhar em dict aqui,
        # rerank() já devolve [{"text": str, "score": float}]
        docs = retrieve(req.question)
        ranked = rerank(req.question, docs)
        top = ranked[: req.top_k]
        context = "\n\n".join(d["text"] for d in top)
        answer = generate_llm_answer(req.model, req.question, context)
    except Exception:
        logger.exception("query falhou")  # traceback completo no log do container
        raise HTTPException(502, "Pipeline RAG indisponível (retrieve/rerank/LLM)")

    return {
        "model_used": req.model,
        "question": req.question,
        "context_docs": top,
        "answer": answer,
    }

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
def get_available_models_route():
    # 1. Executa o healthcheck para pegar o estado em tempo real
    health_results = healthcheck_all()

    available_models = []

    # 2. Varre os modelos configurados e busca o status real de dentro do health_results
    for model_name, url in LLM_MODELS.items():
        # Captura o status gerado na função healthcheck_all
        model_status = health_results["models"].get(model_name, "unhealthy")

        available_models.append({
            "name": model_name,
            "url": url,
            # Se for "healthy", define como True. Caso contrário (unhealthy ou error), como False
            "healthy": model_status == "healthy"
        })

    return {
        "available_models": available_models,
        "fallback_order": FALLBACK_ORDER
    }

# ---------------------------------------------------------
# ENDPOINT PRINCIPAL DE CHAT (COM FALLBACK)
# ---------------------------------------------------------
@app.post("/chat")
def chat(req: ChatRequest):
    result = generate_answer(
        question=req.question,
        model=req.model,
        context=req.context,
        temperature=req.temperature,
        max_tokens=req.max_tokens
    )
    return result

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
        logger.exception("direct_chat falhou para modelo %s", model)
        raise HTTPException(
            status_code=502,
            detail=f"Erro ao chamar modelo '{model}': {str(e)}"
        )
