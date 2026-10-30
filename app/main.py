from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict, Any
from app.ingestion.pipeline import ingest_folder, ingest_text, ingest_text_content
from app.retrieval.retrieve import retrieve
from app.retrieval.rerank import rerank
from app.chat.chat import generate_answer
from app.chat.schemas import ChatRequest  # vamos colocar a classe em schemas.py
from app.llm import generate_llm_answer, _call_model  # _call_model é usado para bypass do fallback
from app.settings import (
    healthcheck_all,
    LLM_MODELS,
    FALLBACK_ORDER,
    get_model_url
)
from app.ingestion.loader import load_file

class Message(BaseModel):
    role: str
    content: str

#class ChatRequest(BaseModel):
#    model: str = "llama3"
#    question: str
#    context: str = ""
#    temperature: float = 0.2
#    max_tokens: int = 512

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
def ingest_text_endpoint(req: dict):
    text = req["text"]
    result = ingest_text_content(text, source="text_input")
    return result

@app.post("/ingest_file")
async def ingest_file(file: UploadFile = File(...)):
    content = await file.read()
    text = load_file(file.filename)
    result = ingest_text_content(text, source=file.filename)
    return {"filename": file.filename, **result}

# ---------------------------------------------------------
# QUERY DO ORQUESTRADOR + BACKENDS
# ---------------------------------------------------------
@app.post("/query")
def query(req: dict):

    if "question" not in req:
        raise ValueError("Campo 'question' é obrigatório")

    question = req["question"]
    model_name = req.get("model", "llama_cpp")

    # 1. Recuperação REAL (lista de textos)
    docs = retrieve(question)

    print("DOCS:", docs)
    for i, d in enumerate(docs):
        print(i, type(d), d[:200] if isinstance(d, str) else d)

    # 2. Rerank
    ranked = rerank(question, docs)

    # 3. Contexto REAL do RAG (top 5)
    context = "\n\n".join([d["text"] for d in ranked[:5]])

    # 4. Geração
    answer = generate_llm_answer(
        model_name=model_name,
        question=question,
        context=context
    )

    return {
        "model_used": model_name,
        "question": question,
        "context_docs": ranked[:5],
        "answer": answer
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
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao chamar modelo '{model}': {str(e)}"
        )
