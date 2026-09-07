import os
from urllib.parse import urlparse
import chromadb
from sentence_transformers import SentenceTransformer

# =========================================================
# LLM ENDPOINTS — MULTI-MODEL + FALLBACK
# =========================================================

# URLs dos modelos servidos pelo docker-compose
VLLM_URL = os.getenv("VLLM_URL", "http://vllm-server:8000")
LLAMA_CPP_URL  = os.getenv("LLAMA_CPP_URL",  "http://llama-server:8080")

# Dicionário determinístico de modelos
LLM_MODELS = {
    "vllm": VLLM_URL,
    "llama_cpp":  LLAMA_CPP_URL
}

# Ordem de fallback (determinística)
FALLBACK_ORDER = ["vllm", "llama_cpp"]

# =========================================================
# GET MODEL URL
# =========================================================
def get_model_url(model_name: str) -> str:
    """
    Retorna a URL do modelo solicitado.
    """
    if model_name not in LLM_MODELS:
        raise ValueError(f"Modelo '{model_name}' não está configurado.")
    return LLM_MODELS[model_name]

# =========================================================
# HEALTHCHECK CONFIG
# =========================================================
HEALTHCHECK_TIMEOUT = 3  # segundos

# =========================================================
# ChromaDB
# =========================================================
CHROMADB_URL = os.getenv("CHROMADB_URL", "http://chromadb:8000")

parsed = urlparse(CHROMADB_URL)
chroma_host = parsed.hostname
chroma_port = parsed.port

client = chromadb.HttpClient(host=chroma_host, port=chroma_port)

# =========================================================
# Embeddings CPU
# =========================================================
embedder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

class CPUEmbedder:
    def __init__(self, model):
        self.model = model

    def __call__(self, input):
        return self.model.encode(
            input,
            device="cpu",
            convert_to_numpy=True,
            show_progress_bar=False
        )

    def embed_query(self, input):
        return self.model.encode(
            input,
            device="cpu",
            convert_to_numpy=True,
            show_progress_bar=False
        )

    def name(self):
        return "cpu_embedder"

cpu_embedder = CPUEmbedder(embedder)

# =========================================================
# ChromaDB Collection
# =========================================================
collection = client.get_or_create_collection(
    name="rag_cpu",
    embedding_function=cpu_embedder
)

def healthcheck_all():
    status = {
        "status": "ok",
        "collection": collection.name if collection else None,
        "embedder": embedder.__class__.__name__ if embedder else None,
        "models": {}
    }

    for model_name, url in LLM_MODELS.items():
        try:
            if _healthcheck(url):
                status["models"][model_name] = "healthy"
            else:
                status["models"][model_name] = "unhealthy"
        except Exception:
            status["models"][model_name] = "error"

    return status
