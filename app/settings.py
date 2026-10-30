import os
from urllib.parse import urlparse
import urllib.request  # <--- Adicionado para fazer a requisição HTTP nativa
import chromadb
from sentence_transformers import SentenceTransformer
import json # Adicionado no topo se necessário
import traceback  # <--- Adicionado para capturar o erro exato
import logging
import urllib.request

# Configuração básica de log para exibir no console (STDOUT)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("RAG-Healthcheck")

# =========================================================
# LLM ENDPOINTS — MULTI-MODEL + FALLBACK
# =========================================================

LLAMA_CPP_URL  = os.getenv("LLAMA_CPP_URL",  "http://llama-server:8080")

LLM_MODELS = {
    "llama_cpp":  LLAMA_CPP_URL
}

FALLBACK_ORDER = [
    "llama_cpp"
]

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
HEALTHCHECK_TIMEOUT = 5  # Aumentado para 5 segundos por segurança

def _healthcheck(url: str) -> bool:
    health_url = f"{url.rstrip('/')}/health"

    logger.info(f"Iniciando healthcheck na URL: {health_url}")

    req = urllib.request.Request(
        health_url,
        headers={'Accept': 'application/json', 'User-Agent': 'RAG-Healthcheck'}
    )

    try:
        with urllib.request.urlopen(req, timeout=HEALTHCHECK_TIMEOUT) as response:
            status_code = response.status
            body_content = response.read().decode('utf-8')

            # Log da resposta recebida com sucesso
            logger.info(f"Resposta recebida de {health_url} -> Status HTTP: {status_code} | Body: {body_content}")

            if status_code in (200, 201, 204):
                res_json = json.loads(body_content)
                if res_json.get("status") == "ok" or res_json.get("health") == "ok":
                    return True

            raise ValueError(f"HTTP Status ou JSON inválido. Status: {status_code}")

    except Exception as e:
        # Log detalhado caso ocorra qualquer exceção (Timeout, ConnectionRefused, etc)
        logger.error(f"Falha ao conectar/ler {health_url}. Erro: {type(e).__name__} - {str(e)}")
        raise e

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
        except Exception as e:
            # Captura o erro técnico para visualização no JSON de saída
            error_line = traceback.format_exc().splitlines()[-2] if traceback.format_exc().splitlines() else ""
            error_msg = f"{type(e).__name__}: {str(e)} | Line: {error_line}"
            status["models"][model_name] = f"error -> {error_msg}"
            status["status"] = "error"

    return status

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
