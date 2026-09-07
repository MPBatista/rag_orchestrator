# =========================================================
# RETRIEVAL PIPELINE (CORRIGIDO)
# =========================================================
from app.settings import collection
import json
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# Normalização do contexto (EVITA ERRO 400 NO VLLM)
# ---------------------------------------------------------
def normalize_context(context):
    """
    Converte qualquer retorno do Chroma em string limpa.
    Evita listas, dicts, objetos, etc.
    """

    if context is None:
        return ""

    # Caso seja lista de documentos
    if isinstance(context, list):
        return "\n\n".join([str(x) for x in context])

    # Caso seja dict (retorno bruto do Chroma)
    if isinstance(context, dict):
        docs = context.get("documents")
        if docs:
            return "\n\n".join([str(x) for x in docs])
        return json.dumps(context, ensure_ascii=False)

    # Qualquer outra coisa vira string
    return str(context)

# ---------------------------------------------------------
# Função principal de retrieval
# ---------------------------------------------------------
def retrieve(query: str, k: int = 10) -> list[str]:
    """
    Retorna SOMENTE os textos dos documentos recuperados do Chroma.
    Isso é o que o rerank espera.
    """

    results = collection.query(
        query_texts=[query],
        n_results=k
    )

    # results["documents"] é sempre uma lista de strings
    docs = results.get("documents", [])

    # 🔥 Achatar listas internas
    flat_docs = []
    for d in docs:
        if isinstance(d, list):
            flat_docs.extend(d)
        else:
            flat_docs.append(d)

    return flat_docs
