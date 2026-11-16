from app.retrieval.retrieve import normalize_context
import requests
from app.settings import (
    _healthcheck,
    get_model_url,
    LLM_MODELS,
    FALLBACK_ORDER,
    HEALTHCHECK_TIMEOUT,
)

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# =========================================================
# RESOLVE MODEL ID DINAMICAMENTE
# =========================================================
def _resolve_model_id(url: str) -> str:
    """
    Consulta /v1/models no momento da inferência para obter o ID ativo.
    """
    try:
        resp = requests.get(f"{url.rstrip('/')}/v1/models", timeout=5.0)
        if resp.status_code == 200:
            models = resp.json().get("data", [])
            if models and "id" in models[0]:
                return models[0]["id"]
    except Exception as e:
        logger.warning(f"Não foi possível obter /v1/models de {url}, usando '/model': {e}")
    return "/model"

# =========================================================
# GENERATE LLM ANSWER CONFIG
# =========================================================
def generate_llm_answer(model_name: str, question: str, context: str):
    """
    Gera resposta usando fallback automático.
    """

    errors = {}

    # Normaliza contexto ANTES de qualquer tentativa
    context = normalize_context(context)

    logger.info("=== generate_llm_answer() chamado ===")
    logger.info(f"Pergunta: {question[:200]}")
    logger.info(f"Contexto normalizado (200 chars): {context[:200]}")
    logger.info(f"Modelo solicitado: {model_name}")

    try:
        url = get_model_url(model_name)
        if _healthcheck(url):
            return _call_model(model_name, question, context)
        else:
            errors[model_name] = "unhealthy"
    except Exception as e:
        errors[model_name] = str(e)

    # Fallbacks (se quiser manter)
    for fallback in FALLBACK_ORDER:
        if fallback == model_name:
            continue

        try:
            url = get_model_url(fallback)
            if _healthcheck(url):
                return _call_model(fallback, question, context)
            else:
                errors[fallback] = "unhealthy"
        except Exception as e:
            errors[fallback] = str(e)

    raise RuntimeError(f"Todos os modelos falharam: {errors}")

# =========================================================
# CALL MODEL CONFIG
# =========================================================
def _call_model(model_name: str, question: str, context: str) -> str:
    """
    Chama um backend específico (sem fallback).
    """

    # Normalização do contexto ANTES de montar o prompt
    context = normalize_context(context)

    if not question:
        raise ValueError("Pergunta vazia ou inválida")

    prompt = (
        "You are a helpful assistant.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

    prompt = sanitize(prompt)
    url = get_model_url(model_name)
    active_model_id = _resolve_model_id(url)

    payload = {
        "model": active_model_id,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 512,
        "temperature": 0.2,
        "stream": False
    }

    logger.info(f"Chamando modelo '{active_model_id}' ({model_name}) em {url}")
    logger.warning("PAYLOAD ENVIADO AO MODELO:\n" + str(payload))

    # CORREÇÃO: Aumentado timeout para 180 segundos para suportar processamento de contextos longos em CPU
    r = requests.post(f"{url}/v1/chat/completions", json=payload, timeout=180.0)
    r.raise_for_status()
    data = r.json()

    logger.info(f"Resposta recebida do modelo {model_name}")

    return (
        data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
    )

# =========================================================
# Sanitiza o contexto
# =========================================================
def sanitize(text: str) -> str:
    return (
        text.encode("utf-8", "ignore")   # remove bytes inválidos
            .decode("utf-8", "ignore")
            .replace("\x00", "")         # remove null bytes
    )
