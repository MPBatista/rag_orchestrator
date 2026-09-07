import requests
import time
from app.chat.retrieval import retrieve
from app.settings import get_model_url, FALLBACK_ORDER

#MODEL_ID = "bartowski/Meta-Llama-3-8B-Instruct-AWQ"
MODEL_ID = "/model"

def call_model(prompt: str, model: str, temperature: float, max_tokens: int):
    base_url = get_model_url(model)  # ex: http://vllm_llama3:8001
    url = f"{base_url}/v1/chat/completions"
    for attempt in range(5):
        try:
            response = requests.post(
                url,
                json={
                    "model": MODEL_ID,
                    "messages": [
                         {
                            "role": "user",
                            "content": prompt
                         }
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens
                },
                timeout=20
            )
            response.raise_for_status()
            return extract_answer(response.json())
        except Exception as e:
            print(f"[chat] Modelo {model} falhou: {e}")
            time.sleep(2)

    return None

def extract_answer(data: dict) -> str | None:
    """
    Extrai a resposta no formato OpenAI ChatCompletion.
    """
    try:
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = message.get("content", "")
        return content.strip() if content else None
    except Exception:
        return None

def generate_answer(
    question: str,
    model: str,
    context: str,
    temperature: float,
    max_tokens: int
):
    # 1. Recupera contexto do RAG
    retrieved = retrieve(question)
    rag_context = "\n".join([f"- {item['text']}" for item in retrieved])

    # 2. Junta contexto manual + contexto RAG
    full_context = f"{context}\n{rag_context}".strip()

    prompt = f"""
Você é um assistente especializado em responder perguntas com base no contexto abaixo.

Contexto:
{full_context}

Pergunta:
{question}

Resposta:
"""

    # 3. Monta lista de modelos com fallback
    models_to_try = []

    if model in FALLBACK_ORDER:
        models_to_try.append(model)

    for m in FALLBACK_ORDER:
        if m not in models_to_try:
            models_to_try.append(m)

    # 4. Tenta cada modelo até um responder
    for m in models_to_try:
        answer = call_model(prompt, m, temperature, max_tokens)
        if answer:
            return {
                "answer": answer,
                "model_used": m
            }

    # 5. Se todos falharem
    return {
        "answer": "Nenhum modelo conseguiu responder no momento.",
        "model_used": None
    }
