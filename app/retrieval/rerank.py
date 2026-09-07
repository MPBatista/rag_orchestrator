from app.settings import embedder

def rerank(query, docs):
    if not docs:
        return []

    query_emb = embedder.encode(
        [query],
        device="cpu",
        convert_to_numpy=True,
        show_progress_bar=False
    )[0]

    doc_embs = embedder.encode(
        docs,
        device="cpu",
        convert_to_numpy=True,
        show_progress_bar=False
    )

    scores = (doc_embs @ query_emb)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

    return [{"text": d, "score": float(s)} for d, s in ranked]
