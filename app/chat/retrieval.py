from app.settings import collection, embedder

def retrieve(query: str, top_k: int = 5):
    """
    Recupera os chunks mais relevantes do ChromaDB usando embeddings.
    Retorna uma lista de dicts: { "text": ..., "metadata": ... }
    """

    # 1. Gera embedding da query
    query_embedding = embedder.encode([query]).tolist()

    # 2. Consulta o ChromaDB
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )

    # 3. Extrai documentos e metadados
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    retrieved = []
    for text, meta in zip(docs, metas):
        retrieved.append({
            "text": text,
            "metadata": meta
        })

    return retrieved

#    return [
#        {"text": text, "metadata": meta}
#        for text, meta in zip(docs, metas)
#    ]
