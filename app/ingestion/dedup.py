import hashlib

def deduplicate(texts, embeddings):
    seen = set()
    unique_texts = []
    unique_embeddings = []

    for text, emb in zip(texts, embeddings):
        h = hashlib.md5(text.encode("utf-8")).hexdigest()
        if h not in seen:
            seen.add(h)
            unique_texts.append(text)
            unique_embeddings.append(emb)

    return unique_texts, unique_embeddings
