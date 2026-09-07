from app.ingestion.loader import load_documents_recursive,clean_text,compute_hash
from app.ingestion.chunking import semantic_chunk
from app.ingestion.dedup import deduplicate
from app.settings import collection, embedder

# =========================================================
# INGEST FOLDER
# =========================================================
def ingest_folder(folder):
    docs = load_documents_recursive(folder)

    total_chunks = 0
    inserted = 0
    skipped = 0

    for path, content, _hash in docs:
        result = ingest_text_content(content, source=path)
        total_chunks += result["chunks_total"]
        inserted += result["chunks_inserted"]
        skipped += result["chunks_skipped_duplicates"]

    return {
        "files": len(docs),
        "chunks_total": total_chunks,
        "chunks_inserted": inserted,
        "chunks_skipped_duplicates": skipped
    }

def ingest_text_content(text: str, source: str = "text_input"):
    # 1. Chunking semântico
    chunks = semantic_chunk(text)

    inserted = 0
    skipped = 0

    for i, chunk in enumerate(chunks):
        # 2. Normalização e limpeza (já feita no loader, mas reforçamos)
        chunk = clean_text(chunk)

        if not chunk:
            continue

        # 3. Hash único por chunk
        hash_id = compute_hash(chunk)

        # 4. Verificar duplicação no Chroma
        if exists_in_chroma(hash_id):
            skipped += 1
            continue

        # 5. Gerar embedding
        embedding = embedder.encode(
            chunk,
            device="cpu",
            convert_to_numpy=True,
            show_progress_bar=False
        )

        # 6. Inserir no Chroma
        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            metadatas=[{
                "source": source,
                "chunk_index": i,
                "hash": hash_id
            }],
            ids=[hash_id]
        )

        inserted += 1

    return {
        "chunks_total": len(chunks),
        "chunks_inserted": inserted,
        "chunks_skipped_duplicates": skipped
    }

def exists_in_chroma(hash_id: str) -> bool:
    """
    Verifica se um documento com esse hash já existe no Chroma.
    """
    results = collection.get(
        ids=[hash_id],
        include=[]
    )

    return len(results["ids"]) > 0

# =========================================================
# INGEST FOLDER
# =========================================================
def ingest_folder_old(folder):
    docs = load_documents_recursive(folder)

    all_chunks = []
    for path, content in docs:
        chunks = semantic_chunk(content)
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "id": f"{path}_{i}",
                "text": chunk,
                "metadata": {"source": path, "chunk": i}
            })

    # extrai textos
    texts = [c["text"] for c in all_chunks]

    # embed antes da deduplicação final
    embeddings = embedder.encode(
        texts,
        device="cpu",
        convert_to_numpy=True,
        show_progress_bar=False
    )

    # deduplica textos + embeddings juntos
    texts, embeddings = deduplicate(texts, embeddings)

    # insere no Chroma
    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=[{"source": "ingest"} for _ in texts],
        ids=[f"doc_{i}" for i in range(len(texts))]
    )

    return {
        "files": len(docs),
        "chunks": len(all_chunks),
        "unique_chunks": len(texts)
    }

# =========================================================
# INGEST TEXT
# =========================================================
def ingest_text(text: str):
    # 1. Chunking semântico
    chunks = semantic_chunk(text)

    # 2. Criar estrutura igual à usada no ingest_folder
    all_chunks = []
    for i, chunk in enumerate(chunks):
        all_chunks.append({
            "id": f"text_{i}",
            "text": chunk,
            "metadata": {"source": "text_input", "chunk": i}
        })

    # 3. Extrair textos
    texts = [c["text"] for c in all_chunks]

    # 4. Gerar embeddings
    embeddings = embedder.encode(
        texts,
        device="cpu",
        convert_to_numpy=True,
        show_progress_bar=False
    )

    # 5. Deduplicar textos + embeddings
    texts, embeddings = deduplicate(texts, embeddings)

    # 6. Inserir no Chroma
    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=[{"source": "text_input"} for _ in texts],
        ids=[f"text_{i}" for i in range(len(texts))]
    )

    return {
        "chunks": len(all_chunks),
        "unique_chunks": len(texts)
    }
