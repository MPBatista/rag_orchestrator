import re

def semantic_chunk(text, max_len=500):
    # quebra por parágrafos
    paragraphs = re.split(r"\n\s*\n", text)

    chunks = []
    buffer = ""

    for p in paragraphs:
        if len(buffer) + len(p) < max_len:
            buffer += " " + p
        else:
            chunks.append(buffer.strip())
            buffer = p

    if buffer.strip():
        chunks.append(buffer.strip())

    return chunks
