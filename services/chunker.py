


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Splits the input text into chunks of specified size with overlap."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    print(f"Chunked text into {len(chunks)} chunks.")

    return chunks