import re

def chunk_text(text: str, max_words_per_chunk: int = 150) -> list:
    """
    Splits text into chunks of ~150 words only if text exceeds 500 words.
    Otherwise returns the entire text as a single chunk.
    """
    words = text.strip().split()
    word_count = len(words)

    # Do not split if total word count is less than 500
    if word_count < 500:
        return [text.strip()]

    # Otherwise split into ~150-word chunks
    chunks = []
    for i in range(0, word_count, max_words_per_chunk):
        chunk = " ".join(words[i:i + max_words_per_chunk])
        chunks.append(chunk.strip())

    return chunks
