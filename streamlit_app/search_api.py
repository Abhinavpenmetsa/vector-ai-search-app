# search_api.py
from fastapi import FastAPI, Query
from sentence_transformers import SentenceTransformer, util
import numpy as np
import oracledb
import os
from dotenv import load_dotenv
import json
from typing import List, Dict, Any
import re

load_dotenv()

app = FastAPI(title="Vector Search API")

model = SentenceTransformer("all-MiniLM-L6-v2")

dsn = oracledb.makedsn(
    os.getenv("ORACLE_HOST"),
    os.getenv("ORACLE_PORT"),
    service_name=os.getenv("ORACLE_SERVICE")
)

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def extract_best_sentence(chunk_text: str, query: str) -> str:
    import re
    from difflib import SequenceMatcher

    sentences = re.split(r'(?<=[.!?])\s+', chunk_text.strip())
    if not sentences:
        return chunk_text.strip()

    sentence_embeddings = model.encode(sentences)
    query_embedding = model.encode(query)
    scores = util.pytorch_cos_sim(query_embedding, sentence_embeddings)[0]

    keyword_matches = []
    query_words = set(query.lower().split())

    for i, sentence in enumerate(sentences):
        sentence_clean = sentence.lower()
        word_overlap = len(query_words.intersection(sentence_clean.split()))
        contains_digit = any(char.isdigit() for char in sentence_clean)

        # Strong candidate if it matches many query terms and includes numbers
        if word_overlap >= 2 and contains_digit:
            keyword_matches.append((i, scores[i].item()))

    # If we found keyword-rich sentences, return the highest scoring among them
    if keyword_matches:
        best_i = max(keyword_matches, key=lambda x: x[1])[0]
        return sentences[best_i].strip()

    # Else fallback to highest cosine similarity
    best_index = scores.argmax().item()
    return sentences[best_index].strip()


def keyword_filter(results: List[Dict[str, Any]], keyword: str) -> List[Dict[str, Any]]:
    keyword = keyword.lower()
    for item in results:
        item["keyword_match"] = keyword in item["text"].lower()
    return sorted(results, key=lambda x: (x["keyword_match"], x["score"]), reverse=True)

@app.get("/search")
def search_chunks(
    query: str = Query(..., description="Search query string"),
    top_k: int = Query(3, description="Number of top results to return"),
    min_score: float = Query(0.25, description="Minimum similarity score to consider")
) -> Dict[str, Any]:

    query_vector = model.encode([query])[0]

    connection = oracledb.connect(
        user=os.getenv("ORACLE_USER"),
        password=os.getenv("ORACLE_PASSWORD"),
        dsn=dsn
    )
    cursor = connection.cursor()

    cursor.execute("SELECT file_name, chunk_index, chunk_text, embedding FROM pdf_embeddings")
    rows = cursor.fetchall()

    seen_chunks = set()
    scored_chunks = []

    for file_name, chunk_index, chunk_text_lob, embedding_lob in rows:
        try:
            chunk_text = chunk_text_lob.read()
            embedding_vector = np.array(json.loads(embedding_lob.read()))
            score = cosine_similarity(query_vector, embedding_vector)

            chunk_key = f"{file_name}#{chunk_index}"
            if score >= min_score and chunk_key not in seen_chunks:
                seen_chunks.add(chunk_key)
                scored_chunks.append({
                    "score": round(score, 4),
                    "file": file_name,
                    "chunk_index": chunk_index,
                    "text": chunk_text
                })
        except Exception as e:
            print(f"\u26a0\ufe0f Skipping chunk due to error: {e}")
            continue

    filtered_results = keyword_filter(scored_chunks, query)

    for item in filtered_results[:top_k]:
        item["best_sentence"] = extract_best_sentence(item["text"], query)

    final_results = [
        {
            "file": item["file"],
            "chunk_index": item["chunk_index"],
            "score": item["score"],
            "best_sentence": item["best_sentence"]
        }
        for item in filtered_results[:top_k]
    ]

    return {"query": query, "results": final_results}