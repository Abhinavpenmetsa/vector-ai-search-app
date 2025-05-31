from fastapi import FastAPI, Query
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
from openai_utils import get_openai_embedding, summarize_text, answer_question
load_dotenv()
import numpy as np
import oracledb
import os
import json
from typing import List, Dict, Any
import re

app = FastAPI(title="Vector Search API")

sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
use_openai = os.getenv("USE_OPENAI_EMBEDDINGS", "false").lower() == "true"

dsn = oracledb.makedsn(
    os.getenv("ORACLE_HOST"),
    os.getenv("ORACLE_PORT"),
    service_name=os.getenv("ORACLE_SERVICE")
)

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def get_embedding(text: str) -> np.ndarray:
    if use_openai:
        return get_openai_embedding(text)
    return sentence_model.encode([text])[0]

def extract_best_sentence(chunk_text: str, query: str) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', chunk_text.strip())
    if not sentences:
        return chunk_text.strip()

    if use_openai:
        query_embedding = get_embedding(query)
        sentence_embeddings = [get_embedding(s) for s in sentences]
        scores = [cosine_similarity(query_embedding, s_emb) for s_emb in sentence_embeddings]
    else:
        sentence_embeddings = sentence_model.encode(sentences)
        query_embedding = sentence_model.encode(query)
        scores = util.pytorch_cos_sim(query_embedding, sentence_embeddings)[0]

    best_index = np.argmax(scores)
    best_sentence = sentences[best_index].strip()

    if len(best_sentence.split()) > 50:
        return summarize_text(best_sentence)
    return best_sentence

@app.get("/search")
def search_chunks(
    query: str = Query(..., description="Search query string"),
    top_k: int = Query(3, description="Number of top results to return"),
    min_score: float = Query(0.25, description="Minimum similarity score to consider"),
    summarize: bool = Query(False, description="Whether to summarize the results")
) -> Dict[str, Any]:
    query_vector = get_embedding(query)

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
            print(f"⚠️ Skipping chunk due to error: {e}")
            continue

    scored_chunks.sort(key=lambda x: x["score"], reverse=True)

    final_results = []
    for item in scored_chunks[:top_k]:
        best_sentence = extract_best_sentence(item["text"], query)

        result = {
            "file": item["file"],
            "chunk_index": item["chunk_index"],
            "score": item["score"],
            "best_sentence": best_sentence,
            "text": item["text"]  # ✅ include full text so `/ask` can use it
        }

        if summarize:
            result["summary"] = summarize_text(item["text"])

        final_results.append(result)

    return {"query": query, "results": final_results}

@app.get("/ask")
def ask_question(
    question: str = Query(..., description="Question to ask about the documents"),
    top_k: int = Query(3, description="Number of relevant chunks to use for answering")
) -> Dict[str, Any]:
    search_results = search_chunks(question, top_k=top_k, min_score=0.2)

    context = "\n\n".join([item["text"] for item in search_results["results"] if "text" in item])

    answer = answer_question(context, question)

    return {
        "question": question,
        "answer": answer,
        "sources": [
            {"file": item["file"], "chunk_index": item["chunk_index"]}
            for item in search_results["results"]
        ]
    }
