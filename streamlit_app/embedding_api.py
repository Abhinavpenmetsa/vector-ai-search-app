# embedding_api.py

from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import numpy as np
import oracledb
import os

# ✅ Explicitly load .env from correct path
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=dotenv_path)

# ✅ Debug: Print loaded environment values to confirm
print("✅ ORACLE_HOST:", os.getenv("ORACLE_HOST"))
print("✅ ORACLE_PORT:", os.getenv("ORACLE_PORT"))
print("✅ ORACLE_SERVICE:", os.getenv("ORACLE_SERVICE"))
print("✅ ORACLE_USER:", os.getenv("ORACLE_USER"))
print("✅ ORACLE_PASSWORD:", os.getenv("ORACLE_PASSWORD"))

def embed_and_store(file_name, chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)

    dsn = oracledb.makedsn(
        os.getenv("ORACLE_HOST"),
        os.getenv("ORACLE_PORT"),
        service_name=os.getenv("ORACLE_SERVICE")
    )

    conn = oracledb.connect(
        user=os.getenv("ORACLE_USER"),
        password=os.getenv("ORACLE_PASSWORD"),
        dsn=dsn
    )
    cursor = conn.cursor()

    for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
        cursor.execute("""
            INSERT INTO pdf_embeddings (file_name, chunk_index, chunk_text, embedding)
            VALUES (:1, :2, :3, :4)
        """, (file_name, i, chunk, str(vector.tolist())))

    conn.commit()
    cursor.close()
    conn.close()
