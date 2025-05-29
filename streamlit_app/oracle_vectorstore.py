# oracle_vectorstore.py

import oracledb
import os
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
import json

load_dotenv()

# Oracle DSN setup
dsn = oracledb.makedsn(
    os.getenv("ORACLE_HOST"),
    os.getenv("ORACLE_PORT"),
    service_name=os.getenv("ORACLE_SERVICE")
)

# Connect to Oracle and fetch documents
def load_documents_from_oracle():
    conn = oracledb.connect(
        user=os.getenv("ORACLE_USER"),
        password=os.getenv("ORACLE_PASSWORD"),
        dsn=dsn
    )
    cursor = conn.cursor()
    cursor.execute("SELECT file_name, chunk_index, chunk_text FROM pdf_embeddings")
    
    documents = []
    for file_name, chunk_index, chunk_text in cursor.fetchall():
        metadata = {"file": file_name, "chunk_index": chunk_index}
        documents.append(Document(page_content=chunk_text.read(), metadata=metadata))
    
    cursor.close()
    conn.close()
    return documents

# Build FAISS vectorstore using LangChain
def get_oracle_vectorstore():
    documents = load_documents_from_oracle()
    embeddings = OpenAIEmbeddings()  # Uses your OPENAI_API_KEY from .env
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore
