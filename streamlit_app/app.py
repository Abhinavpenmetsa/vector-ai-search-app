# app.py
import streamlit as st
from utils.pdf_parser import extract_text_from_pdf
from utils.chunking import chunk_text
from embedding_api import embed_and_store

st.set_page_config(page_title="Vector AI Embedder", layout="wide")

st.markdown("<h1 style='text-align: center;'>🧠 Vector AI Embedder</h1>", unsafe_allow_html=True)

st.markdown("### Upload & Embed")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)
    st.write(f"📄 Extracted **{len(text.split())}** words.")

    chunks = chunk_text(text)
    st.write(f"📑 Generated **{len(chunks)}** chunks.")

    if st.button("🔄 Embed"):
        embed_and_store(uploaded_file.name, chunks)
        st.success("✅ Embedding complete!")