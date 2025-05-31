# app.py
import streamlit as st
import requests
from utils.pdf_parser import extract_text_from_pdf
from utils.chunking import chunk_text
from embedding_api import embed_and_store
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API endpoint
API_URL = "http://localhost:8000"  # Update this if your API runs on a different port

st.set_page_config(page_title="Vector AI Search", layout="wide")

st.markdown("<h1 style='text-align: center;'>🧠 Vector AI Search</h1>", unsafe_allow_html=True)

# Sidebar for settings
with st.sidebar:
    st.markdown("### Settings")
    use_openai = st.toggle("Use OpenAI Embeddings", value=os.getenv("USE_OPENAI_EMBEDDINGS", "false").lower() == "true")
    summarize_results = st.toggle("Summarize Results", value=False)
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This app allows you to:
    1. Upload and embed PDF documents
    2. Search through documents using semantic search
    3. Ask questions about the documents
    4. Get AI-powered summaries of search results
    """)

# Main content
tab1, tab2 = st.tabs(["📚 Upload & Embed", "🔍 Search & Ask"])

with tab1:
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

with tab2:
    st.markdown("### Search & Ask")
    
    # Search section
    st.markdown("#### Search Documents")
    search_query = st.text_input("Enter your search query")
    if search_query:
        try:
            response = requests.get(
                f"{API_URL}/search",
                params={
                    "query": search_query,
                    "top_k": 3,
                    "summarize": summarize_results
                }
            )
            results = response.json()
            
            for i, result in enumerate(results["results"], 1):
                with st.expander(f"Result {i} (Score: {result['score']:.2f}) - {result['file']}"):
                    st.markdown(f"**Best matching text:**\n{result['best_sentence']}")
                    if summarize_results and "summary" in result:
                        st.markdown(f"**Summary:**\n{result['summary']}")
                    st.markdown(f"*Chunk index: {result['chunk_index']}*")
        except Exception as e:
            st.error(f"Error searching documents: {str(e)}")
    
    st.markdown("---")
    
    # Question answering section
    st.markdown("#### Ask Questions")
    question = st.text_input("Ask a question about your documents")
    if question:
        try:
            with st.spinner("Thinking..."):
                response = requests.get(
                    f"{API_URL}/ask",
                    params={
                        "question": question,
                        "top_k": 3
                    }
                )
                result = response.json()
                
                st.markdown("### Answer")
                st.markdown(result["answer"])
                
                with st.expander("View Sources"):
                    for source in result["sources"]:
                        st.markdown(f"- {source['file']} (Chunk {source['chunk_index']})")
        except Exception as e:
            st.error(f"Error getting answer: {str(e)}")