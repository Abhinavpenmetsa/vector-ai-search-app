# qa_api.py

from fastapi import FastAPI, Query
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from oracle_vectorstore import get_oracle_vectorstore
from dotenv import load_dotenv
import os

load_dotenv()
app = FastAPI(title="LangChain Q&A API")

@app.get("/ask")
def ask_question(query: str = Query(..., description="User question")):
    try:
        # Load vector DB
        vectorstore = get_oracle_vectorstore()
        
        # Initialize LangChain QA chain
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")  # Requires OPENAI_API_KEY
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        # Ask the question
        response = qa_chain.run(query)

        return {
            "query": query,
            "answer": response
        }

    except Exception as e:
        return {"error": str(e)}
