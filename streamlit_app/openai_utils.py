import os
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Use model from .env or fallback to gpt-4.1-mini
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")


def get_openai_embedding(text: str) -> np.ndarray:
    """Get embeddings using OpenAI's text-embedding-3-small model."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding)


def summarize_text(text: str, max_length: int = 150) -> str:
    """Summarize text using OpenAI's GPT model."""
    prompt = f"""Please provide a concise summary of the following text in {max_length} words or less:

{text}

Summary:"""

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes text concisely."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100,
        temperature=0.3
    )
    return response.choices[0].message.content.strip()


def answer_question(context: str, question: str) -> str:
    """Answer a question based on the provided context using OpenAI's GPT model."""
    prompt = f"""Based on the following context, please answer the question. If the answer cannot be found in the context, say "I cannot find the answer in the provided context."

Context:
{context}

Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0.3
    )
    return response.choices[0].message.content.strip()
