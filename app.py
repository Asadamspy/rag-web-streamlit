import streamlit as st
import requests
from bs4 import BeautifulSoup
import numpy as np
import faiss
import re
from sentence_transformers import SentenceTransformer
from groq import Groq
import os

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(page_title="Web RAG with Groq", layout="wide")

st.title("🌐 Web-based RAG Application")
st.write("Ask questions from live web pages using Retrieval-Augmented Generation (RAG).")

# ----------------------------
# API KEY
# ----------------------------
st.sidebar.header("🔑 Groq API Key")
groq_key = st.sidebar.text_input(
    "Enter your Groq API Key",
    type="password"
)

if groq_key:
    os.environ["GROQ_API_KEY"] = groq_key

# ----------------------------
# INITIALIZE MODELS
# ----------------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

def load_llm():
    return Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ----------------------------
# WEB LOADER (CLEAN)
# ----------------------------
def load_web_page(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    html = requests.get(url, headers=headers, timeout=10).text
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    main = soup.find("article") or soup.find("main")
    text = main.get_text(" ") if main else soup.get_text(" ")
    return " ".join(text.split())

# ----------------------------
# CHUNKING WITH NOISE CONTROL
# ----------------------------
def chunk_text(
    text,
    chunk_size=400,
    overlap=80,
    min_words=120
):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk = " ".join(chunk_words)

        if len(chunk_words) >= min_words and chunk.count(".") >= 2:
            chunks.append(chunk)

        start = end - overlap

    return chunks

# ----------------------------
# BUILD FAISS
# ----------------------------
def build_faiss(chunks):
    embeddings = embedder.encode(chunks, show_progress_bar=False)
    embeddings = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

# ----------------------------
# RETRIEVAL
# ----------------------------
def retrieve(query, index, chunks, k=5):
    q_emb = embedder.encode([query]).astype("float32")
    faiss.normalize_L2(q_emb)
    _, idx = index.search(q_emb, k)
    return [chunks[i] for i in idx[0]]

# ----------------------------
# PROMPT
# ----------------------------
def build_prompt(context, question):
    return f"""
You are a document-based assistant.

Answer the question strictly using the context below.
If the answer is not present, say:
"Information not available on the provided webpage."

Context:
{context}

Question:
{question}

Answer:
"""

# ----------------------------
# RAG PIPELINE
# ----------------------------
def ask_web_rag(question, index, chunks):
    retrieved = retrieve(question, index, chunks)
    context = "\n\n".join(retrieved)

    prompt = build_prompt(context, question)

    client = load_llm()
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content, retrieved

# ----------------------------
# STREAMLIT UI
# ----------------------------
url = st.text_input("🌍 Enter Web URL")
question = st.text_input("❓ Ask a question")

if st.button("Run RAG"):
    if not groq_key:
        st.error("Please enter your Groq API Key.")
    elif not url or not question:
        st.warning("Please enter both URL and question.")
    else:
        with st.spinner("Processing webpage..."):
            text = load_web_page(url)
            chunks = chunk_text(text)

            if not chunks:
                st.error("No meaningful content found on this page.")
            else:
                index = build_faiss(chunks)
                answer, sources = ask_web_rag(question, index, chunks)

                st.subheader("✅ Answer")
                st.write(answer)

                st.subheader("📌 Retrieved Context")
                for i, src in enumerate(sources, 1):
                    st.markdown(f"**Chunk {i}:** {src[:300]}...")

