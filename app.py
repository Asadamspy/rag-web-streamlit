import streamlit as st
import requests
from bs4 import BeautifulSoup
import numpy as np
import faiss
import os
from sentence_transformers import SentenceTransformer
from groq import Groq

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Web RAG Application",
    layout="wide"
)

st.title("🌐 Web-based RAG Application")
st.caption("Ask questions from live webpages using Retrieval-Augmented Generation (RAG)")

# --------------------------------------------------
# LOAD GROQ API KEY (FROM STREAMLIT SECRETS / ENV)
# --------------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found. Please set it in Streamlit Secrets.")
    st.stop()

# --------------------------------------------------
# LOAD MODELS (CACHED)
# --------------------------------------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_llm():
    return Groq(api_key=GROQ_API_KEY)

embedder = load_embedder()
client = load_llm()

# --------------------------------------------------
# WEB PAGE LOADER (CLEAN CONTENT)
# --------------------------------------------------
def load_web_page(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove noisy elements
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    # Prefer main/article content
    main = soup.find("article") or soup.find("main")
    text = main.get_text(" ") if main else soup.get_text(" ")

    return " ".join(text.split())

# --------------------------------------------------
# CHUNKING WITH NOISE CONTROL
# --------------------------------------------------
def chunk_text(
    text: str,
    chunk_size: int = 400,
    overlap: int = 80,
    min_words: int = 120
):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk = " ".join(chunk_words)

        # Filters to reduce noise
        if len(chunk_words) >= min_words and chunk.count(".") >= 2:
            chunks.append(chunk)

        start = end - overlap

    return chunks

# --------------------------------------------------
# BUILD FAISS INDEX
# --------------------------------------------------
def build_faiss(chunks):
    embeddings = embedder.encode(chunks, show_progress_bar=False)
    embeddings = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

# --------------------------------------------------
# RETRIEVAL
# --------------------------------------------------
def retrieve(query, index, chunks, k=5):
    q_emb = embedder.encode([query]).astype("float32")
    faiss.normalize_L2(q_emb)
    _, indices = index.search(q_emb, k)
    return [chunks[i] for i in indices[0]]

# --------------------------------------------------
# PROMPT
# --------------------------------------------------
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

# --------------------------------------------------
# RAG PIPELINE
# --------------------------------------------------
def ask_web_rag(question, index, chunks):
    retrieved_chunks = retrieve(question, index, chunks)
    context = "\n\n".join(retrieved_chunks)

    prompt = build_prompt(context, question)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content, retrieved_chunks

# --------------------------------------------------
# STREAMLIT UI
# --------------------------------------------------
url = st.text_input("🌍 Enter Web URL")
question = st.text_input("❓ Ask a question")

if st.button("Run RAG"):
    if not url or not question:
        st.warning("Please enter both URL and question.")
    else:
        with st.spinner("🔎 Processing webpage and building RAG pipeline..."):
            try:
                text = load_web_page(url)
                chunks = chunk_text(text)

                if not chunks:
                    st.error("No meaningful content found on this webpage.")
                else:
                    index = build_faiss(chunks)
                    answer, sources = ask_web_rag(question, index, chunks)

                    st.subheader("✅ Answer")
                    st.write(answer)

                    st.subheader("📌 Retrieved Context")
                    for i, src in enumerate(sources, 1):
                        st.markdown(f"**Chunk {i}:** {src[:300]}...")

            except Exception as e:
                st.error(f"Error: {e}")
