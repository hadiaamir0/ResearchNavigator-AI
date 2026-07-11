"""
app.py
Main Streamlit application — ties together PDF processing,
vector search, and the local LLM.
"""

import streamlit as st

from pdf_utils import extract_text, chunk_text
from vector_store import load_embedding_model, build_vector_db, retrieve_context
from llm import check_ollama_available, ask_llama, DEFAULT_MODEL

st.set_page_config(page_title="ResearchMind AI", layout="wide")

model = load_embedding_model()

# ---------- Session state setup ----------

if "index" not in st.session_state:
    st.session_state.index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "filename" not in st.session_state:
    st.session_state.filename = None
if "history" not in st.session_state:
    st.session_state.history = []

# ---------- UI ----------

st.title("ResearchMind AI")
st.subheader("Intelligent PDF Research Assistant using Retrieval-Augmented Generation (RAG)")

with st.sidebar:
    st.markdown("### Settings")
    top_k = st.slider("Chunks to retrieve", 1, 10, 3)
    chunk_size = st.slider("Chunk size (chars)", 500, 2000, 1000, step=100)

    if check_ollama_available():
        st.success("Ollama is reachable ✅")
    else:
        st.error(f"Ollama not reachable — start it with `ollama run {DEFAULT_MODEL}`")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file and uploaded_file.name != st.session_state.filename:
    with st.spinner("Extracting text and building vector index..."):
        text = extract_text(uploaded_file)

        if not text.strip():
            st.error("No extractable text found — this PDF may be scanned images. Try an OCR'd file.")
        else:
            chunks = chunk_text(text, chunk_size=chunk_size)
            index = build_vector_db(chunks, model)

            st.session_state.index = index
            st.session_state.chunks = chunks
            st.session_state.filename = uploaded_file.name
            st.session_state.history = []

    if st.session_state.index is not None:
        st.success(f"Processed '{uploaded_file.name}' into {len(st.session_state.chunks)} chunks")

if st.session_state.index is not None:
    question = st.text_input("Ask a question about the PDF")

    if st.button("Generate Answer") and question.strip():
        with st.spinner("Retrieving context and generating answer..."):
            contexts = retrieve_context(
                question, st.session_state.index, st.session_state.chunks, model, top_k=top_k
            )
            answer = ask_llama(question, contexts)

        st.session_state.history.append((question, answer, contexts))

    for q, a, contexts in reversed(st.session_state.history):
        st.markdown(f"### Q: {q}")
        st.write(a)
        with st.expander("Retrieved sources"):
            for i, c in enumerate(contexts):
                st.markdown(f"**Source {i+1}**")
                st.write(c)
        st.divider()
else:
    st.info("Upload a PDF to get started.")
