  import streamlit as st
import fitz  # PyMuPDF
import faiss
import ollama
import numpy as np
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="ResearchMind AI",
    layout="wide"
)

# ---------- Cached resources (load once, not on every rerun) ----------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


model = load_embedding_model()


# ---------- Core functions ----------

def extract_text(pdf_file):
    pdf = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in pdf:
        text += page.get_text()
    pdf.close()
    return text


def chunk_text(text, chunk_size=1000, overlap=200):
    """
    Word-boundary-aware chunking so we don't cut words in half,
    which hurts embedding quality.
    """
    words = text.split()
    chunks = []
    current_chunk = []
    current_len = 0

    for word in words:
        current_chunk.append(word)
        current_len += len(word) + 1

        if current_len >= chunk_size:
            chunks.append(" ".join(current_chunk))
            # keep the tail of the chunk for overlap
            overlap_words = current_chunk[-(overlap // 6):]  # rough word estimate
            current_chunk = overlap_words
            current_len = sum(len(w) + 1 for w in overlap_words)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def build_vector_db(chunks):
    embeddings = model.encode(chunks, show_progress_bar=False)
    embeddings = np.array(embeddings).astype("float32")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


def retrieve_context(question, index, chunks, top_k=3):
    q_embedding = model.encode([question])
    q_embedding = np.array(q_embedding).astype("float32")
    top_k = min(top_k, len(chunks))
    distances, indices = index.search(q_embedding, top_k)
    return [chunks[idx] for idx in indices[0]]


def check_ollama_available(model_name="llama3.2:1b"):
    try:
        ollama.list()
        return True
    except Exception:
        return False


def ask_llama(question, contexts, model_name="llama3.2:1b"):
    context_text = "\n\n".join(contexts)

    prompt = f"""Use ONLY the context below to answer. If the answer isn't in the context, say so.

Context:
{context_text}

Question:
{question}

Answer:"""

    try:
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"]
    except Exception as e:
        return f"⚠️ Could not reach Ollama ({e}). Make sure `ollama run {model_name}` works in a terminal."


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
    ollama_ok = check_ollama_available()
    if ollama_ok:
        st.success("Ollama is reachable ✅")
    else:
        st.error("Ollama not reachable — start it with `ollama run llama3.1`")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file and uploaded_file.name != st.session_state.filename:
    with st.spinner("Extracting text and building vector index..."):
        text = extract_text(uploaded_file)

        if not text.strip():
            st.error("No extractable text found — this PDF may be scanned images. Try an OCR'd file.")
        else:
            chunks = chunk_text(text, chunk_size=chunk_size)
            index = build_vector_db(chunks)

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
                question, st.session_state.index, st.session_state.chunks, top_k=top_k
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
