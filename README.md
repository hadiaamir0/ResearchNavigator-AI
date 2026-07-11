# # ResearchNavigtor AI

An intelligent PDF research assistant built with Retrieval-Augmented Generation (RAG):

- Streamlit (UI)
- Ollama + Llama 3.1 (local LLM)
- FAISS (vector search)
- Sentence Transformers (embeddings)
- PyMuPDF (PDF text extraction)

## Features

- PDF upload and text extraction
- Word-boundary-safe chunking
- Semantic search over document chunks
- Local, private question answering (no data leaves your machine)
- Persistent session — ask multiple questions without reprocessing
- Ollama connection check in the sidebar
- Adjustable chunk size and retrieval depth

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
ollama pull llama3.1
```

## Run

```bash
streamlit run streamlit_app.py
```

