# # ResearchMind AI

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

## Notes

- If the PDF is scanned (image-only), text extraction will fail — you'd need OCR first (e.g. `pytesseract`), which isn't included here.
- Ollama must be running locally (`ollama serve` or via the desktop app) before you ask questions.

