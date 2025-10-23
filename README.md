# RAG-Playground

A small collection of RAG (Retrieval-Augmented Generation) implementations:
- Dense (FAISS + embeddings)
- Sparse (BM25)
- Hybrid (BM25 + Dense)
- Rerank (dense retrieval + model-based re-ranking)

All modules are organized under `src/rag/`. Use `cli.py` to run different modes.

## Quickstart

1. Create a virtualenv and install requirements:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
