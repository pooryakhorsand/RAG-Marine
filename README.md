# ⚙️ RAG-Marine  
### _A Unified, Explainable Retrieval-Augmented Generation Toolkit_

RAG-Marine is a **unified, explainable, and modular RAG framework** that implements four powerful retrieval strategies for high-precision document question answering:

- 🧠 **Dense Retrieval** — Semantic search using embeddings + FAISS  
- 🔍 **Sparse Retrieval (BM25)** — Lexical matching for exact terms, IDs, and formulas  
- ⚡ **Hybrid Retrieval** — Combines dense + sparse scores (late fusion, α = 0.65)  
- 🎯 **Hybrid-Rank Retrieval** — Hybrid pipeline + Cross-Encoder reranking for top precision  

This framework is designed for **technical and regulatory texts** (e.g., maritime classification rules) where **traceability and explainability** matter as much as accuracy.

---

## 🌊 Key Highlights

- 🧩 Modular architecture (`src/rag/`)  
- 🔎 Explainable outputs (shows rule ID, section, and page for each source)  
- 🚫 Abstains under low confidence — avoids hallucinated answers  
- ⚙️ Configurable via environment variables  
- 🧠 Designed for reproducibility (temperature=0.0)  

---

## 🚀 Quickstart

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/pooryakhorsand/RAG-Marine.git
cd RAG-Marine
