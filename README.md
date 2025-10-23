# ⚙️ RAG-Marine  
### _A Unified, Explainable Retrieval-Augmented Generation Toolkit_

<p align="center">
  <video src="https://github.com/user-attachments/assets/7d4e911d-e31e-4c07-9aba-93227cd2c9f7" width="100%" controls>
    Your browser does not support the video tag.
  </video>
</p>

> 🎥 *Watch the demo above to see how RAG-Marine builds indices and runs all four retrieval modes (Dense, Sparse, Hybrid, and Hybrid-Rank).*

---

## 🌊 Overview

**RAG-Marine** is a research-grade and production-ready framework for  
**Retrieval-Augmented Generation (RAG)** with a focus on explainability,  
auditability, and modular design.  
It provides **four complementary retrieval strategies** optimized for technical, legal, and regulatory texts:

- 🧠 **Dense Retrieval** — semantic search via embeddings (FAISS)  
- 🔍 **Sparse Retrieval (BM25)** — exact-term lexical search  
- ⚡ **Hybrid Retrieval** — weighted fusion of Dense + Sparse results (α = 0.65)  
- 🎯 **Hybrid-Rank Retrieval** — hybrid pipeline with cross-encoder re-ranking  

All components are cleanly organized under `src/rag/` and controlled through a single CLI.

---

## 🚀 Quickstart

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/pooryakhorsand/RAG-Marine.git
cd RAG-Marine
