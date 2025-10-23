# cli.py
"""
Command-line interface for the RAG toolkit.
Allows running dense, sparse, hybrid, and rerank flows from the terminal.
"""

import argparse
import logging
import os
from pathlib import Path
import json

# set up basic logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("rag-cli")

# local imports
from src.rag.config import get_config
from src.rag.utils import read_jsonl, safe_text, ensure_dir
from src.rag.embed import Embedder
from src.rag.dense import DenseIndex
from src.rag.sparse import SparseIndex
from src.rag.hybrid import HybridRetriever
from src.rag.rerank import ReRanker
from openai import OpenAI

def build_indices(cfg, client):
    """
    Build indices for all strategies using the same source JSONL.
    Returns instantiated DenseIndex and SparseIndex objects (built in-memory).
    """
    logger.info("Loading source JSONL: %s", cfg.data_jsonl)
    raw = list(read_jsonl(cfg.data_jsonl))

    # prepare texts and records
    texts = []
    records = []
    for i, r in enumerate(raw):
        txt = safe_text(r.get("content", ""))
        if not txt or len(txt) < 4:
            continue
        r["_idx"] = i  # keep stable index
        texts.append(txt)
        records.append(r)

    logger.info("Preparing embedder and indices...")
    embedder = Embedder(OpenAI(), model=cfg.embed_model, batch_size=cfg.batch_size)
    dense_idx = DenseIndex(OpenAI(), cfg.embed_model, embedder, dim=cfg.embed_dim, workdir=cfg.workdir)
    sparse_idx = SparseIndex(workdir=Path(cfg.workdir)/"sparse")

    # build
    dense_idx.build(texts, records)
    sparse_idx.build(texts, records)
    return dense_idx, sparse_idx

def run_dense(cfg, query: str):
    client = OpenAI()
    dense = DenseIndex(client, cfg.embed_model, None, dim=cfg.embed_dim, workdir=cfg.workdir)
    results = dense.search(query, top_k=cfg.top_k)
    print_results(results)

def run_sparse(cfg, query: str):
    client = OpenAI()
    sparse = SparseIndex(workdir=Path(cfg.workdir)/"sparse")
    # Note: sparse requires build() to have been called previously (or you must call build here)
    # For convenience, try to load records from disk if present:
    try:
        import json
        with open(Path(cfg.workdir)/"sparse"/"records.json", "r", encoding="utf-8") as fh:
            records = json.load(fh)
        sparse._records = records
        # naive tokenized corpus rebuild
        tokenized = [r.get("content", "").split() for r in records]
        from rank_bm25 import BM25Okapi
        sparse._bm25 = BM25Okapi(tokenized)
    except Exception:
        raise RuntimeError("Sparse index not found on disk; run build first using --build")

    results = sparse.search(query, top_k=cfg.top_k)
    print_results(results)

def run_hybrid(cfg, query: str, alpha: float = 0.5):
    client = OpenAI()
    dense = DenseIndex(client, cfg.embed_model, None, dim=cfg.embed_dim, workdir=cfg.workdir)
    sparse = SparseIndex(workdir=Path(cfg.workdir)/"sparse")
    # load dense and sparse data
    dense.load()
    # rebuild sparse similar to run_sparse()
    import json
    with open(Path(cfg.workdir)/"sparse"/"records.json", "r", encoding="utf-8") as fh:
        sparse._records = json.load(fh)
    tokenized = [r.get("content", "").split() for r in sparse._records]
    from rank_bm25 import BM25Okapi
    sparse._bm25 = BM25Okapi(tokenized)

    hybrid = HybridRetriever(dense, sparse, alpha=alpha)
    results = hybrid.search(query, top_k=cfg.top_k)
    print_results(results)

def run_rerank(cfg, query: str):
    client = OpenAI()
    dense = DenseIndex(client, cfg.embed_model, None, dim=cfg.embed_dim, workdir=cfg.workdir)
    dense.load()
    candidates = dense.search(query, top_k=cfg.top_k * 3)
    reranker = ReRanker(client, cfg.rerank_model)
    reranked = reranker.rerank(query, candidates)
    print_results(reranked)

def print_results(results):
    """
    Nicely print list of (record, score)
    """
    for i, (rec, score) in enumerate(results, start=1):
        snippet = rec.get("content", "")[:400].replace("\n", " ").strip()
        print(f"[{i}] score={score:.4f}\n  id={rec.get('id', 'n/a')}\n  snippet={snippet}\n")

def main():
    parser = argparse.ArgumentParser(description="RAG Toolkit CLI")
    parser.add_argument("--mode", choices=["build", "dense", "sparse", "hybrid", "rerank"], required=True)
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--build-force", action="store_true", help="Force rebuild indices from data")
    args = parser.parse_args()

    cfg = get_config()

    # Ensure OPENAI_API_KEY present
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not found in environment. Please export it.")
        return

    if args.mode == "build":
        build_indices(cfg, OpenAI())
        logger.info("Build finished.")
        return

    if args.query is None:
        logger.error("Please supply --query for modes other than build.")
        return

    if args.mode == "dense":
        run_dense(cfg, args.query)
    elif args.mode == "sparse":
        run_sparse(cfg, args.query)
    elif args.mode == "hybrid":
        run_hybrid(cfg, args.query, alpha=args.alpha)
    elif args.mode == "rerank":
        run_rerank(cfg, args.query)

if __name__ == "__main__":
    main()
