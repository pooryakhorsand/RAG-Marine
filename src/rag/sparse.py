# src/rag/sparse.py
"""
Sparse retrieval using BM25 (rank_bm25).
Provides index building and search API with consistent return type.
"""

from typing import List, Tuple
from rank_bm25 import BM25Okapi
import logging
from .utils import safe_text
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class SparseIndex:
    def __init__(self, workdir: str = "data/workdir/sparse"):
        self.workdir = Path(workdir)
        self._bm25 = None
        self._records = None

    def build(self, texts: List[str], records: List[dict]):
        """
        Build BM25 index from tokenized texts.
        """
        tokenized = [t.split() for t in texts]
        self._bm25 = BM25Okapi(tokenized)
        self._records = records
        self.workdir.mkdir(parents=True, exist_ok=True)
        # persist minimal state
        with open(self.workdir / "records.json", "w", encoding="utf-8") as fh:
            json.dump(records, fh, ensure_ascii=False)
        logger.info("BM25 index built with %d docs", len(records))

    def load(self):
        """
        Load records (BM25 must be rebuilt in memory — requires tokenized corpus).
        """
        raise NotImplementedError("BM25 reload not implemented: rebuild from texts in memory")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[dict, float]]:
        """
        Return a list of (record, score). Scores are BM25 raw scores.
        """
        if self._bm25 is None or self._records is None:
            raise RuntimeError("Sparse index not built")
        q_tokens = query.split()
        scores = self._bm25.get_scores(q_tokens)
        import numpy as np
        top_idx = list(np.argsort(scores)[::-1][:top_k])
        return [(self._records[i], float(scores[i])) for i in top_idx]
