# src/rag/hybrid.py
"""
Simple hybrid retrieval: combine BM25 (sparse) and Dense (embedding) scores.
Scores are normalized and combined via alpha weighting.
"""

from typing import List, Tuple
import numpy as np
import logging
from .dense import DenseIndex
from .sparse import SparseIndex

logger = logging.getLogger(__name__)

def normalize_scores(arr: np.ndarray) -> np.ndarray:
    """
    Min-max normalize a 1D numpy array to [0, 1], safe against constant arrays.
    """
    if arr.size == 0:
        return arr
    mn, mx = arr.min(), arr.max()
    if abs(mx - mn) < 1e-8:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn + 1e-12)

class HybridRetriever:
    def __init__(self, dense: DenseIndex, sparse: SparseIndex, alpha: float = 0.5):
        """
        alpha controls weight of dense vs sparse: final = alpha*dense + (1-alpha)*sparse
        """
        self.dense = dense
        self.sparse = sparse
        self.alpha = float(alpha)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[dict, float]]:
        """
        Produce a hybrid ranking over the union of documents.
        Implementation note: we compute dense top-M and sparse scores over full corpus, then combine.
        For efficiency in large corpora, consider inverted idx + approximate dense search.
        """
        # dense top_k
        dense_results = self.dense.search(query, top_k=top_k*5)  # get more candidates from dense
        dense_indices = [int(r[0].get("_idx", i)) for i, r in enumerate(dense_results)]
        dense_scores = np.zeros(len(self.dense._records))
        for rec, sc in dense_results:
            idx = self.dense._records.index(rec)
            dense_scores[idx] = sc

        # sparse scores for all docs (this may be expensive for large corpora)
        sparse_results = self.sparse.search(query, top_k=len(self.sparse._records))
        sparse_scores = np.zeros(len(self.sparse._records))
        for rec, sc in sparse_results:
            idx = self.sparse._records.index(rec)
            sparse_scores[idx] = sc

        # normalize both
        dense_n = normalize_scores(dense_scores)
        sparse_n = normalize_scores(sparse_scores)

        hybrid = self.alpha * dense_n + (1 - self.alpha) * sparse_n
        top_idx = np.argsort(hybrid)[::-1][:top_k]
        return [(self.dense._records[i], float(hybrid[i])) for i in top_idx]
