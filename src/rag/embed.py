# src/rag/embed.py
"""
Embedding utilities: batch embeddings, save/load numpy arrays.
This module wraps calls to the OpenAI embeddings API client.
"""

from typing import List, Sequence
import numpy as np
from pathlib import Path
from openai import OpenAI
from .config import get_config
from .utils import ensure_dir
import logging

logger = logging.getLogger(__name__)
cfg = get_config()

class Embedder:
    """
    Simple Embedder wrapper for batching and persistence.
    """
    def __init__(self, client: OpenAI = None, model: str = None, batch_size: int = None):
        self.client = client or OpenAI()
        self.model = model or cfg.embed_model
        self.batch_size = batch_size or cfg.batch_size

    def embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        """
        Create embeddings for a list of texts. Returns a float32 numpy array (N, D).
        """
        vectors = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            logger.info("Embedding batch %d..%d", i, i + len(batch) - 1)
            resp = self.client.embeddings.create(model=self.model, input=batch)
            vectors.extend([d.embedding for d in resp.data])
        arr = np.array(vectors, dtype=np.float32)
        return arr

    def save(self, arr: np.ndarray, out_path: str):
        ensure_dir(Path(out_path).parent.as_posix())
        np.save(out_path, arr)
        logger.info("Saved embeddings to %s", out_path)

    def load(self, in_path: str) -> np.ndarray:
        arr = np.load(in_path)
        logger.info("Loaded embeddings from %s", in_path)
        return arr
