# src/rag/config.py
"""
Central configuration object for the RAG project.
Load secrets from environment variables and provide defaults for models and paths.
"""

from dataclasses import dataclass
import os
from typing import Optional

@dataclass
class Config:
    # I/O
    data_jsonl: str = os.getenv("RAG_DATA_JSONL", "data/merged.jsonl")
    workdir: str = os.getenv("RAG_WORKDIR", "data/workdir")

    # Models
    embed_model: str = os.getenv("RAG_EMBED_MODEL", "text-embedding-3-small")
    chat_model: str = os.getenv("RAG_CHAT_MODEL", "gpt-4o-mini")
    rerank_model: str = os.getenv("RAG_RERANK_MODEL", "gpt-4o-mini")

    # Embedding dimension (default for text-embedding-3-small; adjust if using another model)
    embed_dim: int = int(os.getenv("RAG_EMBED_DIM", 1536))

    # Misc
    batch_size: int = int(os.getenv("RAG_BATCH_SIZE", 100))
    top_k: int = int(os.getenv("RAG_TOP_K", 5))

def get_config() -> Config:
    return Config()
