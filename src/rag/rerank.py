# src/rag/rerank.py
"""
Reranking wrapper: given candidate documents (dense search), use the chat model
to return an ordering of the candidates. The model is instructed to return JSON.
"""

from typing import List, Tuple
import json
import logging
from openai import OpenAI
from .config import get_config

logger = logging.getLogger(__name__)
cfg = get_config()

class ReRanker:
    def __init__(self, client: OpenAI = None, model: str = None):
        self.client = client or OpenAI()
        self.model = model or cfg.rerank_model

    def rerank(self, query: str, candidates: List[Tuple[dict, float]]) -> List[Tuple[dict, float]]:
        """
        Ask the chat model to rank candidates. Expects the model to return a JSON array
        of indices (0-based) in order of relevance.
        """
        # Build a compact prompt
        docs_text = "\n\n".join([f"[{i}] {c[0].get('content', '')}" for i, c in enumerate(candidates)])
        prompt = (
            f"Query: {query}\n\n"
            f"Documents:\n{docs_text}\n\n"
            "Task: Rank the above documents from most to least relevant to the query. "
            "Return a JSON array of document indices (0-based) in descending order of relevance. "
            "Respond ONLY with valid JSON."
        )

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a ranking assistant. Return only JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=256,
        )
        text = resp.choices[0].message.content.strip()
        try:
            order = json.loads(text)
            # Validate indices
            ordered = []
            for idx in order:
                if not (0 <= idx < len(candidates)):
                    logger.warning("Index out of range in re-rank response: %s", idx)
                    continue
                ordered.append(candidates[idx])
            # If model returned partial or invalid output, fall back gracefully
            if not ordered:
                return candidates
            return ordered
        except Exception as e:
            logger.warning("Failed to parse re-rank output: %s -- falling back to original order", e)
            return candidates
