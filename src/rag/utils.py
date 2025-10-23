# src/rag/utils.py
"""
Utility functions: JSONL loader, safe tokenization, logging helpers.
"""

import json
from pathlib import Path
from typing import Iterable, Dict, Any, List

def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    """
    Read a JSONL file and yield objects. Ignores empty lines.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def safe_text(obj: Any) -> str:
    """
    Normalize different content types into a clean string for embedding/indexing.
    """
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj.strip()
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return str(obj)

def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)
