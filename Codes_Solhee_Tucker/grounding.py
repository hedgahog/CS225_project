"""Thin Gilda wrapper with file-backed cache.

Cache key: MD5(text + "\x00" + context), so the same surface form in
different sentences gets independent disambiguation, but repeated calls
within the same run are instant.

Call flush_cache() after each normalize script finishes to persist results.
"""

import atexit
import hashlib
import json
from pathlib import Path
from typing import Optional

import gilda

from .config import GROUNDING_CACHE, SCORE_THRESHOLD

# Module-level cache loaded once on import.
_cache: dict[str, dict] = {}
_dirty: bool = False


def _load() -> None:
    global _cache
    if GROUNDING_CACHE.exists():
        with GROUNDING_CACHE.open(encoding="utf-8") as f:
            _cache = json.load(f)


def flush_cache() -> None:
    global _dirty
    if _dirty:
        GROUNDING_CACHE.parent.mkdir(exist_ok=True)
        with GROUNDING_CACHE.open("w", encoding="utf-8") as f:
            json.dump(_cache, f)
        _dirty = False


atexit.register(flush_cache)
_load()


def _key(text: str, context: Optional[str]) -> str:
    raw = text + "\x00" + (context or "")
    return hashlib.md5(raw.encode()).hexdigest()


def ground_mention(
    text: str,
    context: Optional[str] = None,
    threshold: float = SCORE_THRESHOLD,
) -> dict:
    """Ground one surface form with Gilda.

    Returns a dict with keys:
        db       (str | None)  — namespace, e.g. "HGNC"
        curie    (str | None)  — canonical ID, e.g. "HGNC:11998"
        name     (str | None)  — canonical name, e.g. "TP53"
        score    (float)       — top-match score (0.0 if no match)
        grounded (bool)        — True iff score >= threshold
    """
    global _dirty

    key = _key(text, context)
    if key in _cache:
        cached = _cache[key]
        # Re-apply threshold in case config changed since last run.
        result = dict(cached)
        result["grounded"] = (result["score"] >= threshold and result["db"] is not None)
        return result

    try:
        matches = gilda.ground(text, context=context)
    except Exception:
        matches = []

    if matches:
        top = matches[0]
        entry = {
            "db":    top.term.db,
            "curie": f"{top.term.db}:{top.term.id}",
            "name":  top.term.entry_name,
            "score": float(top.score),
        }
    else:
        entry = {"db": None, "curie": None, "name": None, "score": 0.0}

    _cache[key] = entry
    _dirty = True

    result = dict(entry)
    result["grounded"] = (result["score"] >= threshold and result["db"] is not None)
    return result