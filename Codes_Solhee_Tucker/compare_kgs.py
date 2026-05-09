"""Before / after KG comparison report.

Compares the original string-normalized KG (produced by merged_kg.py)
against the Gilda-normalized KG built by build_normalized_kg.py.

Usage (from repo root):
    python -m Codes_Solhee_Tucker.compare_kgs
"""

import json
from pathlib import Path

import networkx as nx


# ── Baseline KG (re-built from raw inputs, same logic as merged_kg.py) ──────

def _build_baseline_kg(chemprot_path: Path, euadr_path: Path) -> nx.DiGraph:
    """Replicate merged_kg.py + euadr_preprocessing.py (string-norm only)."""
    G = nx.DiGraph()

    with chemprot_path.open(encoding="utf-8") as f:
        chemprot_data = json.load(f)

    for item in chemprot_data:
        head = item["head"].strip().lower()
        tail = item["tail"].strip().lower()
        if head == tail:
            continue
        models = list(set(item.get("models", [])))
        confidence = round(len(models) / 3, 2)
        G.add_node(head)
        G.add_node(tail)
        G.add_edge(head, tail,
                   label=item["relation"],
                   source="chemprot",
                   confidence=confidence,
                   models=", ".join(models))

    with euadr_path.open(encoding="utf-8") as f:
        euadr_raw = json.load(f)

    seen: set[tuple] = set()
    for doc in euadr_raw:
        if not doc.get("relations"):
            continue
        ent_map = {e["id"]: e for e in doc["entities"]}
        for rel in doc["relations"]:
            head_e = ent_map.get(rel["arg1_id"])
            tail_e = ent_map.get(rel["arg2_id"])
            if not head_e or not tail_e:
                continue

            def _txt(e):
                t = e["text"]
                return (t[0] if isinstance(t, list) else t).strip()

            head_str = _txt(head_e).lower()
            tail_str = _txt(tail_e).lower()
            rel_type = rel["type"].lower()
            if not head_str or not tail_str:
                continue
            key = (head_str, tail_str, rel_type)
            if key in seen:
                continue
            seen.add(key)
            if head_str == tail_str:
                continue
            G.add_node(head_str)
            G.add_node(tail_str)
            G.add_edge(head_str, tail_str, label=rel_type, source="euadr", confidence=1.0, models="")

    return G
