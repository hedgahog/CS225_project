"""Before / after KG comparison report.

Compares the original string-normalized KG (produced by merged_kg.py)
against the Gilda-normalized KG built by build_normalized_kg.py.

Usage (from repo root):
    python -m Codes_Solhee_Tucker.compare_kgs
"""

import argparse
import json
import textwrap
from pathlib import Path

import networkx as nx

from .config import (
    CHEMPROT_INPUT,
    COMPARISON_REPORT,
    EUADR_INPUT,
    NORMALIZED_KG,
)


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


# ── Multi-hop path count (same cutoff as multi_hop_reasoning.py) ─────────────

def _count_multihop(G: nx.DiGraph, max_hops: int = 3, sample: int = 200) -> int:
    """Approximate multi-hop path count on a node sample (full graph is slow)."""
    import random
    nodes = list(G.nodes())
    if len(nodes) > sample:
        nodes = random.sample(nodes, sample)
    count = 0
    for src in nodes:
        for tgt in nodes:
            if src == tgt:
                continue
            try:
                paths = list(nx.all_simple_paths(G, src, tgt, cutoff=max_hops))
                count += sum(1 for p in paths if len(p) >= 3)
            except nx.NetworkXNoPath:
                pass
    return count


# ── Merged-node examples ─────────────────────────────────────────────────────

def _merged_node_examples(norm_path: Path, n: int = 10) -> list[str]:
    """Find canonical nodes that absorbed multiple surface forms."""
    with (norm_path.parent / "gilda_normalized_chemprot_triples.json").open() as f:
        cp = json.load(f)
    with (norm_path.parent / "gilda_normalized_euadr_triples.json").open() as f:
        eu = json.load(f)

    curie_to_surfaces: dict[str, set[str]] = {}
    for triple in cp + eu:
        for side in ("head", "tail"):
            curie = triple.get(f"{side}_id")
            surf  = triple.get(f"{side}_text", "").strip()
            if curie and surf:
                curie_to_surfaces.setdefault(curie, set()).add(surf)

    multi = {c: surfs for c, surfs in curie_to_surfaces.items() if len(surfs) > 1}
    top = sorted(multi.items(), key=lambda x: len(x[1]), reverse=True)[:n]
    return [f"{curie}: {sorted(surfs)}" for curie, surfs in top]


# ── Main ─────────────────────────────────────────────────────────────────────

def compare(
    chemprot_input: Path,
    euadr_input: Path,
    norm_kg_path: Path,
    out_path: Path,
) -> None:
    print("[INFO] Building baseline KG ...")
    baseline = _build_baseline_kg(chemprot_input, euadr_input)

    print("[INFO] Loading normalized KG ...")
    normalized = nx.read_graphml(norm_kg_path)

    print("[INFO] Sampling multi-hop paths (baseline) ...")
    mh_baseline = _count_multihop(baseline)
    print("[INFO] Sampling multi-hop paths (normalized) ...")
    mh_norm = _count_multihop(normalized)

    print("[INFO] Finding merged-node examples ...")
    examples = _merged_node_examples(norm_kg_path)

    n_nodes_b = baseline.number_of_nodes()
    n_edges_b = baseline.number_of_edges()
    n_nodes_n = normalized.number_of_nodes()
    n_edges_n = normalized.number_of_edges()

    n_grounded_nodes = sum(1 for n in normalized.nodes() if not str(n).startswith("raw:"))
    grounded_edges   = sum(1 for _, _, d in normalized.edges(data=True) if d.get("grounded"))

    report = textwrap.dedent(f"""\
    # KG Normalization — Before / After Report

    | Metric | Baseline (string-norm) | Normalized (Gilda) | Delta |
    |--------|----------------------|-------------------|-------|
    | Nodes  | {n_nodes_b:,} | {n_nodes_n:,} | {n_nodes_n - n_nodes_b:+,} |
    | Edges  | {n_edges_b:,} | {n_edges_n:,} | {n_edges_n - n_edges_b:+,} |
    | Multi-hop paths (sample) | {mh_baseline:,} | {mh_norm:,} | {mh_norm - mh_baseline:+,} |

    **Grounded nodes**: {n_grounded_nodes:,} / {n_nodes_n:,} ({n_grounded_nodes/n_nodes_n:.1%})
    **Both-grounded edges**: {grounded_edges:,} / {n_edges_n:,} ({grounded_edges/n_edges_n:.1%})

    ## Merged-node examples (top {len(examples)})

    Canonical ID → surface forms collapsed into it:

    """)

    for ex in examples:
        report += f"- {ex}\n"

    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"\n[SUCCESS] Comparison report saved to {out_path}")
    print(report)


def main() -> None:
    parser = argparse.ArgumentParser(description="Before/after KG comparison")
    parser.add_argument("--chemprot-input", default=str(CHEMPROT_INPUT))
    parser.add_argument("--euadr-input",    default=str(EUADR_INPUT))
    parser.add_argument("--normalized",     default=str(NORMALIZED_KG))
    parser.add_argument("--out",            default=str(COMPARISON_REPORT))
    args = parser.parse_args()

    compare(
        Path(args.chemprot_input),
        Path(args.euadr_input),
        Path(args.normalized),
        Path(args.out),
    )


if __name__ == "__main__":
    main()
