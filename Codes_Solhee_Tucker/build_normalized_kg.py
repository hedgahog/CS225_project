"""Build the normalized unified KG from grounded triples.

Node ID strategy:
  - grounded entity  →  canonical CURIE (e.g. "HGNC:11998")
  - ungrounded entity →  "raw:<surface_lower>" (e.g. "raw:ici 182,780")

This ensures grounded variants of the same entity collapse into one node
while ungrounded surface forms stay distinct and are clearly labeled.

Usage (from repo root):
    python -m Codes_Solhee_Tucker.build_normalized_kg
"""

import argparse
import json
from pathlib import Path

import networkx as nx

from .config import (
    NORMALIZED_CHEMPROT,
    NORMALIZED_EUADR,
    NORMALIZED_KG,
)


def _node_id(text: str, curie: str | None) -> str:
    return curie if curie else f"raw:{text.strip().lower()}"


def build_kg(chemprot_triples: list[dict], euadr_triples: list[dict]) -> nx.DiGraph:
    G = nx.DiGraph()

    for triple in chemprot_triples + euadr_triples:
        head_node = _node_id(triple["head_text"], triple["head_id"])
        tail_node = _node_id(triple["tail_text"], triple["tail_id"])

        if head_node == tail_node:
            continue

        # Node attributes — store surface form and canonical name for reference
        if head_node not in G:
            G.add_node(
                head_node,
                surface_form=triple["head_text"],
                canonical_name=triple.get("head_name") or "",
                db=triple.get("head_db") or "",
            )
        if tail_node not in G:
            G.add_node(
                tail_node,
                surface_form=triple["tail_text"],
                canonical_name=triple.get("tail_name") or "",
                db=triple.get("tail_db") or "",
            )

        models = triple.get("models") or []
        G.add_edge(
            head_node,
            tail_node,
            label=triple["relation"],
            source=triple["source"],
            confidence=triple["confidence"],
            models=", ".join(models) if models else "",
            grounded=triple["grounded"],
        )

    return G


def main() -> None:
    parser = argparse.ArgumentParser(description="Build normalized KG from grounded triples")
    parser.add_argument("--chemprot", default=str(NORMALIZED_CHEMPROT))
    parser.add_argument("--euadr",    default=str(NORMALIZED_EUADR))
    parser.add_argument("--out",      default=str(NORMALIZED_KG))
    args = parser.parse_args()

    chemprot_path = Path(args.chemprot)
    euadr_path    = Path(args.euadr)
    out_path      = Path(args.out)

    with chemprot_path.open(encoding="utf-8") as f:
        chemprot = json.load(f)
    with euadr_path.open(encoding="utf-8") as f:
        euadr = json.load(f)

    print(f"[INFO] ChemProt triples: {len(chemprot):,}")
    print(f"[INFO] EU-ADR triples  : {len(euadr):,}")

    G = build_kg(chemprot, euadr)

    n_grounded_nodes = sum(
        1 for n in G.nodes() if not str(n).startswith("raw:")
    )
    print(f"\n[INFO] KG nodes        : {G.number_of_nodes():,}")
    print(f"[INFO] KG edges        : {G.number_of_edges():,}")
    print(f"[INFO] Grounded nodes  : {n_grounded_nodes:,} / {G.number_of_nodes():,}")

    out_path.parent.mkdir(exist_ok=True)
    nx.write_graphml(G, out_path)
    print(f"[SUCCESS] Saved normalized KG to {out_path}")


if __name__ == "__main__":
    main()