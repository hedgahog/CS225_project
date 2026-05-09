"""Normalize EU-ADR triples with Gilda.

In: euadr_full.json (BigBio KB format, 300 docs / 2,891 relations),
deduplicates by (head_lower, tail_lower, relation), grounds every mention,
Out: normalized_euadr_triples_v2.json.

Usage (from repo root):
    python -m Codes_Solhee_Tucker.normalize_euadr
    python -m Codes_Solhee_Tucker.normalize_euadr --score-threshold 0.6 --policy keep_as_raw
"""

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from .config import EUADR_INPUT, NORMALIZED_EUADR, SCORE_THRESHOLD, UNGROUNDED_POLICY
from .grounding import flush_cache, ground_mention


def _passage_text(doc: dict) -> str:
    parts = []
    for p in doc["passages"]:
        txt = p["text"]
        parts.extend(txt if isinstance(txt, list) else [txt])
    return " ".join(parts)


def _ent_text(entity: dict) -> str:
    t = entity["text"]
    return (t[0] if isinstance(t, list) else t).strip()


def _extract_pairs(raw: list[dict]) -> list[tuple]:
    """Return deduplicated list of (head_str, tail_str, rel_type, context)."""
    pairs = []
    seen: set[tuple] = set()

    for doc in raw:
        if not doc.get("relations"):
            continue
        ctx = _passage_text(doc)
        ent_map = {e["id"]: e for e in doc["entities"]}

        for rel in doc["relations"]:
            head_e = ent_map.get(rel["arg1_id"])
            tail_e = ent_map.get(rel["arg2_id"])
            if not head_e or not tail_e:
                continue

            head_str = _ent_text(head_e)
            tail_str = _ent_text(tail_e)
            if not head_str or not tail_str:
                continue

            # euadr_preprocessing.py lowercases the relation type ("PA" → "pa")
            rel_type = rel["type"].lower()

            key = (head_str.lower(), tail_str.lower(), rel_type)
            if key in seen:
                continue
            seen.add(key)
            pairs.append((head_str, tail_str, rel_type, ctx))

    return pairs


def normalize_euadr(
    input_path: Path,
    output_path: Path,
    threshold: float,
    policy: str,
) -> list[dict]:
    with input_path.open(encoding="utf-8") as f:
        raw = json.load(f)
    print(f"[INFO] Loaded {len(raw)} EU-ADR documents from {input_path.name}")

    pairs = _extract_pairs(raw)
    print(f"[INFO] Extracted {len(pairs):,} unique relation pairs")

    results = []
    n_both = 0
    n_head_only = 0
    n_tail_only = 0
    n_neither = 0
    n_dropped = 0

    for head_str, tail_str, rel_type, ctx in tqdm(pairs, desc="EU-ADR grounding"):
        hg = ground_mention(head_str, context=ctx, threshold=threshold)
        tg = ground_mention(tail_str, context=ctx, threshold=threshold)

        both = hg["grounded"] and tg["grounded"]

        if not both and policy == "drop":
            n_dropped += 1
            continue

        if hg["grounded"] and tg["grounded"]:
            n_both += 1
        elif hg["grounded"]:
            n_head_only += 1
        elif tg["grounded"]:
            n_tail_only += 1
        else:
            n_neither += 1

        results.append({
            "sentence":   ctx,
            "head_text":  head_str,
            "head_db":    hg["db"],
            "head_id":    hg["curie"],
            "head_name":  hg["name"],
            "head_score": hg["score"],
            "tail_text":  tail_str,
            "tail_db":    tg["db"],
            "tail_id":    tg["curie"],
            "tail_name":  tg["name"],
            "tail_score": tg["score"],
            "relation":   rel_type,
            "source":     "euadr",
            "models":     [],
            "confidence": 1.0,
            "grounded":   both,
        })

    flush_cache()

    output_path.parent.mkdir(exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    total = len(pairs)
    kept = len(results)
    print(f"\n[RESULTS] EU-ADR normalization (threshold={threshold}, policy={policy})")
    print(f"  Input pairs    : {total:,}")
    if policy == "drop":
        print(f"  Dropped        : {n_dropped:,}")
    print(f"  Output triples : {kept:,}")
    print(f"  Both grounded  : {n_both:,}  ({n_both/total:.1%})")
    print(f"  Head only      : {n_head_only:,}  ({n_head_only/total:.1%})")
    print(f"  Tail only      : {n_tail_only:,}  ({n_tail_only/total:.1%})")
    print(f"  Neither        : {n_neither:,}  ({n_neither/total:.1%})")
    print(f"[SUCCESS] Saved to {output_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize EU-ADR triples with Gilda")
    parser.add_argument("--in",   dest="input",  default=str(EUADR_INPUT))
    parser.add_argument("--out",  dest="output", default=str(NORMALIZED_EUADR))
    parser.add_argument("--score-threshold", type=float, default=SCORE_THRESHOLD)
    parser.add_argument("--policy", choices=["keep_as_raw", "drop"], default=UNGROUNDED_POLICY)
    args = parser.parse_args()

    normalize_euadr(
        Path(args.input),
        Path(args.output),
        args.score_threshold,
        args.policy,
    )


if __name__ == "__main__":
    main()