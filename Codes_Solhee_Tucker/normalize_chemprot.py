"""Normalize ChemProt triples with Gilda.

In: merged_relation_predictions_all_models.json (11,796 triples),
grounds every head/tail mention, 
Out: normalized_chemprot_triples.json.

Usage (from repo root):
    python -m Codes_Solhee_Tucker.normalize_chemprot
    python -m Codes_Solhee_Tucker.normalize_chemprot --score-threshold 0.6 --policy keep_as_raw
"""

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from .config import CHEMPROT_INPUT, NORMALIZED_CHEMPROT, SCORE_THRESHOLD, UNGROUNDED_POLICY
from .grounding import flush_cache, ground_mention


def normalize_chemprot(
    input_path: Path,
    output_path: Path,
    threshold: float,
    policy: str,
) -> list[dict]:
    with input_path.open(encoding="utf-8") as f:
        data = json.load(f)
    print(f"[INFO] Loaded {len(data):,} ChemProt triples from {input_path.name}")

    results = []
    n_both = 0
    n_head_only = 0
    n_tail_only = 0
    n_neither = 0
    n_dropped = 0

    for item in tqdm(data, desc="ChemProt grounding"):
        sentence = item.get("sentence", "")
        models = list(set(item.get("models", [])))
        confidence = round(len(models) / 3, 2)

        hg = ground_mention(item["head"], context=sentence, threshold=threshold)
        tg = ground_mention(item["tail"], context=sentence, threshold=threshold)

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
            "sentence":   sentence,
            "head_text":  item["head"],
            "head_db":    hg["db"],
            "head_id":    hg["curie"],
            "head_name":  hg["name"],
            "head_score": hg["score"],
            "tail_text":  item["tail"],
            "tail_db":    tg["db"],
            "tail_id":    tg["curie"],
            "tail_name":  tg["name"],
            "tail_score": tg["score"],
            "relation":   item["relation"],
            "source":     "chemprot",
            "models":     models,
            "confidence": confidence,
            "grounded":   both,
        })

    flush_cache()

    output_path.parent.mkdir(exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    total = len(data)
    kept = len(results)
    print(f"\n[RESULTS] ChemProt normalization (threshold={threshold}, policy={policy})")
    print(f"  Input triples  : {total:,}")
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
    parser = argparse.ArgumentParser(description="Normalize ChemProt triples with Gilda")
    parser.add_argument("--in",   dest="input",     default=str(CHEMPROT_INPUT))
    parser.add_argument("--out",  dest="output",    default=str(NORMALIZED_CHEMPROT))
    parser.add_argument("--score-threshold", type=float, default=SCORE_THRESHOLD)
    parser.add_argument("--policy", choices=["keep_as_raw", "drop"], default=UNGROUNDED_POLICY)
    args = parser.parse_args()

    normalize_chemprot(
        Path(args.input),
        Path(args.output),
        args.score_threshold,
        args.policy,
    )


if __name__ == "__main__":
    main()