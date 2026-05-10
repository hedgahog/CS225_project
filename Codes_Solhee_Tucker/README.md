# Entity Normalization Module (Codes_Solhee_Tucker)

Adds Gilda-based entity normalization to the upstream ChemProt + EU-ADR
knowledge-graph pipeline. Standalone post-processing — no upstream code is
modified.

Addresses reviewer comment #8: *"A critical gap in the methodology is the
complete absence of entity alignment or normalization before knowledge graph
construction."*

## What it does

1. Ground every head/tail surface form with [Gilda](https://github.com/gyorilab/gilda)
   (sentence supplied as disambiguation context).
2. Accept matches with score ≥ 0.6; mark or drop the rest per
   `UNGROUNDED_POLICY` in `config.py`.
3. Rebuild the unified KG with canonical CURIEs as node IDs (original surface
   forms preserved as node attributes).
4. Re-run the original authors' downstream reasoning unchanged on each KG for
   a like-for-like comparison.

## Module layout

```
Codes_Solhee_Tucker/
├── config.py                   # threshold, paths, ungrounded policy
├── grounding.py                # Gilda wrapper with on-disk caching
├── normalize_chemprot.py       # → outputs/gilda_normalized_chemprot_triples.json
├── normalize_euadr.py          # → outputs/gilda_normalized_euadr_triples.json
├── build_normalized_kg.py      # → outputs/normalized_unified_kg{,_drop}.graphml
├── compare_kgs.py              # baseline KG builder (string-norm only)
├── summary.ipynb               # analysis & reviewer-response report (source of truth)
├── poc/                        # 200-sample threshold study
├── Chemprot_pipeline_edited/   # downstream reasoning, re-run on normalized KGs
└── outputs/                    # JSON triples, GraphML KGs, comparison_report.md
```

## Installation

Tested on Python 3.10 (matches the upstream pipeline's torch / CUDA stack).

```bash
python3.10 -m venv .venv-norm
source .venv-norm/bin/activate
pip install -r Codes_Solhee_Tucker/requirements.txt
```

Gilda ships with ~2M pre-grounded terms (HGNC, UniProt, FamPlex, ChEBI, MeSH,
GO, DOID, EFO, HP) — no extra resource download needed.

## Usage

```bash
# 1. Generate normalized triples (defaults from config.py).
python -m Codes_Solhee_Tucker.normalize_chemprot
python -m Codes_Solhee_Tucker.normalize_euadr

# 2. Build the unified normalized KG.
python -m Codes_Solhee_Tucker.build_normalized_kg

# 3. (Optional) Re-run downstream reasoning on the normalized KG.
cd Codes_Solhee_Tucker/Chemprot_pipeline_edited
python multi_hop_reasoning_v2.py             # all paths,    max_hops=4
python filtered_multi_hop.reasoning_v2.py    # CPR-anchored, max_hops=3
# Add `--drop` to either to use the drop-policy KG.

# 4. Open summary.ipynb for the full analysis.
```

Configure threshold and ungrounded policy in `config.py`:

```python
SCORE_THRESHOLD    = 0.6
UNGROUNDED_POLICY  = "keep_as_raw"   # or "drop"
```

## Output schema (normalized triple)

```json
{
  "sentence":   "Androgen antagonistic effect of estramustine ...",
  "head_text":  "Androgen",
  "head_db":    "CHEBI",
  "head_id":    "CHEBI:CHEBI:50113",
  "head_name":  "androgen",
  "head_score": 0.76,
  "tail_text":  "estramustine",
  "tail_db":    "CHEBI",
  "tail_id":    "CHEBI:CHEBI:4868",
  "tail_name":  "estramustine",
  "tail_score": 0.78,
  "relation":   "CPR:3",
  "source":     "chemprot",
  "models":     ["bert-base-cased", "biobert", "biogpt"],
  "confidence": 1.0,
  "grounded":   true
}
```

When grounding fails, `*_db` / `*_id` / `*_name` are `null`, `*_score` is
`0.0`, and `grounded` is `false`.

## Reviewer-response artifacts

- `outputs/comparison_report.md` — quantitative before/after summary
- `summary.ipynb` — reproducible analysis behind the report
- `outputs/normalized_unified_kg.graphml` — viewable in Gephi / Cytoscape

## Reproducibility

- Python 3.10
- Gilda pinned in `requirements.txt` (`>=1.4.0`)
- Grounding cache (`outputs/.grounding_cache.json`) keys results by
  `(text, context_hash)` so re-runs are deterministic and fast.
- All thresholds and policies live in `config.py` — change there, not in
  individual scripts.

## Citation

> Gyori, B. M., Hoyt, C. T., & Steppi, A. (2022). *Gilda: biomedical entity
> text normalization with machine-learned disambiguation as a service.*
> Bioinformatics Advances, 2(1), vbac034.
> <https://doi.org/10.1093/bioadv/vbac034>
