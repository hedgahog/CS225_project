# Entity Normalization Module (Codes_Solhee_Tucker)

Adds biomedical **entity normalization** (a.k.a. *grounding*) to the
ChemProt + EU-ADR knowledge-graph pipeline of `chemprot-relexner-pipeline-main/`.

This module is a **standalone post-processing layer** — the existing
pipeline code is not modified. It reads the relation-extraction outputs
already produced upstream, grounds every head/tail mention to a canonical
biomedical identifier with [Gilda](https://github.com/gyorilab/gilda),
rebuilds the unified KG using those identifiers as node IDs, and produces
a before/after report for the reviewer response.

---

## Why this module exists

The upstream pipeline performs only **string normalization**
(`text.strip().lower()`) on extracted entities before constructing the KG
(see `merged_kg.py`, `euadr_preprocessing.py`). As a result, surface-form
variants of the same biological entity become **distinct KG nodes** —
for example `"TP53"`, `"p53"`, and `"tumor protein 53"` are three separate
nodes rather than one. This fragmentation:

- inflates the node count and dilutes edge density,
- breaks multi-hop reasoning paths that should connect through a shared
  entity,
- prevents alignment with external resources (HGNC, UniProt, ChEBI, MeSH).

A reviewer comment on the manuscript flagged this gap. This module
addresses it by introducing a proper **entity normalization** step.

---

## What this module does

For every triple emerging from the upstream NER + RE pipeline:

1. Submit the head/tail surface form to `gilda.ground(text, context=sentence)`.
2. Take the top match above a configurable score threshold; record
   `(db, id, name, score)` plus the original surface form.
3. Drop or flag triples that fail to ground (policy decided after a small
   pilot study — see `poc/`).
4. Rebuild the unified KG with `f"{db}:{id}"` as the canonical node
   identifier; original surface forms are preserved as a node attribute.
5. Produce a before/after comparison report (node/edge counts, merged-node
   examples, multi-hop path delta).

---

## Pipeline placement

```
Upstream (existing, unchanged)
  ChemProt corpus ─► NER ─► RE (3 models) ─► merged_relation_predictions_all_models.json
  EU-ADR (raw)    ─► euadr_preprocessing.py ─► normalized_euadr_triples.json   (string-only)

                                  │
                                  ▼
This module (Codes_Solhee_Tucker)
  normalize_chemprot.py  ─► normalized_chemprot_triples.json
  normalize_euadr.py     ─► normalized_euadr_triples_v2.json
  build_normalized_kg.py ─► normalized_unified_kg.graphml
  compare_kgs.py         ─► comparison_report.md
                                  │
                                  ▼
Downstream (existing, re-run unchanged on the new KG)
  multi_hop_reasoning.py / graph_multi_hop.py
```

---

## Module layout

```
Codes_Solhee_Tucker/
├── README.md                  ← you are here
├── requirements.txt
├── config.py                  ← shared constants (paths, thresholds, namespaces)
├── grounding.py               ← thin Gilda wrapper with caching
├── normalize_chemprot.py
├── normalize_euadr.py
├── build_normalized_kg.py
├── compare_kgs.py
├── poc/
│   ├── poc_threshold_study.py ← runs on a 200-triple sample
│   └── poc_report.md          ← hit rate + score histogram (input to threshold decision)
├── tests/
│   ├── test_normalize.py
│   └── test_kg_build.py
└── outputs/                   ← generated artifacts (gitignored)
    ├── normalized_chemprot_triples.json
    ├── normalized_euadr_triples_v2.json
    ├── normalized_unified_kg.graphml
    └── comparison_report.md
```

---

## Installation

Tested on **Python 3.10** (matches the upstream pipeline's torch/CUDA stack).

```bash
# from the repo root
python3.10 -m venv .venv-norm
source .venv-norm/bin/activate
pip install -r Codes_Solhee_Tucker/requirements.txt
```

`gilda` ships with ~2M pre-grounded terms from HGNC, UniProt, FamPlex,
ChEBI, MeSH, GO, DOID, EFO, HP — no extra resource download is needed.

---

## Usage

All scripts are CLI-friendly with sensible defaults. Paths default to the
locations used by the upstream pipeline; override with flags.

```bash
# 1. Pilot study to choose the score threshold (run once, ~minutes).
python -m Codes_Solhee_Tucker.poc.poc_threshold_study \
    --chemprot ../chemprot-relexner-pipeline-main/chemprot-relexner-pipeline-main/rl_files/merged_relation_predictions_all_models.json \
    --euadr    ../298Code_Tran_Brian/euadr_full.json \
    --n 200 \
    --out Codes_Solhee_Tucker/poc/poc_report.md

# 2. Normalize ChemProt triples (uses sentence context for disambiguation).
python -m Codes_Solhee_Tucker.normalize_chemprot \
    --in  ../chemprot-relexner-pipeline-main/chemprot-relexner-pipeline-main/rl_files/merged_relation_predictions_all_models.json \
    --out Codes_Solhee_Tucker/outputs/normalized_chemprot_triples.json \
    --score-threshold 0.7

# 3. Normalize EU-ADR triples (uses raw bigbio_kb so sentence context is available).
python -m Codes_Solhee_Tucker.normalize_euadr \
    --out Codes_Solhee_Tucker/outputs/normalized_euadr_triples_v2.json \
    --score-threshold 0.7

# 4. Rebuild the unified KG with canonical node IDs.
python -m Codes_Solhee_Tucker.build_normalized_kg \
    --chemprot Codes_Solhee_Tucker/outputs/normalized_chemprot_triples.json \
    --euadr    Codes_Solhee_Tucker/outputs/normalized_euadr_triples_v2.json \
    --out      Codes_Solhee_Tucker/outputs/normalized_unified_kg.graphml

# 5. Before/after comparison against the legacy KGs.
python -m Codes_Solhee_Tucker.compare_kgs \
    --baseline-dir ../chemprot-relexner-pipeline-main/chemprot-relexner-pipeline-main/output_kgs \
    --normalized   Codes_Solhee_Tucker/outputs/normalized_unified_kg.graphml \
    --out          Codes_Solhee_Tucker/outputs/comparison_report.md
```

---

## Output schema (normalized triple)

```json
{
  "sentence": "Androgen antagonistic effect of estramustine ...",
  "head_text": "Androgen",
  "head_db":   "HGNC",
  "head_id":   "HGNC:417",
  "head_name": "AR",
  "head_score": 0.97,
  "tail_text": "estramustine",
  "tail_db":   "CHEBI",
  "tail_id":   "CHEBI:64349",
  "tail_name": "estramustine",
  "tail_score": 0.95,
  "relation":  "CPR:3",
  "source":    "chemprot",
  "models":    ["bert-base-cased", "biobert", "biogpt"],
  "confidence": 1.0,
  "grounded":  true
}
```

When grounding fails, `*_db`/`*_id` are `null`, `grounded` is `false`,
and the policy on whether to keep the triple is set in `config.py`
(decided after the POC).

---

## Reviewer-response artifacts

After running the pipeline, two files are intended for the manuscript
revision:

- `outputs/comparison_report.md` — quantitative before/after table
  (node count, edge count, merged-node examples, multi-hop path count).
- The `outputs/normalized_unified_kg.graphml` itself, openable in Gephi /
  Cytoscape for visual inspection.

---

## Citation

If you use this module, please cite Gilda:

> Gyori, B. M., Hoyt, C. T., & Steppi, A. (2022).
> *Gilda: biomedical entity text normalization with machine-learned
> disambiguation as a service.* **Bioinformatics Advances**, 2(1), vbac034.
> https://doi.org/10.1093/bioadv/vbac034

---

## Reproducibility

- Python version: 3.10
- Gilda version: pinned in `requirements.txt` (`>=1.4.0`)
- A grounding cache (`outputs/.grounding_cache.json`) keys results by
  `(text, context_hash)` so re-runs are deterministic and fast.
- All thresholds and namespace policies live in `config.py` — change
  there, not in individual scripts.
