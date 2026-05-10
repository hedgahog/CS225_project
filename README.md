# CS225 Project — Biomedical NER & Relation Extraction Pipeline

A pipeline for Named Entity Recognition (NER) and Relation Extraction (RE) on biomedical text using BERT, BioBERT, and BioGPT models, with downstream knowledge graph construction and multi-hop reasoning.

## Repository Structure

```
CS225_project/
├── chemprot-relexner-pipeline-main/   # Main NER + RE pipeline
│   ├── ChemProt_Corpus/               # ← Place downloaded ChemProt data here (see below)
│   ├── ner_files/                     # NER training scripts and model outputs
│   ├── rl_files/                      # Relation extraction scripts and outputs
│   ├── output_kgs/                    # Generated knowledge graph visualizations
│   └── old_experiments/               # Earlier experiment scripts and outputs
├── 298Code_Tran_Brian/                # LLM fine-tuning experiments (Google Colab)
│   ├── Datasets.ipynb                 # Dataset exploration notebook
│   └── LLM_Training.ipynb             # Fine-tuning notebook
│                                      # ← Place ChemProt/EU-ADR JSON files here (see below)
└── Codes_Solhee_Tucker/               # Entity normalization & normalized KG pipeline
    ├── normalize_chemprot.py          # GILDA-based normalization for ChemProt triples
    ├── normalize_euadr.py             # GILDA-based normalization for EU-ADR triples
    ├── build_normalized_kg.py         # Builds unified KG from normalized triples
    ├── grounding.py                   # GILDA grounding utilities
    ├── compare_kgs.py                 # Baseline KG builder (string-norm only)
    ├── outputs/                       # Normalized triples (.json) and KG (.graphml)
    ├── poc/                           # PoC threshold study notebook and results
    └── Chemprot_pipeline_edited/      # Downstream pipeline re-run on normalized KG
        ├── multi_hop_reasoning_v2.py          # Multi-hop reasoning (input: normalized_unified_kg)
        ├── filtered_multi_hop.reasoning_v2.py # Filtered reasoning starting from ChemProt edges
        └── graph_multi_hop_v2.py              # Graph visualization of multi-hop paths
```

## Data Setup

Model weights (`.safetensors`) and raw datasets are excluded from this repo due to size. Follow the steps below to download them before running the code.

---

### 1. ChemProt Corpus

Used by: `ner_files/`, `rl_files/create_rl_data.py`

**Download:**

Option A — Hugging Face (recommended):
```python
from datasets import load_dataset
ds = load_dataset("bigbio/chemprot", "chemprot_bigbio_kb")
```

Option B — Official BioCreative VI release:  
Go to https://biocreative.bioinformatics.udel.edu/tasks/biocreative-vi/track-5/ and download the ChemProt corpus zip.

**Where to place:**  
Extract the zip so that the directory structure looks like this:

```
chemprot-relexner-pipeline-main/
└── ChemProt_Corpus/
    ├── chemprot_training/
    │   ├── chemprot_training_abstracts.tsv
    │   ├── chemprot_training_entities.tsv
    │   ├── chemprot_training_relations.tsv
    │   └── chemprot_training_gold_standard.tsv
    ├── chemprot_development/
    │   ├── chemprot_development_abstracts.tsv
    │   ├── chemprot_development_entities.tsv
    │   ├── chemprot_development_relations.tsv
    │   └── chemprot_development_gold_standard.tsv
    ├── chemprot_test_gs/
    │   ├── chemprot_test_abstracts_gs.tsv
    │   ├── chemprot_test_entities_gs.tsv
    │   ├── chemprot_test_relations_gs.tsv
    │   └── chemprot_test_gold_standard.tsv
    └── chemprot_sample/
        └── ...
```

---

### 2. EU-ADR Dataset

Used by: `euadr_preprocessing.py`

**Download:**  
The script downloads EU-ADR automatically via the Hugging Face `datasets` library — no manual download needed:

```python
from datasets import load_dataset
dataset = load_dataset("bigbio/euadr", "euadr_bigbio_kb")
```

Running `euadr_preprocessing.py` will fetch the data and produce `normalized_euadr_triples.json`.

---

### 3. JSON Data Files for `298Code_Tran_Brian/` Notebooks

Used by: `Datasets.ipynb`, `LLM_Training.ipynb` (run on Google Colab)

These notebooks expect the following files to be uploaded to `/content/` in Colab (or placed in `298Code_Tran_Brian/` if running locally):

| File | Source |
|------|--------|
| `chemprot_train.json` | `load_dataset("bigbio/chemprot", "chemprot_bigbio_kb")["train"]` → save as JSON |
| `chemprot_validation.json` | `load_dataset("bigbio/chemprot", "chemprot_bigbio_kb")["validation"]` → save as JSON |
| `chemprot_test.json` | `load_dataset("bigbio/chemprot", "chemprot_bigbio_kb")["test"]` → save as JSON |
| `euadr_full.json` | `load_dataset("bigbio/euadr", "euadr_bigbio_kb")["train"]` → save as JSON |

Quick export snippet:
```python
import json
from datasets import load_dataset

chem = load_dataset("bigbio/chemprot", "chemprot_bigbio_kb")
for split in ["train", "validation", "test"]:
    with open(f"chemprot_{split}.json", "w") as f:
        json.dump(list(chem[split]), f)

euadr = load_dataset("bigbio/euadr", "euadr_bigbio_kb")
with open("euadr_full.json", "w") as f:
    json.dump(list(euadr["train"]), f)
```

**Where to place:** Upload to `/content/` in Google Colab, or put in `298Code_Tran_Brian/` if running locally.

---

## Installation

```bash
pip install -r chemprot-relexner-pipeline-main/requirements.txt
```

Key dependencies: `transformers`, `datasets`, `torch`, `networkx`, `scikit-learn`, `seqeval`

## Pipeline Overview

### Phase 1 — NER & RE (chemprot-relexner-pipeline-main)

1. **Preprocess ChemProt** — run `rl_files/create_rl_data.py` to generate `rl_files/preprocessed_data/*.csv`
2. **Train NER models** — `ner_files/{bert-base-cased,bio-bert,biogpt}/ner_train.py`
3. **Train RE models** — `rl_files/{bert-base-cased,bio-bert,bio-gpt}/train_rl.py`
4. **Extract entities** — `ner_files/entity_extraction_3_models.py`
5. **Extract relations** — `rl_files/relation_extraction_3_models.py`
6. **Build knowledge graph** — `merged_kg.py`
7. **Multi-hop reasoning** — `multi_hop_reasoning.py`, `graph_multi_hop.py`

### Phase 2 — Entity Normalization & Normalized KG (Codes_Solhee_Tucker)

8. **Normalize ChemProt triples** — `normalize_chemprot.py` → `outputs/gilda_normalized_chemprot_triples.json`
9. **Normalize EU-ADR triples** — `normalize_euadr.py` → `outputs/gilda_normalized_euadr_triples.json`
10. **Build normalized unified KG** — `build_normalized_kg.py` → `outputs/normalized_unified_kg.graphml`
11. **Compare raw vs. normalized KG** — see `Codes_Solhee_Tucker/summary.ipynb`; results in `outputs/comparison_report.md`

### Phase 3 — Downstream Pipeline on Normalized KG (Codes_Solhee_Tucker/Chemprot_pipeline_edited)

12. **Multi-hop reasoning (normalized)** — `multi_hop_reasoning_v2.py` reads `normalized_unified_kg.graphml`, outputs `inferred_multi_hop_links_normalized.json`
13. **Filtered multi-hop reasoning** — `filtered_multi_hop.reasoning_v2.py` restricts paths to those starting from ChemProt edges
14. **Graph visualization (normalized)** — `graph_multi_hop_v2.py` reads `filtered_inferred_links_normalized.json` and plots cross-dataset multi-hop paths
