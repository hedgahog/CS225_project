# Pipeline Run Commands

---

## 1. Gilda Normalization

**Run from: project root** (`CS225_project/`)

### 1-1. ChemProt triple normalization
- Input: `chemprot-relexner-pipeline-main/.../merged_relation_predictions_all_models.json` (11,796 triples)
- Output: `Codes_Solhee_Tucker/outputs/gilda_normalized_chemprot_triples.json`

```bash
python -m Codes_Solhee_Tucker.normalize_chemprot
```

### 1-2. EU-ADR triple normalization
- Input: `298Code_Tran_Brian/euadr_full.json` (300 docs, 2,891 relations)
- Output: `Codes_Solhee_Tucker/outputs/gilda_normalized_euadr_triples.json`

```bash
python -m Codes_Solhee_Tucker.normalize_euadr
```

### 1-3. Build normalized KG
- Input: the two JSON files above
- Output: `Codes_Solhee_Tucker/outputs/normalized_unified_kg.graphml`

```bash
python -m Codes_Solhee_Tucker.build_normalized_kg
```

---

## 2. Following Pipeline

**Run from: `Codes_Solhee_Tucker/Chemprot_pipeline_edited/`**

```bash
cd Codes_Solhee_Tucker/Chemprot_pipeline_edited
```

### 2-1. Multi-hop reasoning
- Input: `../outputs/normalized_unified_kg.graphml`
- Output: `../outputs/inferred_multi_hop_links_normalized.json`
- Up to 4-hop paths; only valid CPR/EU-ADR relations are kept

```bash
python multi_hop_reasoning_v2.py
```

### 2-2. Filtered multi-hop
- Input: `../outputs/normalized_unified_kg.graphml`
- Output: `../outputs/filtered_inferred_links_normalized.json`
- Only paths whose first edge is a CPR relation are kept

```bash
python "filtered_multi_hop.reasoning_v2.py"
```

### 2-3. Visualization
- Input: `../outputs/filtered_inferred_links_normalized.json`
- Output: `../outputs/multi_hop_reasoning_subset_normalized.png`

```bash
python graph_multi_hop_v2.py
```

---

## 1-drop. Gilda Normalization (drop policy)

> Variant that **excludes** triples containing any entity below the score threshold.  
> The normalize step completes almost instantly when the grounding cache is warm.

**Run from: project root** (`CS225_project/`)

```bash
python -m Codes_Solhee_Tucker.normalize_chemprot \
    --policy drop \
    --out Codes_Solhee_Tucker/outputs/gilda_normalized_chemprot_triples_drop.json

python -m Codes_Solhee_Tucker.normalize_euadr \
    --policy drop \
    --out Codes_Solhee_Tucker/outputs/gilda_normalized_euadr_triples_drop.json

python -m Codes_Solhee_Tucker.build_normalized_kg \
    --chemprot Codes_Solhee_Tucker/outputs/gilda_normalized_chemprot_triples_drop.json \
    --euadr    Codes_Solhee_Tucker/outputs/gilda_normalized_euadr_triples_drop.json \
    --out      Codes_Solhee_Tucker/outputs/normalized_unified_kg_drop.graphml
```

---

## 2-drop. Following Pipeline (drop)

**Run from: `Codes_Solhee_Tucker/Chemprot_pipeline_edited/`**

### 2-drop-1. Multi-hop reasoning
- Input: `../outputs/normalized_unified_kg_drop.graphml`
- Output: `../outputs/inferred_multi_hop_links_normalized_drop.json`

```bash
python multi_hop_reasoning_v2.py --drop
```

### 2-drop-2. Filtered multi-hop
- Input: `../outputs/normalized_unified_kg_drop.graphml`
- Output: `../outputs/filtered_inferred_links_normalized_drop.json`

```bash
python "filtered_multi_hop.reasoning_v2.py" --drop
```

### 2-drop-3. Visualization
- Input: `../outputs/filtered_inferred_links_normalized_drop.json`
- Output: `../outputs/multi_hop_reasoning_subset_normalized_drop.png`

```bash
python graph_multi_hop_v2.py --drop
```

---

## 3. Before / After Comparison

**Run from: project root** (`CS225_project/`)

### keep_as_raw
- Output: `Codes_Solhee_Tucker/outputs/comparison_report.md`

```bash
python -m Codes_Solhee_Tucker.compare_kgs
```

### drop
- Output: `Codes_Solhee_Tucker/outputs/comparison_report_drop.md`

```bash
python -m Codes_Solhee_Tucker.compare_kgs \
    --normalized Codes_Solhee_Tucker/outputs/normalized_unified_kg_drop.graphml \
    --out        Codes_Solhee_Tucker/outputs/comparison_report_drop.md
```
