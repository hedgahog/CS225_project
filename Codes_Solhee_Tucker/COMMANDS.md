# Pipeline Run Commands

---

## 1. Gilda Normalization

**실행 위치: 프로젝트 루트** (`CS225_project/`)

### 1-1. ChemProt 트리플 정규화
- 입력: `chemprot-relexner-pipeline-main/.../merged_relation_predictions_all_models.json` (11,796 triples)
- 출력: `Codes_Solhee_Tucker/outputs/gilda_normalized_chemprot_triples.json`

```bash
python -m Codes_Solhee_Tucker.normalize_chemprot
```

### 1-2. EU-ADR 트리플 정규화
- 입력: `298Code_Tran_Brian/euadr_full.json` (300 docs, 2,891 relations)
- 출력: `Codes_Solhee_Tucker/outputs/gilda_normalized_euadr_triples.json`

```bash
python -m Codes_Solhee_Tucker.normalize_euadr
```

### 1-3. Normalized KG 빌드
- 입력: 위 두 JSON
- 출력: `Codes_Solhee_Tucker/outputs/normalized_unified_kg.graphml`

```bash
python -m Codes_Solhee_Tucker.build_normalized_kg
```

---

## 2. Following Pipeline

**실행 위치: `Codes_Solhee_Tucker/Chemprot_pipeline_edited/`**

```bash
cd Codes_Solhee_Tucker/Chemprot_pipeline_edited
```

### 2-1. Multi-hop reasoning
- 입력: `../outputs/normalized_unified_kg.graphml`
- 출력: `../outputs/inferred_multi_hop_links_normalized.json`
- 최대 4-hop, 유효한 CPR/EU-ADR 관계만 유지

```bash
python multi_hop_reasoning_v2.py
```

### 2-2. Filtered multi-hop
- 입력: `../outputs/normalized_unified_kg.graphml`
- 출력: `../outputs/filtered_inferred_links_normalized.json`
- 첫 엣지가 CPR 관계인 경로만 유지

```bash
python "filtered_multi_hop.reasoning_v2.py"
```

### 2-3. 시각화
- 입력: `../outputs/filtered_inferred_links_normalized.json`
- 출력: `../outputs/multi_hop_reasoning_subset_normalized.png`

```bash
python graph_multi_hop_v2.py
```

---

## 1-drop. Gilda Normalization (drop policy)

> threshold 미만 엔티티가 포함된 트리플을 **제외**하는 버전.  
> normalize 단계는 캐시가 warm 상태라 거의 즉시 완료.

**실행 위치: 프로젝트 루트** (`CS225_project/`)

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

**실행 위치: `Codes_Solhee_Tucker/Chemprot_pipeline_edited/`**

### 2-drop-1. Multi-hop reasoning
- 입력: `../outputs/normalized_unified_kg_drop.graphml`
- 출력: `../outputs/inferred_multi_hop_links_normalized_drop.json`

```bash
python multi_hop_reasoning_v2.py --drop
```

### 2-drop-2. Filtered multi-hop
- 입력: `../outputs/normalized_unified_kg_drop.graphml`
- 출력: `../outputs/filtered_inferred_links_normalized_drop.json`

```bash
python "filtered_multi_hop.reasoning_v2.py" --drop
```

### 2-drop-3. 시각화
- 입력: `../outputs/filtered_inferred_links_normalized_drop.json`
- 출력: `../outputs/multi_hop_reasoning_subset_normalized_drop.png`

```bash
python graph_multi_hop_v2.py --drop
```

---

## 3. Before / After 비교

**실행 위치: 프로젝트 루트** (`CS225_project/`)

### keep_as_raw
- 출력: `Codes_Solhee_Tucker/outputs/comparison_report.md`

```bash
python -m Codes_Solhee_Tucker.compare_kgs
```

### drop
- 출력: `Codes_Solhee_Tucker/outputs/comparison_report_drop.md`

```bash
python -m Codes_Solhee_Tucker.compare_kgs \
    --normalized Codes_Solhee_Tucker/outputs/normalized_unified_kg_drop.graphml \
    --out        Codes_Solhee_Tucker/outputs/comparison_report_drop.md
```