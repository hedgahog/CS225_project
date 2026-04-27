import json

import json
import networkx as nx

def load_chemprot_triples(path):
    with open(path) as f:
        chemprot_data = json.load(f)

    chemprot_triples = []
    for item in chemprot_data:
        models = list(set(item.get("models", [])))  # remove duplicates
        confidence = round(len(models) / 3, 2)       # normalize confidence

        chemprot_triples.append({
            "head": item["head"].strip().lower(),
            "tail": item["tail"].strip().lower(),
            "relation": item["relation"],
            "confidence": confidence,
            "source": "chemprot",
            "models": models
        })
    return chemprot_triples

def load_euadr_triples(path):
    with open(path) as f:
        euadr_data = json.load(f)

    euadr_triples = []
    for item in euadr_data:
        euadr_triples.append({
            "head": item["head"].strip().lower(),
            "tail": item["tail"].strip().lower(),
            "relation": item["relation"],
            "confidence": item["confidence"],
            "source": "euadr",
            "models": []  # EU-ADR is curated
        })
    return euadr_triples

def compute_entity_overlap(chemprot_triples, euadr_triples):
    chemprot_entities = set()
    euadr_entities = set()

    for triple in chemprot_triples:
        chemprot_entities.add(triple["head"])
        chemprot_entities.add(triple["tail"])

    for triple in euadr_triples:
        euadr_entities.add(triple["head"])
        euadr_entities.add(triple["tail"])

    overlap = chemprot_entities.intersection(euadr_entities)
    print(f"[INFO] Unique entities in CHEMPROT: {len(chemprot_entities)}")
    print(f"[INFO] Unique entities in EU-ADR: {len(euadr_entities)}")
    print(f"[INFO] Unique overlapping entities between CHEMPROT and EU-ADR: {len(overlap)}")
    return overlap

if __name__ == "__main__":
    chemprot_path = "/data/akshatkrishna/chemprot-relexner-pipeline-main/rl_files/merged_relation_predictions_all_models.json"
    euadr_path = "normalized_euadr_triples.json"

    chemprot_triples = load_chemprot_triples(chemprot_path)
    euadr_triples = load_euadr_triples(euadr_path)

    all_triples = chemprot_triples + euadr_triples
    compute_entity_overlap(chemprot_triples, euadr_triples)
    print(f"[INFO] Triples in CHEMPROT: {len(chemprot_triples)}")
    print(f"[INFO] Triples in EU_ADR: {len(euadr_triples)}")
    print(f"[INFO] Total merged triples: {len(all_triples)}")
