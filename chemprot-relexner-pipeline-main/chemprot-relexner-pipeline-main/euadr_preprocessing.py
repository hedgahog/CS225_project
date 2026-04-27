from datasets import load_dataset
import json

def normalize_eu_adr_bigbio_kb():
    dataset = load_dataset("bigbio/euadr", "euadr_bigbio_kb")["train"]
    triples = []
    seen = set()
    skipped = 0
    relation_types = set()

    for record in dataset:
        entity_map = {e["id"]: e["text"] for e in record["entities"]}
        
        for rel in record["relations"]:
            head_id = rel["arg1_id"]
            tail_id = rel["arg2_id"]
            head_text = entity_map.get(head_id)
            tail_text = entity_map.get(tail_id)

            if not head_text or not tail_text:
                skipped += 1
                continue

            # Convert list -> string if needed
            head_str = " ".join(head_text).strip() if isinstance(head_text, list) else head_text.strip()
            tail_str = " ".join(tail_text).strip() if isinstance(tail_text, list) else tail_text.strip()

            if not head_str or not tail_str:
                skipped += 1
                continue

            rel_type = rel["type"].lower().replace(" ", "_")
            relation_types.add(rel_type)

            key = (head_str.lower(), tail_str.lower(), rel_type)
            if key not in seen:
                seen.add(key)
                triples.append({
                    "head": head_str,
                    "tail": tail_str,
                    "relation": rel_type,
                    "source": "euadr",
                    "confidence": 1.0
                })

    print(f"[INFO] Extracted {len(triples)} unique triples from EU-ADR.")
    print(f"[INFO] Skipped {skipped} entries due to missing or invalid data.")
    print(f"[INFO] Found {len(relation_types)} unique relation types: {sorted(relation_types)}")
    return triples

def save_triples_to_json(triples, output_path="normalized_euadr_triples.json"):
    with open(output_path, "w") as f:
        json.dump(triples, f, indent=2)
    print(f"[SUCCESS] Saved normalized EU-ADR triples to {output_path}")

if __name__ == "__main__":
    triples = normalize_eu_adr_bigbio_kb()
    save_triples_to_json(triples)