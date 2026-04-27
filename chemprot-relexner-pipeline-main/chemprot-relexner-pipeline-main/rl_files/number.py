import json
from collections import Counter, defaultdict

# Load JSON file
with open("merged_relation_predictions_all_models.json") as f:
    data = json.load(f)

# Initialize containers
unique_relations = set()
relation_type_counter = Counter()
confidence_bins = Counter()
model_counter = Counter()
entity_pair_set = set()

# Analyze each triple
for item in data:
    head = item["head"].strip().lower()
    tail = item["tail"].strip().lower()
    relation = item["relation"].strip()
    confidence = item.get("confidence", 0)
    models = item.get("models", [])

    # Unique (head, relation, tail)
    unique_relations.add((head, relation, tail))

    # Count relation type
    relation_type_counter[relation] += 1

    # Count confidence levels
    if confidence >= 0.9:
        confidence_bins["0.9–1.0"] += 1
    elif confidence >= 0.67:
        confidence_bins["0.67–0.89"] += 1
    elif confidence >= 0.5:
        confidence_bins["0.5–0.66"] += 1
    else:
        confidence_bins["<0.5"] += 1

    # Count models
    for model in models:
        model_counter[model] += 1

    # Count entity pairs (head, tail)
    entity_pair_set.add((head, tail))

# Print results
print(f"✅ Total relation triples: {len(data)}")
print(f"✅ Unique (head, relation, tail) triples: {len(unique_relations)}")
print(f"✅ Unique entity pairs (head–tail): {len(entity_pair_set)}\n")

print("📊 Relation Type Distribution:")
for rel, count in relation_type_counter.items():
    print(f"  {rel}: {count}")

print("\n📈 Confidence Score Distribution:")
for bin_label, count in confidence_bins.items():
    print(f"  {bin_label}: {count}")

print("\n🤖 Model Participation:")
for model, count in model_counter.items():
    print(f"  {model}: {count} triples")