import json
from collections import defaultdict

# Load the merged entity file
with open("merged_entity_output.json", "r") as f:
    data = json.load(f)

# Sets to collect unique entity texts
chemicals = set()
genes = set()

# Count by entity type
for entry in data:
    for entity in entry.get("entities", []):
        entity_type = entity.get("type", "").upper()
        entity_text = entity.get("text", "").strip().lower()

        if not entity_text:
            continue

        if entity_type == "CHEMICAL":
            chemicals.add(entity_text)
        elif entity_type in {"GENE", "GENE-Y", "GENE-N"}:
            genes.add(entity_text)

# Results
print(f"✅ Unique chemicals: {len(chemicals)}")
print(f"✅ Unique gene/protein mentions: {len(genes)}")