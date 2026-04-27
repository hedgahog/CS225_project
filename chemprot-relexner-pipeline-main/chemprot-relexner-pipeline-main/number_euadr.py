import json

# Path to your EU-ADR JSON file
with open("normalized_euadr_triples.json", "r") as f:
    data = json.load(f)

# Total number of relation triples
total_triples = len(data)

print(f"✅ Total relation triples in EU-ADR dataset: {total_triples}")