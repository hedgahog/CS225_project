import json
import random
with open("filtered_inferred_links.json") as f:
    all_paths = json.load(f)

'''# Filter to only cross-dataset paths (contain euadr relations)
euadr_paths = [
    p for p in all_paths 
    if any(r[1] == "euadr" for r in p["relations"])
]

print(f"Cross-dataset paths: {len(euadr_paths)}")  # 2178'''

# Sample 150 for judging
random.seed(42)  # set seed for reproducibility
sample = random.sample(all_paths, 20)
print(sample)


def format_path_for_judging(path): 
    return json.dumps(path, indent=2)

print(format_path_for_judging(sample[0]))