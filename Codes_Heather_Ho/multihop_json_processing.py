import json
import random
with open("filtered_inferred_links_normalized.json") as f:
    all_paths = json.load(f)

'''# Filter to only cross-dataset paths (contain euadr relations)
euadr_paths = [
    p for p in all_paths 
    if any(r[1] == "euadr" for r in p["relations"])
]

print(f"Cross-dataset paths: {len(euadr_paths)}")  # 2178'''

# Sample 150 for judging
random.seed(42)  # set seed for reproducibility
samples = random.sample(all_paths, 5)
#Add path ids to samples
for i, sample in enumerate(samples):
    print(f"Path_number: {i+1}")
    print(json.dumps(sample, indent=2))
    print("\n\n")

    




'''def format_path_for_judging(path): 
    return json.dumps(path, indent=2)

print(format_path_for_judging(sample[0]))'''