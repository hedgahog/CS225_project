import os
import json
import random

with open("../chemprot-relexner-pipeline-main/chemprot-relexner-pipeline-main/filtered_inferred_links.json") as f:
    all_paths = json.load(f)

# Filter to only cross-dataset paths (contain euadr relations)
euadr_paths = [
    p for p in all_paths 
    if any(r[1] == "euadr" for r in p["relations"])
]

print(f"Cross-dataset paths: {len(euadr_paths)}")  # 2178'''

# Sample 150 for judging
random.seed(42)  # set seed for reproducibility
sample = random.sample(euadr_paths, 30)
#print(sample)

api_key = os.getenv("OPENAI_API_KEY")

from openai import OpenAI
client = OpenAI()

system_prompt = "You are a helpful and precise assistant for checking the correctness of inferred relations between chemical entities and diseases. Given a path of relations, determine if the final inferred relation is correct based on the intermediate relations. Output 'PASS' if the final relation is supported by the intermediate relations, and 'FAIL' otherwise. Explain your reasoning briefly. Please state the overall number of PASSES and FAILS at the end of your evaluation."

results = []

for i, sample in enumerate(sample):
    path_string = json.dumps(sample, indent=2)
    

    response = client.chat.completions.create(
        model="gpt-5.4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": path_string}]
    )
    
    verdict = response.choices[0].message.content.strip()

    # Track progress and results
    print(f"Path_number: {i+1}")
    print(path_string)
    print(f"Verdict: {verdict}")
    print("\n\n")

    # Store results
    results.append({
        "path_number": i+1,
        "path": sample,
        "verdict": verdict
    })
    
