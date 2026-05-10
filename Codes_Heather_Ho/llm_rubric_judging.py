import os
import json
import random
# TODO: calculate total passes and fails
with open("../chemprot-relexner-pipeline-main/chemprot-relexner-pipeline-main/filtered_inferred_links.json") as f:
    all_paths = json.load(f)

# Filter to only cross-dataset paths (contain euadr relations)
euadr_paths = [
    p for p in all_paths 
    if any(r[1] == "euadr" for r in p["relations"])
]

#print(f"Cross-dataset paths: {len(euadr_paths)}")  # 2178'''

# load prompt.txt
with open("prompt.txt") as f:
    system_prompt = f.read()

print(system_prompt)

# Sample 150 for judging
random.seed(42)  # set seed for reproducibility
sample = random.sample(euadr_paths, 2)
#print(sample)

api_key = os.getenv("OPENAI_API_KEY")

from openai import OpenAI
 # Using openai
client = OpenAI()


def llm_judging(sample, system_prompt, client, model="gpt-5.4"):
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

        # Load results into a text file
    with open(f"judging_results_{model}.txt", "w") as f:
        for result in results:
            f.write(f"Path_number: {result['path_number']}\n")
            f.write(json.dumps(result['path'], indent=2) + "\n")
            f.write(f"Verdict: {result['verdict']}\n\n")
    return "Done judging and results saved to judging_results.txt"

   
status = llm_judging(sample, system_prompt, client)
print(status)
        
