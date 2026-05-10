import os
import json
import random
from openai import OpenAI
from openairubric_judging import llm_judging

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
sample_lst = random.sample(euadr_paths, 4)
#print(sample)

api_key = os.getenv("OPENAI_API_KEY")

 # Using openai
client = OpenAI()

if __name__ == "__main__":
    status = llm_judging(sample_lst, system_prompt, client, model="gpt-5.4")
    print(status)