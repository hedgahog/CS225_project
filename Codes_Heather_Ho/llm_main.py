import os
import json
import random
from openai import OpenAI
from openai_rubric_judging import llm_judging

with open("../chemprot-relexner-pipeline-main/chemprot-relexner-pipeline-main/filtered_inferred_links.json") as f:
    all_paths = json.load(f)

# Filter to only cross-dataset paths (contain euadr relations)
euadr_paths = [
    p for p in all_paths 
    if any(r[1] == "euadr" for r in p["relations"])
]

#print(f"Cross-dataset paths: {len(euadr_paths)}")  # 2178'''

# load regular prompt.txt
with open("prompt.txt") as f:
    system_prompt = f.read()

# load rubric prompt.txt
with open("rubric_prompt.txt") as f:
    rubric_system_prompt = f.read()

# Sample 150 for judging
random.seed(42)  # set seed for reproducibility
sample_lst = random.sample(euadr_paths, 150)
#print(sample)

api_key = os.getenv("OPENAI_API_KEY")

 # Using openai
client = OpenAI()

if __name__ == "__main__":
    print("-----Running basic prompt on LLM-----")
    basic_results = llm_judging(sample_lst, system_prompt, client, model="gpt-5.4")
    print("-----Running rubric prompt on LLM-----")
    rubric_results = llm_judging(sample_lst, rubric_system_prompt, client, model="gpt-5.4")

    # Save results to text files
    with open("basic_judging_results.txt", "w") as f:
        json.dump(basic_results, f, indent=2, ensure_ascii=False)
    with open("rubric_judging_results.txt", "w") as f:
        json.dump(rubric_results, f, indent=2, ensure_ascii=False)
    
    # Compare results

    basic_passes = sum(1 for r in basic_results if "PASS" in r["verdict"])
    basic_fails = sum(1 for r in basic_results if "FAIL" in r["verdict"])

    rubric_passes = sum(1 for r in rubric_results if "PASS" in r["verdict"])
    rubric_fails = sum(1 for r in rubric_results if "FAIL" in r["verdict"])

    print(f"Basic Prompt Results:")
    print(f"Passes: {basic_passes}, Fails: {basic_fails}, Total: {basic_passes + basic_fails}")

    print(f"\nRubric Prompt Results:")
    print(f"Passes: {rubric_passes}, Fails: {rubric_fails}, Total: {rubric_passes + rubric_fails}")