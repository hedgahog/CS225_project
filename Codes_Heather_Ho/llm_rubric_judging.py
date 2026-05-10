import os
import json
import random

with open("filtered_inferred_links.json") as f:
    all_paths = json.load(f)

# Filter to only cross-dataset paths (contain euadr relations)
euadr_paths = [
    p for p in all_paths 
    if any(r[1] == "euadr" for r in p["relations"])
]

print(f"Cross-dataset paths: {len(euadr_paths)}")  # 2178

# Sample 150 for judging
random.seed(42)  # set seed for reproducibility
sample = random.sample(euadr_paths, 5)
#print(sample)

'''api_key = os.getenv("OPENAI_API_KEY")

from openai import OpenAI
client = OpenAI()

response = client.responses.create(
    model="gpt-5.4",
    instructions="You are a helpful and precise assistant for checking the correctness of inferred relations between chemical entities and diseases. Given a path of relations, determine if the final inferred relation is correct based on the intermediate relations. Output 'Correct' if the final relation is supported by the intermediate relations, and 'Incorrect' otherwise.",
    input=sample
)

print(response.output_text)'''