def llm_judging(sample, system_prompt, client, model="gpt-5.4"):
    results = []

    for i, sample in enumerate(sample):
        path_string = json.dumps(sample, indent=2)
        

        response = client.chat.completions.create(
            model=model,
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
        
