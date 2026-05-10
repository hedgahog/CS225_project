import json
def llm_judging(path_json, system_prompt, client, model="gpt-5.4"):
    results = []
    passes = 0
    fails = 0

    for i, path in enumerate(path_json):
        path_string = json.dumps(path, indent=2)
        

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": path_string}]
        )
        
        verdict = response.choices[0].message.content.strip()
        if "pass" in verdict.lower():
            passes += 1
        elif "fail" in verdict.lower():
            fails += 1

        # Track progress and results
        print(f"Path_number: {i+1}")
        print(path_string)
        print(verdict)
        print("\n\n")

        # Store results
        results.append({
            "path_number": i+1,
            "path": path,
            "verdict": verdict
        })
    print(f"Total Passes: {passes}")
    print(f"Total Fails: {fails}")
        # Load results into a text file
    with open(f"judging_results_{model}.txt", "w") as f:
        for result in results:
            f.write(f"Path_number: {result['path_number']}\n")
            f.write(json.dumps(result['path'], indent=2) + "\n")
            f.write(f"Verdict: {result['verdict']}\n\n")
        f.write(f"Total Passes: {passes}\n")
        f.write(f"Total Fails: {fails}\n")
    return "Done judging and results saved to judging_results.txt"

   

        
