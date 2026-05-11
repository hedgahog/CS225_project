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
        if "FAIL" in verdict:
            fails += 1
        elif "PASS" in verdict:
            passes += 1

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
    
   
    return results

   

        
