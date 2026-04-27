import networkx as nx
import json

# Example rules as path patterns
RULES = [
    {
        "pattern": ["CPR:5", "CPR:3"],
        "inferred": "CPR:3"
    },
    {
        "pattern": ["CPR:6", "CPR:4"],
        "inferred": "CPR:6"
    },
    {
        "pattern": ["CPR:9", "CPR:4"],
        "inferred": "CPR:9"
    }
]

def match_rule(path_relations):
    for rule in RULES:
        if path_relations == rule["pattern"]:
            return rule["inferred"]
    return None

def apply_rule_based_inference(graph, max_hops=3):
    inferred_relations = []

    for source in graph.nodes():
        for target in graph.nodes():
            if source == target:
                continue
            paths = list(nx.all_simple_paths(graph, source=source, target=target, cutoff=max_hops))
            for path in paths:
                if len(path) == 3:  # 2-hop
                    rel_1 = graph[path[0]][path[1]].get('label')
                    rel_2 = graph[path[1]][path[2]].get('label')
                    inferred = match_rule([rel_1, rel_2])
                    if inferred:
                        inferred_relations.append({
                            "head": path[0],
                            "tail": path[2],
                            "relation": inferred,
                            "via": path,
                            "path_relations": [rel_1, rel_2],
                            "inference_type": "rule-based"
                        })

    return inferred_relations

def save_inferred_edges_json(inferred_relations, output_path):
    with open(output_path, "w") as f:
        json.dump(inferred_relations, f, indent=2)
    print(f"[✓] Saved {len(inferred_relations)} rule-based inferred edges to {output_path}")

def main():
    input_graphml = "combined_output_kgs/combined_kg.graphml"
    output_json = "combined_output_kgs/rule_based_inferred_edges.json"

    graph = nx.read_graphml(input_graphml)
    graph = nx.DiGraph(graph)

    inferred = apply_rule_based_inference(graph, max_hops=3)
    save_inferred_edges_json(inferred, output_json)

if __name__ == "__main__":
    main()