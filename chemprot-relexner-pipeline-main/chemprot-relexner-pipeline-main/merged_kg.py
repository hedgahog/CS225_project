import json
import networkx as nx

def load_chemprot_triples(path):
    with open(path) as f:
        chemprot_data = json.load(f)

    chemprot_triples = []
    for item in chemprot_data:
        models = list(set(item.get("models", [])))  # remove duplicates
        confidence = round(len(models) / 3, 2)       # normalize confidence

        chemprot_triples.append({
            "head": item["head"].strip().lower(),
            "tail": item["tail"].strip().lower(),
            "relation": item["relation"],
            "confidence": confidence,
            "source": "chemprot",
            "models": models
        })
    return chemprot_triples

def load_euadr_triples(path):
    with open(path) as f:
        euadr_data = json.load(f)

    euadr_triples = []
    for item in euadr_data:
        euadr_triples.append({
            "head": item["head"].strip().lower(),
            "tail": item["tail"].strip().lower(),
            "relation": item["relation"],
            "confidence": item["confidence"],
            "source": "euadr",
            "models": []  # EU-ADR is curated
        })
    return euadr_triples

def build_knowledge_graph(all_triples):
    G = nx.DiGraph()

    for triple in all_triples:
        head = triple["head"]
        tail = triple["tail"]
        relation = triple["relation"]

        if head == tail:
            continue  # ❌ skip self-loops

        G.add_node(head)
        G.add_node(tail)

        G.add_edge(head, tail, 
                   label=relation,
                   source=triple.get("source", "chemprot"),
                   confidence=triple.get("confidence", 1.0),
                   models=triple.get("models", []))
    
    return G

def save_graph(G, path="unified_kg.graphml"):
    # Convert lists to strings for GraphML compatibility
    for u, v, data in G.edges(data=True):
        if isinstance(data.get("models"), list):
            data["models"] = ", ".join(data["models"])

    nx.write_graphml(G, path)
    print(f"[SUCCESS] Saved unified KG to {path}")

if __name__ == "__main__":
    chemprot_path = "/data/akshatkrishna/chemprot-relexner-pipeline-main/rl_files/merged_relation_predictions_all_models.json"
    euadr_path = "normalized_euadr_triples.json"

    chemprot_triples = load_chemprot_triples(chemprot_path)
    euadr_triples = load_euadr_triples(euadr_path)

    all_triples = chemprot_triples + euadr_triples
    print(f"[INFO] Total merged triples: {len(all_triples)}")

    G = build_knowledge_graph(all_triples)
    print(f"[INFO] Final KG has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    save_graph(G, "unified_kg.graphml")