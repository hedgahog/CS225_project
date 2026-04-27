import json
import os
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import hashlib

def infer_entity_type(entity):
    """Simple heuristic for demo purposes. Can be extended."""
    entity = entity.lower()
    if "factor" in entity or "receptor" in entity:
        return "receptor"
    elif "protein" in entity or "kinase" in entity:
        return "protein"
    elif "cell" in entity:
        return "cell"
    elif "drug" in entity or "treatment" in entity:
        return "drug"
    elif "virus" in entity or "infection" in entity:
        return "disease"
    elif "gene" in entity:
        return "gene"
    else:
        return "other"

def get_color(entity_type):
    color_map = {
        "protein": "skyblue",
        "receptor": "lightgreen",
        "cell": "orange",
        "drug": "violet",
        "disease": "tomato",
        "gene": "gold",
        "other": "gray"
    }
    return color_map.get(entity_type, "gray")

def create_combined_kg(all_models_data, output_dir="output_kgs"):
    G = nx.DiGraph()
    edge_sources = defaultdict(set)
    node_types = {}

    for model_name, predictions in all_models_data.items():
        for item in predictions:
            for rel in item.get("relations", []):
                head = rel["head"].strip()
                tail = rel["tail"].strip()
                relation = rel["relation"].strip()

                if relation != "no_relation":
                    edge_key = (head, tail, relation)
                    edge_sources[edge_key].add(model_name)

                    # Store node type
                    node_types[head] = infer_entity_type(head)
                    node_types[tail] = infer_entity_type(tail)

    # Build the final graph
    for (head, tail, relation), models in edge_sources.items():
        G.add_node(head, entity_type=node_types[head])
        G.add_node(tail, entity_type=node_types[tail])
        G.add_edge(head, tail, label=relation, models=", ".join(sorted(models)))

    # Save GraphML
    graphml_path = os.path.join(output_dir, "combined_kg.graphml")
    nx.write_graphml(G, graphml_path)

    # Visualization
    pos = nx.spring_layout(G, k=0.5, seed=42)
    plt.figure(figsize=(18, 14))

    node_colors = [get_color(G.nodes[node]["entity_type"]) for node in G.nodes()]
    nx.draw(
        G, pos,
        with_labels=True,
        node_color=node_colors,
        node_size=2500,
        edge_color="gray",
        font_size=10,
        font_weight="bold"
    )

    # Combine relation + model info for edge labels
    edge_labels = {
        (u, v): f"{d['label']} ({d['models']})"
        for u, v, d in G.edges(data=True)
    }
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black', font_size=8)

    plt.axis("off")
    plt.tight_layout()

    # Save PNG
    image_path = os.path.join(output_dir, "combined_kg.png")
    plt.savefig(image_path, dpi=300)
    plt.close()

    print(f"✅ Combined KG saved:\n- GraphML: {graphml_path}\n- PNG: {image_path}")

def main():
    input_json = "/data/akshatkrishna/chemprot-relexner-pipeline-main/rl_files/relation_predictions_all_models.json"
    output_dir = "combined_output_kgs"
    os.makedirs(output_dir, exist_ok=True)

    # Load all model data
    with open(input_json, "r") as f:
        all_models_data = json.load(f)

    # Build combined KG
    create_combined_kg(all_models_data, output_dir=output_dir)

if __name__ == "__main__":
    main()