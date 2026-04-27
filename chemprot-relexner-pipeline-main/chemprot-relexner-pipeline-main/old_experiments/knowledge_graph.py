import json
import os
import networkx as nx
import matplotlib.pyplot as plt

def create_kg_for_model(model_name, predictions, output_dir="output_kgs"):
    G = nx.DiGraph()

    for item in predictions:
        for rel in item.get("relations", []):
            head = rel["head"].strip()
            tail = rel["tail"].strip()
            relation = rel["relation"].strip()

            if relation != "no_relation":
                G.add_node(head)
                G.add_node(tail)
                G.add_edge(head, tail, label=relation)

    # File names
    graphml_path = os.path.join(output_dir, f"{model_name}_kg.graphml")
    image_path = os.path.join(output_dir, f"{model_name}_kg.png")

    # Save .graphml
    nx.write_graphml(G, graphml_path)

    # Save PNG image
    pos = nx.spring_layout(G, k=0.5, seed=42)
    plt.figure(figsize=(16, 12))
    nx.draw(
        G, pos,
        with_labels=True,
        node_color="skyblue",
        node_size=2000,
        edge_color="gray",
        font_size=10,
        font_weight="bold"
    )
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=8)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(image_path, dpi=300)
    plt.close()

    print(f"✅ Saved KG for {model_name}:\n- GraphML: {graphml_path}\n- PNG: {image_path}")


def main():
    input_json = "/data/akshatkrishna/chemprot-relexner-pipeline-main/rl_files/relation_predictions_all_models.json"
    output_dir = "output_kgs"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load the JSON
    with open(input_json, "r") as f:
        all_models_data = json.load(f)

    # Build KG per model
    for model_name, predictions in all_models_data.items():
        print(f"\n🔍 Building KG for model: {model_name}")
        create_kg_for_model(model_name, predictions, output_dir=output_dir)

if __name__ == "__main__":
    main()