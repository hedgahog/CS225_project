import json
import os
from pyvis.network import Network
from collections import defaultdict

def infer_entity_type(entity):
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
        "protein": "#87CEEB",   # skyblue
        "receptor": "#90EE90",  # lightgreen
        "cell": "#FFA500",      # orange
        "drug": "#EE82EE",      # violet
        "disease": "#FF6347",   # tomato
        "gene": "#FFD700",      # gold
        "other": "#A9A9A9"      # dark gray
    }
    return color_map.get(entity_type, "#A9A9A9")

def create_interactive_kg(all_models_data, output_html="output_kgs/interactive_combined_kg.html"):
    net = Network(height="900px", width="100%", bgcolor="#ffffff", font_color="black", notebook=False, directed=True)
    net.force_atlas_2based(gravity=-25, central_gravity=0.001, spring_length=200, spring_strength=0.001)

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

                    node_types[head] = infer_entity_type(head)
                    node_types[tail] = infer_entity_type(tail)

    nodes_added = set()
    for (head, tail, relation), models in edge_sources.items():
        if head not in nodes_added:
            net.add_node(head, label=head, color=get_color(node_types[head]), title=f"Entity type: {node_types[head]}")
            nodes_added.add(head)
        if tail not in nodes_added:
            net.add_node(tail, label=tail, color=get_color(node_types[tail]), title=f"Entity type: {node_types[tail]}")
            nodes_added.add(tail)

        model_list = ", ".join(sorted(models))
        net.add_edge(head, tail, label=f"{relation} ({model_list})", title=f"{relation} from {model_list}")

    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    net.write_html(output_html)
    print(f"✅ Interactive KG saved at: {output_html}")

def main():
    input_json = "/data/akshatkrishna/chemprot-relexner-pipeline-main/rl_files/relation_predictions_all_models.json"
    output_html = "interactive_output_kgs/interactive_combined_kg.html"

    with open(input_json, "r") as f:
        all_models_data = json.load(f)

    create_interactive_kg(all_models_data, output_html)

if __name__ == "__main__":
    main()