import os
import json
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt

def load_combined_graph(graphml_path):
    G = nx.read_graphml(graphml_path, node_type=str)
    return nx.DiGraph(G)  # Ensure directional graph for reasoning

def find_multi_hop_paths(graph, max_hops=3):
    inferred_links = set()

    for source in graph.nodes():
        for target in graph.nodes():
            if source == target:
                continue
            paths = list(nx.all_simple_paths(graph, source=source, target=target, cutoff=max_hops))
            for path in paths:
                if len(path) > 2:
                    # Avoid cycles and self-loops
                    if len(set(path)) == len(path):
                        inferred_links.add((path[0], path[-1], f"inferred_{len(path)-1}_hop"))

    return inferred_links

def add_inferred_links_to_graph(G, inferred_links):
    for head, tail, relation in inferred_links:
        if not G.has_edge(head, tail):
            G.add_edge(head, tail, label=relation, models="multi-hop")

def visualize_graph(G, output_path):
    pos = nx.spring_layout(G, k=0.5, seed=42)
    plt.figure(figsize=(20, 15))

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

    node_colors = [get_color(G.nodes[n].get("entity_type", "other")) for n in G.nodes()]
    nx.draw(
        G, pos,
        with_labels=True,
        node_color=node_colors,
        node_size=2500,
        edge_color="gray",
        font_size=10,
        font_weight="bold"
    )
    edge_labels = {(u, v): d['label'] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[✓] Multi-hop KG saved to {output_path}")

def main():
    graphml_path = "kg_subgraph_bio-gpt.graphml"
    output_graphml = "multi_hop_kg_biogpt.graphml"
    output_png = "multi_hop_kg_biogpt.png"

    G = load_combined_graph(graphml_path)

    # Step 1: Multi-hop reasoning (2-hop and 3-hop)
    inferred_links = find_multi_hop_paths(G, max_hops=3)
    print(f"[INFO] Found {len(inferred_links)} multi-hop inferred relations")

    # Step 2: Add to graph
    add_inferred_links_to_graph(G, inferred_links)

    # Step 3: Save and visualize
    nx.write_graphml(G, output_graphml)
    visualize_graph(G, output_png)

if __name__ == "__main__":
    main()