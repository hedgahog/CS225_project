import argparse
import json
import random

import matplotlib.pyplot as plt
import networkx as nx

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--drop", action="store_true")
    args = parser.parse_args()

    suffix = "_drop" if args.drop else ""

    with open(f"../outputs/filtered_inferred_links_normalized{suffix}.json") as f:
        data = json.load(f)

    # Filter for paths that contain BOTH ChemProt and EU-ADR edges
    combined_paths = [
        entry for entry in data
        if "chemprot" in [source for _, source in entry["relations"]] and
           "euadr" in [source for _, source in entry["relations"]]
    ]

    # Sample a small subset (adjust size for clarity)
    sample_size = 15
    sampled_paths = random.sample(combined_paths, min(sample_size, len(combined_paths)))

    # Build the graph
    G = nx.DiGraph()
    for entry in sampled_paths:
        path = entry["path"]
        relations = entry["relations"]
        for i in range(len(path) - 1):
            src = path[i]
            tgt = path[i + 1]
            label = f"{relations[i][0]} ({relations[i][1]})"
            color = "red" if relations[i][1] == "chemprot" else "blue"
            G.add_edge(src, tgt, label=label, color=color)

    edge_colors = [G[u][v]["color"] for u, v in G.edges()]
    # Draw the graph
    plt.figure(figsize=(13, 9))
    pos = nx.spring_layout(G, k=0.6, seed=42)

    nx.draw_networkx_nodes(G, pos, node_color="lightyellow", node_size=1000, edgecolors="black")
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, arrows=True, arrowsize=20)
    nx.draw_networkx_labels(G, pos, font_size=9, font_family="sans-serif")

    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

    plt.title("Sampled Multi-hop Reasoning Paths (ChemProt + EU-ADR)", fontsize=14)
    plt.axis("off")
    plt.tight_layout()

    out_path = f"../outputs/multi_hop_reasoning_subset_normalized{suffix}.png"
    plt.savefig(out_path, dpi=300)
    print(f"✅ Graph saved to '{out_path}'")