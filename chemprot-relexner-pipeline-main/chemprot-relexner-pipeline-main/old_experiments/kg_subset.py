import json
import networkx as nx
import matplotlib.pyplot as plt

# === USER PARAMETERS ===
MODEL_NAME = "bio-gpt"     # Options: 'bert-base-cased', 'biobert', 'bio-gpt'
INPUT_FILE = "/data/akshatkrishna/chemprot-relexner-pipeline-main/rl_files/relation_predictions_bio_gpt.json"              # Limit to this many sentences for visualization
MIN_NODE_DEGREE = 2               # Only show nodes with this minimum degree
OUTPUT_IMAGE = f"kg_subgraph_{MODEL_NAME}.png"

# === LOAD DATA ===
with open(INPUT_FILE, "r") as f:
    all_model_data = json.load(f)

if MODEL_NAME not in all_model_data:
    raise ValueError(f"Model '{MODEL_NAME}' not found in JSON.")

data_subset = all_model_data[MODEL_NAME]

# === BUILD GRAPH ===
G = nx.Graph()

for entry in data_subset:
    for rel in entry["relations"]:
        head = rel["head"].strip().lower()
        tail = rel["tail"].strip().lower()
        label = rel["relation"]
        G.add_edge(head, tail, label=label)

# === FILTER BY MINIMUM DEGREE ===
nodes_to_keep = [node for node, deg in G.degree() if deg >= MIN_NODE_DEGREE]
G_sub = G.subgraph(nodes_to_keep)

# === SAVE TO GRAPHML === 🔹
GRAPHML_OUTPUT = f"kg_subgraph_{MODEL_NAME}.graphml"
nx.write_graphml(G_sub, GRAPHML_OUTPUT)
print(f"✅ Saved GraphML to {GRAPHML_OUTPUT}")

# === DRAW GRAPH ===
plt.figure(figsize=(18, 18))
pos = nx.spring_layout(G_sub, k=0.4)

# Node and edge styling
edge_labels = nx.get_edge_attributes(G_sub, 'label')
nx.draw_networkx_nodes(G_sub, pos, node_size=700, node_color='lightblue')
nx.draw_networkx_edges(G_sub, pos, width=1.5, alpha=0.7)
nx.draw_networkx_labels(G_sub, pos, font_size=8)
nx.draw_networkx_edge_labels(G_sub, pos, edge_labels=edge_labels, font_color='red', font_size=7)

plt.title(f"Knowledge Graph Subset – {MODEL_NAME} (sentences)", fontsize=14)
plt.axis('off')
plt.tight_layout()
plt.savefig(OUTPUT_IMAGE, dpi=300)
plt.show()

print(f"✅ Saved graph to {OUTPUT_IMAGE}")