import networkx as nx
from collections import Counter

# Load the GraphML file
G = nx.read_graphml("unified_kg.graphml")

# Total nodes and edges
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()

# Degree statistics
degrees = [deg for node, deg in G.degree()]
avg_degree = sum(degrees) / len(degrees)
max_degree = max(degrees)
min_degree = min(degrees)

# Relation types (edge labels)
relation_counter = Counter()
for _, _, data in G.edges(data=True):
    label = data.get("label", "unknown")
    relation_counter[label] += 1

# Connected components (undirected view)
undirected = G.to_undirected()
num_components = nx.number_connected_components(undirected)

# Print metrics
print(f"🔢 Total nodes: {num_nodes}")
print(f"🔗 Total edges: {num_edges}")
print(f"📈 Avg node degree: {avg_degree:.2f}")
print(f"📉 Min node degree: {min_degree}, Max node degree: {max_degree}")
print(f"🔍 Connected components (undirected view): {num_components}\n")

print("📊 Relation Type Distribution (Top 10):")
for rel, count in relation_counter.most_common(10):
    print(f"  {rel}: {count}")