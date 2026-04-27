import networkx as nx

# Define relation categories (adjust as needed)
CPR_RELATIONS = {"CPR:3", "CPR:4", "CPR:5", "CPR:6", "CPR:9"}
EUADR_RELATIONS = {"pa", "sa", "na"}

def load_graph(graph_path="unified_kg.graphml"):
    G = nx.read_graphml(graph_path)
    return G

def is_valid_relation(rel, source, allow_chemprot=True, allow_euadr=True):
    if allow_chemprot and source == "chemprot" and rel in CPR_RELATIONS:
        return True
    if allow_euadr and source == "euadr" and rel in EUADR_RELATIONS:
        return True
    return False

def find_inferred_links(G, max_hops=3):
    inferred_links = []

    for src in G.nodes():
        for tgt in G.nodes():
            if src == tgt:
                continue

            try:
                paths = list(nx.all_simple_paths(G, source=src, target=tgt, cutoff=max_hops))
            except nx.NetworkXNoPath:
                continue

            for path in paths:
                if len(path) < 3:
                    continue  # skip direct links and self-loops

                # extract relations along the path
                relations = []
                valid_path = True

                for i in range(len(path) - 1):
                    edge = G.get_edge_data(path[i], path[i+1])
                    if edge:
                        rel = edge.get("label")
                        source = edge.get("source")
                        if not is_valid_relation(rel, source):
                            valid_path = False
                            break
                        relations.append((rel, source))
                    else:
                        valid_path = False
                        break

                if valid_path:
                    inferred_links.append({
                        "path": path,
                        "relations": relations,
                        "inferred": f"{path[0]} → {path[-1]}"
                    })

    return inferred_links

def save_inferred_links(links, output_path="inferred_multi_hop_links_max_hop_4.json"):
    import json
    with open(output_path, "w") as f:
        json.dump(links, f, indent=2)
    print(f"[SUCCESS] Saved {len(links)} inferred multi-hop paths to {output_path}")

if __name__ == "__main__":
    G = load_graph("unified_kg.graphml")
    print(f"[INFO] Loaded KG with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    inferred_links = find_inferred_links(G, max_hops=4)
    print(f"[INFO] Found {len(inferred_links)} valid multi-hop paths")

    save_inferred_links(inferred_links)