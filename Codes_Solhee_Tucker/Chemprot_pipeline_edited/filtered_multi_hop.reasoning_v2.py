# Experiment file -> The path starts with CHEMPROT only

import networkx as nx
import json

# Relation categories
CPR_RELATIONS = {"CPR:3", "CPR:4", "CPR:5", "CPR:6", "CPR:9"}
EUADR_RELATIONS = {"pa", "sa", "na"}

def load_graph(graph_path="unified_kg.graphml"):
    G = nx.read_graphml(graph_path)
    return G

def is_valid_edge(rel, source):
    if source == "chemprot" and rel in CPR_RELATIONS:
        return True
    if source == "euadr" and rel in EUADR_RELATIONS:
        return True
    return False

def is_meaningful_path(relations):
    """
    Define meaningful logic:
    - Start with CPR from chemprot
    - Followed by one or more EUADR relations
    - All relations must be valid
    """
    if not relations:
        return False

    # Must start with CPR
    first_rel, first_source = relations[0]
    if not (first_source == "chemprot" and first_rel in CPR_RELATIONS):
        return False

    # Subsequent edges can be EU-ADR or more CPRs
    for rel, source in relations[1:]:
        if not is_valid_edge(rel, source):
            return False

    return True

def find_filtered_paths(G, max_hops=3):
    inferred_links = []
    seen = set()

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
                    continue  # Skip direct or too short

                relations = []
                valid = True
                for i in range(len(path) - 1):
                    edge = G.get_edge_data(path[i], path[i + 1])
                    if edge:
                        rel = edge.get("label")
                        source = edge.get("source")
                        relations.append((rel, source))
                    else:
                        valid = False
                        break

                if valid and is_meaningful_path(relations):
                    key = (path[0], path[-1])  # avoid duplicates
                    if key not in seen:
                        seen.add(key)
                        inferred_links.append({
                            "path": path,
                            "relations": relations,
                            "inferred": f"{path[0]} → {path[-1]}"
                        })

    return inferred_links

def save_paths(paths, output_path="filtered_inferred_links.json"):
    with open(output_path, "w") as f:
        json.dump(paths, f, indent=2)
    print(f"[SUCCESS] Saved {len(paths)} filtered multi-hop paths to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--drop", action="store_true")
    args = parser.parse_args()

    suffix = "_drop" if args.drop else ""
    G = load_graph(f"../outputs/normalized_unified_kg{suffix}.graphml")
    print(f"[INFO] Loaded KG with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    filtered_paths = find_filtered_paths(G, max_hops=3)
    print(f"[INFO] Found {len(filtered_paths)} meaningful multi-hop paths")

    save_paths(filtered_paths, f"../outputs/filtered_inferred_links_normalized{suffix}.json")