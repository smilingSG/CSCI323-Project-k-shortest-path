# main.py
# -----------------------------------------------
# SHORTEST PATH FINDER PROGRAM
# -----------------------------------------------
# Supports multiple shortest path algorithms:
# 1. Dijkstra’s Algorithm (single shortest path)
# 2. Yen’s Algorithm (K shortest paths)
# 3. AI Comparison Suite (Yen-Dijkstra, Yen-Bellman-Ford, A*, K*)
# -----------------------------------------------
# Features:
# - Can read both directed and undirected graphs
# - Supports coordinates for heuristic algorithms
# - Detects negative weights for safe algorithm selection

from typing import Dict, List, Tuple, Any
import networkx as nx
import os

# Import helper and AI modules
from helper import dijkstra_shortest_path
from ai import (
    compare_and_recommend,
    print_summary,
    default_heuristic,
    yen_dijkstra_adapter,
    yen_bellmanford_adapter,
    astar_adapter,
    kstar_adapter,
    has_negative_edge,
    set_coords
)

# Loads a graph from a text file
# - Coordinates are stored for use by heuristic algorithms (A*, K*)
# -----------------------------------------------------
def load_graph_from_file(filename: str) -> nx.DiGraph:
    Gnx = nx.DiGraph()       # initialize directed graph
    coords = {}              # store node coordinates
    directed = False         # track if graph is directed

    with open(filename, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue  # skip empty lines

            # Detect and set graph type
            if line.upper().startswith("# DIRECTED"):
                directed = True
                continue
            if line.upper().startswith("# UNDIRECTED"):
                directed = False
                continue

            # Skip other comment lines
            if line.startswith("#"):
                continue

            parts = line.split()

            # Try to parse coordinate line: node x y
            if len(parts) == 3:
                n, xs, ys = parts
                try:
                    x = float(xs); y = float(ys)
                    coords[n] = (x, y)    # store coordinates
                    Gnx.add_node(n)       # ensure node is added
                    continue
                except ValueError:
                    pass  # if not numeric, treat it as an edge instead

            # Parse edge line: u v w
            if len(parts) == 3:
                u, v, w = parts
                Gnx.add_edge(u, v, weight=float(w))
                # If undirected, add reverse edge
                if not directed:
                    Gnx.add_edge(v, u, weight=float(w))
            else:
                print(f"Skipping invalid line: {line}")

    # Store coordinates for heuristic
    set_coords(coords)
    return Gnx


# Displays basic statistics about the loaded graph.
# Shows number of nodes, edges, and range of edge weights.
# -----------------------------------------------------
def display_graph_info(G: nx.DiGraph) -> None:
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    weights = [data.get("weight", 1) for _, _, data in G.edges(data=True)]
    w_min = min(weights) if weights else None
    w_max = max(weights) if weights else None

    print(f"\nGraph loaded: {num_nodes} nodes, {num_edges} edges")
    if weights:
        print(f"Edge weights: min={w_min}, max={w_max}")
    else:
        print("No edge weights; defaulting to 1")


# Converts a NetworkX graph into a dictionary format:
#   { node: [(neighbor, weight), ...], ... }
# Therefore is easier for custom AI algorithms to process.
# -----------------------------------------------------
def nx_to_dict(G: nx.DiGraph) -> Dict[Any, List[Tuple[Any, float]]]:
    d: Dict[Any, List[Tuple[Any, float]]] = {}
    for u, v, data in G.edges(data=True):
        d.setdefault(u, []).append((v, data.get("weight", 1.0)))
    for n in G.nodes:
        d.setdefault(n, [])
    return d


# Handles user input and coordinates.
# -----------------------------------------------------
def main():
    print("=====================================")
    print("          SHORTEST PATH FINDER       ")
    print("=====================================\n")

    # Load Graph File
    graph_file = input("Enter graph file name (e.g. map.txt): ").strip()
    try:
        G = load_graph_from_file(graph_file)
    except FileNotFoundError:
        print("Error: file not found.")
        return

    # Display summary of loaded graph
    display_graph_info(G)

    # User input for start and goal nodes
    start = input("\nEnter START node: ").strip()
    goal = input("Enter GOAL node: ").strip()

    # Choose algorithm to run
    print("\nChoose Algorithm:")
    print("1. Shortest Path (Dijkstra)")
    print("2. K Shortest Paths (Yen's Algorithm)")
    print("3. AI Compare & Recommend (runs Yen-Dijkstra, Yen-Bellman-Ford, A*, K*)")
    algo_choice = input("Enter choice (1/2/3): ").strip()


    # Option 1: Dijkstra Algorithm (Single Path)
    if algo_choice == "1":
        print("\nComputing shortest path using Dijkstra...\n")
        path, cost = dijkstra_shortest_path(G, start, goal)
        if path:
            print(f"✅ Shortest Path: {' -> '.join(path)}")
            print(f"✅ Total Cost: {cost}")
        else:
            print("❌ No path found.")

 
    # Option 2: Yen’s Algorithm (K Shortest Paths)
    elif algo_choice == "2":
        k = int(input("Enter number of paths (k): "))
        print(f"\nComputing top {k} shortest paths using Yen's Algorithm\n")
        G_dict = nx_to_dict(G)
        pairs = yen_dijkstra_adapter(G_dict, start, goal, k)
        if pairs:
            for i, (cost, path) in enumerate(pairs, start=1):
                print(f"{i}) Path: {' -> '.join(path)}   Cost: {cost}")
        else:
            print("❌ No paths found.")


    # Option 3: AI Comparison & Recommendation Suite
    elif algo_choice == "3":
        k = int(input("Enter number of paths (k) for K*/Yen: "))
        print("\nRunning AI comparison suite...\n")
        G_dict = nx_to_dict(G)

        # Check if negative edges exist
        neg = has_negative_edge(G_dict)

        # Run multiple algorithms and compare their performance
        recommendation, results = compare_and_recommend(
            G_dict, start, goal, k,
            yen_dijkstra_fn=yen_dijkstra_adapter,
            yen_bellmanford_fn=yen_bellmanford_adapter,
            astar_fn=astar_adapter,
            kstar_fn=kstar_adapter,
            heuristic=default_heuristic,

            # Disable unsafe algorithms if negative edges exist
            run_yen_dijkstra=not neg,
            run_yen_bellmanford=True,
            run_astar=not neg,
            run_kstar=not neg,
        )

        # Print results and recommended algorithm
        print_summary(recommendation, results)

    else:
        print("❌ Invalid choice.")

    # Optional AI Suggestion (Quick Recommendation)
    ai_decision = input("\nNeed AI travel suggestion? (y/n): ").lower()
    if ai_decision == "y":
        k = 3  # default to 3 paths
        G_dict = nx_to_dict(G)
        recommendation, _ = compare_and_recommend(
            G_dict, start, goal, k,
            yen_dijkstra_fn=yen_dijkstra_adapter,
            yen_bellmanford_fn=yen_bellmanford_adapter,
            astar_fn=astar_adapter,
            kstar_fn=kstar_adapter,
            heuristic=default_heuristic,
        )
        print(f"\n💡 AI Suggestion: {recommendation}")

    print("\nProgram finished. Goodbye!")


# Entry Point program starts here
if __name__ == "__main__":
    main()
