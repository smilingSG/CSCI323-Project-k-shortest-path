""" Yen Algorithm """

from helper import dijkstra_shortest_path, bellman_ford_shortest_path
import heapq
import networkx as nx


def yen_k_shortest_paths(graph, source, target, k, algorithm='dijkstra'):
    # Choose which algorithm to use for shortest path calculation
    if algorithm == 'dijkstra':
        shortest_path_func = dijkstra_shortest_path
    elif algorithm == 'bellman-ford':
        shortest_path_func = bellman_ford_shortest_path
    else:
        raise ValueError("Algorithm must be 'dijkstra' or 'bellman-ford'")

    # start & end are the same (no movement)
    if source == target:
        return [([source], 0)]

    # stores the final K shortest paths found so far
    paths = []

    # priority queue that stores temporary path candidates
    potential_paths = []

    # ---------------------------------------------------------------
    # Find the very first shortest path
    # ---------------------------------------------------------------
    first_path, first_cost = shortest_path_func(graph, source, target)

    # If no path exists, it stops
    if not first_path:
        return []

    # Add 1st path into 'paths' list
    paths.append((first_path, first_cost))

    # ---------------------------------------------------------------
    # Loop to find the next shortest paths
    # ---------------------------------------------------------------
    for _ in range(1, k):
        # Use most recent path as base for new paths
        last_path = paths[-1][0]

        # Go through each node in that path (except the last one)
        for j in range(len(last_path) - 1):
            spur_node = last_path[j]       # node to branch off from
            root_path = last_path[:j + 1]  # path from source to spur_node

            # Make temporary copy of the graph (so removals don't affect the real graph)
            g_copy = graph.copy()

            # -------------------------------------------------------
            # Remove edges that would recreate previous paths
            # -------------------------------------------------------
            for path, _ in paths:
                if path[:j + 1] == root_path and len(path) > j + 1:
                    u, v = path[j], path[j + 1]
                    try:
                        g_copy.remove_edge(u, v)
                    except Exception:
                        # Edge might already be removed in a prior spur attempt if the 
                        # paths share the same prefix, ignore if this is the case
                        pass

            # -------------------------------------------------------
            # Remove nodes in root path (except spur node)
            # -------------------------------------------------------
            for node in root_path[:-1]:
                try:
                    g_copy.remove_node(node)
                except Exception:
                    # Node might already be removed; ignore
                    pass

            # -------------------------------------------------------
            # Find new path (spur path) from spur node to target
            # -------------------------------------------------------
            spur_path, spur_cost = shortest_path_func(g_copy, spur_node, target)

            # If new spur path found, join with root
            if spur_path:
                total_path = root_path[:-1] + spur_path
                total_cost = calc_path_cost(graph, total_path)
                heapq.heappush(potential_paths, (total_cost, total_path))

        # -------------------------------------------------------
        # After checking all spur nodes: pick shortest candidate
        # -------------------------------------------------------
        if potential_paths:
            total_weight, new_path = heapq.heappop(potential_paths)
            paths.append((new_path, total_weight))
        else:
            # No more new paths found
            break

    return paths


""" K* Algorithm + A* search"""

import heapq
import networkx as nx

# ---------------------------
# A* search: returns (path, cost)
# ---------------------------
def a_star_search(graph, start, goal, heuristic):
    # g_score: best-known cost from start to node
    g_score = {n: float('inf') for n in graph.nodes}
    g_score[start] = 0

    # f_score = g + h
    f_score = {n: float('inf') for n in graph.nodes}
    f_score[start] = heuristic(start, goal)

    # open set: heap of (f_score, node)
    open_heap = []
    heapq.heappush(open_heap, (f_score[start], start))

    came_from = {}  # to reconstruct path

    while open_heap:
        _, current = heapq.heappop(open_heap)

        # If goal reached, rebuild and return
        if current == goal:
            path = _reconstruct_path(came_from, current)
            return path, g_score[goal]

        # Explore neighbor nodes
        for nbr in graph.neighbors(current):
            weight = graph[current][nbr].get('weight', 1)
            tentative_g = g_score[current] + weight

            if tentative_g < g_score[nbr]:
                came_from[nbr] = current
                g_score[nbr] = tentative_g
                f = tentative_g + heuristic(nbr, goal)
                f_score[nbr] = f
                heapq.heappush(open_heap, (f, nbr))

    # No path found
    return None, float('inf')


# ---------------------------------------------------
# Reconstructs full path (trace backwards from goal)
# ---------------------------------------------------
def _reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


# ---------------------------
# Compute cost of a path
# ---------------------------
def calc_path_cost(graph, path):
    return sum(graph[u][v].get('weight', 1) for u, v in zip(path, path[1:]))


# ----------------------------------------
# K* algorithm (Run A* Search repeatedly)
# ----------------------------------------
def k_star_algorithm(graph, source, target, k, heuristic):
    # 1st shortest path
    first_path, first_cost = a_star_search(graph, source, target, heuristic)
    if not first_path:
        return []

    results = [(first_path, first_cost)]
    candidates = []  # min-heap of (cost, path)

    # Loop to find the next shortest paths
    for _ in range(1, k):
        # Use most recent path as base for new paths
        last_path = results[-1][0]

        for j in range(len(last_path) - 1):
            spur_node = last_path[j]         # node that branch off
            root_path = last_path[: j + 1]   # path from source to spur_node

            # Make temporary copy of the graph
            # (to remove edges/nodes without affecting the real graph)
            g_copy = graph.copy()

            # Remove edges that would recreate previous paths.
            for p, _ in results:
                if len(p) > j and p[: j + 1] == root_path:
                    u, v = p[j], p[j + 1]
                    if g_copy.has_edge(u, v):
                        g_copy.remove_edge(u, v)

            # Remove nodes in root_path except spur_node to avoid loops
            for node in root_path[:-1]:
                if g_copy.has_node(node):
                    g_copy.remove_node(node)

            # Run A* from spur_node to target on modified graph
            spur_path, spur_cost = a_star_search(g_copy, spur_node, target, heuristic)

            if spur_path:
                # Combine root_path (without duplicate spur node) + spur_path
                total_path = root_path[:-1] + spur_path
                total_cost = calc_path_cost(graph, total_path)

                # Avoid duplicates in candidate heaps
                heapq.heappush(candidates, (total_cost, total_path))

        # Choose next shortest path from candidates
        while candidates:
            cost, path = heapq.heappop(candidates)
            # Checks path is not in results
            if not any(path == existing_path for existing_path, _ in results):
                results.append((path, cost))
                break
        else:
            # No more paths
            break

    return results