"""helper.py

Provides two shortest-path routines to be used by `algorithm.py`:
- dijkstra_shortest_path(graph, source, target) -> (path, cost)
- bellman_ford_shortest_path(graph, source, target) -> (path, cost)

Both accept a NetworkX graph (Graph or DiGraph) where edge weights are stored
under the 'weight' attribute. If an edge has no 'weight', weight defaults to 1.
If no path exists the functions return (None, float('inf')). Bellman-Ford
detects negative cycles reachable from the source; if a negative cycle that
affects the target is detected the function returns (None, float('-inf')).
"""

from typing import Tuple, List, Any
import heapq
import math


#Return the weight for edge as number (u, v).
#If edge has no weight ,weight 1.
def _edge_weight(graph, u, v):
    return graph[u][v].get('weight', 1)


#Dijkstra's algorithm (binary-heap implementation).
#Assumes non-negative edge weights. Returns (path, cost) or (None, inf).
def dijkstra_shortest_path(graph, source, target) -> Tuple[List[Any], float]:
    """Compute shortest path using Dijkstra's algorithm.

    Returns (path, cost). If no path exists returns (None, float('inf')).
    Works for non-negative weights.
    """
    if source not in graph or target not in graph:
        return None, float('inf')

    dist = {n: float('inf') for n in graph.nodes}
    prev = {}
    dist[source] = 0

    heap = [(0, source)]

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        if u == target:
            break

        for v in graph.neighbors(u):
            w = _edge_weight(graph, u, v)
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))

    if dist[target] == float('inf'):
        return None, float('inf')

    #Reconstruct path
    path = []
    cur = target
    while cur != source:
        path.append(cur)
        cur = prev.get(cur)
        if cur is None:
            #Shouldn't happen, but just in case
            return None, float('inf')
    path.append(source)
    path.reverse()
    return path, dist[target]


#Bellman-Ford algorithm: handle negative-weight edges and detect negative cycles that can affect target. returns (path, cost),
#(None, inf) if no path, or (None, -inf) if negative-cycle affects target.
def bellman_ford_shortest_path(graph, source, target) -> Tuple[List[Any], float]:
    """Compute shortest path using Bellman-Ford algorithm.

    Returns (path, cost). If no path exists returns (None, float('inf')).
    If a negative-weight cycle is detected that can affect the target, returns
    (None, float('-inf')).
    """
    if source not in graph or target not in graph:
        return None, float('inf')

    nodes = list(graph.nodes)
    dist = {n: float('inf') for n in nodes}
    prev = {n: None for n in nodes}
    dist[source] = 0

    edges = []
    for u, v, data in graph.edges(data=True):
        w = data.get('weight', 1)
        edges.append((u, v, w))

    #Relax edges |V|-1 times
    for _ in range(len(nodes) - 1):
        updated = False
        for u, v, w in edges:
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                prev[v] = u
                updated = True
        if not updated:
            break

    #Check for neg cycles reachable from source
    neg_cycle_nodes = set()
    for u, v, w in edges:
        if dist[u] != float('inf') and dist[u] + w < dist[v]:
            neg_cycle_nodes.add(v)

    if neg_cycle_nodes:
        #If any neg cycle node is reachable from source and can reach target then shortest path to target is undefine
        #Do small BFS/DFS from negative-cycle-affected nodes to see if target is reachable.
        stack = list(neg_cycle_nodes)
        visited = set(neg_cycle_nodes)
        while stack:
            n = stack.pop()
            if n == target:
                return None, float('-inf')
            for nbr in graph.neighbors(n):
                if nbr not in visited:
                    visited.add(nbr)
                    stack.append(nbr)

    if dist[target] == float('inf'):
        return None, float('inf')

    #Reconstruct path
    path = []
    cur = target
    while cur is not None:
        path.append(cur)
        if cur == source:
            break
        cur = prev.get(cur)
    if path[-1] != source:
        #No path
        return None, float('inf')
    path.reverse()
    return path, dist[target]


if __name__ == "__main__":
    #Self-test
    try:
        import networkx as nx
    except Exception:
        print("networkx not available")
        raise

    G = nx.DiGraph()
    G.add_weighted_edges_from([
        ('A', 'B', 1),
        ('B', 'C', 2),
        ('A', 'C', 4),
        ('C', 'D', 1),
    ])

    print("Dijkstra on simple positive-weights graph:")
    p, c = dijkstra_shortest_path(G, 'A', 'D')
    print("path:", p, "cost:", c)

    print("Bellman-Ford on same graph:")
    p, c = bellman_ford_shortest_path(G, 'A', 'D')
    print("path:", p, "cost:", c)

    #Negative weight example
    G2 = nx.DiGraph()
    G2.add_weighted_edges_from([
        ('S', 'A', 4),
        ('S', 'B', 5),
        ('A', 'B', -6),  #neg edge
        ('B', 'T', 3),
    ])
    print('\nBellman-Ford on graph with negative cycle affecting target?')
    p, c = bellman_ford_shortest_path(G2, 'S', 'T')
    print("path:", p, "cost:", c)
