
#Enables me to code a function related to the class before the class functions
from __future__ import annotations
#@dataclass automatically creates the init and print methods, and field() lets us safely set default values like empty lists.”
from dataclasses import dataclass, field

from typing import Any, Dict, List, Tuple, Optional, Callable
import time
import math


import networkx as nx

#set the format of the coords
_coords: Dict[Any, Tuple[float, float]] = {}

#Sets/Update the corrds with inputed data if no data it will use fallback heuristic (1 or 0)
def set_coords(d: Dict[Any, Tuple[float, float]]) -> None:
    global _coords
    _coords = dict(d)

#Make sure the graph is in NetworkX directed graph format (nx.Digraph)
def _ensure_nx(G):
    if isinstance(G, (nx.DiGraph, nx.Graph)):
        return G if isinstance(G, nx.DiGraph) else G.to_directed()

#If it’s already a NetworkX graph, it keeps or converts it to directed.
#If it’s just a normal dictionary, it builds a directed graph by adding nodes and weighted edges.
    Gnx = nx.DiGraph()
    for u, nbrs in G.items():
        Gnx.add_node(u)
        for v, w in nbrs:
            Gnx.add_edge(u, v, weight=float(w))
    return Gnx

#flips (path,cost) so that the format in this code is always consistent
def _flip_list_paths(paths_pc):
    """Convert algorithm.py output (path,cost) -> (cost,path)"""
    return [(float(c), list(p) if p else []) for (p, c) in paths_pc]

def _flip_single(path_cost):
    """Convert single result (path,cost) -> (cost,path) or None"""
    p, c = path_cost
    if p is None or c is None or c == float('inf'):
        return None
    return float(c), list(p)

#Removes duplicate path, invalid or looping path and sorting valid ones neatly (cost, length, alphabet order)
def _sanitize_k_paths(pairs):

    #Removes empty path and repeated path
    pairs = [(c, p) for (c, p) in pairs if p and len(p) == len(set(p))]
    #Tracks path that is already added and store unique path
    seen, uniq = set(), []
    #It loops through and converts it into a Tuple if its duplicated it will skip, else add to uniq
    for c, p in pairs:
        key = tuple(p)
        if key in seen:
            continue
        seen.add(key)
        uniq.append((c, p))
    #Sort by cost(lowest cost first), path length (shorter first) , alphabet order (based on node sequence)
    uniq.sort(key=lambda x: (x[0], len(x[1]), tuple(x[1])))
    return uniq



#This are adpater functions that calls algorithm from algorithm.py and using _ensure_nx(GD)
#and _sanitize_k_path to fomat the input and output so everything is consistent
def yen_dijkstra_adapter(Gd, s, t, k):
    from algorithm import yen_k_shortest_paths
    G = _ensure_nx(Gd)
    paths_pc = yen_k_shortest_paths(G, s, t, k, algorithm='dijkstra')  # [(path,cost)]
    flipped = [(float(c), list(p) if p is not None else []) for (p, c) in paths_pc]
    return _sanitize_k_paths(flipped)[:k]

def yen_bellmanford_adapter(Gd, s, t, k):
    from algorithm import yen_k_shortest_paths
    G = _ensure_nx(Gd)
    paths_pc = yen_k_shortest_paths(G, s, t, k, algorithm='bellman-ford')
    flipped = [(float(c), list(p) if p is not None else []) for (p, c) in paths_pc]
    return _sanitize_k_paths(flipped)[:k]


def astar_adapter(G, s, t, heuristic):
    from algorithm import a_star_search
    return _flip_single(a_star_search(_ensure_nx(G), s, t, heuristic))

def kstar_adapter(G, s, t, k, heuristic):
    from algorithm import k_star_algorithm
    return _flip_list_paths(k_star_algorithm(_ensure_nx(G), s, t, k, heuristic))
# ==== END WRAPPERS ====

# ---------------------------
# Types
# ---------------------------
Graph = Dict[Any, List[Tuple[Any, float]]]
Path = List[Any]

# Structure for both algor (Yen/djikstra, Yen/Bellman-ford) + advance algor (astart,kstar)
# yen_dijkstra_fn(graph, s, t, k) -> List[(cost: float, path: List[Any])]
# yen_bellmanford_fn(graph, s, t, k) -> List[(cost: float, path: List[Any])]
# astar_fn(graph, s, t, heuristic) -> Optional[(cost: float, path: List[Any])]
# kstar_fn(graph, s, t, k, heuristic) -> List[(cost: float, path: List[Any])]

# ---------------------------
# Utilities
# ---------------------------

#Check for negative edge weight (aka bellman-ford)
def has_negative_edge(graph: Graph) -> bool:
    # Accept dict {u:[(v,w),...]} or a NetworkX graph
    try:
        # dict-style
        for _, nbrs in graph.items():
            for _, w in nbrs:
                if w < 0:
                    return True
        return False
    #if its not a dict assume its a NetworkX graph
    except AttributeError:
        for u, v, data in graph.edges(data=True):
            w = data.get('weight', 1)
            if w < 0:
                return True
        return False

#Caculates the path cost
def path_cost(graph: Graph, path: Path) -> float:
    # no movement so cost = 0
    if not path or len(path) == 1:
        return 0.0
    # if the graph has a .get() method means its a dictionary
    if hasattr(graph, "get"):
        #cost counter starts at 0
        total = 0.0
        for u, v in zip(path, path[1:]):
            found = False
            for vv, w in graph.get(u, []):
                if vv == v:
                    total += w
                    found = True
                    break
            if not found:
                raise ValueError(f"Edge ({u}->{v}) missing when computing cost.")
        return total
    # Fallback: networkx-style if the graph is not a dictionary
    total = 0.0
    for u, v in zip(path, path[1:]):
        if not graph.has_edge(u, v):
            raise ValueError(f"Edge ({u}->{v}) missing when computing cost.")
        total += graph[u][v].get('weight', 1)
    return total

#check if paths has cycle. if length = equal means no duplicate if not means there is a cycle
def is_simple(path: Path) -> bool:
    return len(path) == len(set(path))

def default_heuristic(u: Any, v: Any) -> float:
    """
    If both nodes have coordinates -> return Manhattan distance.
    Otherwise -> safe fallback used in your project (0 for goal, else 1).
    """
    #check if both nodes has coords stored in _coords
    if u in _coords and v in _coords:
        #retrieve coords from __coords
        (x1, y1) = _coords[u]
        (x2, y2) = _coords[v]
        #calculate manhattan distance between two points (aka how far apart if can only move horizontal or vertical)
        return abs(x1 - x2) + abs(y1 - y2)
    # 0 if it reaches the goal, 1 means its a "guess"
    return 0.0 if u == v else 1.0

def diversity_score(paths: List[Tuple[float, Path]]) -> float:
    """Crude diversity: unique edge-structures + a small bonus for unique lengths."""
    # if empty list = no diversity = return 0
    if not paths:
        return 0.0
    #converts them example ( [A,B,C] -> [A,B], [B,C] )
    #add them into a set to remove duplicates means only unique paths are left
    edge_structs = set(tuple(zip(p, p[1:])) for _, p in paths)
    #Calculates unique path length
    uniq_lengths = len(set(len(p) for _, p in paths))
    # diversity = (number of unique edge struct) + (0.25 x number of unique path length)
    return float(len(edge_structs)) + 0.25 * float(uniq_lengths)

#returns in miliseconds for runtime used later for reccomendation logic
def now_ms() -> float:
    return time.perf_counter() * 1000.0

# ---------------------------
# Result model
# ---------------------------
@dataclass
class AlgoResult:
    name: str
    ok: bool
    runtime_ms: float = 0.0
    k: int = 0
    paths: List[Tuple[float, Path]] = field(default_factory=list)
    notes: str = ""
    diversity: float = 0.0

    # summary of algorithm performance
    def brief(self) -> str:
        #if algorithm failed
        if not self.ok:
            return f"{self.name}: unavailable or failed. {self.notes}"
        #building the summary string
        s = f"{self.name}: {len(self.paths)} path(s) in {self.runtime_ms:.2f} ms"
        #shows the best cost (aka lowest)
        if self.paths:
            s += f", best_cost={self.paths[0][0]}"
        #Adds diversity score if applicable
        if self.diversity:
            s += f", diversity={self.diversity:.2f}"
        if self.notes:
            s += f" | {self.notes}"
        return s

# ---------------------------
# Core API (call this from main.py)
# ---------------------------
#run all algorithm and compare and recomend
#This function sets up and prepares everything for testing the algorithms.
#It takes the graph, start and end nodes, and the algorithm functions (which are empty for now but will be linked later).
#Then it checks for negative edges, sets the heuristic, and gets ready to store all results.
def compare_and_recommend(
    graph: Graph,
    s: Any,
    t: Any,
    k: int,


    # insert algor.py functions here in none (for linkning)
    #it will be called from main.py
    yen_dijkstra_fn: Optional[Callable[[Graph, Any, Any, int], List[Tuple[float, Path]]]] = None,
    yen_bellmanford_fn: Optional[Callable[[Graph, Any, Any, int], List[Tuple[float, Path]]]] = None,
    astar_fn: Optional[Callable[[Graph, Any, Any, Callable[[Any, Any], float]], Optional[Tuple[float, Path]]]] = None,
    kstar_fn: Optional[Callable[[Graph, Any, Any, int, Callable[[Any, Any], float]], List[Tuple[float, Path]]]] = None,
    heuristic: Optional[Callable[[Any, Any], float]] = None,
    run_yen_dijkstra: bool = True,
    run_yen_bellmanford: bool = True,
    run_astar: bool = True,
    run_kstar: bool = True,
) -> Tuple[str, List[AlgoResult]]:

    # Executes whatever algorithms are provided, compares runtime & basic quality,
    # and returns a recommendation plus per-algorithm results.

    heuristic = heuristic or default_heuristic
    neg_edges = has_negative_edge(graph)
    results: List[AlgoResult] = []

    # ---- Yen(Dijkstra)
    # it runs Yen (Dijkstra) and check for runtime, results, cost etc
    if run_yen_dijkstra:
        if yen_dijkstra_fn is None:
            results.append(AlgoResult("Yen(Dijkstra)", ok=False, notes="function not provided"))
        elif neg_edges:
            results.append(AlgoResult("Yen(Dijkstra)", ok=False, notes="negative edges present; Dijkstra invalid"))
        else:
            t0 = now_ms()
            try:
                yd = yen_dijkstra_fn(graph, s, t, k) or []
                t1 = now_ms()
                results.append(AlgoResult(
                    name="Yen(Dijkstra)",
                    ok=len(yd) > 0,
                    runtime_ms=t1 - t0,
                    k=k,
                    paths=yd,
                    notes="loopless; requires non-negative weights",
                    diversity=diversity_score(yd),
                ))
            except Exception as e:
                results.append(AlgoResult("Yen(Dijkstra)", ok=False, notes=f"Exception: {e!r}"))

    # ---- Yen(Bellman-Ford)
    # it runs Yen (Bellman-Ford) and check for runtime, results, cost etc
    if run_yen_bellmanford:
        if yen_bellmanford_fn is None:
            results.append(AlgoResult("Yen(Bellman-Ford)", ok=False, notes="function not provided"))
        else:
            t0 = now_ms()
            try:
                yb = yen_bellmanford_fn(graph, s, t, k) or []
                t1 = now_ms()
                results.append(AlgoResult(
                    name="Yen(Bellman-Ford)",
                    ok=len(yb) > 0,
                    runtime_ms=t1 - t0,
                    k=k,
                    paths=yb,
                    notes="loopless; supports negative edges; should skip routes involving reachable negative cycles",
                    diversity=diversity_score(yb),
                ))
            except Exception as e:
                results.append(AlgoResult("Yen(Bellman-Ford)", ok=False, notes=f"Exception: {e!r}"))

    # ---- A* (single best)
    # it runs A* (single best) and check for runtime, results, cost etc
    if run_astar:
        if astar_fn is None:
            results.append(AlgoResult("A*", ok=False, notes="function not provided"))
        else:
            t0 = now_ms()
            try:
                one = astar_fn(graph, s, t, heuristic)
                t1 = now_ms()
                if one is None:
                    results.append(AlgoResult("A*", ok=False, runtime_ms=t1 - t0, notes="no path"))
                else:
                    cost, path = one
                    note = "simple path" if is_simple(path) else "may contain cycles"
                    results.append(AlgoResult(
                        name="A*",
                        ok=True,
                        runtime_ms=t1 - t0,
                        k=1,
                        paths=[(cost, path)],
                        notes=note,
                        diversity=1.0,
                    ))
            except Exception as e:
                results.append(AlgoResult("A*", ok=False, notes=f"Exception: {e!r}"))

    # ---- K* (heuristic K-best)
    # it runs K* (heuristic K-best) and check for runtime, results, cost etc
    if run_kstar:
        if kstar_fn is None:
            results.append(AlgoResult("K*", ok=False, notes="function not provided"))
        else:
            t0 = now_ms()
            try:
                kp = kstar_fn(graph, s, t, k, heuristic) or []
                t1 = now_ms()
                # Recompute costs from graph to validate
                checked = []
                for c, p in kp:
                    try:
                        c2 = path_cost(graph, p)
                        checked.append((c2, p))
                    except Exception:
                        checked.append((c, p))
                results.append(AlgoResult(
                    name="K*",
                    ok=len(checked) > 0,
                    runtime_ms=t1 - t0,
                    k=k,
                    paths=checked,
                    notes="heuristic K-best",
                    diversity=diversity_score(checked),
                ))
            except Exception as e:
                results.append(AlgoResult("K*", ok=False, notes=f"Exception: {e!r}"))

    # ---------------------------
    # Recommendation logic
    # ---------------------------
    res_map = {r.name: r for r in results}

    def pick_best(a: AlgoResult, b: AlgoResult) -> AlgoResult:
        if not a.ok and b.ok:
            return b
        if a.ok and not b.ok:
            return a
        if not a.ok and not b.ok:
            return a
        # both ok: prefer faster; then higher diversity; then more paths
        if abs(a.runtime_ms - b.runtime_ms) > 1e-6:
            return a if a.runtime_ms < b.runtime_ms else b
        if abs(a.diversity - b.diversity) > 1e-6:
            return a if a.diversity > b.diversity else b
        if a.k != b.k:
            return a if a.k > b.k else b
        return a
    #if both algorithm failed
    recommendation = "No algorithm produced a valid result."
    rationale = ""

    if neg_edges:
        cands = [res_map.get("Yen(Bellman-Ford)"), res_map.get("K*")]
        cands = [x for x in cands if x is not None]
        if cands:
            best = cands[0]
            for r in cands[1:]:
                best = pick_best(best, r)
            if best.ok:
                recommendation = f"Use {best.name}"
                rationale = "Negative edges detected; Bellman–Ford variant is robust."
            else:
                recommendation = "No valid paths found with negative edges."
    else:
        cands = [res_map.get("Yen(Dijkstra)"), res_map.get("K*")]
        cands = [x for x in cands if x is not None]
        if not cands:
            cands = [res_map.get("A*"), res_map.get("Yen(Bellman-Ford)")]
            cands = [x for x in cands if x is not None]
        if cands:
            best = cands[0]
            for r in cands[1:]:
                best = pick_best(best, r)
            if best.ok:
                recommendation = f"Use {best.name}"
                rationale = "Non-negative weights; chose based on runtime, diversity, and number of valid paths."

    # ---------------------------
    # Accuracy vs baseline metric (cost gap)
    # ---------------------------
    #gets best cost (aka lowest)
    def _best_cost(r: AlgoResult) -> Optional[float]:
        return r.paths[0][0] if (r.ok and r.paths) else None
    #Use Dijkstra first if not then use bellman-ford
    yen_res = res_map.get("Yen(Dijkstra)") or res_map.get("Yen(Bellman-Ford)")
    #Get result from K*
    kstar_res = res_map.get("K*")
    #Compare ONLY if both ran successfully
    if yen_res and kstar_res and yen_res.ok and kstar_res.ok:
        #Get best path cost (aka lowest)
        base = _best_cost(yen_res)
        kbest = _best_cost(kstar_res)
        if base is not None and kbest is not None:
            gap = kbest - base
            results.append(AlgoResult(name="Accuracy", ok=True, notes=f"K* cost gap vs baseline: {gap:+.4f}"))

    # Human-friendly appendix (optional)
    if rationale:
        results.append(AlgoResult(name="Reason", ok=True, notes=rationale))

    return recommendation, results

# ---------------------------
# Output
# ---------------------------
def print_summary(recommendation: str, results: List[AlgoResult]) -> None:
    print("=== Algorithm Comparison Summary ===")
    for r in results:
        if r.name in ("Reason", "Accuracy"):
            continue
        print(" - " + r.brief())
        if r.ok and r.paths:
            for i, (c, p) in enumerate(r.paths[:3], 1):
                print(f"    {i}. cost={c}, path={p}")

    # Show rationale and accuracy (if present)
    reason = next((r for r in results if r.name == "Reason"), None)
    if reason:
        print(f"\n{reason.notes}")
    acc = next((r for r in results if r.name == "Accuracy"), None)
    if acc:
        print(f"{acc.notes}")

    # Build a direct two-way comparison line if we have at least 2 ideal results
    ok = [r for r in results if r.ok and r.paths and r.name not in ("Reason", "Accuracy")]
    if len(ok) >= 2:
        # Take first two for a clean, human-friendly comparison
        a, b = ok[0], ok[1]
        cost_a, path_a = a.paths[0]
        cost_b, path_b = b.paths[0]

        # Decide which of (a, b) matches the recommendation string
        winner = a if a.name in recommendation else b
        w_cost, w_path = (cost_a, path_a) if winner is a else (cost_b, path_b)
        l_cost, l_path = (cost_b, path_b) if winner is a else (cost_a, path_a)

        print(
            f"\nRecommendation: {recommendation} — best cost {w_cost} vs {l_cost}, "
            f"path length {len(w_path)} vs {len(l_path)}; {recommendation} has the lower overall values."
        )
    else:
        print(f"\nRecommendation: {recommendation}")


"""
SAMPLE OUTPUT:
=== Algorithm Comparison Summary ===
 - Yen(Dijkstra): 3 path(s) in 2.08 ms, best_cost=3.0, diversity=2.25 | loopless; requires non-negative weights
    1. cost=3.0, path=['A', 'C', 'D']
    2. cost=3.0, path=['A', 'B', 'C', 'D']
    3. cost=4.0, path=['A', 'B', 'D']

 - K*: 3 path(s) in 1.85 ms, best_cost=3.0, diversity=2.25 | heuristic K-best;
    1. cost=3.0, path=['A', 'C', 'D']
    2. cost=3.0, path=['A', 'B', 'C', 'D']
    3. cost=4.0, path=['A', 'B', 'D']

Non-negative weights; chose based on runtime, diversity, and number of valid paths.
K* cost gap vs baseline: +0.0000

Recommendation: Use K* — best cost 3.0 vs 3.0, path length 3 vs 3; Use K* has the lower overall values.


"""
