"""Exact current-access completion on finite undirected physical multigraphs.

An edge has a reference direction, but a physical stationary current may have
either sign.  Conservation is B j = 0, with -1 at the tail and +1 at the head.
All arithmetic is rational.  Max-flow/min-cut and spanning-tree facts are
established mathematics; this module supplies their scoped access interpretation.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from fractions import Fraction
import json
from pathlib import Path


def rat(x):
    if isinstance(x, bool) or isinstance(x, float):
        raise TypeError("Use exact integers, rational strings, or Fraction values")
    return Fraction(x)


@dataclass(frozen=True)
class Edge:
    key: str
    tail: str
    head: str


@dataclass(frozen=True)
class Graph:
    vertices: tuple[str, ...]
    edges: tuple[Edge, ...]

    def __post_init__(self):
        if len(set(self.vertices)) != len(self.vertices) or not self.vertices:
            raise ValueError("Vertices must be distinct and nonempty")
        if len({e.key for e in self.edges}) != len(self.edges):
            raise ValueError("Edge identifiers must be distinct")
        if any(e.tail == e.head or e.tail not in self.vertices or
               e.head not in self.vertices for e in self.edges):
            raise ValueError("Every edge joins two distinct declared vertices")

    def edge(self, key):
        for e in self.edges:
            if e.key == key:
                return e
        raise KeyError(key)

    def bundle(self, edge):
        e = self.edge(edge) if isinstance(edge, str) else edge
        return tuple(sorted((e.tail, e.head), key=self.vertices.index))

    def bundles(self):
        answer = {}
        for e in self.edges:
            answer.setdefault(self.bundle(e), []).append(e)
        return answer


def _canonical_bundles(graph, known):
    answer = set()
    for pair in known:
        if len(pair) != 2 or pair[0] == pair[1] or any(v not in graph.vertices for v in pair):
            raise ValueError("Invalid aggregate bundle")
        answer.add(tuple(sorted(pair, key=graph.vertices.index)))
    if not answer.issubset(graph.bundles()):
        raise ValueError("Aggregate bundle not present in graph")
    return answer


def _divergence(graph, flow):
    result = {v: Fraction(0) for v in graph.vertices}
    for e in graph.edges:
        value = rat(flow.get(e.key, 0))
        result[e.tail] -= value
        result[e.head] += value
    return result


def _reachable(graph, edges, source):
    reached = {source}
    queue = deque([source])
    while queue:
        vertex = queue.popleft()
        for e in edges:
            other = e.head if e.tail == vertex else e.tail if e.head == vertex else None
            if other is not None and other not in reached:
                reached.add(other)
                queue.append(other)
    return reached


def _components(graph, edges):
    unseen = set(graph.vertices)
    count = 0
    while unseen:
        reached = _reachable(graph, edges, next(v for v in graph.vertices if v in unseen))
        unseen -= reached
        count += 1
    return count


def target_cut(graph, target, capacities):
    """Minimum cut avoiding target, with unspecified edges unavailable to a cut.

    This is (1) minimum additive acquisition cost when capacities are costs,
    or (2) exact deterministic minimax error when capacities are sensor errors.
    Target must not be in capacities.  A finite result includes a cut and a
    matching circulation, independently checkable without rerunning max-flow.
    """
    e0 = graph.edge(target)
    capacities = {str(k): rat(v) for k, v in capacities.items()}
    if target in capacities or not set(capacities).issubset({e.key for e in graph.edges}):
        raise ValueError("Capacities must name nontarget edges")
    if any(v < 0 for v in capacities.values()):
        raise ValueError("Capacities must be nonnegative")
    others = tuple(e for e in graph.edges if e.key != target)
    # Unboundedness means a circulation survives with zero on every sensor.
    unsensed = tuple(e for e in others if e.key not in capacities)
    source, sink = e0.head, e0.tail
    reached = {source}
    parents = {}
    queue = deque([source])
    while queue and sink not in reached:
        v = queue.popleft()
        for e in unsensed:
            w, sign = (e.head, 1) if e.tail == v else (e.tail, -1) if e.head == v else (None, 0)
            if w is not None and w not in reached:
                parents[w] = (v, e.key, sign)
                reached.add(w)
                queue.append(w)
    if sink in reached:
        flow = {e.key: Fraction(0) for e in graph.edges}
        flow[target] = Fraction(1)
        v = sink
        while v != source:
            prev, key, sign = parents[v]
            flow[key] += sign
            v = prev
        return {"status": "unidentifiable", "ambiguity": {k: str(v) for k, v in flow.items()}}
    large = sum(capacities.values(), Fraction(0)) + 1
    cap = {e.key: capacities.get(e.key, large) for e in others}
    flow = {e.key: Fraction(0) for e in graph.edges}
    total = Fraction(0)
    while True:
        reached = {source}
        parents = {}
        queue = deque([source])
        while queue and sink not in reached:
            v = queue.popleft()
            for e in others:
                w, sign = (e.head, 1) if e.tail == v else (e.tail, -1) if e.head == v else (None, 0)
                residual = cap[e.key] - sign * flow[e.key]
                if w is not None and w not in reached and residual > 0:
                    parents[w] = (v, e.key, sign, residual)
                    reached.add(w)
                    queue.append(w)
        if sink not in reached:
            break
        amount = large
        v = sink
        while v != source:
            prev, key, sign, residual = parents[v]
            amount = min(amount, residual)
            v = prev
        v = sink
        while v != source:
            prev, key, sign, _ = parents[v]
            flow[key] += sign * amount
            v = prev
        total += amount
    flow[target] = total
    cut = set(graph.vertices) - reached  # includes target tail, excludes head
    coefficients = {}
    for e in others:
        if (e.tail in cut) != (e.head in cut):
            coefficients[e.key] = -1 if e.tail in cut else 1
    result = {
        "status": "identified", "value": str(total),
        "cut": [v for v in graph.vertices if v in cut],
        "coefficients": coefficients,
        "circulation": {k: str(v) for k, v in flow.items()},
    }
    if not verify_target_cut(graph, target, capacities, result):
        raise AssertionError("Internal max-flow certificate failed")
    return result


def verify_target_cut(graph, target, capacities, certificate):
    """Check cut/flow dual bounds or a surviving exact ambiguity; no optimization."""
    try:
        e0 = graph.edge(target)
        capacities = {k: rat(v) for k, v in capacities.items()}
        edgekeys = {e.key for e in graph.edges}
        if target in capacities or not set(capacities).issubset(edgekeys) or any(c < 0 for c in capacities.values()):
            return False
        if certificate["status"] == "unidentifiable":
            f = {k: rat(v) for k, v in certificate["ambiguity"].items()}
            return (set(f) == edgekeys and f[target] == 1 and
                    all(f[k] == 0 for k in capacities) and
                    all(v == 0 for v in _divergence(graph, f).values()))
        if certificate["status"] != "identified":
            return False
        cut = set(certificate["cut"])
        if not cut.issubset(graph.vertices) or e0.tail not in cut or e0.head in cut:
            return False
        coefficients = {}
        for e in graph.edges:
            if e.key != target and ((e.tail in cut) != (e.head in cut)):
                if e.key not in capacities:
                    return False
                coefficients[e.key] = -1 if e.tail in cut else 1
        if certificate["coefficients"] != coefficients:
            return False
        value = rat(certificate["value"])
        if value != sum((capacities[k] for k in coefficients), Fraction(0)):
            return False
        f = {k: rat(v) for k, v in certificate["circulation"].items()}
        return (set(f) == edgekeys and f[target] == value and
                all(abs(f[k]) <= c for k, c in capacities.items()) and
                all(v == 0 for v in _divergence(graph, f).values()))
    except (ValueError, TypeError, KeyError, ZeroDivisionError):
        return False


def mixed_observability(graph, sensed=(), known_aggregates=()):
    """Exact ambiguity classification with channel and aggregate current access."""
    sensed = set(sensed)
    if not sensed.issubset({e.key for e in graph.edges}):
        raise ValueError("Unknown sensed edge")
    known = _canonical_bundles(graph, known_aggregates)
    bundles = graph.bundles()
    residual = {b: [e for e in es if e.key not in sensed] for b, es in bundles.items()}
    aggregate_graph = [Edge(str(i), *b) for i, (b, es) in enumerate(residual.items()) if es and b not in known]
    cycle_rank = len(aggregate_graph) - len(graph.vertices) + _components(graph, aggregate_graph)
    internal = sum(max(len(es) - 1, 0) for es in residual.values())
    identified = []
    for e in graph.edges:
        b = graph.bundle(e)
        if e.key in sensed:
            identified.append(e.key)
        elif len(residual[b]) == 1:
            if b in known:
                identified.append(e.key)
            else:
                without = [f for f in aggregate_graph if (f.tail, f.head) != b]
                if b[1] not in _reachable(graph, without, b[0]):
                    identified.append(e.key)
    return {
        "ambiguity_dimension": internal + cycle_rank,
        "internal_channel_dimension": internal,
        "network_cycle_dimension": cycle_rank,
        "identifiable_channels": identified,
    }


def target_completion(graph, target, sensor_costs, known_aggregates=()):
    """Minimum extra exact individual-current sensors with exact bundle totals.

    All nontarget channels must have a supplied nonnegative acquisition cost.
    Exact aggregate readings are already available and have no added cost here.
    """
    graph.edge(target)
    costs = {k: rat(v) for k, v in sensor_costs.items()}
    needed = {e.key for e in graph.edges if e.key != target}
    if set(costs) != needed or any(v < 0 for v in costs.values()):
        raise ValueError("Supply one nonnegative cost for each nontarget channel")
    known = _canonical_bundles(graph, known_aggregates)
    bundles = graph.bundles()
    b0 = graph.bundle(target)
    forced = [e.key for e in bundles[b0] if e.key != target]
    selected = list(forced)
    cut_certificate = None
    if b0 not in known:
        residual = []
        cap = {}
        mapping = {}
        for i, (b, es) in enumerate(bundles.items()):
            if b in known:
                continue
            key = "target" if b == b0 else "bundle_" + str(i)
            residual.append(Edge(key, *b))
            mapping[key] = [e.key for e in es]
            if b != b0:
                cap[key] = sum((costs[e.key] for e in es), Fraction(0))
        cut_certificate = target_cut(Graph(graph.vertices, tuple(residual)), "target", cap)
        for key in cut_certificate["coefficients"]:
            selected.extend(mapping[key])
    return {
        "sensors": selected,
        "cost": str(sum((costs[k] for k in selected), Fraction(0))),
        "forced_parallel_channels": forced,
        "bundle_cut_certificate": cut_certificate,
    }


def all_target_completion(graph, sensor_costs, known_aggregates=()):
    """Minimum exact sensor cost to recover every channel current.

    Retain one maximum-cost unsensed channel in each known-total bundle, and
    maximum-cost representatives of a maximum-weight forest of unknown bundles.
    """
    costs = {k: rat(v) for k, v in sensor_costs.items()}
    if set(costs) != {e.key for e in graph.edges} or any(v < 0 for v in costs.values()):
        raise ValueError("Supply one nonnegative cost for every channel")
    known = _canonical_bundles(graph, known_aggregates)
    parent = {v: v for v in graph.vertices}

    def find(v):
        while parent[v] != v:
            v = parent[v]
        return v

    kept = []
    candidates = []
    for b, es in graph.bundles().items():
        representative = max(es, key=lambda e: (costs[e.key], e.key))
        if b in known:
            kept.append(representative.key)
        else:
            candidates.append((costs[representative.key], representative.key, b))
    forest = []
    for value, key, b in sorted(candidates, reverse=True):
        left, right = find(b[0]), find(b[1])
        if left != right:
            parent[left] = right
            kept.append(key)
            forest.append(key)
    selected = [e.key for e in graph.edges if e.key not in set(kept)]
    return {
        "sensors": selected, "unsensed_channels": kept,
        "unknown_bundle_forest": forest,
        "cost": str(sum((costs[k] for k in selected), Fraction(0))),
    }


def realize_currents(graph, currents, stationary=None, traffic=None):
    """Positive bidirectional channel rates realizing a conserved rational j.

    Traffic t_e is half the total stationary jump traffic; no global traffic,
    rate, energy or fuel ceiling is assumed.  Supply it to impose a checked cap.
    """
    j = {k: rat(v) for k, v in currents.items()}
    if set(j) != {e.key for e in graph.edges} or any(_divergence(graph, j).values()):
        raise ValueError("Current must be supplied on every edge and conserved")
    pi = ({v: Fraction(1, len(graph.vertices)) for v in graph.vertices}
          if stationary is None else {v: rat(x) for v, x in stationary.items()})
    if set(pi) != set(graph.vertices) or sum(pi.values()) != 1 or any(x <= 0 for x in pi.values()):
        raise ValueError("Stationary distribution must be strictly positive and normalized")
    t = ({k: abs(x) + 1 for k, x in j.items()} if traffic is None
         else {k: rat(x) for k, x in traffic.items()})
    if set(t) != set(j) or any(t[k] <= abs(j[k]) / 2 for k in j):
        raise ValueError("Each traffic must strictly exceed half absolute current")
    return {
        e.key: {"forward": str((t[e.key] + j[e.key] / 2) / pi[e.tail]),
                "reverse": str((t[e.key] - j[e.key] / 2) / pi[e.head])}
        for e in graph.edges
    }


def main():
    graph = Graph(("A", "B", "C", "D"), (
        Edge("target", "A", "B"), Edge("AB_hidden", "A", "B"),
        Edge("AC", "A", "C"), Edge("BC", "B", "C"),
        Edge("CD", "C", "D"), Edge("DA", "D", "A"),
    ))
    costs = {e.key: Fraction(i + 1) for i, e in enumerate(graph.edges)}
    completion = target_completion(graph, "target", {k: v for k, v in costs.items() if k != "target"})
    errors = {"AB_hidden": Fraction(1, 100), "AC": Fraction(2, 100), "DA": Fraction(3, 100)}
    radius = target_cut(graph, "target", errors)
    result = {
        "model": "stationary signed physical-channel circulation; declared exact or box access",
        "single_target_completion": completion,
        "all_target_completion": all_target_completion(graph, costs),
        "all_aggregates_known": mixed_observability(graph, known_aggregates=graph.bundles()),
        "one_parallel_channel_revealed": mixed_observability(graph, sensed=("AB_hidden",), known_aggregates=graph.bundles()),
        "robust_target_radius": radius,
        "certificate_valid": verify_target_cut(graph, "target", errors, radius),
        "physical_realization_of_dual": realize_currents(graph, radius["circulation"]),
        "claims_of_novelty": "No priority claim for standard flow/cut/tree or linear-identification facts",
    }
    path = Path(__file__).resolve().parents[1] / "results" / "graph_access.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"result": str(path), "certificate_valid": result["certificate_valid"]}))


if __name__ == "__main__":
    main()
