"""Independent exact-rank, exhaustive-subset and cut checks for graph access."""
import copy
from fractions import Fraction
from itertools import combinations
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from graph_access import (Edge, Graph, all_target_completion, mixed_observability,
                          realize_currents, target_completion, target_cut,
                          verify_target_cut)


def rank(rows, columns):
    """Independent Gaussian elimination, not production graph routines."""
    a = [[Fraction(x) for x in row] for row in rows]
    pivot = 0
    for col in range(columns):
        found = next((i for i in range(pivot, len(a)) if a[i][col]), None)
        if found is None:
            continue
        a[pivot], a[found] = a[found], a[pivot]
        scale = a[pivot][col]
        a[pivot] = [x / scale for x in a[pivot]]
        for i in range(pivot + 1, len(a)):
            scale = a[i][col]
            if scale:
                a[i] = [x - scale * y for x, y in zip(a[i], a[pivot])]
        pivot += 1
    return pivot


def constraints(graph, sensors=(), aggregates=()):
    rows = [[int(e.head == v) - int(e.tail == v) for e in graph.edges] for v in graph.vertices]
    rows.extend([[int(e.key == key) for e in graph.edges] for key in sensors])
    for u, v in aggregates:
        rows.append([1 if (e.tail, e.head) == (u, v) else -1 if (e.tail, e.head) == (v, u) else 0 for e in graph.edges])
    return rows


def identified_rank(graph, target, sensors=(), aggregates=()):
    rows = constraints(graph, sensors, aggregates)
    return rank(rows, len(graph.edges)) == rank(rows + [[int(e.key == target) for e in graph.edges]], len(graph.edges))


def powerset(values):
    values = list(values)
    for n in range(len(values) + 1):
        yield from combinations(values, n)


def random_graph(rng, max_edges=8):
    n = rng.randrange(2, 6)
    vertices = tuple(str(i) for i in range(n))
    count = rng.randrange(1, max_edges + 1)
    edges = []
    for i in range(count):
        u, v = rng.sample(vertices, 2)
        edges.append(Edge("e" + str(i), u, v))
    return Graph(vertices, tuple(edges))


class GraphAccessTests(unittest.TestCase):
    def test_exact_cut_and_radius(self):
        graph = Graph(("A", "B", "C"), (Edge("t", "A", "B"), Edge("x", "B", "C"), Edge("y", "C", "A")))
        result = target_cut(graph, "t", {"x": "2/3", "y": "1/4"})
        self.assertEqual(Fraction(result["value"]), Fraction(1, 4))
        self.assertEqual(result["coefficients"], {"y": 1})
        self.assertTrue(verify_target_cut(graph, "t", {"x": "2/3", "y": "1/4"}, result))

    def test_unobserved_parallel_channel_ambiguity(self):
        graph = Graph(("A", "B"), (Edge("t", "A", "B"), Edge("p", "B", "A")))
        result = target_cut(graph, "t", {})
        self.assertEqual(result["status"], "unidentifiable")
        self.assertTrue(verify_target_cut(graph, "t", {}, result))
        self.assertEqual(mixed_observability(graph, known_aggregates=(("A", "B"),))["ambiguity_dimension"], 1)

    def test_bridge_zero_current(self):
        graph = Graph(("A", "B", "C"), (Edge("t", "A", "B"), Edge("x", "B", "C")))
        result = target_cut(graph, "t", {})
        self.assertEqual(result["value"], "0")
        self.assertEqual(result["coefficients"], {})

    def test_certificate_tampering(self):
        graph = Graph(("A", "B"), (Edge("t", "A", "B"), Edge("x", "A", "B")))
        cap = {"x": 3}
        original = target_cut(graph, "t", cap)
        for field, value in (("value", "2"), ("cut", ["B"]), ("coefficients", {"x": 1}), ("circulation", {"t": "3", "x": "3"})):
            altered = copy.deepcopy(original)
            altered[field] = value
            self.assertFalse(verify_target_cut(graph, "t", cap, altered))
        altered = copy.deepcopy(original)
        altered["circulation"]["invented"] = "0"
        self.assertFalse(verify_target_cut(graph, "t", cap, altered))

    def test_seeded_cuts_against_all_vertex_partitions(self):
        rng = random.Random(193307)
        for _ in range(160):
            graph = random_graph(rng)
            target = graph.edges[0]
            cap = {e.key: Fraction(rng.randrange(0, 8), rng.randrange(1, 5)) for e in graph.edges[1:] if rng.randrange(3)}
            result = target_cut(graph, target.key, cap)
            self.assertTrue(verify_target_cut(graph, target.key, cap, result))
            valid = []
            free = [v for v in graph.vertices if v not in (target.tail, target.head)]
            for part in powerset(free):
                cut = {target.tail, *part}
                cross = [e.key for e in graph.edges[1:] if (e.tail in cut) != (e.head in cut)]
                if all(k in cap for k in cross):
                    valid.append(sum((cap[k] for k in cross), Fraction(0)))
            if valid:
                self.assertEqual(result["status"], "identified")
                self.assertEqual(Fraction(result["value"]), min(valid))
            else:
                self.assertEqual(result["status"], "unidentifiable")

    def test_mixed_dimension_and_targets_against_exact_rank(self):
        rng = random.Random(100318)
        for _ in range(240):
            graph = random_graph(rng)
            sensors = [e.key for e in graph.edges if rng.randrange(3) == 0]
            aggregates = [b for b in graph.bundles() if rng.randrange(2)]
            result = mixed_observability(graph, sensors, aggregates)
            self.assertEqual(result["ambiguity_dimension"], len(graph.edges) - rank(constraints(graph, sensors, aggregates), len(graph.edges)))
            expected = [e.key for e in graph.edges if identified_rank(graph, e.key, sensors, aggregates)]
            self.assertEqual(result["identifiable_channels"], expected)

    def test_target_completion_against_all_sensor_subsets(self):
        rng = random.Random(91531)
        for _ in range(32):
            graph = random_graph(rng, max_edges=6)
            target = graph.edges[0].key
            costs = {e.key: rng.randrange(0, 8) for e in graph.edges[1:]}
            aggregates = [b for b in graph.bundles() if rng.randrange(2)]
            result = target_completion(graph, target, costs, aggregates)
            costs_of_valid = [sum(costs[k] for k in subset) for subset in powerset(costs) if identified_rank(graph, target, subset, aggregates)]
            self.assertEqual(Fraction(result["cost"]), min(costs_of_valid))
            self.assertTrue(identified_rank(graph, target, result["sensors"], aggregates))

    def test_all_target_completion_against_all_sensor_subsets(self):
        rng = random.Random(72253)
        for _ in range(32):
            graph = random_graph(rng, max_edges=6)
            costs = {e.key: rng.randrange(0, 8) for e in graph.edges}
            aggregates = [b for b in graph.bundles() if rng.randrange(2)]
            result = all_target_completion(graph, costs, aggregates)
            costs_of_valid = [sum(costs[k] for k in subset) for subset in powerset(costs) if rank(constraints(graph, subset, aggregates), len(graph.edges)) == len(graph.edges)]
            self.assertEqual(Fraction(result["cost"]), min(costs_of_valid))
            self.assertEqual(rank(constraints(graph, result["sensors"], aggregates), len(graph.edges)), len(graph.edges))

    def test_physical_realization(self):
        graph = Graph(("A", "B", "C"), (Edge("ab", "A", "B"), Edge("bc", "B", "C"), Edge("ca", "C", "A")))
        current = {"ab": Fraction(7, 3), "bc": Fraction(7, 3), "ca": Fraction(7, 3)}
        pi = {"A": Fraction(1, 2), "B": Fraction(1, 3), "C": Fraction(1, 6)}
        rates = realize_currents(graph, current, pi)
        for e in graph.edges:
            forward, reverse = Fraction(rates[e.key]["forward"]), Fraction(rates[e.key]["reverse"])
            self.assertGreater(forward, 0)
            self.assertGreater(reverse, 0)
            self.assertEqual(pi[e.tail] * forward - pi[e.head] * reverse, current[e.key])
        with self.assertRaises(ValueError):
            realize_currents(graph, {**current, "ab": 1}, pi)
        with self.assertRaises(ValueError):
            realize_currents(graph, current, pi, {k: 1 for k in current})

    def test_invalid_contracts(self):
        with self.assertRaises(ValueError):
            Graph(("A", "B"), (Edge("x", "A", "A"),))
        graph = Graph(("A", "B"), (Edge("t", "A", "B"), Edge("x", "A", "B")))
        with self.assertRaises(TypeError):
            target_cut(graph, "t", {"x": 0.5})
        with self.assertRaises(ValueError):
            target_cut(graph, "t", {"t": 1})
        with self.assertRaises(ValueError):
            target_cut(graph, "t", {"x": -1})


if __name__ == "__main__":
    unittest.main()
