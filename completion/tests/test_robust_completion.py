"""Exact adversarial tests for robust access completion and energy contracts."""
from copy import deepcopy
from fractions import Fraction as Q
from itertools import combinations, product
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"code"))
from graph_access import Edge, Graph, target_cut
from robust_completion import (allocate_cut_energy, enumerate_floor_design,
    enumerate_partition_design, floor_energy_bound, minimum_ideal_energy, partition_energy_instance,
    robust_target_certificate, verify_cut_energy, verify_ideal_energy,
    verify_floor_energy_bound, verify_robust_certificate)


class RobustCompletionTests(unittest.TestCase):
    def setUp(self):
        self.model = {"conservation": [[-1, 0, 1], [1, -1, 0], [0, 1, -1]],
                      "target": [1, 0, 0], "epsilon": ["1/10", "1/5"],
                      "observation": [[0, 1, 0], [0, 0, -1]], "nuisance": [[1], [1]]}
        self.graph = Graph(("s", "a", "t"), (Edge("target", "s", "t"), Edge("single", "s", "a"),
                                             Edge("pair_1", "a", "t"), Edge("pair_2", "a", "t")))
        self.q = {"single": 4, "pair_1": 1, "pair_2": 2}

    def test_shared_bias_cancellation(self):
        certificate = robust_target_certificate(self.model)
        self.assertEqual(certificate["weights"], ["1/2", "-1/2"])
        self.assertEqual(certificate["absolute_error"], "3/20")
        self.assertTrue(verify_robust_certificate(self.model, certificate)["verified"])

    def test_common_bias_with_same_orientation_is_unidentifiable(self):
        self.model["observation"][1] = [0, 0, 1]
        certificate = robust_target_certificate(self.model)
        self.assertFalse(certificate["finite"])
        self.assertTrue(verify_robust_certificate(self.model, certificate)["verified"])

    def test_independent_biases_restore_ambiguity(self):
        self.model["nuisance"] = [[1, 0], [0, 1]]
        self.assertFalse(robust_target_certificate(self.model)["finite"])

    def test_tampered_robust_certificates_rejected(self):
        original = robust_target_certificate(self.model)
        for field, value in (("weights", ["1", "0"]), ("absolute_error", "0"),
                             ("current_direction", ["1", "0", "0"]),
                             ("nuisance_direction", ["100"])):
            with self.subTest(field=field):
                certificate = deepcopy(original)
                certificate[field] = value
                with self.assertRaises(ValueError):
                    verify_robust_certificate(self.model, certificate)

    def test_zero_target_and_exact_observations(self):
        self.model["target"] = [0, 0, 0]
        self.assertEqual(robust_target_certificate(self.model)["absolute_error"], "0")
        self.model["target"] = [1, 0, 0]
        self.model["epsilon"] = [0, 0]
        self.assertEqual(robust_target_certificate(self.model)["absolute_error"], "0")

    def test_bad_numeric_inputs_rejected(self):
        for value in (True, 0.1, "-1"):
            self.model["epsilon"][0] = value
            with self.assertRaises(ValueError):
                robust_target_certificate(self.model)

    def test_cut_allocation_and_floor_boundary(self):
        result = allocate_cut_energy([1, 2], [0, 0], "1/10")
        self.assertEqual(result["energies"], ["900", "1800"])
        self.assertEqual(result["errors"], ["1/30", "1/15"])
        self.assertEqual(verify_cut_energy(result)["total_energy"], "2700")
        for threshold in ("1/100", "3/100"):
            result = allocate_cut_energy([1, 2], ["1/100", "1/50"], threshold)
            self.assertFalse(result["feasible"])
            self.assertTrue(verify_cut_energy(result)["verified"])
        self.assertTrue(allocate_cut_energy([1, 2], ["1/100", "1/50"], "31/1000")["feasible"])

    def test_global_optimum_and_floor_induced_sensor_switch(self):
        result = minimum_ideal_energy(self.graph, "target", self.q, "1/10")
        self.assertEqual(result["sensor_edges"], ["pair_1", "pair_2"])
        self.assertEqual(verify_ideal_energy(self.graph, "target", self.q, "1/10", result)["total_energy"], "2700")
        floor = {"single": 0, "pair_1": "3/50", "pair_2": "1/50"}
        result = enumerate_floor_design(self.graph, "target", self.q, floor, "1/10")
        self.assertEqual(result["sensor_edges"], ["single"])
        self.assertEqual(result["total_energy"], "6400")

    def test_global_certificate_forgery_rejected(self):
        original = minimum_ideal_energy(self.graph, "target", self.q, "1/10")
        certificate = deepcopy(original)
        certificate["cut_certificate"]["circulation"]["target"] = "0"
        with self.assertRaises(ValueError):
            verify_ideal_energy(self.graph, "target", self.q, "1/10", certificate)
        certificate = deepcopy(original)
        certificate["allocation"]["total_energy"] = "1"
        with self.assertRaises(ValueError):
            verify_ideal_energy(self.graph, "target", self.q, "1/10", certificate)
        with self.assertRaises(ValueError):
            verify_ideal_energy(self.graph, "target", self.q, "1/5", original)

    def test_floor_supporting_certificate_and_duality_gap(self):
        floor = {"single": 0, "pair_1": "3/50", "pair_2": "1/50"}
        certificate = floor_energy_bound(self.graph, "target", self.q, floor, "1/10", 40)
        checked = verify_floor_energy_bound(self.graph, "target", self.q, floor, "1/10", certificate)
        self.assertEqual(checked["lower_energy"], "6400")
        self.assertEqual(checked["upper_energy"], "6400")
        self.assertTrue(checked["global_optimum_certified"])
        corrupted = deepcopy(certificate)
        corrupted["lower_energy"] = "6401"
        with self.assertRaises(ValueError):
            verify_floor_energy_bound(self.graph, "target", self.q, floor, "1/10", corrupted)
        instance = partition_energy_instance([1, 2, 4])
        graph = Graph(("s", "t", "v0", "v1", "v2"), tuple([Edge("target", "s", "t")]+
                      [Edge(e["id"], e["u"], e["v"]) for e in instance["sensor_edges"]]))
        q = {e["id"]: e["q"] for e in instance["sensor_edges"]}
        floor = {e["id"]: e["floor"] for e in instance["sensor_edges"]}
        certificate = floor_energy_bound(graph, "target", q, floor, instance["accuracy"], "3/2")
        checked = verify_floor_energy_bound(graph, "target", q, floor, instance["accuracy"], certificate)
        self.assertEqual(checked["lower_energy"], instance["threshold_energy"])
        true_optimum = Q(enumerate_partition_design(instance)["best_energy"])
        self.assertLess(Q(checked["lower_energy"]), true_optimum)
        self.assertLessEqual(true_optimum, Q(checked["upper_energy"]))

    def test_supporting_bounds_against_full_cut_enumeration(self):
        rng = random.Random(5521)
        for _ in range(20):
            q = {key: Q(rng.randint(1, 10), 3) for key in self.q}
            floor = {key: Q(rng.randint(0, 3), 20) for key in self.q}
            accuracy = Q(2, 5)
            optimum = enumerate_floor_design(self.graph, "target", q, floor, accuracy)
            for t in (Q(1, 2), Q(2), Q(7), Q(50)):
                certificate = floor_energy_bound(self.graph, "target", q, floor, accuracy, t)
                checked = verify_floor_energy_bound(self.graph, "target", q, floor, accuracy, certificate)
                self.assertLessEqual(Q(checked["lower_energy"]), Q(optimum["total_energy"]))
                if checked["upper_energy"] is not None:
                    self.assertGreaterEqual(Q(checked["upper_energy"]), Q(optimum["total_energy"]))

    def test_bridge_and_unavailable_sensors(self):
        bridge = Graph(("s", "t"), (Edge("target", "s", "t"),))
        result = minimum_ideal_energy(bridge, "target", {}, 1)
        self.assertEqual(verify_ideal_energy(bridge, "target", {}, 1, result)["total_energy"], "0")
        result = minimum_ideal_energy(self.graph, "target", {}, 1)
        self.assertFalse(result["feasible"])
        self.assertTrue(verify_ideal_energy(self.graph, "target", {}, 1, result)["verified"])

    def test_partition_reduction_against_independent_subset_sums(self):
        rng = random.Random(49170)
        for _ in range(24):
            values = [rng.randint(1, 12) for _ in range(rng.randint(1, 7))]
            instance = partition_energy_instance(values)
            result = enumerate_partition_design(instance)
            sums = {0}
            for value in values:
                sums |= {s+value for s in sums}
            expected = sum(values) % 2 == 0 and sum(values)//2 in sums
            self.assertEqual(result["partition_exists"], expected)
            graph = Graph(tuple(["s", "t"]+[f"v{i}" for i in range(len(values))]),
                          tuple([Edge("target", "s", "t")]+[Edge(e["id"], e["u"], e["v"]) for e in instance["sensor_edges"]]))
            q = {e["id"]: e["q"] for e in instance["sensor_edges"]}
            floor = {e["id"]: e["floor"] for e in instance["sensor_edges"]}
            independently_enumerated = enumerate_floor_design(graph, "target", q, floor, instance["accuracy"])
            self.assertEqual(independently_enumerated["total_energy"], result["best_energy"])

    def test_linear_minimax_agrees_with_cut_on_seeded_small_graphs(self):
        rng = random.Random(191)
        for _ in range(12):
            vertices = ("a", "b", "c", "d")
            possible = list(combinations(vertices, 2))
            chosen = [possible[0]]+[pair for pair in possible[1:] if rng.randrange(2)]
            graph = Graph(vertices, tuple(Edge(str(i), *pair) for i, pair in enumerate(chosen)))
            sensors = {e.key: Q(rng.randint(0, 5), 7) for e in graph.edges[1:] if rng.randrange(3)}
            incidence = [[-int(e.tail == v)+int(e.head == v) for e in graph.edges] for v in vertices]
            observations = [[int(e.key == sensor) for e in graph.edges] for sensor in sensors]
            model = {"conservation": incidence, "target": [1]+[0]*(len(graph.edges)-1),
                     "observation": observations, "epsilon": list(sensors.values())}
            linear = robust_target_certificate(model)
            cut = target_cut(graph, "0", sensors)
            self.assertEqual(linear["finite"], cut["status"] == "identified")
            if linear["finite"]:
                self.assertEqual(linear["absolute_error"], cut["value"])


if __name__ == "__main__":
    unittest.main()
