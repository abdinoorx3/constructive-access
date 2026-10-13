import copy
from fractions import Fraction
from itertools import combinations
import unittest
from constructive_access.suite import SuiteError, design_suite, infer_target, verify_suite


def fixture():
    targets = {"A": "even", "A_copy": "even", "B": "even", "C": "odd", "D": "odd"}
    probabilities = {
        "left": {"A": "1/4", "A_copy": "1/4", "B": "3/4", "C": "1/4", "D": "3/4"},
        "right": {"A": "1/4", "A_copy": "1/4", "B": "3/4", "C": "3/4", "D": "1/4"},
        "direct": {"A": "1/4", "A_copy": "1/4", "B": "1/4", "C": "3/4", "D": "3/4"}}
    return targets, [{"name": name, "charge": "3" if name == "direct" else "1",
                      "laws": {h: {"0": str(1-Fraction(p)), "1": p} for h, p in ps.items()},
                      "tv_radii": {h: "1/100" for h in targets}}
                     for name, ps in probabilities.items()]


class SuiteTests(unittest.TestCase):
    def test_minimum_cover_and_redundant_mechanisms(self):
        targets, menu = fixture()
        report = design_suite(targets, menu)
        self.assertEqual(report["status"], "certified_suite")
        self.assertEqual(report["selection_charge"], "2")
        self.assertEqual([e["name"] for e in report["selected"]], ["left", "right"])
        self.assertEqual(report["pair_count"], 6)
        # Independent exhaustive subsets verify the known menu optimum.
        candidates = []
        for n in range(len(menu)+1):
            for subset in combinations(menu, n):
                if all(any(e["laws"][i] != e["laws"][j] for e in subset)
                       for i, j in combinations(targets, 2) if targets[i] != targets[j]):
                    candidates.append(sum(Fraction(e["charge"]) for e in subset))
        self.assertEqual(Fraction(report["selection_charge"]), min(candidates))
        self.assertTrue(verify_suite(targets, menu, report)["valid"])
        counts = {}
        for e in report["selected"]:
            n = e["records"]
            counts[e["name"]] = {"1": n//4, "0": n-n//4}
        decision = infer_target(targets, menu, report, counts)
        self.assertEqual(decision["target"], "even")
        self.assertEqual(decision["retained_models"], ["A", "A_copy"])

    def test_no_single_sensor_can_identify_target(self):
        targets, menu = fixture()
        for experiment in menu[:2]:
            self.assertEqual(design_suite(targets, [experiment])["status"], "insufficient_robust_menu")

    def test_calibration_collapse(self):
        targets, menu = fixture()
        for e in menu:
            e["tv_radii"] = {h: "1/4" for h in targets}
        self.assertEqual(design_suite(targets, menu)["status"], "insufficient_robust_menu")

    def test_rejects_mismatched_model_and_tampering(self):
        targets, menu = fixture()
        report = design_suite(targets, menu)
        corrupt = copy.deepcopy(report)
        corrupt["selection_charge"] = "0"
        with self.assertRaises(SuiteError):
            verify_suite(targets, menu, corrupt)
        modified = copy.deepcopy(menu)
        modified[0]["charge"] = "10"
        with self.assertRaises(SuiteError):
            verify_suite(targets, modified, report)

    def test_requested_sampling_budget_is_bound(self):
        targets, menu = fixture()
        report = design_suite(targets, menu, delta="1/10")
        with self.assertRaises(SuiteError):
            verify_suite(targets, menu, report)
        self.assertTrue(verify_suite(targets, menu, report, expected_delta="1/10")["valid"])

    def test_sampling_certificate_and_separation(self):
        targets, menu = fixture()
        report = design_suite(targets, menu)
        self.assertLessEqual(Fraction(report["sampling_failure_upper"]), Fraction(1, 20))
        for selected in report["selected"]:
            r, n = Fraction(selected["sampling_radius"]), selected["records"]
            self.assertGreaterEqual(2*n*r*r, report["union_bound_exponent"])
        for pair in report["pair_certificates"]:
            r = next(Fraction(e["sampling_radius"]) for e in report["selected"] if e["name"] == pair["experiment"])
            self.assertGreater(Fraction(pair["robust_gap"]), 2*r)

    def test_explicit_failure_on_unmodeled_data(self):
        targets, menu = fixture()
        report = design_suite(targets, menu)
        counts = {e["name"]: {"1": 0, "0": e["records"]} for e in report["selected"]}
        self.assertEqual(infer_target(targets, menu, report, counts)["status"], "abstain_all_rejected")
        counts["left"]["0"] -= 1
        with self.assertRaises(SuiteError):
            infer_target(targets, menu, report, counts)

    def test_float_and_budget_rejection(self):
        targets, menu = fixture()
        bad = copy.deepcopy(menu)
        bad[0]["charge"] = .1
        with self.assertRaises(SuiteError):
            design_suite(targets, bad)
        with self.assertRaises(SuiteError):
            design_suite(targets, menu, max_pairs=1)
        with self.assertRaises(SuiteError):
            design_suite(targets, menu, max_states=1)

    def test_constant_target_needs_no_data(self):
        targets, menu = fixture()
        targets = {h: "same" for h in targets}
        report = design_suite(targets, menu)
        self.assertEqual(report["status"], "target_constant")
        self.assertEqual(infer_target(targets, menu, report, {})["target"], "same")

    def test_zero_charge_tie_uses_shorter_suite(self):
        targets, menu = fixture()
        for e in menu:
            e["charge"] = "0"
        report = design_suite(targets, menu)
        self.assertEqual([x["name"] for x in report["selected"]], ["direct"])


if __name__ == "__main__":
    unittest.main()
