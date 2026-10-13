"""End-to-end and adversarial checks of the supported JSON workflow."""
from copy import deepcopy
from fractions import Fraction as Q
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from constructive_access import ContractError, VerificationError, analyze, load_contract, verify_report
from constructive_access.workflow import _log_upper

ROOT = Path(__file__).resolve().parents[1]


def example(name="active"):
    return json.loads((ROOT/"examples"/(name+".json")).read_text())


class WorkflowTests(unittest.TestCase):
    def test_complete_active_workflow(self):
        data = example()
        report = analyze(data)
        self.assertEqual(report["witness"]["actions"], ["drive", "read"])
        self.assertEqual(report["witness"]["minimum_action_charge"], "3")
        experiment = report["experiment"]
        self.assertEqual(experiment["total_variation"], "1/2")
        self.assertEqual(experiment["single_record_bayes_error"], "1/4")
        self.assertEqual(len(experiment["transcripts"]), 2)
        self.assertEqual(sum(Q(row["probability0"]) for row in experiment["transcripts"]), 1)
        self.assertEqual(sum(Q(row["probability1"]) for row in experiment["transcripts"]), 1)
        self.assertEqual(experiment["sufficient_records"], 12)
        self.assertLessEqual(Q(experiment["certified_average_error_upper"]), Q(1, 10))
        self.assertEqual(experiment["per_record_declared_charge"], "4")
        self.assertEqual(experiment["nominal_batch_declared_charge"], "48")
        self.assertTrue(verify_report(data, report)["verified"])

    def test_passive_equivalence(self):
        data = example("passive")
        report = analyze(data)
        self.assertEqual(report["status"], "equivalent_under_declared_access")
        self.assertIsNone(report["experiment"])
        self.assertEqual(report["certificate_verification"]["mode"], "equivalent")
        self.assertTrue(verify_report(data, report)["verified"])

    def test_robust_calibrated_classifier(self):
        report = analyze(example("calibrated"))
        robust = report["robust_experiment"]
        self.assertEqual(robust["robust_gap_lower"], "12/25")
        self.assertEqual(robust["records"], 21)
        self.assertEqual(robust["event_frequency_threshold"], "1/2")
        self.assertEqual(robust["conditional_each_hypothesis_error_upper"], "9/100")
        self.assertEqual(robust["unconditional_each_hypothesis_error_upper"], "1/10")
        self.assertGreaterEqual(Q(robust["hoeffding_exponent"]), Q(robust["log_inverse_conditional_error_upper"]))
        self.assertTrue(verify_report(example("calibrated"), report)["verified"])

    def test_calibration_failure_modes_do_not_claim_universal_impossibility(self):
        report = analyze(example("calibration_gap_closed"))
        self.assertEqual(report["robust_experiment"]["status"], "calibration_gap_not_certified")
        self.assertNotIn("records", report["robust_experiment"])
        self.assertEqual(report["status"], "distinguishable_under_declared_access")
        data = example("calibrated")
        data["calibration"]["joint_failure_probability"] = "1/10"
        self.assertEqual(analyze(data)["robust_experiment"]["status"], "calibration_failure_exhausts_error_budget")

    def test_non_iid_has_no_repetition_certificate(self):
        data = example("calibrated")
        data["preparation"]["iid_records"] = False
        report = analyze(data)
        self.assertIsNone(report["experiment"]["sufficient_records"])
        self.assertIsNone(report["experiment"]["certified_average_error_upper"])
        self.assertEqual(report["experiment"]["single_record_bayes_error"], "1/4")
        self.assertEqual(report["robust_experiment"]["status"], "iid_assumption_not_declared")

    def test_transcript_cap_preserves_exact_access(self):
        data = example()
        data["design"]["max_transcripts"] = 1
        report = analyze(data)
        self.assertEqual(report["experiment"]["status"], "transcript_limit_exceeded")
        self.assertEqual(report["status"], "distinguishable_under_declared_access")
        self.assertTrue(verify_report(data, report)["verified"])

    def test_grammar_and_empty_language(self):
        data = example("grammar")
        report = analyze(data)
        self.assertEqual(report["witness"]["actions"], ["drive", "read"])
        self.assertTrue(verify_report(data, report)["verified"])
        data["grammar"]["accepting"] = []
        report = analyze(data)
        self.assertEqual(report["status"], "equivalent_under_declared_access")
        self.assertTrue(verify_report(data, report)["verified"])

    def test_zero_cost_actions_still_terminate_with_shortest_witness(self):
        data = example()
        data["resources"]["action_costs"] = {"drive": 0, "read": 0}
        report = analyze(data)
        self.assertEqual(report["witness"]["minimum_action_charge"], "0")
        self.assertEqual(report["witness"]["length"], 2)
        self.assertTrue(verify_report(data, report)["verified"])

    def test_verifier_does_not_invoke_synthesis(self):
        data = example()
        report = analyze(data)
        with patch("constructive_access.workflow.cc.generate_certificate", side_effect=AssertionError("search called")), \
             patch("constructive_access.workflow.cc._coordinates", side_effect=AssertionError("rank called")):
            self.assertFalse(verify_report(data, report)["synthesis_repeated"])

    def test_tamper_rejection(self):
        data = example("calibrated")
        original = analyze(data)
        def altered(path, value):
            report = deepcopy(original)
            cursor = report
            for key in path[:-1]:
                cursor = cursor[key]
            cursor[path[-1]] = value
            return report
        for path, value in [(["witness", "minimum_action_charge"], "0"),
                            (["experiment", "sufficient_records"], 1),
                            (["robust_experiment", "records"], 1),
                            (["experiment", "transcripts", 0, "probability0"], "1"),
                            (["certificate", "certificate", "witness", "cost"], "0"),
                            (["resources", "meaning"], "energy_upper_bound"),
                            (["certificate_verification", "expected_model_bound"], False),
                            (["toolkit_version"], "arbitrary")]:
            with self.subTest(path=path), self.assertRaises(VerificationError):
                verify_report(data, altered(path, value))
        extra = deepcopy(original)
        extra["perfect_discrimination"] = True
        with self.assertRaises(VerificationError):
            verify_report(data, extra)
        extra = deepcopy(original)
        extra["certificate"]["certificate"]["unsupported_claim"] = True
        with self.assertRaises(VerificationError):
            verify_report(data, extra)

    def test_removed_closure_and_model_substitution_rejected(self):
        data = example()
        report = analyze(data)
        report["certificate"]["certificate"]["closure"].pop()
        with self.assertRaises(VerificationError):
            verify_report(data, report)
        substitute = example()
        substitute["models"][0]["actions"]["drive"]["done"] = [["2/3", "1/3"], ["2/3", "1/3"]]
        report = analyze(substitute)
        report["contract_sha256"] = load_contract(data).sha256
        with self.assertRaises(VerificationError):
            verify_report(data, report)

    def test_contract_is_detached_and_canonical(self):
        data = example()
        contract = load_contract(data)
        digest = contract.sha256
        data["models"][0]["initial"][0] = 0
        contract.to_dict()["models"][0]["initial"][0] = 0
        self.assertEqual(contract.sha256, digest)
        equivalent = example()
        equivalent["resources"]["action_costs"]["read"] = "2/2"
        self.assertEqual(load_contract(equivalent).sha256, digest)

    def test_invalid_contracts(self):
        changes = [lambda d: d["models"][0]["initial"].__setitem__(0, .5),
                   lambda d: d["models"][0]["initial"].__setitem__(0, True),
                   lambda d: d["models"][0]["initial"].__setitem__(0, "1/0"),
                   lambda d: d["resources"]["action_costs"].__setitem__("read", -1),
                   lambda d: d["models"][0]["actions"]["read"].pop("0"),
                   lambda d: d["models"][0]["actions"]["drive"]["done"].__setitem__(0, [1, 1]),
                   lambda d: d["models"][0]["actions"]["read"]["0"].__setitem__(0, [1]),
                   lambda d: d["target"]["labels"].__setitem__(1, d["target"]["labels"][0]),
                   lambda d: d.__setitem__("unknown_key", 1),
                   lambda d: d["design"].__setitem__("target_average_error", 0),
                   lambda d: d["design"].__setitem__("precision_bits", True),
                   lambda d: d["preparation"].__setitem__("iid_records", "yes")]
        for index, change in enumerate(changes):
            data = example()
            change(data)
            with self.subTest(index=index), self.assertRaises(ContractError):
                load_contract(data)

    def test_duplicate_json_key_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"bad.json"
            path.write_text('{"schema":"one","schema":"two"}')
            with self.assertRaises(ContractError):
                load_contract(path)

    def test_cli_success_and_error(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory)/"report.json"
            command = [sys.executable, "-m", "constructive_access"]
            built = subprocess.run(command+["analyze", str(ROOT/"examples/active.json"), "-o", str(report)], capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stderr)
            checked = subprocess.run(command+["verify", str(ROOT/"examples/active.json"), str(report)], capture_output=True, text=True)
            self.assertEqual(checked.returncode, 0, checked.stderr)
            self.assertTrue(json.loads(checked.stdout)["verified"])
            wrong = subprocess.run(command+["verify", str(ROOT/"examples/passive.json"), str(report)], capture_output=True, text=True)
            self.assertEqual(wrong.returncode, 2)
            self.assertEqual(json.loads(wrong.stderr)["status"], "error")

    def test_log_enclosure_against_independent_decimal(self):
        from decimal import Decimal, localcontext
        with localcontext() as ctx:
            ctx.prec = 160
            for value in (Q(1), Q(2), Q(100, 9), Q(10**30), Q(1001, 1000)):
                bound = _log_upper(value)
                decimal_bound = Decimal(bound.numerator)/Decimal(bound.denominator)
                target = (Decimal(value.numerator)/Decimal(value.denominator)).ln()
                self.assertGreaterEqual(decimal_bound, target)

    def test_linear_target_and_exact_minimax_public_apis(self):
        from constructive_access import identify_linear_target, minimax_certificate, verify_minimax_certificate
        fingerprint = [[1, 1], ["1/2", 1]]
        target = [1, 0]
        radii = ["1/100", "1/100"]
        self.assertTrue(identify_linear_target(fingerprint, target)["identifiable"])
        self.assertFalse(identify_linear_target([[1, 1]], target)["identifiable"])
        self.assertTrue(identify_linear_target([[1, 1]], [1, 1])["identifiable"])
        certificate = minimax_certificate(fingerprint, target, radii)
        self.assertEqual(certificate["absolute_error"], "1/25")
        self.assertTrue(verify_minimax_certificate(fingerprint, target, radii, certificate)["verified"])
        certificate["absolute_error"] = "0"
        with self.assertRaises(ValueError):
            verify_minimax_certificate(fingerprint, target, radii, certificate)

    def test_counting_energy_envelope_public_apis(self):
        from constructive_access import count_cap, verify_count_cap, declared_energy_cap
        certificate = count_cap(10, "1/100")
        self.assertEqual(certificate["count_cap"], 22)
        self.assertTrue(verify_count_cap(certificate))
        self.assertEqual(declared_energy_cap(certificate, "3/2", 4), 37)
        bad = dict(certificate, count_cap=10)
        self.assertFalse(verify_count_cap(bad))
        with self.assertRaises(ValueError):
            declared_energy_cap(bad, 1, 0)
        self.assertEqual(count_cap(0)["count_cap"], 0)


if __name__ == "__main__":
    unittest.main()
