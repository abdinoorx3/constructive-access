"""Adversarial checks authored independently of the integrated implementation.

Expected values below use direct rational products or elementary identities,
not the producing routines.  These are internal software checks, not physical
experiments or external peer review.
"""
from copy import deepcopy
from fractions import Fraction as Q
from itertools import product
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import sympy as sp

from constructive_access.contract import ContractError, load_contract
from constructive_access._vendor import cost_certificate as certificate
from constructive_access._vendor.model_adequacy import AdequacyMonitor, Interval
from constructive_access import analyze, verify_report
from constructive_access import (
    count_cap, verify_count_cap, declared_energy_cap,
    identify_linear_target, minimax_certificate, verify_minimax_certificate,
    LinearRepresentation, RationalInstrument, experiment_law,
)
from constructive_access.workflow import VerificationError
from constructive_access.suite import design_suite, infer_target, verify_suite, SuiteError


def specification(p0="1/4", p1="3/4"):
    return {
        "schema": "constructive-access-contract-v1",
        "name": "independent Bernoulli contrast",
        "question": "Which declared output bias is present?",
        "target": {"name": "bias", "labels": ["low", "high"]},
        "models": [
            {"name": name, "initial": [1], "actions": {"read": {
                "0": [[str(1-Q(p))]], "1": [[p]]}}}
            for name, p in (("model0", p0), ("model1", p1))
        ],
        "resources": {"charge_name": "read calls", "unit": "call",
            "meaning": "operation_count", "action_costs": {"read": 1},
            "per_record_overhead": 0, "calibration": "Literal call counting",
            "included": ["read calls"], "excluded": ["apparatus energy"]},
        "preparation": {"description": "Independent fresh preparations under the same fixed hypothesis",
            "iid_records": True},
        "design": {"target_average_error": "1/10"},
    }


def paired_bernoulli():
    # alpha diag(p0,p1) beta = p0-p1; no pairing implementation is used.
    letters = []
    for a, p0, p1, cost in (("cheap", Q(49,100), Q(51,100), 1),
                             ("strong", Q(1,5), Q(4,5), 3)):
        for outcome in (0, 1):
            x, y = (p0, p1) if outcome else (1-p0, 1-p1)
            letters.append({"id": a+str(outcome), "action": a,
                "outcome": str(outcome), "cost": str(cost),
                "matrix": [[str(x), "0"], ["0", str(y)]]})
    return {"alpha": ["1", "-1"], "beta": ["1", "1"], "letters": letters}


class IndependentContractTests(unittest.TestCase):
    def test_input_is_detached_and_float_is_rejected(self):
        raw = specification()
        loaded = load_contract(raw)
        before = loaded.sha256
        raw["models"][0]["initial"][0] = 0
        self.assertEqual(loaded.sha256, before)
        raw = specification()
        raw["models"][0]["actions"]["read"]["1"][0][0] = .25
        with self.assertRaises(ContractError):
            load_contract(raw)

    def test_incomplete_resource_and_wrong_calibration_scope_fail(self):
        raw = specification()
        del raw["resources"]["excluded"]
        with self.assertRaises(ContractError):
            load_contract(raw)
        raw = specification()
        raw["calibration"] = {"record_tv_radii": ["0", "0"],
            "scope": "one_unconditional_marginal", "justification": "insufficient",
            "joint_failure_probability": "0"}
        with self.assertRaises(ContractError):
            load_contract(raw)

    def test_duplicate_json_keys_are_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"duplicate.json"
            path.write_text('{"schema":"first","schema":"second"}')
            with self.assertRaises(ContractError):
                load_contract(path)

    def test_negative_or_nonstochastic_kernel_fails(self):
        raw = specification()
        raw["models"][0]["actions"]["read"]["0"] = [["-1/4"]]
        with self.assertRaises(ContractError):
            load_contract(raw)
        raw = specification()
        raw["models"][0]["actions"]["read"]["0"] = [[1]]
        with self.assertRaises(ContractError):
            load_contract(raw)


class IndependentCertificateTests(unittest.TestCase):
    def test_minimum_witness_and_independent_model_binding(self):
        model = paired_bernoulli()
        bundle = certificate.generate_certificate(model)
        checked = certificate.check_certificate(bundle, expected_model=model)
        self.assertEqual(checked["minimum_action_cost"], "1")
        self.assertEqual(checked["shortest_length_at_minimum_cost"], 1)
        self.assertEqual(abs(Q(checked["contrast"])), Q(1,50))
        alternate = deepcopy(model)
        alternate["letters"][0]["cost"] = "2"
        alternate["letters"][1]["cost"] = "2"
        with self.assertRaises(certificate.CertificateError):
            certificate.check_certificate(bundle, expected_model=alternate)

    def test_valid_but_expensive_witness_cannot_claim_optimality(self):
        model = paired_bernoulli()
        bundle = certificate.generate_certificate(model)
        # This is a genuine witness, with exactly stated cost and contrast;
        # it fails because cheaper witnesses exist, not because its gap is fake.
        bundle["certificate"]["witness"] = {
            "word": ["strong1"], "cost": "3", "contrast": "-3/5"}
        with self.assertRaises(certificate.CertificateError):
            certificate.check_certificate(bundle, expected_model=model)

    def test_checker_does_not_call_generator_or_span_solver(self):
        model = paired_bernoulli()
        bundle = certificate.generate_certificate(model)
        saved_generator, saved_coordinates = certificate.generate_certificate, certificate._coordinates
        def forbidden(*args, **kwargs):
            raise AssertionError("certificate checker called synthesis")
        try:
            certificate.generate_certificate = forbidden
            certificate._coordinates = forbidden
            self.assertTrue(certificate.check_certificate(bundle, expected_model=model)["verified"])
        finally:
            certificate.generate_certificate = saved_generator
            certificate._coordinates = saved_coordinates


class IndependentInferenceTests(unittest.TestCase):
    def test_fixed_nuisance_is_not_redrawn(self):
        # Under a fixed p in {1/10,9/10}, a 50:50 mixture of TWO-record laws
        # has agreement probability 41/50; redrawing p each trial gives 1/2.
        fixed = sum(((p*p+(1-p)*(1-p))/2 for p in (Q(1,10), Q(9,10))), Q())
        redrawn_p = (Q(1,10)+Q(9,10))/2
        redrawn = redrawn_p**2+(1-redrawn_p)**2
        self.assertEqual(fixed, Q(41,50))
        self.assertEqual(redrawn, Q(1,2))
        self.assertNotEqual(fixed, redrawn)

    def test_prediction_lifecycle_and_target_specific_commitment(self):
        monitor = AdequacyMonitor({"mechanism_a": "same_target", "mechanism_b": "same_target"})
        self.assertEqual(monitor.decision().status, "commit")
        self.assertEqual(len(monitor.decision().survivors), 2)
        with self.assertRaises(RuntimeError):
            monitor.observe(1)
        monitor.prepare("read", {k: Interval(Q(0), Q(1)) for k in monitor.targets})
        with self.assertRaises(RuntimeError):
            monitor.prepare("read", {k: Interval(Q(0), Q(1)) for k in monitor.targets})
        monitor.observe(1)
        self.assertEqual(len(monitor.decision().survivors), 2)

    def test_stationary_marginal_is_not_a_conditional_envelope(self):
        # Actual sequence Y_t=Z for one fair bit. Marginal p=1/2 is true at
        # each time but is an invalid conditional prediction after time one.
        rejected = 0
        for bit in (0, 1):
            monitor = AdequacyMonitor({"incorrect_envelope": "target"})
            for _ in range(40):
                monitor.prepare("read", {"incorrect_envelope": Interval(Q(1,2), Q(1,2))})
                monitor.observe(bit)
            rejected += bool(monitor.excluded)
        self.assertEqual(rejected, 2)  # false rejection probability one

    def test_hidden_ring_has_identical_laws_with_different_hidden_current(self):
        # Each emission instrument K_y=T/2. T is stochastic, so every word
        # has law 2^-n, independently of the unobserved transition current.
        t = [[Q(0), Q(3,4), Q(1,4)], [Q(1,4), Q(0), Q(3,4)],
             [Q(3,4), Q(1,4), Q(0)]]
        initial = [Q(1,3)]*3
        self.assertEqual(initial[0]*t[0][1]-initial[1]*t[1][0], Q(1,6))
        for n in range(6):
            for word in product((0, 1), repeat=n):
                v = initial[:]
                for _ in word:
                    v = [sum((v[i]*t[i][j]/2 for i in range(3)), Q()) for j in range(3)]
                self.assertEqual(sum(v), Q(1,2)**n)


class IndependentWorkflowTests(unittest.TestCase):
    def test_nominal_and_calibrated_report_have_independent_expected_values(self):
        raw = specification()
        raw["calibration"] = {"record_tv_radii": ["1/100", "1/100"],
            "scope": "whole_record_for_returned_protocol",
            "justification": "Synthetic assumed conditional record-law bounds",
            "joint_failure_probability": "1/100"}
        report = analyze(raw)
        self.assertEqual(report["experiment"]["total_variation"], "1/2")
        self.assertEqual(report["experiment"]["single_record_bayes_error"], "1/4")
        robust = report["robust_experiment"]
        self.assertEqual(robust["robust_gap_lower"], "12/25")
        self.assertEqual(robust["event_frequency_threshold"], "1/2")
        # ceil[2 log(100/9)/(12/25)^2] = 21, well away from an integer boundary.
        self.assertEqual(robust["records"], 21)
        self.assertEqual(robust["unconditional_each_hypothesis_error_upper"], "1/10")
        self.assertTrue(verify_report(raw, report)["verified"])
        forged = deepcopy(report)
        forged["robust_experiment"]["records"] = 1
        with self.assertRaises(VerificationError):
            verify_report(raw, forged)

    def test_calibration_can_block_the_experiment(self):
        raw = specification()
        raw["calibration"] = {"record_tv_radii": ["1/4", "1/4"],
            "scope": "whole_record_for_returned_protocol", "justification": "Deliberately wide",
            "joint_failure_probability": "0"}
        result = analyze(raw)
        self.assertEqual(result["status"], "distinguishable_under_declared_access")
        self.assertEqual(result["robust_experiment"]["status"], "calibration_gap_not_certified")
        self.assertNotIn("records", result["robust_experiment"])
        raw["calibration"]["record_tv_radii"] = ["0", "0"]
        raw["calibration"]["joint_failure_probability"] = "1/10"
        result = analyze(raw)
        self.assertEqual(result["robust_experiment"]["status"], "calibration_failure_exhausts_error_budget")

    def test_no_iid_assumption_no_repeated_record_guarantee(self):
        raw = specification()
        raw["preparation"]["iid_records"] = False
        result = analyze(raw)
        self.assertEqual(result["experiment"]["single_record_bayes_error"], "1/4")
        self.assertIsNone(result["experiment"]["sufficient_records"])
        self.assertIsNone(result["experiment"]["certified_average_error_upper"])
        self.assertIsNone(result["experiment"]["nominal_batch_declared_charge"])

    def test_different_hidden_dimensions_are_not_recovered_from_identical_laws(self):
        raw = specification("1/2", "1/2")
        t = [[Q(0), Q(3,4), Q(1,4)], [Q(1,4), Q(0), Q(3,4)],
             [Q(3,4), Q(1,4), Q(0)]]
        ring = raw["models"][1]
        ring["initial"] = ["1/3"]*3
        ring["actions"]["read"] = {str(y): [[str(x/2) for x in row] for row in t] for y in (0,1)}
        result = analyze(raw)
        self.assertEqual(result["status"], "equivalent_under_declared_access")
        self.assertEqual(result["certificate_verification"]["dimension"], 4)
        self.assertIsNone(result["experiment"])
        self.assertTrue(verify_report(raw, result)["verified"])

    def test_cli_runs_from_an_unrelated_working_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contract, report = root/"input.json", root/"report.json"
            contract.write_text(json.dumps(specification()))
            # Source-tree tests may be invoked with a relative PYTHONPATH.
            # Freeze it before changing cwd; clean-wheel import is a separate
            # release test that runs without any source-tree PYTHONPATH.
            env = dict(os.environ)
            paths = [str(Path(__file__).resolve().parents[1]/"src")]
            paths += [str(Path(p).resolve()) for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
            env["PYTHONPATH"] = os.pathsep.join(paths)
            first = subprocess.run([sys.executable, "-m", "constructive_access", "analyze",
                str(contract), "-o", str(report)], cwd=root, env=env, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            second = subprocess.run([sys.executable, "-m", "constructive_access", "verify",
                str(contract), str(report)], cwd=root, env=env, capture_output=True, text=True)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertTrue(json.loads(second.stdout)["verified"])


class IndependentTargetSuiteTests(unittest.TestCase):
    @staticmethod
    def fixture():
        targets = {"a1": "A", "a2": "A", "b1": "B", "b2": "B"}
        def experiment(name, charge, probabilities):
            return {"name": name, "charge": charge,
                "laws": {h: {"0": str(1-Q(p)), "1": p} for h, p in probabilities.items()},
                "tv_radii": {h: "0" for h in targets}}
        menu = [experiment("target_read", 3, {"a1": "1/4", "a2": "1/4", "b1": "3/4", "b2": "3/4"}),
                experiment("index_read", 1, {"a1": "1/4", "a2": "3/4", "b1": "1/4", "b2": "3/4"})]
        return targets, menu

    def test_target_suite_selects_useful_access_without_reconstructing_mechanism(self):
        targets, menu = self.fixture()
        report = design_suite(targets, menu)
        self.assertEqual(report["selection_charge"], "3")
        self.assertEqual([x["name"] for x in report["selected"]], ["target_read"])
        self.assertEqual(report["total_records"], 192)
        decision = infer_target(targets, menu, report, {"target_read": {"0":144, "1":48}})
        self.assertEqual(decision["status"], "commit")
        self.assertEqual(decision["target"], "A")
        self.assertEqual(decision["retained_models"], ["a1", "a2"])
        absent = infer_target(targets, menu, report, {"target_read": {"0":96, "1":96}})
        self.assertEqual(absent["status"], "abstain_all_rejected")

    def test_insufficient_menu_and_caller_bound_risk(self):
        targets, menu = self.fixture()
        failed = design_suite(targets, menu[1:])
        self.assertEqual(failed["status"], "insufficient_robust_menu")
        self.assertEqual(infer_target(targets, menu[1:], failed, {})["status"], "abstain_insufficient_access")
        relaxed = design_suite(targets, menu, delta="1/5")
        with self.assertRaises(SuiteError):
            verify_suite(targets, menu, relaxed)
        self.assertTrue(verify_suite(targets, menu, relaxed, expected_delta="1/5")["valid"])


class IndependentPhysicalFormulaTests(unittest.TestCase):
    def test_hidden_gate_coefficient_by_direct_joint_matrix_products(self):
        # Unequal controller stationary weights avoid relying on a symmetric
        # two-state gate special case. No production physical module is used.
        q = sp.Matrix([[-2, sp.Rational(4,3), sp.Rational(2,3)],
                       [sp.Rational(2,3), -2, sp.Rational(4,3)],
                       [sp.Rational(4,3), sp.Rational(2,3), -2]])
        r = sp.Matrix([[-2,2],[1,-1]])
        l = sp.Matrix([[-1,1,0],[1,-1,0],[0,0,0]])
        g = sp.kronecker_product(r, sp.eye(3))+sp.kronecker_product(sp.eye(2), q)+sp.kronecker_product(sp.diag(0,1),l)
        stationary = sp.Matrix([[sp.Rational(1,9)]*3+[sp.Rational(2,9)]*3])
        d = sp.diag(1,0,0,1,0,0)
        one = sp.ones(6,1)
        self.assertEqual(stationary*g, sp.zeros(1,6))
        def coefficient(a,b):
            return ((stationary*d*(g**a)*d*(g**b)*d*one)[0] -
                    (stationary*d*(g**b)*d*(g**a)*d*one)[0])/sp.factorial(a)/sp.factorial(b)
        self.assertEqual(coefficient(1,3), sp.Rational(2,81))
        self.assertEqual(coefficient(3,1), -sp.Rational(2,81))
        for a,b in ((1,1),(1,2),(2,1),(2,2)):
            self.assertEqual(coefficient(a,b), 0)

    def test_quantum_stationary_state_and_activation_coefficients(self):
        # Independently construct the eight-dimensional Lindblad action.
        delta = sp.Symbol("delta", real=True)
        pauli = {"I": sp.eye(2), "X": sp.Matrix([[0,1],[1,0]]),
                 "Y": sp.Matrix([[0,-sp.I],[sp.I,0]]), "Z": sp.diag(1,-1)}
        def word(s):
            return sp.kronecker_product(*(pauli[x] for x in s))
        def local(a,site):
            return sp.kronecker_product(*[a if j == site else sp.eye(2) for j in range(3)])
        lowering = sp.Matrix([[0,1],[0,0]])
        h = sum((word(s)/2 for s in ("XXI","YYI","IXX","IYY")), sp.zeros(8))
        jumps = []
        for site,f in ((0,sp.Rational(1,2)+delta),(2,sp.Rational(1,2)-delta)):
            sm = local(lowering,site)
            jumps.extend(((1-f,sm),(f,sm.H)))
        def generator(rho):
            out = -sp.I*(h*rho-rho*h)
            for rate,a in jumps:
                out += rate*(a*rho*a.H-(a.H*a*rho+rho*a.H*a)/2)
            return out.applyfunc(sp.expand)
        rho = (word("III")+2*delta/5*(word("IIZ")-word("ZII"))
            +4*delta/5*(-word("IXY")+word("IYX")-word("XYI")+word("YXI"))
            -16*delta**2/25*(word("IZZ")+word("XIX")+word("YIY")+word("ZZI"))
            -4*delta**2/25*word("ZIZ")
            +8*delta**2/25*(-word("XYZ")+word("YXZ")+word("ZXY")-word("ZYX")))/8
        self.assertEqual(sp.trace(rho), 1)
        self.assertEqual(generator(rho), sp.zeros(8))
        px, pz = (sp.eye(8)+word("IXI"))/2, (sp.eye(8)+word("IZI"))/2
        xx = px*rho*px
        zxz = word("IXI")*pz*rho*pz*word("IXI")
        for state,projector,expected in (
            (xx,px,(sp.Rational(1,2),0,-sp.Rational(1,4)+4*delta**2/25)),
            (zxz,pz,(0,0,sp.Rational(1,2)-16*delta**2/25))):
            current = state
            for k in range(3):
                observed = sp.expand(sp.trace(projector*current)/sp.factorial(k))
                self.assertEqual(sp.expand(observed-expected[k]),0)
                current = generator(current)


class IndependentAdvancedAPITests(unittest.TestCase):
    def test_linear_target_not_total_parameter_recovery(self):
        self.assertTrue(identify_linear_target([[1,1]],[1,1])["identifiable"])
        blocked = identify_linear_target([[1,1]],[1,0])
        self.assertFalse(blocked["identifiable"])
        left, right = [[Q(v) for v in x] for x in blocked["indistinguishable_positive_parameters"]]
        self.assertEqual(sum(left),sum(right))
        self.assertNotEqual(left[0],right[0])
        self.assertTrue(all(x>0 for x in left+right))

    def test_minimax_exact_primal_dual_and_false_precision(self):
        f = [[1,1],["1/2",1],[2,1]]
        result = minimax_certificate(f,[1,0],["1/100"]*3)
        self.assertEqual(result["absolute_error"],"1/75")
        w = [Q(v) for v in result["weights"]]
        h = [Q(v) for v in result["dual_direction"]]
        # Separate exact arithmetic for feasibility and equality of values.
        self.assertEqual([sum(w[i]*Q(f[i][j]) for i in range(3)) for j in range(2)],[1,0])
        self.assertLessEqual(max(abs(sum(Q(row[j])*h[j] for j in range(2))) for row in f),Q(1,100))
        self.assertEqual(sum(abs(x)/100 for x in w),h[0])
        self.assertTrue(verify_minimax_certificate(f,[1,0],["1/100"]*3,result)["verified"])
        false = deepcopy(result)
        false["absolute_error"] = "0"
        with self.assertRaises(ValueError):
            verify_minimax_certificate(f,[1,0],["1/100"]*3,false)
        with self.assertRaises(ValueError):
            verify_minimax_certificate(f,[1,0],["1/200"]*3,result)

    def test_counting_cap_is_not_mean_or_an_unqualified_total_energy(self):
        result = count_cap(150,"1/100")
        self.assertEqual(result["count_cap"],189)
        self.assertEqual(Q(result["mu"]),150)
        self.assertEqual(Q(result["beta"]),Q(1,100))
        self.assertTrue(verify_count_cap(result))
        self.assertEqual(declared_energy_cap(result,"3/2",7),Q(581,2))
        false = deepcopy(result)
        false["count_cap"] = 150
        self.assertFalse(verify_count_cap(false))
        with self.assertRaises(ValueError):
            declared_energy_cap(false,"3/2",7)
        with self.assertRaises(ValueError):
            declared_energy_cap(result,-1,7)
        zero = count_cap(0,"1/100")
        self.assertEqual(declared_energy_cap(zero,99,7),7)

    def test_signed_algebra_does_not_validate_physical_probabilities(self):
        signed = LinearRepresentation([1], {("read","0"):[[-1]], ("read","1"):[[2]]}, [1])
        self.assertEqual(signed.word_probability((("read","0"),)),-1)
        with self.assertRaises(ValueError):
            experiment_law(signed,["read"])
        with self.assertRaises(ValueError):
            RationalInstrument([1],{"read":{"0":[[-1]],"1":[[2]]}})


if __name__ == "__main__":
    unittest.main()
