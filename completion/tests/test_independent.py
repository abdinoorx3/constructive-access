"""Independent adversarial checks; producer optimizers are not used as oracles.

Run: python tests/test_independent.py
Exact row elimination, exhaustive subsets/cuts and Taylor enclosures below are
separate implementations from the producing modules. Numerical checks, when
present, are supplementary and explicitly labelled.
"""
from __future__ import annotations

from fractions import Fraction as F
from itertools import combinations, product
import importlib.util
import hashlib
import json
from math import factorial, comb
from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = {}
sys.path.insert(0, str(ROOT / "code"))


def module(name):
    spec = importlib.util.spec_from_file_location("redteam_" + name, ROOT / "code" / (name + ".py"))
    obj = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = obj
    spec.loader.exec_module(obj)
    return obj


def rank(rows, width):
    a = [[F(x) for x in row] for row in rows]
    r = 0
    for c in range(width):
        pivot = next((i for i in range(r, len(a)) if a[i][c]), None)
        if pivot is None:
            continue
        a[r], a[pivot] = a[pivot], a[r]
        den = a[r][c]
        a[r] = [x / den for x in a[r]]
        for i in range(r + 1, len(a)):
            mul = a[i][c]
            if mul:
                a[i] = [x - mul*y for x, y in zip(a[i], a[r])]
        r += 1
        if r == len(a):
            break
    return r


def row_model(graph, sensed=(), totals=()):
    keys = [e.key for e in graph.edges]
    rows = [[int(e.head == v) - int(e.tail == v) for e in graph.edges] for v in graph.vertices]
    for key in sensed:
        rows.append([int(k == key) for k in keys])
    for u, v in totals:
        rows.append([1 if (e.tail, e.head) == (u, v) else -1 if (e.tail, e.head) == (v, u) else 0 for e in graph.edges])
    return rows


def identified_exact(graph, sensed=(), totals=()):
    m = len(graph.edges)
    rows = row_model(graph, sensed, totals)
    r = rank(rows, m)
    answer = []
    for k, edge in enumerate(graph.edges):
        target = [int(i == k) for i in range(m)]
        if rank(rows + [target], m) == r:
            answer.append(edge.key)
    return m-r, answer


def subsets(items):
    for bits in product((False, True), repeat=len(items)):
        yield tuple(x for x, bit in zip(items, bits) if bit)


def matmul(a, b):
    return [[sum((x*y for x, y in zip(row, col)), F(0)) for col in zip(*b)] for row in a]


def exponential_enclosure(a, t, terms=80):
    """Exact Taylor matrix enclosure with geometric operator-norm tail."""
    n = len(a)
    a = [[F(x)*t for x in row] for row in a]
    norm = max(sum(abs(x) for x in row) for row in a)
    if norm >= terms + 2:
        raise ValueError("Insufficient terms for geometric remainder")
    part = [[F(i == j) for j in range(n)] for i in range(n)]
    total = [row[:] for row in part]
    for k in range(1, terms+1):
        part = [[x/k for x in row] for row in matmul(part, a)]
        total = [[x+y for x, y in zip(row, add)] for row, add in zip(total, part)]
    tail = norm**(terms+1) / factorial(terms+1) / (1 - norm/F(terms+2))
    return total, tail


def partitions(items):
    if not items:
        yield ()
        return
    first, rest = items[0], items[1:]
    for parts in partitions(rest):
        yield ((first,),) + parts
        for k in range(len(parts)):
            yield parts[:k] + ((first,) + parts[k],) + parts[k+1:]


def refines(fine, coarse):
    return all(any(set(block).issubset(parent) for parent in coarse) for block in fine)


def is_stable(model, blocks):
    for matrices in model.values():
        for m in matrices.values():
            for block in blocks:
                for destination in blocks:
                    totals = {sum(m[x][y] for y in destination) for x in block}
                    if len(totals) != 1:
                        return False
    return True


def binomial_law(n,p):
    return [F(comb(n,k))*p**k*(1-p)**(n-k) for k in range(n+1)]


def exact_three_arm_rejection(design,p0,p1,p2):
    n = design["records_per_arm"]
    base,early,late = (binomial_law(n,p) for p in (p0,p1,p2))
    t1,t2 = design["early_threshold"],design["late_threshold"]
    result = F(0)
    for k,pk in enumerate(base):
        early_pass = sum((p for j,p in enumerate(early) if F(k-j,n)>t1),F(0))
        late_pass = sum((p for j,p in enumerate(late) if F(j-k,n)>t2),F(0))
        result += pk*early_pass*late_pass
    return result


class IndependentReview(unittest.TestCase):
    def test_three_arm_pulse_test_exact_binomial_laws(self):
        p = module("pulse_test")
        contract = dict(gap_early=F(2,5),gap_late=F(2,5),
                        eta_baseline=F(1,100),eta_early=F(1,100),eta_late=F(1,100),
                        alpha=F(1,5),beta=F(1,5))
        design = p.plan(**contract)
        self.assertTrue(p.verify_plan(design,**contract))
        nulls = ((F(51,100),F(49,100),F(91,100)),
                 (F(49,100),F(9,100),F(51,100)),
                 (F(1,2),F(1,2),F(1,2)))
        probabilities = []
        for triple in nulls:
            probability = exact_three_arm_rejection(design,*triple)
            self.assertLessEqual(probability,contract["alpha"])
            probabilities.append(float(probability))
        # Nominal (p0,p1,p2)=(.5,.1,.9). Both extreme common-baseline shifts
        # and the adverse early/late shifts are included, with the same fixed
        # nuisance throughout each experiment.
        powers = []
        for p0 in (F(49,100),F(51,100)):
            probability = exact_three_arm_rejection(design,p0,F(11,100),F(89,100))
            self.assertGreaterEqual(probability,1-contract["beta"])
            powers.append(float(probability))
        changed = dict(design)
        changed["records_per_arm"] = 1; changed["total_records"] = 3
        self.assertFalse(p.verify_plan(changed,**contract))
        with self.assertRaises(ValueError):
            p.plan(F(4,5),F(4,5))
        RESULTS["three_arm_exact_binomial_checks"] = {
            "records_per_arm":design["records_per_arm"],
            "null_rejection_probabilities":probabilities,
            "adverse_calibration_powers":powers,
            "alpha":str(contract["alpha"]),"beta":str(contract["beta"]),
            "method":"exact rational binomial convolution conditional on shared baseline count",
        }

    def test_short_pulse_bias_and_variance_against_exact_ctmc(self):
        p = module("physical_access")
        native_current = F(13,10)
        q = [[-F(41,10),F(4),F(1,10)],
             [F(1,10),-F(41,10),F(4)],
             [F(4),F(1,10),-F(41,10)]]
        q[0][1] -= 2; q[0][0] += 2
        q[1][0] -= F(1,20); q[1][1] += F(1,20)
        pi = [[F(1,3)]*3]
        examples = []
        for error in (F(1,5),F(1,10),F(1,20)):
            design = p.short_pulse_current_plan(error,native_current,F(41,10),F(1,2),F(1,20),F(1,3))
            t = F(design["duration"])
            evolution,radius = exponential_enclosure(q,t)
            mean = F(1,3)-matmul(pi,evolution)[0][1]
            second = 2*F(1,3)*(1-evolution[1][1])-mean
            self.assertLessEqual(abs(mean/(t/2)-native_current)+radius/(t/2),error/2)
            self.assertLessEqual(second+(1+2*F(1,3))*radius,F(design["variance_cap"]))
            examples.append({"error":str(error),"duration":str(t),"records":design["records"]})
        RESULTS["short_pulse_exact_ctmc_bias_variance_checks"] = examples

    def test_floor_energy_partition_reduction(self):
        r = module("robust_completion")
        count = 0
        for length in range(1,5):
            for values in product(range(1,5),repeat=length):
                expected = any(2*sum(s) == sum(values) for s in subsets(values))
                instance = r.partition_energy_instance(list(values))
                actual = r.enumerate_partition_design(instance)
                self.assertEqual(actual["partition_exists"],expected)
                self.assertEqual(F(actual["best_energy"]) <= F(instance["threshold_energy"]),expected)
                count += 1
        RESULTS["partition_reduction_exhaustive_small_instances"] = count

    def test_energy_support_bounds_against_exhaustive_cuts(self):
        g = module("graph_access"); r = module("robust_completion")
        rng = random.Random(88013)
        count = 0
        for _ in range(35):
            pairs = list(combinations(("A","B","C","D","E"),2))
            edges = tuple(g.Edge("e"+str(k),*p) for k,p in enumerate(rng.sample(pairs,rng.randrange(2,9))))
            graph = g.Graph(("A","B","C","D","E"),edges)
            target = edges[0]
            q = {e.key:F(rng.randrange(1,5)) for e in edges[1:]}
            floors = {e.key:F(rng.randrange(3),5) for e in edges[1:]}
            accuracy = F(3)
            optima = []
            for selected in subsets(graph.vertices):
                selected = set(selected)
                if target.tail not in selected or target.head in selected:
                    continue
                crossing = [e.key for e in edges[1:] if (e.tail in selected)!=(e.head in selected)]
                total_q = sum((q[k] for k in crossing),F(0))
                total_floor = sum((floors[k] for k in crossing),F(0))
                if not crossing:
                    optima.append(F(0))
                elif total_floor < accuracy:
                    optima.append(total_q**3/(accuracy-total_floor)**2)
            optimum = min(optima)
            exact = r.enumerate_floor_design(graph,target.key,q,floors,accuracy)
            self.assertEqual(F(exact["total_energy"]),optimum)
            for support in (F(1,4),F(1),F(4)):
                cert = r.floor_energy_bound(graph,target.key,q,floors,accuracy,support)
                verified = r.verify_floor_energy_bound(graph,target.key,q,floors,accuracy,cert)
                self.assertLessEqual(F(verified["lower_energy"]),optimum)
                if verified["upper_energy"] is not None:
                    self.assertGreaterEqual(F(verified["upper_energy"]),optimum)
                count += 1
        RESULTS["energy_support_vs_exhaustive_cuts"] = count

    def test_interface_completion_against_all_partitions(self):
        c = module("composition")
        parts = tuple(partitions((0,1,2)))
        count = 0
        for choices in product(range(6), repeat=3):
            model = {"a": {"0": [[F(0)]*3 for _ in range(3)], "1": [[F(0)]*3 for _ in range(3)]}}
            for x, value in enumerate(choices):
                model["a"][str(value//3)][x][value%3] = F(1)
            for initial in parts:
                answer = c.complete_partition(model, initial)
                found = answer["partition"]
                self.assertTrue(refines(found, initial))
                self.assertTrue(is_stable(model, found))
                for candidate in parts:
                    if refines(candidate, initial) and is_stable(model, candidate):
                        self.assertTrue(refines(candidate, found))
                self.assertTrue(c.check_partition_certificate(model, initial, answer))
                count += 1
        RESULTS["interface_completion_exhaustive_partition_cases"] = count

    def test_correlated_preparations_defeat_marginal_composition(self):
        # Every local one-bit marginal is fair under both preparations, while
        # the joint readout laws are disjoint. Initial interface data must be joint.
        left = {"00":F(1,2),"11":F(1,2)}
        right = {"01":F(1,2),"10":F(1,2)}
        for index in (0,1):
            for bit in "01":
                self.assertEqual(sum(p for word,p in left.items() if word[index]==bit),
                                 sum(p for word,p in right.items() if word[index]==bit))
        tv = sum(abs(left.get(k,0)-right.get(k,0)) for k in set(left)|set(right))/2
        self.assertEqual(tv,1)
        RESULTS["correlated_preparation_negative_control"] = "equal local marginals; disjoint joint records"

    def test_mixed_graph_against_exact_ranks(self):
        g = module("graph_access")
        count = 0
        # Every loop-free multigraph on three vertices with up to two channels
        # per pair, all exact aggregate subsets, and all individual sensor sets.
        pairs = (("A", "B"), ("A", "C"), ("B", "C"))
        for multiplicities in product(range(3), repeat=3):
            edges = tuple(g.Edge(str(i)+"_"+str(k), *(pair if k % 2 == 0 else pair[::-1]))
                          for i, (pair, m) in enumerate(zip(pairs, multiplicities)) for k in range(m))
            graph = g.Graph(("A", "B", "C"), edges)
            present = tuple(p for p, m in zip(pairs, multiplicities) if m)
            for totals in subsets(present):
                for sensed in subsets(tuple(e.key for e in edges)):
                    expected_dim, expected_targets = identified_exact(graph, sensed, totals)
                    actual = g.mixed_observability(graph, sensed, totals)
                    self.assertEqual(actual["ambiguity_dimension"], expected_dim)
                    self.assertEqual(actual["identifiable_channels"], expected_targets)
                    count += 1
        RESULTS["mixed_graph_exact_rank_cases"] = count

    def test_completion_against_exhaustive_sensor_subsets(self):
        g = module("graph_access")
        rng = random.Random(127003)
        count = 0
        for _ in range(65):
            pairs = list(combinations(("A", "B", "C", "D"), 2))
            edges = tuple(g.Edge("e"+str(k), *(p if rng.randrange(2) else p[::-1]))
                          for k, p in enumerate(rng.choices(pairs, k=rng.randrange(1, 8))))
            graph = g.Graph(("A", "B", "C", "D"), edges)
            totals = tuple(p for p in graph.bundles() if rng.randrange(2))
            costs = {e.key: F(rng.randrange(6), rng.randrange(1, 4)) for e in edges}
            target = edges[0].key
            best_all = None
            best_target = None
            for sensed in subsets(tuple(costs)):
                dim, identified = identified_exact(graph, sensed, totals)
                price = sum((costs[k] for k in sensed), F(0))
                if dim == 0:
                    best_all = price if best_all is None else min(best_all, price)
                if target not in sensed and target in identified:
                    best_target = price if best_target is None else min(best_target, price)
            a = g.all_target_completion(graph, costs, totals)
            b = g.target_completion(graph, target, {k:v for k,v in costs.items() if k != target}, totals)
            self.assertEqual(F(a["cost"]), best_all)
            self.assertEqual(F(b["cost"]), best_target)
            self.assertEqual(identified_exact(graph, a["sensors"], totals)[0], 0)
            self.assertIn(target, identified_exact(graph, b["sensors"], totals)[1])
            count += 1
        RESULTS["completion_exhaustive_sensor_cases"] = count

    def test_cut_against_exhaustive_partitions_and_tampering(self):
        g = module("graph_access")
        rng = random.Random(63111)
        count = 0
        for _ in range(160):
            pairs = list(combinations(("A", "B", "C", "D", "E"), 2))
            edges = tuple(g.Edge("e"+str(k), *p) for k, p in enumerate(rng.choices(pairs, k=rng.randrange(1, 11))))
            graph = g.Graph(("A", "B", "C", "D", "E"), edges)
            target = edges[0]
            caps = {e.key:F(rng.randrange(8), rng.randrange(1,5)) for e in edges[1:] if rng.randrange(3)}
            feasible = []
            for chosen in subsets(graph.vertices):
                chosen = set(chosen)
                if target.tail not in chosen or target.head in chosen:
                    continue
                crossing = [e.key for e in edges[1:] if (e.tail in chosen) != (e.head in chosen)]
                if all(k in caps for k in crossing):
                    feasible.append(sum((caps[k] for k in crossing), F(0)))
            actual = g.target_cut(graph, target.key, caps)
            self.assertTrue(g.verify_target_cut(graph, target.key, caps, actual))
            if feasible:
                self.assertEqual(actual["status"], "identified")
                self.assertEqual(F(actual["value"]), min(feasible))
                bad = json.loads(json.dumps(actual)); bad["value"] = str(F(actual["value"])+1)
            else:
                self.assertEqual(actual["status"], "unidentifiable")
                bad = json.loads(json.dumps(actual)); bad["ambiguity"][target.key] = "0"
            self.assertFalse(g.verify_target_cut(graph, target.key, caps, bad))
            count += 1
        RESULTS["cut_exhaustive_partition_cases"] = count

    def test_physical_realization_and_hidden_parallel_failure(self):
        g = module("graph_access")
        graph = g.Graph(("A","B"), (g.Edge("one","A","B"),g.Edge("two","B","A")))
        currents = {"one":F(3,2), "two":F(3,2)}
        pi = {"A":F(1,3), "B":F(2,3)}
        rates = g.realize_currents(graph, currents, pi)
        for e in graph.edges:
            recovered = pi[e.tail]*F(rates[e.key]["forward"])-pi[e.head]*F(rates[e.key]["reverse"])
            self.assertEqual(recovered, currents[e.key])
        actual = g.mixed_observability(graph, known_aggregates=(("A","B"),))
        self.assertEqual(actual["ambiguity_dimension"], 1)
        self.assertEqual(actual["identifiable_channels"], [])
        with self.assertRaises(ValueError):
            g.realize_currents(graph, currents, pi, {"one":F(1,2),"two":F(1,2)})
        RESULTS["parallel_channel_negative_control"] = "passed"

    def test_nonstationary_and_global_pulse_counterexamples(self):
        q = [[F(-3),F(2),F(1)],[F(1),F(-3),F(2)],[F(2),F(1),F(-3)]]
        channel = [[F(-2),F(2),F(0)],[F(1),F(-1),F(0)],[F(0),F(0),F(0)]]
        qr = [[a-b/2 for a,b in zip(row,ch)] for row,ch in zip(q,channel)]
        pi = [[F(1,3)]*3]
        self.assertEqual(matmul(pi,q)[0], [0,0,0])
        self.assertEqual(matmul(pi,qr)[0][1], -F(1,6))
        self.assertEqual(qr[1][1]-q[1][1], F(1,2))
        global_q = [[x/2 for x in row] for row in q]
        self.assertEqual(matmul(pi,global_q)[0], [0,0,0])
        RESULTS["pulse_failure_controls"] = {
            "native_current": "1/3", "stationary_endpoint_derivative": "-1/6",
            "nonstationary_matched_control_derivative": "1/2",
            "global_slowing_stationary_response": "identically zero",
        }

    def test_coarse_classical_detector_can_cross_baseline(self):
        # Physics-agent proposed negative control, checked through an independent
        # rational matrix exponential rather than its numerical implementation.
        q = [[F(0)]*4 for _ in range(4)]
        for i,j,a,b in ((0,1,2,1),(0,2,10,1),(1,3,10,1),(2,3,1,3)):
            q[i][j] += a; q[j][i] += b
            q[i][i] -= a; q[j][j] -= b
        pi = [[F(7,121),F(4,121),F(80,121),F(30,121)]]
        self.assertEqual(matmul(pi,q)[0],[0,0,0,0])
        qr = [row[:] for row in q]
        qr[0][1] -= 1; qr[0][0] += 1
        qr[1][0] -= F(1,2); qr[1][1] += F(1,2)
        values = []
        for time in (F(1,10),F(1,2)):
            p, error = exponential_enclosure(qr,time,terms=100)
            evolved = matmul(pi,p)[0]
            coarse = evolved[1]+evolved[2]-pi[0][1]-pi[0][2]
            singleton = evolved[1]-pi[0][1]
            self.assertLess(singleton+error,0)
            values.append((coarse,error))
        self.assertLess(values[0][0]+2*values[0][1],0)
        self.assertGreater(values[1][0]-2*values[1][1],0)
        RESULTS["coarse_classical_detector_counterexample"] = {
            "native_current":"10/121", "detector":"union of states 1 and 2",
            "early_time":"1/10", "early_shift":float(values[0][0]),
            "late_time":"1/2", "late_shift":float(values[1][0]),
            "singleton_destination_remains_negative":True,
            "signs_certified_by":"100-term rational Taylor enclosures",
            "consequence":"Pulse sign crossing alone cannot certify quantumness against coarse or hidden-state classical alternatives",
        }

    def test_classical_postpulse_readout_delay_can_reverse_sign(self):
        q = [[-F(41,10),F(4),F(1,10)],
             [F(1,10),-F(41,10),F(4)],
             [F(4),F(1,10),-F(41,10)]]
        qr = [row[:] for row in q]
        qr[0][1] -= 2; qr[0][0] += 2
        qr[1][0] -= F(1,20); qr[1][1] += F(1,20)
        pi = [[F(1,3)]*3]
        pulse, e1 = exponential_enclosure(qr,F(1,10))
        delay, e2 = exponential_enclosure(q,F(2,5))
        immediate = matmul(pi,pulse)[0][1]-F(1,3)
        late = matmul(matmul(pi,pulse),delay)[0][1]-F(1,3)
        error = e1+e2+e1*e2
        self.assertLess(immediate+e1,0)
        self.assertGreater(late-error,0)
        RESULTS["postpulse_delay_counterexample"] = {
            "native_clockwise_rate":"4", "native_counterclockwise_rate":"1/10",
            "native_current":"13/10", "suppression_factor":"1/2",
            "pulse_duration":"1/10", "native_evolution_delay":"2/5",
            "immediate_shift":float(immediate), "delayed_shift":float(late),
            "signs_certified_by":"Exact rational matrix Taylor enclosures and stochastic product error bound",
            "consequence":"Suppression must persist until readout, or a delay must be covered by the model discrepancy budget",
        }

    def test_physical_quantum_sign_reversal_exact_enclosure(self):
        # H=X; channel1(up,down)=(3/100,1/100), channel2=(1/100,3/100).
        # Native state I/2; channel1 slowed by 1/2. z=p_A-p_B and y chosen
        # so z'=-Γz+2y+f, y'=-2z-(Γ/2)y, Γ=3/50, f=1/100.
        gamma = F(3,50); forcing = F(1,100)
        a = [[-gamma,F(2),forcing],[-F(2),-gamma/2,F(0)],[F(0),F(0),F(0)]]
        late, late_error = exponential_enclosure(a,F(2))
        early, early_error = exponential_enclosure(a,F(1,2))
        self.assertLess(late[0][2]+late_error,0)
        self.assertGreater(early[0][2]-early_error,0)
        # Endpoint B displacement = -z/2. Native channel A→B current +1/100.
        lower = -(late[0][2]+late_error)/2
        upper = -(late[0][2]-late_error)/2
        RESULTS["quantum_counterexample"] = {
            "native_channel_current":"1/100", "suppression_factor":"1/2",
            "time":"2", "endpoint_B_displacement_positive":True,
            "lower_exact":str(lower), "upper_exact":str(upper),
            "lower_float":float(lower),"upper_float":float(upper),
            "enclosure_method":"80-term rational Taylor sum; geometric induced-infinity-norm tail",
            "scope":"Physical coherent counterexample to a universal classical pulse sign law; not a device-independent quantum witness",
        }
        # Larger proposed benchmark signal: κ=1/10, r=1/10, H=X.
        gamma = F(11,25); forcing = F(9,50)
        a = [[-gamma,F(2),forcing],[-F(2),-gamma/2,F(0)],[F(0),F(0),F(0)]]
        preferred_late, le = exponential_enclosure(a,F(5,2))
        preferred_early, ee = exponential_enclosure(a,F(1,2))
        self.assertLess(preferred_late[0][2]+le,0)
        self.assertGreater(preferred_early[0][2]-ee,0)
        RESULTS["quantum_preferred_benchmark"] = {
            "kappa":"1/10", "r":"1/10", "g":"1",
            "native_channel_current":"1/10",
            "early_time":"1/2", "early_endpoint_B_shift":float(-preferred_early[0][2]/2),
            "late_time":"5/2", "late_endpoint_B_shift":float(-preferred_late[0][2]/2),
            "late_lower_exact":str(-(preferred_late[0][2]+le)/2),
            "late_upper_exact":str(-(preferred_late[0][2]-le)/2),
            "early_strictly_negative":True,"late_strictly_positive":True,
        }


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(IndependentReview)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    RESULTS["tests_run"] = result.testsRun
    RESULTS["failures"] = len(result.failures)
    RESULTS["errors"] = len(result.errors)
    RESULTS["success"] = result.wasSuccessful()
    RESULTS["producer_source_sha256"] = {
        name: hashlib.sha256((ROOT/"code"/(name+".py")).read_bytes()).hexdigest()
        for name in ("graph_access","composition","robust_completion","physical_access","quantum_completion","pulse_test")
    }
    output = ROOT / "results" / "redteam.json"
    output.write_text(json.dumps(RESULTS,indent=2,sort_keys=True)+"\n")
    sys.exit(not result.wasSuccessful())
