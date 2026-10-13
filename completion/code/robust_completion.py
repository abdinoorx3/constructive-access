"""Exact robust access completion and conditional acquisition-energy design.

The linear minimax construction extends the exact vertex enumeration in
application_audit/code/intervention_criterion.py to conservation constraints
and fixed, shared calibration nuisance.  It implements established LP duality,
not a new general optimization algorithm.  Certificate checks use Fraction
identities only and do not call synthesis.

Energy formulas assume the separately calibrated deterministic contract
epsilon_e = delta_e + sqrt(q_e**3 / E_e).  Energy is acquisition energy only.
They do not assert that arbitrary physical detectors obey this contract.
"""
from fractions import Fraction as Q
from itertools import combinations, product
import json
from pathlib import Path


def rational(value):
    if isinstance(value, bool) or not isinstance(value, (int, str, Q)):
        raise ValueError("Exact integer, fraction, or rational string required")
    return Q(value)


def dot(a, b):
    if len(a) != len(b):
        raise ValueError("Vector dimensions differ")
    return sum((x*y for x, y in zip(a, b)), Q(0))


def mv(matrix, vector):
    return [dot(row, vector) for row in matrix]


def rref(matrix, columns):
    a = [[rational(x) for x in row] for row in matrix]
    if any(len(row) < columns for row in a):
        raise ValueError("Short matrix row")
    pivots, row = [], 0
    for column in range(columns):
        chosen = next((i for i in range(row, len(a)) if a[i][column]), None)
        if chosen is None:
            continue
        a[row], a[chosen] = a[chosen], a[row]
        scale = a[row][column]
        a[row] = [x/scale for x in a[row]]
        for i in range(len(a)):
            if i != row and a[i][column]:
                scale = a[i][column]
                a[i] = [x-scale*y for x, y in zip(a[i], a[row])]
        pivots.append(column)
        row += 1
    return a, pivots


def solve(matrix, rhs, columns):
    if len(matrix) != len(rhs) or any(len(row) != columns for row in matrix):
        raise ValueError("Linear-system dimensions differ")
    reduced, pivots = rref([list(row)+[rational(y)] for row, y in zip(matrix, rhs)], columns)
    if any(not any(row[:columns]) and row[columns] for row in reduced):
        return None
    result = [Q(0)]*columns
    for row, column in enumerate(pivots):
        result[column] = reduced[row][columns]
    return result


def normalize_model(model):
    b = [rational(x) for x in model["target"]]
    if not b:
        raise ValueError("Positive current dimension required")
    h = [[rational(x) for x in row] for row in model["observation"]]
    conservation = [[rational(x) for x in row] for row in model["conservation"]]
    if any(len(row) != len(b) for row in h+conservation):
        raise ValueError("Current dimension mismatch")
    nuisance = [[rational(x) for x in row] for row in model.get("nuisance", [[] for _ in h])]
    nuisance_dimension = len(nuisance[0]) if nuisance else 0
    if len(nuisance) != len(h) or any(len(row) != nuisance_dimension for row in nuisance):
        raise ValueError("Nuisance dimensions differ")
    epsilon = [rational(x) for x in model["epsilon"]]
    if len(epsilon) != len(h) or any(x < 0 for x in epsilon):
        raise ValueError("Nonnegative error per observation required")
    return h, conservation, nuisance, b, epsilon


def verify_robust_certificate(model, certificate):
    """Verify minimax primal/dual or an unbounded ambiguity witness exactly."""
    h, conservation, nuisance, b, epsilon = normalize_model(model)
    n, m = len(b), len(h)
    k = len(nuisance[0]) if nuisance else 0
    direction = [rational(x) for x in certificate["current_direction"]]
    bias = [rational(x) for x in certificate["nuisance_direction"]]
    if len(direction) != n or len(bias) != k:
        raise ValueError("Dual certificate dimensions differ")
    if any(mv(conservation, direction)):
        raise ValueError("Dual direction violates conservation")
    image = [dot(row, direction)+dot(a, bias) for row, a in zip(h, nuisance)]
    if not certificate["finite"]:
        if any(image) or not dot(b, direction):
            raise ValueError("Invalid unbounded ambiguity direction")
        return {"verified": True, "finite": False, "absolute_error": "infinity"}
    weights = [rational(x) for x in certificate["weights"]]
    potential = [rational(x) for x in certificate["conservation_multiplier"]]
    if len(weights) != m or len(potential) != len(conservation):
        raise ValueError("Primal certificate dimensions differ")
    reconstructed = [sum((weights[i]*h[i][j] for i in range(m)), Q(0))+
                     sum((potential[i]*conservation[i][j] for i in range(len(conservation))), Q(0))
                     for j in range(n)]
    if reconstructed != b:
        raise ValueError("Estimator fails target identity")
    if any(sum((weights[i]*nuisance[i][j] for i in range(m)), Q(0)) for j in range(k)):
        raise ValueError("Estimator does not cancel shared nuisance")
    if any(abs(x) > eps for x, eps in zip(image, epsilon)):
        raise ValueError("Dual record perturbation outside calibrated box")
    upper = sum((abs(w)*eps for w, eps in zip(weights, epsilon)), Q(0))
    lower = dot(b, direction)
    if upper != lower or rational(certificate["absolute_error"]) != upper:
        raise ValueError("Primal and dual values differ")
    return {"verified": True, "finite": True, "absolute_error": str(upper),
            "least_favorable_targets": [str(-lower), str(lower)],
            "shared_record": ["0"]*m,
            "scope": "Full real conserved-current space, unrestricted fixed nuisance, full deterministic coordinatewise error box"}


def robust_target_certificate(model):
    """Small exact LP by support/vertex enumeration; no scaling claim."""
    h, conservation, nuisance, b, epsilon = normalize_model(model)
    k = len(nuisance[0]) if nuisance else 0
    f = [row+a for row, a in zip(h, nuisance)] + [row+[Q(0)]*k for row in conservation]
    target, bounds = b+[Q(0)]*k, epsilon+[Q(0)]*len(conservation)
    n, m = len(target), len(f)
    reduced, pivots = rref(f, n)
    transposed = [[f[i][j] for i in range(m)] for j in range(n)]
    if solve(transposed, target, m) is None:
        for free in range(n):
            if free in pivots:
                continue
            direction = [Q(0)]*n
            direction[free] = Q(1)
            for row, pivot in enumerate(pivots):
                direction[pivot] = -reduced[row][free]
            if dot(target, direction):
                certificate = {"finite": False, "current_direction": list(map(str, direction[:len(b)])),
                               "nuisance_direction": list(map(str, direction[len(b):]))}
                certificate["verification"] = verify_robust_certificate(model, certificate)
                return certificate
        raise AssertionError("Missing kernel witness")
    rank = len(pivots)
    primal = []
    for size in range(rank+1):
        for support in combinations(range(m), size):
            if len(rref([f[i] for i in support], n)[1]) != size:
                continue
            small = solve([[f[i][j] for i in support] for j in range(n)], target, size)
            if small is None:
                continue
            w = [Q(0)]*m
            for index, weight in zip(support, small):
                w[index] = weight
            primal.append((sum((abs(x)*e for x, e in zip(w, bounds)), Q(0)), tuple(w)))
    value, weights = min(primal)
    if rank:
        a = [[row[j] for j in pivots] for row in f]
        dual = []
        for active in combinations(range(m), rank):
            selected = [a[i] for i in active]
            if len(rref(selected, rank)[1]) != rank:
                continue
            for signs in product((-1, 1), repeat=rank):
                candidate = solve(selected, [sign*bounds[i] for sign, i in zip(signs, active)], rank)
                if any(abs(x) > e for x, e in zip(mv(a, candidate), bounds)):
                    continue
                direction = [Q(0)]*n
                for j, x in zip(pivots, candidate):
                    direction[j] = x
                dual.append((dot(target, direction), tuple(direction)))
        dual_value, direction = max(dual)
        if value != dual_value:
            raise AssertionError("Exact primal/dual gap")
    else:
        direction = [Q(0)]*n
    certificate = {"finite": True, "weights": list(map(str, weights[:len(h)])),
                   "conservation_multiplier": list(map(str, weights[len(h):])),
                   "current_direction": list(map(str, direction[:len(b)])),
                   "nuisance_direction": list(map(str, direction[len(b):])),
                   "absolute_error": str(value)}
    certificate["verification"] = verify_robust_certificate(model, certificate)
    return certificate


def allocate_cut_energy(q, floor, accuracy):
    """Exact globally optimal allocation ON a supplied cut, q = a**(1/3)>0."""
    q, floor, accuracy = list(map(rational, q)), list(map(rational, floor)), rational(accuracy)
    if len(q) != len(floor) or any(x <= 0 for x in q) or any(x < 0 for x in floor) or accuracy <= 0:
        raise ValueError("Positive q, positive accuracy and nonnegative floors required")
    total_q, total_floor = sum(q, Q(0)), sum(floor, Q(0))
    if not q:
        return {"feasible": True, "q": [], "floor": [], "accuracy": str(accuracy),
                "errors": [], "energies": [], "total_energy": "0"}
    if accuracy <= total_floor:
        return {"feasible": False, "q": list(map(str, q)), "floor": list(map(str, floor)),
                "accuracy": str(accuracy), "calibration_floor": str(total_floor)}
    slack = accuracy-total_floor
    energies = [total_q**2*x/slack**2 for x in q]
    errors = [d+slack*x/total_q for d, x in zip(floor, q)]
    return {"feasible": True, "q": list(map(str, q)), "floor": list(map(str, floor)),
            "accuracy": str(accuracy), "errors": list(map(str, errors)),
            "energies": list(map(str, energies)), "total_energy": str(total_q**3/slack**2)}


def verify_cut_energy(certificate, expected_q=None, expected_floor=None, expected_accuracy=None):
    """Check a fixed-cut optimum without a numerical optimizer or square roots."""
    q = list(map(rational, certificate["q"]))
    floor = list(map(rational, certificate["floor"]))
    accuracy = rational(certificate["accuracy"])
    if expected_q is not None and q != list(map(rational, expected_q)):
        raise ValueError("Unexpected precision parameters")
    if expected_floor is not None and floor != list(map(rational, expected_floor)):
        raise ValueError("Unexpected calibration floors")
    if expected_accuracy is not None and accuracy != rational(expected_accuracy):
        raise ValueError("Unexpected target accuracy")
    if len(q) != len(floor) or any(x <= 0 for x in q) or any(x < 0 for x in floor) or accuracy <= 0:
        raise ValueError("Invalid calibration contract")
    total_q, slack = sum(q, Q(0)), accuracy-sum(floor, Q(0))
    if not certificate["feasible"]:
        if not q or slack > 0:
            raise ValueError("False infeasibility claim")
        if rational(certificate["calibration_floor"]) != sum(floor, Q(0)):
            raise ValueError("Wrong calibration floor")
        return {"verified": True, "feasible": False}
    energies = list(map(rational, certificate["energies"]))
    errors = list(map(rational, certificate["errors"]))
    if len(energies) != len(q) or len(errors) != len(q):
        raise ValueError("Allocation dimensions differ")
    if not q:
        if rational(certificate["total_energy"]):
            raise ValueError("Nonzero energy for empty cut")
        return {"verified": True, "feasible": True, "total_energy": "0"}
    if slack <= 0 or any(e <= 0 for e in energies):
        raise ValueError("No finite feasible allocation")
    if any((err-d)**2*energy != x**3 or err <= d
           for x, d, err, energy in zip(q, floor, errors, energies)):
        raise ValueError("Allocation violates precision-energy contract")
    if sum(errors, Q(0)) != accuracy:
        raise ValueError("Cut error does not saturate the target")
    optimum = total_q**3/slack**2
    if sum(energies, Q(0)) != optimum or rational(certificate["total_energy"]) != optimum:
        raise ValueError("Energy is not the fixed-cut optimum")
    return {"verified": True, "feasible": True, "total_energy": str(optimum)}


def minimum_ideal_energy(graph, target, q, accuracy):
    """Global no-floor optimum, by exact mincut plus precision allocation."""
    from graph_access import target_cut
    q = {key: rational(value) for key, value in q.items()}
    accuracy = rational(accuracy)
    if accuracy <= 0 or any(value <= 0 for value in q.values()):
        raise ValueError("Every available sensor needs positive q")
    cut = target_cut(graph, target, q)
    if cut["status"] == "unidentifiable":
        return {"feasible": False, "cut_certificate": cut}
    keys = sorted(cut["coefficients"])
    allocation = allocate_cut_energy([q[key] for key in keys], [0]*len(keys), accuracy)
    return {"feasible": True, "cut_certificate": cut, "sensor_edges": keys,
            "allocation": allocation}


def verify_ideal_energy(graph, target, q, accuracy, certificate):
    """Independent global lower/upper certificate, no mincut optimization."""
    from graph_access import verify_target_cut
    q = {key: rational(value) for key, value in q.items()}
    accuracy = rational(accuracy)
    if accuracy <= 0 or any(value <= 0 for value in q.values()):
        raise ValueError("Every available sensor needs positive q")
    cut = certificate["cut_certificate"]
    if not verify_target_cut(graph, target, q, cut):
        raise ValueError("Invalid independent mincut certificate")
    if cut["status"] == "unidentifiable":
        if certificate["feasible"]:
            raise ValueError("Inaccessible target marked energy-feasible")
        return {"verified": True, "feasible": False}
    if not certificate["feasible"]:
        raise ValueError("Finite positive-capacity cut incorrectly marked infeasible")
    keys = sorted(cut["coefficients"])
    if certificate["sensor_edges"] != keys:
        raise ValueError("Allocation uses a different cut")
    result = verify_cut_energy(certificate["allocation"], [q[key] for key in keys], [0]*len(keys), accuracy)
    if rational(result["total_energy"]) != rational(cut["value"])**3/rational(accuracy)**2:
        raise ValueError("Global energy lower bound not attained")
    return result


def enumerate_floor_design(graph, target, q, floor, accuracy):
    """Exact small-graph design with floors; exponential search is explicit."""
    edge = graph.edge(target)
    if len(graph.vertices) > 16:
        raise ValueError("Exact demonstration enumerator limited to sixteen vertices")
    q = {key: rational(value) for key, value in q.items()}
    floor = {key: rational(value) for key, value in floor.items()}
    if set(q) != set(floor) or target in q or not set(q).issubset({e.key for e in graph.edges}):
        raise ValueError("Matching nontarget sensor parameters required")
    if any(x <= 0 for x in q.values()) or any(x < 0 for x in floor.values()):
        raise ValueError("Positive q and nonnegative floors required")
    free = [v for v in graph.vertices if v not in (edge.tail, edge.head)]
    best, count, feasible = None, 0, 0
    for bits in product((0, 1), repeat=len(free)):
        cut = {edge.head} | {v for v, bit in zip(free, bits) if bit}
        keys = sorted(e.key for e in graph.edges if e.key != target and ((e.tail in cut) != (e.head in cut)))
        count += 1
        if not set(keys).issubset(q):
            continue
        allocation = allocate_cut_energy([q[key] for key in keys], [floor[key] for key in keys], accuracy)
        if not allocation["feasible"]:
            continue
        feasible += 1
        candidate = (rational(allocation["total_energy"]), keys, sorted(cut), allocation)
        if best is None or candidate[:3] < best[:3]:
            best = candidate
    if best is None:
        return {"feasible": False, "cuts_checked": count, "feasible_cuts": feasible}
    energy, keys, cut, allocation = best
    return {"feasible": True, "cuts_checked": count, "feasible_cuts": feasible,
            "sensor_edges": keys, "cut": cut, "allocation": allocation,
            "total_energy": str(energy)}


def floor_energy_bound(graph, target, q, floor, accuracy, support_parameter):
    """One mincut gives a global lower bound and a candidate upper allocation.

    Does not solve the NP-complete floor-aware design problem in general.
    Repeating for chosen positive rational support parameters improves the
    certified bounds without invalidating any earlier bound.
    """
    from graph_access import target_cut
    q = {key: rational(value) for key, value in q.items()}
    floor = {key: rational(value) for key, value in floor.items()}
    accuracy, t = rational(accuracy), rational(support_parameter)
    if set(q) != set(floor) or accuracy <= 0 or t <= 0 or any(x <= 0 for x in q.values()) or any(x < 0 for x in floor.values()):
        raise ValueError("Positive q, accuracy, support and nonnegative matching floors required")
    capacities = {key: 3*t*t*q[key]+2*t**3*floor[key] for key in q}
    cut = target_cut(graph, target, capacities)
    certificate = {"support_parameter": str(t), "cut_certificate": cut}
    if cut["status"] == "unidentifiable":
        certificate["structurally_feasible"] = False
        return certificate
    keys = sorted(cut["coefficients"])
    certificate.update({"structurally_feasible": True, "sensor_edges": keys,
                        "lower_energy": str(max(Q(0), rational(cut["value"])-2*accuracy*t**3)),
                        "candidate_allocation": allocate_cut_energy([q[key] for key in keys],
                                                                    [floor[key] for key in keys], accuracy)})
    return certificate


def verify_floor_energy_bound(graph, target, q, floor, accuracy, certificate):
    """Check a global floor-aware energy bracket without solving an optimizer."""
    from graph_access import verify_target_cut
    q = {key: rational(value) for key, value in q.items()}
    floor = {key: rational(value) for key, value in floor.items()}
    accuracy, t = rational(accuracy), rational(certificate["support_parameter"])
    if set(q) != set(floor) or accuracy <= 0 or t <= 0 or any(x <= 0 for x in q.values()) or any(x < 0 for x in floor.values()):
        raise ValueError("Invalid precision-energy supporting contract")
    capacities = {key: 3*t*t*q[key]+2*t**3*floor[key] for key in q}
    cut = certificate["cut_certificate"]
    if not verify_target_cut(graph, target, capacities, cut):
        raise ValueError("Invalid supporting mincut certificate")
    if cut["status"] == "unidentifiable":
        if certificate["structurally_feasible"]:
            raise ValueError("Incorrect structural feasibility claim")
        return {"verified": True, "structurally_feasible": False}
    if not certificate["structurally_feasible"]:
        raise ValueError("Incorrect structural obstruction claim")
    keys = sorted(cut["coefficients"])
    if certificate["sensor_edges"] != keys:
        raise ValueError("Candidate allocation does not use certified cut")
    lower = max(Q(0), rational(cut["value"])-2*accuracy*t**3)
    if rational(certificate["lower_energy"]) != lower:
        raise ValueError("Incorrect global energy lower bound")
    allocation = certificate["candidate_allocation"]
    verify_cut_energy(allocation, [q[key] for key in keys], [floor[key] for key in keys], accuracy)
    result = {"verified": True, "structurally_feasible": True, "lower_energy": str(lower)}
    if allocation["feasible"]:
        upper = rational(allocation["total_energy"])
        if lower > upper:
            raise ValueError("Lower bound exceeds feasible upper bound")
        result.update({"upper_energy": str(upper), "absolute_gap": str(upper-lower),
                       "global_optimum_certified": lower == upper})
    else:
        result["upper_energy"] = None
    return result


def partition_energy_instance(values):
    """Polynomial rational PARTITION reduction to floor-aware target-edge design.

    A graph with target s--t and paths s--i--t.  Removing the target,
    each s--t cut chooses one edge per path.  Every edge is sensor-capable.
    """
    if not values or any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in values):
        raise ValueError("Positive integer PARTITION instance required")
    total, n = sum(values), len(values)
    base = total
    desired_q = Q(n*base)+Q(total, 2)
    constant = Q(2*n*base+total)
    accuracy = constant-desired_q/3
    threshold = 9*desired_q/4
    edges = []
    for i, value in enumerate(values):
        edges.extend([{"id": f"left_{i}", "u": "s", "v": f"v{i}",
                       "q": str(base), "floor": str(base+value)},
                      {"id": f"right_{i}", "u": f"v{i}", "v": "t",
                       "q": str(base+value), "floor": str(base)}])
    return {"values": values, "target": ["s", "t"], "sensor_edges": edges,
            "accuracy": str(accuracy), "threshold_energy": str(threshold),
            "desired_q_sum": str(desired_q), "constant_q_plus_floor": str(constant)}


def enumerate_partition_design(instance):
    """Small-instance verification only; exponential enumeration is explicit."""
    values = instance["values"]
    if len(values) > 20:
        raise ValueError("Example enumerator limited to twenty paths")
    entries = []
    for bits in product((0, 1), repeat=len(values)):
        selected = [instance["sensor_edges"][2*i+bit] for i, bit in enumerate(bits)]
        allocation = allocate_cut_energy([e["q"] for e in selected], [e["floor"] for e in selected], instance["accuracy"])
        verify_cut_energy(allocation)
        entries.append((rational(allocation["total_energy"]), bits, allocation))
    energy, bits, allocation = min(entries)
    equal_partition = sum(x*bit for x, bit in zip(values, bits))*2 == sum(values)
    decision = energy <= rational(instance["threshold_energy"])
    if decision != equal_partition:
        raise AssertionError("Reduction decision disagrees with PARTITION")
    return {"cuts_checked": len(entries), "best_bits": list(bits), "best_energy": str(energy),
            "threshold_energy": instance["threshold_energy"], "partition_exists": decision,
            "allocation": allocation}


def main():
    from graph_access import Edge, Graph
    b = [[-1, 0, 1], [1, -1, 0], [0, 1, -1]]
    base = {"conservation": b, "target": [1, 0, 0], "epsilon": ["1/10", "1/5"]}
    differential = dict(base, observation=[[0, 1, 0], [0, 0, -1]], nuisance=[[1], [1]])
    same_orientation = dict(base, observation=[[0, 1, 0], [0, 0, 1]], nuisance=[[1], [1]])
    independent_bias = dict(base, observation=[[0, 1, 0], [0, 0, -1]], nuisance=[[1, 0], [0, 1]])
    differential_certificate = robust_target_certificate(differential)
    if differential_certificate["absolute_error"] != "3/20":
        raise AssertionError("Differential shared-bias cancellation changed")
    same_certificate = robust_target_certificate(same_orientation)
    independent_certificate = robust_target_certificate(independent_bias)
    if same_certificate["finite"] or independent_certificate["finite"]:
        raise AssertionError("Unbounded nuisance was incorrectly identifiable")
    allocation = allocate_cut_energy([1, 8], [0, 0], "1/10")
    allocation["verification"] = verify_cut_energy(allocation)
    floors = allocate_cut_energy([1, 8], ["1/100", "1/50"], "1/10")
    floors["verification"] = verify_cut_energy(floors)
    obstruction = allocate_cut_energy([1, 8], ["1/100", "1/50"], "3/100")
    obstruction["verification"] = verify_cut_energy(obstruction)
    instances = []
    for values in ([1, 2, 3], [1, 2, 4], [2, 2, 3, 5], [1, 1, 1], [3, 5, 6, 8], [8]):
        instance = partition_energy_instance(list(values))
        instances.append({"instance": instance, "verification": enumerate_partition_design(instance)})
    graph = Graph(("s", "a", "t"), (Edge("target", "s", "t"), Edge("single", "s", "a"),
                                     Edge("pair_1", "a", "t"), Edge("pair_2", "a", "t")))
    q = {"single": 4, "pair_1": 1, "pair_2": 2}
    ideal = minimum_ideal_energy(graph, "target", q, "1/10")
    ideal["verification"] = verify_ideal_energy(graph, "target", q, "1/10", ideal)
    floor_design = enumerate_floor_design(graph, "target", q,
                                          {"single": 0, "pair_1": "3/50", "pair_2": "1/50"}, "1/10")
    if ideal["allocation"]["total_energy"] != "2700" or floor_design["total_energy"] != "6400":
        raise AssertionError("Calibration-floor sensor choice switch changed")
    floor = {"single": 0, "pair_1": "3/50", "pair_2": "1/50"}
    supporting_bound = floor_energy_bound(graph, "target", q, floor, "1/10", 40)
    supporting_bound["verification"] = verify_floor_energy_bound(graph, "target", q, floor, "1/10", supporting_bound)
    result = {"status": "all exact certificate checks passed",
              "shared_nuisance_cancelled": {"model": differential, "certificate": differential_certificate},
              "same_orientation_failure": {"model": same_orientation, "certificate": same_certificate},
              "independent_nuisance_failure": {"model": independent_bias, "certificate": independent_certificate},
              "ideal_fixed_cut_allocation": allocation, "floor_aware_fixed_cut_allocation": floors,
              "finite_energy_impossible_at_floor": obstruction,
              "global_ideal_energy_design": ideal,
              "calibration_floors_change_best_sensor_set": floor_design,
              "single_mincut_global_floor_optimum_certificate": supporting_bound,
              "matched_precision_benchmarks": {"target_error": "1/10", "optimal_ideal_energy": "2700",
                  "fewest_sensor_choice_energy": "6400", "equal_precision_on_optimal_cut_energy": "3600",
                  "ignoring_floors_then_correcting_same_cut_energy": "67500",
                  "floor_aware_global_energy": "6400",
                  "meaning": "Exact constructed model comparison at equal deterministic target accuracy; no measured device claim"},
              "partition_reduction_cases": instances,
              "limits": ["No empirical sensor energy calibration is supplied.",
                         "Acquisition energy excludes installation, preparation, control, reset and readout overhead.",
                         "The exact LP and subset enumeration are small-example solvers, not scalable general algorithms.",
                         "Complexity result concerns the declared adversarial precision-energy model; it is not a physical impossibility theorem."]}
    destination = Path(__file__).resolve().parents[1]/"results"/"robust_completion.json"
    destination.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"status": result["status"], "partition_cases": len(instances),
                      "differential_error": differential_certificate["absolute_error"]}))


if __name__ == "__main__":
    main()
