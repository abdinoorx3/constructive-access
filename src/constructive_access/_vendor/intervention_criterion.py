"""Exact linear intervention identifiability and deterministic minimax recovery.

Established row-space and linear-programming geometry, with explicit scope.
The model assumes accessible laws identify F theta and observations satisfy a
declared coordinatewise error box. It does not infer either physical assumption.
"""
from fractions import Fraction as Q
from itertools import combinations, product
import json
from pathlib import Path


def rational(x):
    if isinstance(x, bool) or not isinstance(x, (str, int, Q)):
        raise ValueError("Exact rational inputs required")
    return Q(x)


def dot(a, b):
    if len(a) != len(b):
        raise ValueError("Dimension mismatch")
    return sum((x*y for x, y in zip(a, b)), Q(0))


def mv(a, x):
    return [dot(row, x) for row in a]


def rref(a, ncols):
    a = [[rational(x) for x in row] for row in a]
    if any(len(row) < ncols for row in a):
        raise ValueError("Short matrix row")
    pivot_columns, row = [], 0
    for column in range(ncols):
        chosen = next((i for i in range(row, len(a)) if a[i][column]), None)
        if chosen is None:
            continue
        a[row], a[chosen] = a[chosen], a[row]
        pivot = a[row][column]
        a[row] = [v/pivot for v in a[row]]
        for i in range(len(a)):
            if i != row and a[i][column]:
                scale = a[i][column]
                a[i] = [x-scale*y for x, y in zip(a[i], a[row])]
        pivot_columns.append(column)
        row += 1
    return a, pivot_columns


def solve(a, y, ncols):
    if len(a) != len(y) or any(len(row) != ncols for row in a):
        raise ValueError("Linear-system dimensions differ")
    reduced, pivots = rref([list(row)+[rational(v)] for row, v in zip(a, y)], ncols)
    if any(not any(row[:ncols]) and row[ncols] for row in reduced):
        return None
    x = [Q(0)]*ncols
    for row, column in enumerate(pivots):
        x[column] = reduced[row][ncols]
    return x


def inputs(f, b, epsilon=None):
    b = [rational(x) for x in b]
    if not b:
        raise ValueError("Positive parameter dimension required")
    f = [[rational(x) for x in row] for row in f]
    if any(len(row) != len(b) for row in f):
        raise ValueError("Fingerprint dimensions differ")
    if epsilon is None:
        return f, b
    epsilon = [rational(x) for x in epsilon]
    if len(epsilon) != len(f) or any(x < 0 for x in epsilon):
        raise ValueError("One nonnegative error radius per row required")
    return f, b, epsilon


def identify(f, b):
    """Return exact linear-target reconstruction or a positive indistinct pair."""
    f, b = inputs(f, b)
    p, m = len(b), len(f)
    reduced, pivots = rref(f, p)
    transposed = [[f[i][j] for i in range(m)] for j in range(p)]
    w = solve(transposed, b, m)
    result = {"rank": len(pivots), "parameter_dimension": p,
              "identifiable": w is not None}
    if w is not None:
        assert [sum((w[i]*f[i][j] for i in range(m)), Q(0)) for j in range(p)] == b
        result["reconstruction_weights"] = list(map(str, w))
        return result
    for free in range(p):
        if free in pivots:
            continue
        h = [Q(0)]*p
        h[free] = Q(1)
        for row, pivot in enumerate(pivots):
            h[pivot] = -reduced[row][free]
        if dot(b, h):
            center = [abs(x)+1 for x in h]
            low, high = ([x-sign*y for x, y in zip(center, h)] for sign in (1, -1))
            assert all(x > 0 for x in low+high)
            assert mv(f, low) == mv(f, high) and dot(b, low) != dot(b, high)
            result["indistinguishable_positive_parameters"] = [list(map(str, low)), list(map(str, high))]
            result["kernel_direction"] = list(map(str, h))
            result["target_difference"] = str(dot(b, high)-dot(b, low))
            return result
    raise AssertionError("Row-space / null-space alternative failed")


def verify_minimax_certificate(f, b, epsilon, certificate):
    """Check explicit primal/dual identities; no optimizer is called here."""
    f, b, epsilon = inputs(f, b, epsilon)
    w = [rational(x) for x in certificate["weights"]]
    h = [rational(x) for x in certificate["dual_direction"]]
    if len(w) != len(f) or len(h) != len(b):
        raise ValueError("Certificate dimensions differ")
    if [sum((w[i]*f[i][j] for i in range(len(f))), Q(0))
            for j in range(len(b))] != b:
        raise ValueError("Estimator is not exact on the declared model")
    image = mv(f, h)
    if any(abs(x) > radius for x, radius in zip(image, epsilon)):
        raise ValueError("Dual direction is outside the observation error box")
    upper = sum((radius*abs(weight) for radius, weight in zip(epsilon, w)), Q(0))
    lower = dot(b, h)
    if upper != lower or rational(certificate["absolute_error"]) != upper:
        raise ValueError("Primal and dual values differ")
    center = [abs(x)+1 for x in h]
    theta0, theta1 = ([x+sign*y for x, y in zip(center, h)] for sign in (-1, 1))
    observed = mv(f, center)
    assert all(x > 0 for x in theta0+theta1)
    assert dot(b, theta1)-dot(b, theta0) == 2*upper
    return {"verified": True, "minimax_absolute_error": str(upper),
            "shared_observation": list(map(str, observed)),
            "least_favorable_positive_parameters": [list(map(str, theta0)), list(map(str, theta1))],
            "scope": "Global deterministic minimax over the full positive orthant and the full coordinatewise error box"}


def minimax_certificate(f, b, epsilon):
    """Small exact LP by vertex enumeration; intended for worked examples."""
    f, b, epsilon = inputs(f, b, epsilon)
    p, m = len(b), len(f)
    id_result = identify(f, b)
    if not id_result["identifiable"]:
        return {"finite": False, "reason": "Target varies on the observation kernel",
                "identification": id_result}
    _, pivots = rref(f, p)
    rank = len(pivots)
    primal = []
    for size in range(rank+1):
        for support in combinations(range(m), size):
            if len(rref([f[i] for i in support], p)[1]) != size:
                continue
            selected_t = [[f[i][j] for i in support] for j in range(p)]
            small_w = solve(selected_t, b, size)
            if small_w is None:
                continue
            w = [Q(0)]*m
            for i, weight in zip(support, small_w):
                w[i] = weight
            primal.append((sum((epsilon[i]*abs(w[i]) for i in range(m)), Q(0)), tuple(w)))
    if not primal:
        raise AssertionError("No primal representation despite identifiability")
    value, weights = min(primal)
    if not rank:
        h = [Q(0)]*p
    else:
        a = [[row[j] for j in pivots] for row in f]
        dual = []
        for active in combinations(range(m), rank):
            selected = [a[i] for i in active]
            if len(rref(selected, rank)[1]) != rank:
                continue
            for signs in product((-1, 1), repeat=rank):
                candidate = solve(selected, [sign*epsilon[i] for sign, i in zip(signs, active)], rank)
                if any(abs(x) > bound for x, bound in zip(mv(a, candidate), epsilon)):
                    continue
                direction = [Q(0)]*p
                for j, x in zip(pivots, candidate):
                    direction[j] = x
                dual.append((dot(b, direction), tuple(direction)))
        if not dual:
            raise AssertionError("No reduced dual vertex")
        dual_value, h = max(dual)
        if dual_value != value:
            raise AssertionError("Exact LP primal-dual gap")
    certificate = {"finite": True, "weights": list(map(str, weights)),
                   "dual_direction": list(map(str, h)), "absolute_error": str(value)}
    certificate["verification"] = verify_minimax_certificate(f, b, epsilon, certificate)
    return certificate


def main():
    epsilon = [Q(1, 100), Q(1, 100)]
    rows = []
    for r in map(Q, ("0", "1/2", "9/10", "99/100", "999/1000", "1", "1001/1000", "101/100", "11/10", "2", "3")):
        f = [[Q(1), Q(1)], [r, Q(1)]]
        entry = {"selective_response": str(r), "fingerprint": [list(map(str, row)) for row in f],
                 "rank": identify(f, [1, 0])["rank"], "channel_1": minimax_certificate(f, [1, 0], epsilon),
                 "channel_2": minimax_certificate(f, [0, 1], epsilon),
                 "channel_difference": minimax_certificate(f, [1, -1], epsilon)}
        if r != 1:
            assert Q(entry["channel_1"]["absolute_error"]) == Q(1, 50)/abs(r-1)
            assert Q(entry["channel_2"]["absolute_error"]) == (1+abs(r))/100/abs(r-1)
            assert Q(entry["channel_difference"]["absolute_error"]) == (r+3)/100/abs(r-1)
        else:
            assert not entry["channel_1"]["finite"]
        rows.append(entry)
    redundant_f = [[1, 1], [Q(1, 2), 1], [2, 1]]
    redundant = minimax_certificate(redundant_f, [1, 0], [Q(1, 100), Q(1, 100), Q(1, 100)])
    assert Q(redundant["absolute_error"]) == Q(1, 75)
    assert redundant["weights"] == ["0", "-2/3", "2/3"]
    rank_one_target = minimax_certificate([[1, 1], [1, 1]], [1, 1], [Q(1, 100), Q(1, 50)])
    assert rank_one_target["absolute_error"] == "1/100"
    exact_row = minimax_certificate(redundant_f, [1, 1], [0, Q(1, 100), Q(1, 100)])
    assert exact_row["absolute_error"] == "0"
    zero_target = minimax_certificate([], [0, 0], [])
    assert zero_target["absolute_error"] == "0"
    # A coefficient itself cannot be treated as known merely because the
    # corresponding row would make a known matrix full rank.
    theta, theta_prime = [Q(3), Q(1)], [Q(2), Q(2)]
    assert mv([[1, 1], [Q(1, 2), 1]], theta) == mv([[1, 1], [Q(1, 4), 1]], theta_prime)
    unknown_response = {"parameter_pairs": [["3", "1"], ["2", "2"]],
                        "response_pairs": ["1/2", "1/4"], "shared_rates": ["4", "5/2"],
                        "lesson": "Unknown selective response can restore nonidentifiability"}
    # Numerical rank with a tolerance must not replace an exact structural
    # claim. This full-rank example is extremely ill-conditioned in practice.
    close = minimax_certificate([[1, 1], [Q(1000001, 1000000), 1]], [1, 0], epsilon)
    assert close["absolute_error"] == "20000"
    tampered = dict(redundant)
    tampered["absolute_error"] = "0"
    try:
        verify_minimax_certificate(redundant_f, [1, 0], [Q(1, 100)]*3, tampered)
    except ValueError:
        tampering_rejected = True
    else:
        raise AssertionError("Tampered minimax value accepted")
    output = {"status": "all exact assertions passed", "fingerprint_family": rows,
              "redundant_intervention_optimum": redundant, "rank_deficient_but_target_identifiable": rank_one_target,
              "exact_observation_zero_error": exact_row, "zero_target_without_observations": zero_target,
              "unknown_intervention_response_counterexample": unknown_response,
              "nearly_singular_full_rank": close, "tampered_error_rejected": tampering_rejected,
              "scope": "Known rational F, full positive parameter orthant, exact law-factorization/identification assumptions, full deterministic coordinatewise error box; no empirical data"}
    destination = Path(__file__).resolve().parents[1]/"results"/"intervention_criterion.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(output, indent=2)+"\n")
    print(json.dumps({"status": output["status"], "selective_response_cases": len(rows),
                      "redundant_intervention_error": redundant["absolute_error"],
                      "near_singular_error": close["absolute_error"]}, indent=2))


if __name__ == "__main__":
    main()
