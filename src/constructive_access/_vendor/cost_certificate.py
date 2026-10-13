"""Portable exact certificates for equivalence or minimum-cost nonzero words.

The verifier uses only Python's standard library and checks explicit rational
identities. It neither calls a word-search routine nor trusts its execution.
An embedded model is an assumption: bind expected_model or expected_digest when
checking an externally supplied bundle. Signed linear representations are not
automatically physically valid classical or quantum instruments.
"""
from __future__ import annotations

from fractions import Fraction as Q
from hashlib import sha256
from heapq import heappop, heappush
from itertools import count
import json
from pathlib import Path

SCHEMA = "finite-access-cost-certificate-v1"


class CertificateError(ValueError):
    """The declared certificate does not establish its declared result."""


def _q(x):
    if isinstance(x, bool) or not isinstance(x, (int, str, Q)):
        raise CertificateError("Use exact rational strings, integers, or Fractions")
    try:
        return Q(x)
    except (ValueError, ZeroDivisionError) as exc:
        raise CertificateError("Invalid rational coefficient") from exc


def _vector(values, n=None):
    if not isinstance(values, (list, tuple)):
        raise CertificateError("A vector must be a list or tuple")
    out = tuple(_q(x) for x in values)
    if n is not None and len(out) != n:
        raise CertificateError("Vector dimension mismatch")
    return out


def _matrix(values, n):
    if not isinstance(values, (list, tuple)) or len(values) != n:
        raise CertificateError("Matrix dimension mismatch")
    return tuple(_vector(row, n) for row in values)


def _mv(matrix, vector):
    return tuple(sum((a*b for a, b in zip(row, vector)), Q(0)) for row in matrix)


def _dot(a, b):
    return sum((x*y for x, y in zip(a, b)), Q(0))


def _parse_model(model):
    if not isinstance(model, dict):
        raise CertificateError("Model must be an object")
    alpha = _vector(model.get("alpha"))
    if not alpha:
        raise CertificateError("Model dimension must be positive")
    beta = _vector(model.get("beta"), len(alpha))
    raw_letters = model.get("letters")
    if not isinstance(raw_letters, list):
        raise CertificateError("Model letters must be a list")
    letters, action_costs, pairs = {}, {}, set()
    for raw in raw_letters:
        if not isinstance(raw, dict):
            raise CertificateError("Letter must be an object")
        ident, action, outcome = (raw.get(k) for k in ("id", "action", "outcome"))
        if any(not isinstance(s, str) or not s for s in (ident, action, outcome)):
            raise CertificateError("Letter IDs, actions and outcomes must be nonempty strings")
        if ident in letters or (action, outcome) in pairs:
            raise CertificateError("Duplicate letter or action/outcome pair")
        cost = _q(raw.get("cost"))
        if cost < 0 or (action in action_costs and action_costs[action] != cost):
            raise CertificateError("Action costs must be nonnegative and outcome-independent")
        letters[ident] = (action, outcome, cost, _matrix(raw.get("matrix"), len(alpha)))
        action_costs[action] = cost
        pairs.add((action, outcome))
    return alpha, beta, letters


def canonical_model(model):
    alpha, beta, letters = _parse_model(model)
    return {"alpha": list(map(str, alpha)), "beta": list(map(str, beta)),
            "letters": [{"id": k, "action": a, "outcome": y, "cost": str(c),
                         "matrix": [list(map(str, row)) for row in m]}
                        for k, (a, y, c, m) in sorted(letters.items())]}


def model_sha256(model):
    return sha256(json.dumps(canonical_model(model), sort_keys=True,
                             separators=(",", ":")).encode()).hexdigest()


def _word(raw_word, letters):
    if not isinstance(raw_word, (list, tuple)) or any(
            not isinstance(s, str) or s not in letters for s in raw_word):
        raise CertificateError("Word contains an unknown letter")
    return tuple(raw_word)


def _effect(word, beta, letters):
    v = beta
    for symbol in reversed(word):
        v = _mv(letters[symbol][3], v)
    return v


def _weight(word, letters):
    return sum((letters[s][2] for s in word), Q(0)), len(word)


def _combination(coefficients, vectors, dimension):
    return tuple(sum((c*v[i] for c, v in zip(coefficients, vectors)), Q(0))
                 for i in range(dimension))


def check_certificate(bundle, expected_model=None, expected_digest=None):
    """Independently verify a portable bundle or raise CertificateError.

    All matrix products and supplied expansion identities are recomputed.
    No span search, rank calculation, or optimizer is invoked. Exact arithmetic
    decides a declared finite model, not calibration or empirical validity.
    """
    if not isinstance(bundle, dict) or bundle.get("schema") != SCHEMA:
        raise CertificateError("Unrecognized certificate schema")
    model = bundle.get("model")
    alpha, beta, letters = _parse_model(model)
    digest = model_sha256(model)
    if bundle.get("model_sha256") != digest:
        raise CertificateError("Model hash mismatch")
    if expected_model is not None and model_sha256(expected_model) != digest:
        raise CertificateError("Certificate is for a different expected model")
    if expected_digest is not None and expected_digest != digest:
        raise CertificateError("Certificate is for a different expected digest")
    cert = bundle.get("certificate")
    if not isinstance(cert, dict):
        raise CertificateError("Certificate must be an object")
    mode = cert.get("mode")
    if mode not in ("optimal_witness", "equivalent"):
        raise CertificateError("Invalid certificate mode")
    threshold = None
    contrast = None
    witness_word = None
    if mode == "optimal_witness":
        witness = cert.get("witness")
        if not isinstance(witness, dict):
            raise CertificateError("Missing witness")
        witness_word = _word(witness.get("word"), letters)
        threshold = _weight(witness_word, letters)
        if _q(witness.get("cost")) != threshold[0]:
            raise CertificateError("Witness cost mismatch")
        contrast = _dot(alpha, _effect(witness_word, beta, letters))
        if contrast == 0 or _q(witness.get("contrast")) != contrast:
            raise CertificateError("Witness has no claimed nonzero contrast")
    elif cert.get("witness") is not None:
        raise CertificateError("Equivalence certificate must not declare a witness")
    raw_basis = cert.get("basis")
    if not isinstance(raw_basis, list):
        raise CertificateError("Basis must be a list")
    vectors, weights = [], []
    for entry in raw_basis:
        if not isinstance(entry, dict):
            raise CertificateError("Basis entry must be an object")
        word = _word(entry.get("word"), letters)
        weight = _weight(word, letters)
        v = _vector(entry.get("effect"), len(alpha))
        if v != _effect(word, beta, letters) or _q(entry.get("cost")) != weight[0]:
            raise CertificateError("Basis effect or cost is not its declared word")
        if threshold is not None and not weight < threshold:
            raise CertificateError("Basis word is not below the witness threshold")
        if _dot(alpha, v) != 0:
            raise CertificateError("A retained effect is not annihilated")
        vectors.append(v)
        weights.append(weight)
    # Linear independence is unnecessary for soundness: explicit closure and
    # annihilation suffice. The generator produces an independent list anyway.
    def expansion(raw, target, bound):
        coeffs = _vector(raw, len(vectors))
        if bound is not None and any(c and w > bound for c, w in zip(coeffs, weights)):
            raise CertificateError("Expansion uses an effect beyond its allowed weight")
        if _combination(coeffs, vectors, len(alpha)) != target:
            raise CertificateError("Invalid exact expansion identity")
    beta_needed = threshold is None or (Q(0), 0) < threshold
    if beta_needed:
        expansion(cert.get("beta_coefficients"), beta, None if threshold is None else (Q(0), 0))
    elif cert.get("beta_coefficients") is not None:
        raise CertificateError("Empty optimal witness requires no below-threshold beta expansion")
    required = {}
    for i, (v, w) in enumerate(zip(vectors, weights)):
        for symbol, (_, _, cost, matrix) in letters.items():
            bound = (w[0]+cost, w[1]+1)
            if threshold is None or bound < threshold:
                required[(i, symbol)] = (_mv(matrix, v), None if threshold is None else bound)
    raw_closure = cert.get("closure")
    if not isinstance(raw_closure, list):
        raise CertificateError("Closure must be a list")
    seen = set()
    for edge in raw_closure:
        if not isinstance(edge, dict):
            raise CertificateError("Closure edge must be an object")
        i, symbol = edge.get("source"), edge.get("letter")
        if type(i) is not int or not isinstance(symbol, str):
            raise CertificateError("Invalid closure edge address")
        key = (i, symbol)
        if key not in required or key in seen:
            raise CertificateError("Unexpected or duplicate closure edge")
        expansion(edge.get("coefficients"), *required[key])
        seen.add(key)
    if seen != set(required):
        raise CertificateError("Missing required closure edge")
    return {"verified": True, "mode": mode, "model_sha256": digest,
            "expected_model_bound": expected_model is not None or expected_digest is not None,
            "dimension": len(alpha), "retained_effects": len(vectors),
            "checked_closure_identities": len(required),
            "witness": None if witness_word is None else list(witness_word),
            "minimum_action_cost": None if threshold is None else str(threshold[0]),
            "shortest_length_at_minimum_cost": None if threshold is None else threshold[1],
            "contrast": None if contrast is None else str(contrast),
            "scope": "Declared exact rational linear representation; physical validity and calibration are separate"}


def _coordinates(vectors, target):
    """Generator-only exact Gaussian elimination; never used by the checker."""
    k, d = len(vectors), len(target)
    rows = [[v[i] for v in vectors]+[target[i]] for i in range(d)]
    pivots, r = [], 0
    for j in range(k):
        pivot = next((i for i in range(r, d) if rows[i][j]), None)
        if pivot is None:
            continue
        rows[r], rows[pivot] = rows[pivot], rows[r]
        denominator = rows[r][j]
        rows[r] = [x/denominator for x in rows[r]]
        for i in range(d):
            if i != r and rows[i][j]:
                coefficient = rows[i][j]
                rows[i] = [x-coefficient*y for x, y in zip(rows[i], rows[r])]
        pivots.append((r, j))
        r += 1
    if any(not any(row[:k]) and row[k] for row in rows):
        return None
    result = [Q(0)]*k
    for i, j in pivots:
        result[j] = rows[i][k]
    return tuple(result)


def generate_certificate(model):
    """Generate finite filtered-span evidence; verify it separately before use."""
    model = canonical_model(model)
    alpha, beta, letters = _parse_model(model)
    vectors, words, weights = [], [], []
    queue, serial = [], count()
    heappush(queue, (Q(0), 0, next(serial), (), beta))
    witness, threshold = None, None
    processed = 0
    while queue:
        cost, length, _, word, vector = heappop(queue)
        processed += 1
        if _coordinates(vectors, vector) is not None:
            continue
        contrast = _dot(alpha, vector)
        if contrast:
            witness = {"word": list(word), "cost": str(cost), "contrast": str(contrast)}
            threshold = (cost, length)
            break
        vectors.append(vector)
        words.append(word)
        weights.append((cost, length))
        for symbol, (_, _, letter_cost, matrix) in sorted(letters.items()):
            heappush(queue, (cost+letter_cost, length+1, next(serial),
                             (symbol,)+word, _mv(matrix, vector)))
    def coordinates(target, bound):
        eligible = [i for i, w in enumerate(weights) if bound is None or w <= bound]
        small = _coordinates([vectors[i] for i in eligible], target)
        if small is None:
            raise AssertionError("Generator failed its filtered-span invariant")
        result = [Q(0)]*len(vectors)
        for i, x in zip(eligible, small):
            result[i] = x
        return list(map(str, result))
    cert = {"mode": "equivalent" if witness is None else "optimal_witness", "witness": witness,
            "basis": [{"word": list(w), "cost": str(c[0]), "effect": list(map(str, v))}
                      for w, c, v in zip(words, weights, vectors)], "closure": []}
    cert["beta_coefficients"] = (coordinates(beta, None if threshold is None else (Q(0), 0))
                                 if threshold is None or (Q(0), 0) < threshold else None)
    for i, (v, w) in enumerate(zip(vectors, weights)):
        for symbol, (_, _, cost, matrix) in sorted(letters.items()):
            bound = (w[0]+cost, w[1]+1)
            if threshold is None or bound < threshold:
                cert["closure"].append({"source": i, "letter": symbol,
                                        "coefficients": coordinates(_mv(matrix, v),
                                                                     None if threshold is None else bound)})
    return {"schema": SCHEMA, "model": model, "model_sha256": model_sha256(model),
            "certificate": cert, "generator_statistics": {"queue_candidates_processed": processed}}


def pair_model(rep0, rep1, action_costs, grammar=None):
    """Build a signed pair from two portable rational linear representations.

    Each representation has initial, terminal and a list of kernels containing
    action, outcome and matrix. Optional DFA is on actions only. It filters
    complete action protocols, not recorded outcomes. No positivity is inferred.
    """
    def rep(raw):
        initial = _vector(raw.get("initial"))
        if not initial:
            raise CertificateError("Empty representation")
        terminal = _vector(raw.get("terminal"), len(initial))
        matrices = {}
        for kernel in raw.get("kernels", []):
            key = (kernel.get("action"), kernel.get("outcome"))
            if any(not isinstance(s, str) or not s for s in key) or key in matrices:
                raise CertificateError("Invalid representation alphabet")
            matrices[key] = _matrix(kernel.get("matrix"), len(initial))
        return initial, terminal, matrices
    i0, t0, m0 = rep(rep0)
    i1, t1, m1 = rep(rep1)
    if set(m0) != set(m1) or set(action_costs) != {a for a, _ in m0}:
        raise CertificateError("Pair alphabets or action costs differ")
    if grammar is not None:
        states = grammar.get("states")
        if (not isinstance(states, list) or not states or
                any(not isinstance(s, str) or not s for s in states) or len(set(states)) != len(states)):
            raise CertificateError("Invalid DFA states")
        start, accepting = grammar.get("start"), grammar.get("accepting")
        if start not in states or not isinstance(accepting, list) or not set(accepting) <= set(states):
            raise CertificateError("Invalid DFA endpoints")
        transitions = {}
        for edge in grammar.get("transitions", []):
            key, target = (edge.get("state"), edge.get("action")), edge.get("target")
            if (key[0] not in states or target not in states or key[1] not in action_costs
                    or key in transitions):
                raise CertificateError("Invalid DFA transition")
            transitions[key] = target
        def lift(initial, terminal, matrices):
            n = len(initial)
            lifted_i = tuple(x if s == start else Q(0) for s in states for x in initial)
            lifted_t = tuple(x if s in accepting else Q(0) for s in states for x in terminal)
            lifted_m = {}
            for (action, outcome), matrix in matrices.items():
                lifted_m[(action, outcome)] = tuple(
                    tuple(matrix[i][j] if transitions.get((s, action)) == target else Q(0)
                          for target in states for j in range(n))
                    for s in states for i in range(n))
            return lifted_i, lifted_t, lifted_m
        i0, t0, m0 = lift(i0, t0, m0)
        i1, t1, m1 = lift(i1, t1, m1)
    n0, n1 = len(i0), len(i1)
    letters = []
    for index, (action, outcome) in enumerate(sorted(m0)):
        block = [list(row)+[Q(0)]*n1 for row in m0[(action, outcome)]]
        block += [[Q(0)]*n0+list(row) for row in m1[(action, outcome)]]
        letters.append({"id": "s"+str(index), "action": action, "outcome": outcome,
                        "cost": action_costs[action], "matrix": block})
    return canonical_model({"alpha": list(i0)+[-x for x in i1],
                            "beta": list(t0)+list(t1), "letters": letters})


def fixture_models():
    """Small exact models exercise zero costs, DFA restriction and cancellation."""
    def bernoulli(p):
        return {"initial": [1], "terminal": [1],
                "kernels": [{"action": "read", "outcome": "yes", "matrix": [[p]]},
                            {"action": "read", "outcome": "no", "matrix": [[1-Q(p)]]},
                            {"action": "wait", "outcome": "none", "matrix": [[1]]}]}
    p0, p1 = bernoulli(Q(2, 3)), bernoulli(Q(1, 3))
    grammar = {"states": ["start", "waited", "done"], "start": "start", "accepting": ["done"],
               "transitions": [{"state": "start", "action": "wait", "target": "waited"},
                               {"state": "waited", "action": "wait", "target": "waited"},
                               {"state": "waited", "action": "read", "target": "done"}]}
    cancellation = {"alpha": [1, -1, 0], "beta": [1, 1, 1],
                    "letters": [{"id": "a", "action": "drive", "outcome": "none", "cost": 0,
                                 "matrix": [[1, 1, -1], [0, 1, 0], [0, 0, 1]]}]}
    zero_loop = pair_model(p0, p1, {"read": 2, "wait": 0})
    # An expensive one-letter effect loses to a two-letter cheap effect.
    layered = {"alpha": [1, 0, 0], "beta": [0, 0, 1], "letters": [
        {"id": "cheap", "action": "cheap", "outcome": "none", "cost": 1,
         "matrix": [[0, 1, 0], [0, 0, 1], [0, 0, 0]]},
        {"id": "expensive", "action": "expensive", "outcome": "none", "cost": 3,
         "matrix": [[0, 0, 1], [0, 0, 0], [0, 0, 0]]}]}
    return {"zero_cost_loop": zero_loop,
            "grammar_requires_wait": pair_model(p0, p1, {"read": 2, "wait": 0}, grammar),
            "signed_path_cancellation": canonical_model(cancellation),
            "two_cheap_before_one_expensive": canonical_model(layered),
            "empty_language": pair_model(p0, p1, {"read": 2, "wait": 0},
                                         {**grammar, "accepting": []}),
            "empty_word_distinction": canonical_model({"alpha": [1], "beta": [1], "letters": []})}


def main():
    records = {}
    for name, model in fixture_models().items():
        bundle = generate_certificate(model)
        report = check_certificate(bundle, expected_model=model)
        records[name] = {"verification": report, "bundle": bundle}
    output = {"schema": SCHEMA, "fixtures": records,
              "scope": "Exact model-bound mathematical certificates; no empirical or priority claim"}
    destination = Path(__file__).resolve().parents[1]/"results"/"cost_certificate.json"
    destination.write_text(json.dumps(output, indent=2)+"\n")
    print(json.dumps({name: row["verification"] for name, row in records.items()}, indent=2))


if __name__ == "__main__":
    main()
