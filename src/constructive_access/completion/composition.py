"""Exact finite-instrument interface checks and adaptive defect propagation.

This is an implementation of stochastic intertwining/lumpability and elementary
total-variation contraction, not a historical novelty claim.  Row convention.
Rational inputs only.  No free postselection: policy trees retain every outcome.
"""
from __future__ import annotations

from fractions import Fraction as F
from itertools import product
import json
from pathlib import Path


def rational(x):
    if isinstance(x, bool) or isinstance(x, float):
        raise TypeError("exact rational inputs required; bool/float rejected")
    if not isinstance(x, (int, str, F)):
        raise TypeError("expected integer, rational string, or Fraction")
    return F(x)


def matrix(rows):
    out = tuple(tuple(rational(x) for x in row) for row in rows)
    if not out or not out[0] or any(len(row) != len(out[0]) for row in out):
        raise ValueError("nonempty rectangular matrix required")
    return out


def eye(n):
    return tuple(tuple(F(i == j) for j in range(n)) for i in range(n))


def matmul(a, b):
    if len(a[0]) != len(b):
        raise ValueError("incompatible matrix dimensions")
    return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(len(b)))
                       for j in range(len(b[0]))) for i in range(len(a)))


def rowmul(row, m):
    return matmul((tuple(row),), m)[0]


def kron(a, b):
    return tuple(tuple(a[i][j] * b[k][l]
                       for j in range(len(a[0])) for l in range(len(b[0])))
                 for i in range(len(a)) for k in range(len(b)))


def instruments(raw):
    if not isinstance(raw, dict) or not raw:
        raise ValueError("nonempty action dictionary required")
    out = {}
    n = None
    for action, outcomes in raw.items():
        if not isinstance(action, str) or not isinstance(outcomes, dict) or not outcomes:
            raise ValueError("string action names and nonempty outcome dictionaries required")
        out[action] = {}
        for outcome, raw_m in outcomes.items():
            if not isinstance(outcome, str):
                raise ValueError("string outcome names required")
            m = matrix(raw_m)
            if len(m) != len(m[0]) or any(x < 0 for r in m for x in r):
                raise ValueError("instrument matrices must be nonnegative and square")
            if n is None:
                n = len(m)
            if len(m) != n:
                raise ValueError("all instrument dimensions must agree")
            out[action][outcome] = m
        for i in range(n):
            if sum(sum(m[i]) for m in out[action].values()) != 1:
                raise ValueError("sum over outcomes must be stochastic")
    return out


def stochastic(raw):
    out = matrix(raw)
    if any(x < 0 for row in out for x in row) or any(sum(row) != 1 for row in out):
        raise ValueError("row-stochastic interface required")
    return out


def probability(raw, n):
    row = tuple(rational(x) for x in raw)
    if len(row) != n or any(x < 0 for x in row) or sum(row) != 1:
        raise ValueError("invalid initial probability vector")
    return row


def interface_defects(concrete, quotient, interface):
    """Return epsilon_a(x)=TV((y,z)|x, concrete vs quotient interface).

    These defects bound the JOINT outcome and next interface state. Merely
    matching the immediate outcome probabilities is insufficient under feedback.
    """
    concrete, quotient = instruments(concrete), instruments(quotient)
    c = stochastic(interface)
    n = len(next(iter(next(iter(concrete.values())).values())))
    k = len(next(iter(next(iter(quotient.values())).values())))
    if len(c) != n or len(c[0]) != k or set(concrete) != set(quotient):
        raise ValueError("dimensions/actions do not match")
    result = {}
    for action, outcomes in concrete.items():
        if set(outcomes) != set(quotient[action]):
            raise ValueError("outcome alphabets do not match")
        residuals = []
        for outcome, m in outcomes.items():
            left = matmul(m, c)
            right = matmul(c, quotient[action][outcome])
            residuals.append(tuple(tuple(left[i][j] - right[i][j]
                                        for j in range(k)) for i in range(n)))
        result[action] = tuple(sum(abs(r[i][j]) for r in residuals for j in range(k)) / 2
                               for i in range(n))
    return result


def canonical_partition(blocks, n):
    blocks = tuple(sorted((tuple(sorted(block)) for block in blocks), key=lambda b: b[0] if b else -1))
    if not blocks or any(not block for block in blocks):
        raise ValueError("partition blocks must be nonempty")
    flat = [x for block in blocks for x in block]
    if any(isinstance(x, bool) or not isinstance(x, int) for x in flat) or sorted(flat) != list(range(n)):
        raise ValueError("partition must cover every state exactly once")
    return blocks


def partition_interface(blocks, n):
    blocks = canonical_partition(blocks, n)
    return tuple(tuple(F(i in block) for block in blocks) for i in range(n))


def signatures(model, blocks):
    n = len(next(iter(next(iter(model.values())).values())))
    return tuple(tuple(sum(m[i][j] for j in block)
                       for action in sorted(model)
                       for outcome, m in sorted(model[action].items())
                       for block in blocks) for i in range(n))


def complete_partition(raw_model, initial_partition):
    """Coarsest strongly lumpable refinement of a specified initial partition.

    Minimality is among DETERMINISTIC Markov interfaces refining this partition,
    valid for all initial preparations. It does not mean fewest experiments,
    minimum physical sensor cost, or the smallest arbitrary linear realization.
    """
    model = instruments(raw_model)
    n = len(next(iter(next(iter(model.values())).values())))
    blocks = canonical_partition(initial_partition, n)
    trace = [blocks]
    while True:
        sig = signatures(model, blocks)
        nxt = []
        for block in blocks:
            groups = {}
            for i in block:
                groups.setdefault(sig[i], []).append(i)
            nxt.extend(groups.values())
        nxt = canonical_partition(nxt, n)
        if nxt == blocks:
            break
        trace.append(nxt)
        blocks = nxt
    quotient = {a: {y: tuple(tuple(sum(m[block[0]][j] for j in target)
                                        for target in blocks) for block in blocks)
                         for y, m in ys.items()} for a, ys in model.items()}
    return {"partition": blocks, "trace": trace,
            "interface": partition_interface(blocks, n), "quotient": quotient}


def check_partition_certificate(raw_model, initial_partition, result):
    """Check refinement necessity at each stage, final closure, and quotient.

    Does not call complete_partition. The certificate establishes minimality by
    a forced-split induction; every stable refinement must refine every stage.
    """
    model = instruments(raw_model)
    n = len(next(iter(next(iter(model.values())).values())))
    trace = [canonical_partition(t, n) for t in result["trace"]]
    if not trace or len(trace) > n or trace[0] != canonical_partition(initial_partition, n):
        raise ValueError("invalid refinement trace start/length")
    for old, new in zip(trace, trace[1:]):
        sig = signatures(model, old)
        old_of = {x: i for i, b in enumerate(old) for x in b}
        new_of = {x: i for i, b in enumerate(new) for x in b}
        if len(new) <= len(old):
            raise ValueError("each refinement step must strictly split")
        for x in range(n):
            for z in range(n):
                expected_same = old_of[x] == old_of[z] and sig[x] == sig[z]
                if (new_of[x] == new_of[z]) != expected_same:
                    raise ValueError("unsupported or missing refinement split")
    final = canonical_partition(result["partition"], n)
    if final != trace[-1] or matrix(result["interface"]) != partition_interface(final, n):
        raise ValueError("final partition/interface does not match trace")
    defects = interface_defects(model, result["quotient"], result["interface"])
    if any(e for values in defects.values() for e in values):
        raise ValueError("claimed final quotient is not closed")
    return True


def transcript_law(raw_model, initial, policy, max_depth=100):
    """Exact law of ALL terminal records for a deterministic finite policy tree."""
    model = instruments(raw_model)
    n = len(next(iter(next(iter(model.values())).values())))
    initial = probability(initial, n)
    out = {}

    def walk(mu, node, history, depth):
        if node is None:
            out[history] = sum(mu)
            return
        if depth >= max_depth or not isinstance(node, dict) or set(node) != {"action", "children"}:
            raise ValueError("invalid or excessively deep finite policy")
        action = node["action"]
        if action not in model or set(node["children"]) != set(model[action]):
            raise ValueError("policy must retain exactly every declared outcome")
        for y, m in model[action].items():
            walk(rowmul(mu, m), node["children"][y], history + ((action, y),), depth + 1)

    walk(initial, policy, (), 0)
    if sum(out.values()) != 1:
        raise ArithmeticError("terminal record law is not normalized")
    return out


def total_variation(p, q):
    return sum(abs(p.get(x, F(0)) - q.get(x, F(0))) for x in set(p) | set(q)) / 2


def adaptive_defect_bound(concrete, quotient, interface, initial, quotient_initial, policy):
    """Compute TV bound delta0 + E_concrete sum_{before stop} epsilon_a(X).

    Validation and direct transcript comparison are returned alongside the
    bound; the theorem permits arbitrary outcome feedback in this finite tree.
    """
    concrete, quotient = instruments(concrete), instruments(quotient)
    c = stochastic(interface)
    initial = probability(initial, len(c))
    quotient_initial = probability(quotient_initial, len(c[0]))
    defects = interface_defects(concrete, quotient, c)
    p = transcript_law(concrete, initial, policy)
    q = transcript_law(quotient, quotient_initial, policy)
    projected = rowmul(initial, c)
    prep = sum(abs(x-y) for x, y in zip(projected, quotient_initial)) / 2
    terms = []

    def walk(mu, node, history):
        if node is None:
            return
        a = node["action"]
        contribution = sum(mu[x] * defects[a][x] for x in range(len(mu)))
        terms.append({"history": history, "action": a,
                      "reach_probability": sum(mu), "contribution": contribution})
        for y, m in concrete[a].items():
            walk(rowmul(mu, m), node["children"][y], history + ((a, y),))

    walk(initial, policy, ())
    accumulated = sum(t["contribution"] for t in terms)
    bound = min(F(1), prep + accumulated)
    actual = total_variation(p, q)
    if actual > bound:
        raise ArithmeticError("the derived defect bound failed")
    return {"preparation_defect": prep, "accumulated_defect": accumulated,
            "tv_upper_bound": bound, "actual_transcript_tv": actual,
            "terms": terms, "local_defects": defects}


def tensor_instruments(left, right):
    left, right = instruments(left), instruments(right)
    # JSON array names are unambiguous even when user names contain separators.
    return {json.dumps([a, b]): {json.dumps([y, z]): kron(m, n)
                              for y, m in ys.items() for z, n in zs.items()}
            for a, ys in left.items() for b, zs in right.items()}


def tensor_defect_check(concrete_left, quotient_left, interface_left,
                        concrete_right, quotient_right, interface_right):
    dl = interface_defects(concrete_left, quotient_left, interface_left)
    dr = interface_defects(concrete_right, quotient_right, interface_right)
    joint = interface_defects(tensor_instruments(concrete_left, concrete_right),
                              tensor_instruments(quotient_left, quotient_right),
                              kron(stochastic(interface_left), stochastic(interface_right)))
    count = 0
    worst_slack = F(0)
    for a, es in dl.items():
        for b, fs in dr.items():
            values = joint[json.dumps([a, b])]
            for i, e in enumerate(es):
                for j, f in enumerate(fs):
                    upper = e + f - e*f
                    actual = values[i*len(fs)+j]
                    if actual > upper:
                        raise ArithmeticError("product-coupling TV bound failed")
                    worst_slack = max(worst_slack, upper-actual)
                    count += 1
    return {"checked_joint_rows": count, "maximum_slack": worst_slack,
            "bound": "epsilon_left + epsilon_right - epsilon_left*epsilon_right"}


def cnot_example():
    # State index 2*h+o: h is hidden; o is the sole directly observed bit.
    read = {str(y): tuple(tuple(F(i == j and i % 2 == y) for j in range(4))
                               for i in range(4)) for y in range(2)}
    swap = tuple(tuple(F(j == 2*(i//2) + ((i % 2) ^ (i//2)))
                       for j in range(4)) for i in range(4))
    passive = {"read": read, "wait": {"ok": eye(4)}}
    active = {**passive, "couple": {"ok": swap}}
    initial_partition = ((0, 2), (1, 3))
    baseline = complete_partition(passive, initial_partition)
    completed = complete_partition(active, initial_partition)
    for model, result in ((passive, baseline), (active, completed)):
        check_partition_certificate(model, initial_partition, result)
    passive_policy = {"action": "read", "children": {"0": None, "1": None}}
    active_policy = {"action": "couple", "children": {"ok": passive_policy}}
    p0, p1 = (1, 0, 0, 0), (0, 0, 1, 0)
    return {"passive_interface_states": len(baseline["partition"]),
            "completed_interface_states": len(completed["partition"]),
            "passive_tv": total_variation(transcript_law(passive,p0,passive_policy),
                                           transcript_law(passive,p1,passive_policy)),
            "active_tv": total_variation(transcript_law(active,p0,active_policy),
                                          transcript_law(active,p1,active_policy)),
            "coupling_is_permutation": all(sum(row)==1 for row in swap)
                                       and all(sum(swap[i][j] for i in range(4))==1 for j in range(4)),
            "scope": "logically reversible stochastic gate; no hardware energy claim",
            "passive_model": passive, "active_model": active,
            "initial_partition": initial_partition,
            "completion_certificate": completed}


def memory_counterexample():
    # Equal current output probabilities do not imply equal future records.
    zero = ((0,0),(0,0))
    concrete = {"prepare_hidden": {"ok": ((0,1),(0,1))},
                "read": {"0": ((1,0),(0,0)), "1": ((0,0),(0,1))}}
    quotient = {"prepare_hidden": {"ok": ((1,0),(1,0))},
                "read": {"0": ((1,0),(0,0)), "1": ((0,0),(0,1))}}
    policy = {"action":"prepare_hidden", "children":{"ok":
              {"action":"read", "children":{"0":None,"1":None}}}}
    result = adaptive_defect_bound(concrete,quotient,eye(2),(1,0),(1,0),policy)
    result["first_action_output_tv"] = F(0)
    return result


def adaptive_example():
    # Concrete hidden state changes after each outcome; the action at the second
    # stage depends on the first result. All inputs and evaluations are exact.
    concrete = {
        "probe": {"0": ((F(3,5),F(1,5)),(F(1,10),F(1,5))),
                  "1": ((F(1,10),F(1,10)),(F(1,5),F(1,2)))},
        "quiet": {"0": ((F(9,10),0),(0,F(4,5))),
                  "1": ((F(1,10),0),(0,F(1,5)))} }
    quotient = {
        "probe": {"0": ((F(3,4),),),"1":((F(1,4),),)},
        "quiet": {"0": ((F(9,10),),),"1":((F(1,10),),)}}
    interface = ((1,),(1,))
    policy = {"action":"probe", "children": {
        "0": None,
        "1": {"action":"quiet", "children": {"0": None,
              "1":{"action":"probe", "children":{"0":None,"1":None}}}}}}
    result = adaptive_defect_bound(concrete, quotient, interface,
                                   (F(3,4),F(1,4)), (1,), policy)
    result["model"] = concrete
    result["quotient"] = quotient
    result["interface"] = interface
    result["policy"] = policy
    result["tensor_check"] = tensor_defect_check(concrete,quotient,interface,
                                                 concrete,quotient,interface)
    return result


def correlated_preparation_example():
    even = (F(1,2),F(0),F(0),F(1,2))
    odd = (F(0),F(1,2),F(1,2),F(0))
    x = cnot_example()
    read = {"action":"read", "children":{"0":None,"1":None}}
    coupled = {"action":"couple", "children":{"ok":read}}
    return {"both_local_marginals": ["1/2","1/2"],
            "joint_preparation_tv": total_variation(dict(enumerate(even)),dict(enumerate(odd))),
            "direct_visible_tv":total_variation(transcript_law(x["active_model"],even,read),
                                                 transcript_law(x["active_model"],odd,read)),
            "coupled_visible_tv":total_variation(transcript_law(x["active_model"],even,coupled),
                                                  transcript_law(x["active_model"],odd,coupled))}


def jsonable(x):
    if isinstance(x,F):
        return str(x)
    if isinstance(x,dict):
        return {str(k):jsonable(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):
        return [jsonable(y) for y in x]
    return x


def main():
    results = {"status":"exact rational demonstrations; no empirical measurements",
               "novelty":"established lumpability and simulation theory, integrated with access contracts",
               "cnot_completion":cnot_example(),
               "memory_counterexample":memory_counterexample(),
               "adaptive_defect":adaptive_example(),
               "correlated_preparation":correlated_preparation_example()}
    path = Path(__file__).resolve().parents[1]/"results"/"composition.json"
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(jsonable(results),indent=2,sort_keys=True)+"\n")
    print(json.dumps({"output":str(path),
                      "completed_interface_states":results["cnot_completion"]["completed_interface_states"],
                      "adaptive_tv":str(results["adaptive_defect"]["actual_transcript_tv"]),
                      "adaptive_bound":str(results["adaptive_defect"]["tv_upper_bound"])}))


if __name__ == "__main__":
    main()
