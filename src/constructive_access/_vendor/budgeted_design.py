"""Exact finite-horizon Bayes-risk design for rational classical instruments.

Standard belief-state dynamic programming. Budget is a declared pathwise
integer resource; no energy meaning is inferred. A finite horizon is mandatory.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from itertools import combinations, product
import json
from pathlib import Path
from time import perf_counter

from .accesskit import RationalInstrument, ActionDFA, experiment_law, minimum_cost_witness, rational


@dataclass
class DesignResult:
    feasible: bool
    risk: Q | None
    policy: dict | None
    counters: dict


def update(row, matrix):
    return tuple(sum(row[i]*matrix[i][j] for i in range(len(row)))
                 for j in range(len(matrix[0])))


def optimal_design(model0, model1, action_costs, *, budget, horizon,
                   prior0=Q(1, 2), grammar=None, max_states=1000000):
    """Minimum equal-decision-cost Bayesian error among bounded causal trees.

    Stopping requires grammar acceptance. Infeasible root returns feasible=False.
    Initial hidden distributions belong to the supplied normalized instruments.
    Both hypotheses face the same policy, action alphabet and pathwise budget.
    """
    if not isinstance(model0, RationalInstrument) or not isinstance(model1, RationalInstrument):
        raise TypeError("This optimizer accepts validated classical RationalInstrument models only")
    for value in [budget, horizon, max_states]:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError("Budget, mandatory horizon and state limit must be nonnegative integers")
    if set(model0.actions) != set(model1.actions) or set(action_costs) != set(model0.actions):
        raise ValueError("Common action alphabet and a cost for every action are required")
    if any(not isinstance(c, int) or isinstance(c, bool) or c < 0 for c in action_costs.values()):
        raise ValueError("Action costs must be nonnegative integers")
    prior0 = rational(prior0)
    if not 0 < prior0 < 1:
        raise ValueError("Both candidate priors must be strictly positive for both-model policy feasibility")
    actions = tuple(sorted(model0.actions))
    outcomes = {}
    matrices = []
    for model in [model0, model1]:
        converted = {}
        for action in actions:
            if set(model0.actions[action]) != set(model1.actions[action]):
                raise ValueError("Use common explicit outcomes, including zero-probability ones")
            outcomes[action] = tuple(sorted(model.actions[action]))
            converted[action] = {y: tuple(tuple(rational(v) for v in row) for row in k.tolist())
                                 for y, k in model.actions[action].items()}
        matrices.append(converted)
    initial0 = tuple(prior0*rational(v) for v in model0.initial)
    initial1 = tuple((1-prior0)*rational(v) for v in model1.initial)
    cache = {}
    counters = {"states_evaluated": 0, "cache_hits": 0, "action_evaluations": 0,
                "outcome_branches": 0, "maximum_depth": 0}
    started = perf_counter()

    def solve(b0, b1, remaining, depth, state):
        key = (b0, b1, remaining, depth, state)
        if key in cache:
            counters["cache_hits"] += 1
            return cache[key]
        counters["states_evaluated"] += 1
        if counters["states_evaluated"] > max_states:
            raise RuntimeError("Exact search exceeded the declared state limit; no optimum certified")
        counters["maximum_depth"] = max(counters["maximum_depth"], horizon-depth)
        mass0, mass1 = sum(b0), sum(b1)
        if mass0+mass1 == 0:
            result = (Q(0), {"kind": "unreachable"})
            cache[key] = result
            return result
        accepting = grammar is None or state in grammar.accepting
        best = min(mass0, mass1) if accepting else None
        policy = ({"kind": "stop", "decision": 0 if mass0 >= mass1 else 1,
                   "weighted_masses": [str(mass0), str(mass1)]} if accepting else None)
        if depth:
            for action in actions:
                if action_costs[action] > remaining:
                    continue
                next_state = None if grammar is None else grammar.transitions.get((state, action))
                if grammar is not None and next_state is None:
                    continue
                counters["action_evaluations"] += 1
                total, children = Q(0), {}
                for outcome in outcomes[action]:
                    counters["outcome_branches"] += 1
                    n0 = update(b0, matrices[0][action][outcome])
                    n1 = update(b1, matrices[1][action][outcome])
                    risk, child = solve(n0, n1, remaining-action_costs[action], depth-1, next_state)
                    if risk is None:
                        total = None
                        break
                    total += risk
                    children[outcome] = child
                if total is not None and (best is None or total < best):
                    best, policy = total, {"kind": "action", "action": action,
                                           "cost": action_costs[action], "children": children}
        result = (best, policy)
        cache[key] = result
        return result

    risk, policy = solve(initial0, initial1, budget, horizon,
                         None if grammar is None else grammar.start)
    counters["runtime_seconds"] = perf_counter()-started
    counters["cache_size"] = len(cache)
    return DesignResult(risk is not None, risk, policy, counters)


def best_fixed_design(model0, model1, costs, *, budget, horizon, prior0=Q(1, 2), grammar=None):
    best, sequence, tested = None, None, 0
    for length in range(horizon+1):
        for actions in product(sorted(model0.actions), repeat=length):
            if sum(costs[a] for a in actions) > budget or (grammar is not None and not grammar.accepts(actions)):
                continue
            p0, p1 = [experiment_law(model, actions, grammar) for model in [model0, model1]]
            risk = sum(min(prior0*p0[word], (1-prior0)*p1[word]) for word in p0)
            tested += 1
            if best is None or risk < best:
                best, sequence = risk, actions
    return {"risk": best, "actions": sequence, "sequences_evaluated": tested}


def coins(probabilities):
    return RationalInstrument([1], {a: {"1": [[p]], "0": [[1-p]]}
                                    for a, p in probabilities.items()})


def enumerate_trees(actions, depth):
    # Independent exhaustive enumeration for binary outcomes, no grammar.
    yield None
    if depth:
        children = list(enumerate_trees(actions, depth-1))
        for action in actions:
            for left, right in product(children, repeat=2):
                yield (action, left, right)


def tree_risk(tree, p0, p1, m0=Q(1, 2), m1=Q(1, 2)):
    if tree is None:
        return min(m0, m1)
    action, zero, one = tree
    return (tree_risk(zero, p0, p1, m0*(1-p0[action]), m1*(1-p1[action]))
            +tree_risk(one, p0, p1, m0*p0[action], m1*p1[action]))


def summary(result):
    return {"feasible": result.feasible, "risk_exact": None if result.risk is None else str(result.risk),
            "policy": result.policy, "counters": result.counters}


def main():
    # A nonzero witness minimizes a different objective from finite-error risk.
    m0 = coins({"weak": Q(3, 5), "perfect": Q(1)})
    m1 = coins({"weak": Q(2, 5), "perfect": Q(0)})
    costs = {"weak": 1, "perfect": 2}
    witness = minimum_cost_witness(m0, m1, costs)
    budget1 = optimal_design(m0, m1, costs, budget=1, horizon=2)
    budget2 = optimal_design(m0, m1, costs, budget=2, horizon=2)
    assert witness.witness_action_cost == 1
    assert budget1.risk == Q(2, 5)
    assert budget2.risk == 0 and budget2.policy["action"] == "perfect"

    # Deterministic, exact search for a small asymmetric-channel example.
    grid = [Q(i, 5) for i in range(1, 5)]
    channels = [(p, q) for p in grid for q in grid if p > q]
    found, candidates = None, 0
    for aa, bb in combinations(channels, 2):
        p0, p1 = {"a": aa[0], "b": bb[0]}, {"a": aa[1], "b": bb[1]}
        c0, c1 = coins(p0), coins(p1)
        adaptive = optimal_design(c0, c1, {"a": 1, "b": 1}, budget=3, horizon=3)
        fixed = best_fixed_design(c0, c1, {"a": 1, "b": 1}, budget=3, horizon=3)
        candidates += 1
        if adaptive.risk < fixed["risk"]:
            found = (p0, p1, c0, c1, adaptive, fixed)
            break
    assert found is not None
    p0, p1, c0, c1, adaptive, fixed = found
    depth2 = optimal_design(c0, c1, {"a": 1, "b": 1}, budget=2, horizon=2)
    exhaustive = [tree_risk(tree, p0, p1) for tree in enumerate_trees(["a", "b"], 2)]
    assert len(exhaustive) == 19 and depth2.risk == min(exhaustive)
    exhaustive3 = [tree_risk(tree, p0, p1) for tree in enumerate_trees(["a", "b"], 3)]
    assert len(exhaustive3) == 723 and adaptive.risk == min(exhaustive3)
    # A physically redundant two-state refinement of hypothesis1 checks that
    # the two candidate hidden dimensions need not agree.
    mixing = [[Q(1, 2), Q(1, 2)], [Q(1, 2), Q(1, 2)]]
    refined1 = RationalInstrument([Q(1, 2), Q(1, 2)], {
        a: {"1": [[p1[a]*v for v in row] for row in mixing],
            "0": [[(1-p1[a])*v for v in row] for row in mixing]}
        for a in p1})
    refined_design = optimal_design(c0, refined1, {"a": 1, "b": 1}, budget=3, horizon=3)
    assert refined_design.risk == adaptive.risk

    # Zero resource cost still terminates because horizon is mandatory.
    z0, z1 = coins({"free_label": Q(3, 4)}), coins({"free_label": Q(1, 4)})
    zero0 = optimal_design(z0, z1, {"free_label": 0}, budget=0, horizon=0)
    zero2 = optimal_design(z0, z1, {"free_label": 0}, budget=0, horizon=2)
    assert zero0.risk == Q(1, 2) and zero2.risk == Q(1, 4)
    grammar = ActionDFA(["start", "after_weak", "done"], "start", ["done"],
                        {("start", "weak"): "after_weak", ("after_weak", "perfect"): "done"})
    impossible = optimal_design(m0, m1, costs, budget=2, horizon=2, grammar=grammar)
    permitted = optimal_design(m0, m1, costs, budget=3, horizon=2, grammar=grammar)
    assert not impossible.feasible and permitted.risk == 0
    rejected_degenerate_priors = []
    for endpoint in [Q(0), Q(1)]:
        try:
            optimal_design(m0, m1, costs, budget=2, horizon=2, prior0=endpoint)
        except ValueError:
            rejected_degenerate_priors.append(str(endpoint))
        else:
            raise AssertionError("Zero-prior candidate must not be silently excluded from feasibility")

    out = {"status": "passed; standard exact finite-horizon Bayesian dynamic programming",
           "witness_vs_risk": {"minimum_witness_action_cost": str(witness.witness_action_cost),
                               "witness": witness.witness, "budget_1": summary(budget1), "budget_2": summary(budget2)},
           "adaptive_example": {"probability_outcome1_h0": {a: str(p) for a, p in p0.items()},
                                "probability_outcome1_h1": {a: str(p) for a, p in p1.items()},
                                "budget": 3, "horizon": 3, "costs": {"a": 1, "b": 1},
                                "adaptive": summary(adaptive),
                                "best_fixed": {**fixed, "risk": str(fixed["risk"])},
                                "exact_improvement": str(fixed["risk"]-adaptive.risk),
                                "grid_denominator": 5, "candidates_examined": candidates},
           "independent_depth2_policy_enumeration": {"policies": len(exhaustive),
                                                      "risk": str(depth2.risk), "matched": True},
           "independent_depth3_policy_enumeration": {"policies": len(exhaustive3),
                                                      "risk": str(adaptive.risk), "matched": True},
           "zero_cost_finite_horizon": {"horizon0": summary(zero0), "horizon2": summary(zero2)},
           "grammar_controls": {"insufficient_budget": summary(impossible), "sufficient_budget": summary(permitted)},
           "rejected_degenerate_priors": rejected_degenerate_priors,
           "different_hidden_dimensions": {"dimensions": [c0.dimension, refined1.dimension],
                                            "risk_exact": str(refined_design.risk), "matched": True},
           "resource_scope": "pathwise nonnegative integer budget and mandatory finite action horizon; not expected budget or inferred energy",
           "computational_scope": "exact rational enumeration with memoization; no polynomial complexity claim",
           "randomness": "none"}
    dest = Path(__file__).resolve().parents[1]/"results"/"budgeted_design.json"
    dest.write_text(json.dumps(out, indent=2)+"\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
