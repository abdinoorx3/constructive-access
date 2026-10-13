"""Exact finite-instrument access checks and finite experiment evaluation.

This module implements established linear-representation equivalence methods.
Exact rational arithmetic certifies an assumed model; it does not certify its
physical calibration. Floats are rejected rather than silently made exact.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from heapq import heappop, heappush
from itertools import count, product
from math import isqrt, prod
from typing import Mapping, Sequence

import sympy as sp

Symbol = tuple[str, str]
Word = tuple[Symbol, ...]
Q = Fraction


def rational(value) -> Q:
    if isinstance(value, bool):
        raise TypeError("Boolean values are not rational model coefficients")
    if isinstance(value, (int, Q, sp.Rational)):
        return Q(int(value.p), int(value.q)) if isinstance(value, sp.Rational) else Q(value)
    if isinstance(value, str):
        return Q(value)
    raise TypeError("Exact coefficients require int, Fraction, SymPy Rational, or rational string; floats are not certified")


def matrix(value) -> sp.ImmutableMatrix:
    raw = sp.Matrix(value)
    return sp.ImmutableMatrix(raw.rows, raw.cols,
                              [sp.Rational(rational(x).numerator, rational(x).denominator) for x in raw])


def _row(value) -> sp.ImmutableMatrix:
    m = matrix(value)
    if m.cols == 1:
        m = m.T
    if m.rows != 1 or not m.cols:
        raise ValueError("Initial state must be a nonempty row or vector")
    return sp.ImmutableMatrix(m)


def _column(value) -> sp.ImmutableMatrix:
    m = matrix(value)
    if m.rows == 1:
        m = m.T
    if m.cols != 1 or not m.rows:
        raise ValueError("Terminal functional must be a nonempty column or vector")
    return sp.ImmutableMatrix(m)


@dataclass
class LinearRepresentation:
    """Rational weighted representation; positivity is NOT asserted here.

    This generic interface also supports externally certified quantum
    instruments represented in rational real operator coordinates.
    """
    initial: object
    kernels: Mapping[Symbol, object]
    terminal: object
    name: str = ""

    def __post_init__(self):
        self.initial, self.terminal = _row(self.initial), _column(self.terminal)
        self.kernels = dict(self.kernels)
        n = self.initial.cols
        if self.terminal.rows != n:
            raise ValueError("Initial and terminal dimensions differ")
        for symbol, value in list(self.kernels.items()):
            if not isinstance(symbol, tuple) or len(symbol) != 2 or any(not isinstance(x, str) or not x for x in symbol):
                raise ValueError("Symbols must be nonempty (action, outcome) string pairs")
            k = matrix(value)
            if k.shape != (n, n):
                raise ValueError("Each kernel must match the representation dimension")
            self.kernels[symbol] = k

    @property
    def dimension(self):
        return self.initial.cols

    def word_probability(self, word: Sequence[Symbol]) -> Q:
        row = self.initial
        for symbol in word:
            row = row*self.kernels[tuple(symbol)]
        return rational((row*self.terminal)[0])


class RationalInstrument:
    """Finite physical classical instrument with normalized action outcomes.

    actions[action][outcome] is a nonnegative matrix. Each action's outcome
    sum is row-stochastic. Deficits require an explicit recorded abort outcome.
    The state is a row distribution and the terminal functional is all ones.
    """
    def __init__(self, initial, actions, name=""):
        self.initial = _row(initial)
        self.name = name
        n = self.initial.cols
        if min(self.initial) < 0 or sum(self.initial) != 1:
            raise ValueError("Initial probabilities must be nonnegative and sum to one")
        if not actions:
            raise ValueError("At least one action is required")
        self.actions = {}
        kernels = {}
        for action, outcomes in actions.items():
            if not isinstance(action, str) or not action or not outcomes:
                raise ValueError("Each named action requires explicit outcomes")
            total = sp.zeros(n)
            self.actions[action] = {}
            for outcome, value in outcomes.items():
                if not isinstance(outcome, str) or not outcome:
                    raise ValueError("Outcomes must be nonempty strings")
                k = matrix(value)
                if k.shape != (n, n) or min(k) < 0:
                    raise ValueError("Outcome kernels must be nonnegative square matrices of the model dimension")
                if any(sum(k[i, j] for j in range(n)) > 1 for i in range(n)):
                    raise ValueError("Each outcome kernel must be substochastic")
                total += k
                kernels[(action, outcome)] = k
                self.actions[action][outcome] = k
            if total*sp.ones(n, 1) != sp.ones(n, 1):
                raise ValueError("Outcome kernels for every action must sum to a row-stochastic matrix; record missing mass as abort")
        self.representation = LinearRepresentation(self.initial, kernels, sp.ones(n, 1), name)

    @property
    def dimension(self):
        return self.representation.dimension

    def word_probability(self, word):
        return self.representation.word_probability(word)

    def coarsen(self, outcome_maps: Mapping[str, Mapping[str, str]], name=None):
        """Deterministically merge recorded outcomes without changing dynamics."""
        actions = {}
        if set(outcome_maps) != set(self.actions):
            raise ValueError("Supply an outcome map for every action")
        for action, outcomes in self.actions.items():
            if set(outcome_maps[action]) != set(outcomes):
                raise ValueError("Every original outcome must be mapped exactly once")
            actions[action] = {}
            for old, k in outcomes.items():
                new = outcome_maps[action][old]
                if not isinstance(new, str) or not new:
                    raise ValueError("Coarsened outcome names must be nonempty strings")
                actions[action][new] = actions[action].get(new, sp.zeros(self.dimension))+k
        return RationalInstrument(self.initial, actions, name or self.name+" (coarsened)")


@dataclass
class ActionDFA:
    """Partial deterministic automaton on ACTIONS, independent of outcomes.

    Acceptance selects complete action protocols, never outcome postselection.
    Missing transitions make a history inadmissible. Nonaccepting intermediate
    states are allowed; only stopping in an accepting state is evaluated.
    """
    states: Sequence[str]
    start: str
    accepting: Sequence[str]
    transitions: Mapping[tuple[str, str], str]

    def __post_init__(self):
        self.states = tuple(self.states)
        self.accepting = frozenset(self.accepting)
        self.transitions = dict(self.transitions)
        if not self.states or len(set(self.states)) != len(self.states):
            raise ValueError("DFA states must be nonempty and unique")
        if self.start not in self.states or not self.accepting.issubset(self.states):
            raise ValueError("DFA start and acceptance states must exist")
        for (state, action), target in self.transitions.items():
            if state not in self.states or target not in self.states or not isinstance(action, str) or not action:
                raise ValueError("Invalid DFA transition")

    def accepts(self, actions: Sequence[str]) -> bool:
        state = self.start
        for action in actions:
            state = self.transitions.get((state, action))
            if state is None:
                return False
        return state in self.accepting


def _lift(rep: LinearRepresentation, grammar: ActionDFA | None):
    if grammar is None:
        return rep
    s = len(grammar.states)
    indices = {state: j for j, state in enumerate(grammar.states)}
    initial = sp.zeros(1, s)
    initial[0, indices[grammar.start]] = 1
    accept = sp.Matrix([int(state in grammar.accepting) for state in grammar.states])
    kernels = {}
    for symbol, k in rep.kernels.items():
        automaton = sp.zeros(s)
        for state in grammar.states:
            target = grammar.transitions.get((state, symbol[0]))
            if target is not None:
                automaton[indices[state], indices[target]] = 1
        kernels[symbol] = sp.kronecker_product(automaton, k)
    return LinearRepresentation(sp.kronecker_product(initial, rep.initial), kernels,
                                sp.kronecker_product(accept, rep.terminal), rep.name)


def _pair(rep0, rep1, grammar):
    if set(rep0.kernels) != set(rep1.kernels):
        raise ValueError("Models require the same explicit action/outcome alphabet; add zero kernels for impossible outcomes")
    a, b = _lift(rep0, grammar), _lift(rep1, grammar)
    alpha = a.initial.row_join(-b.initial)
    beta = a.terminal.col_join(b.terminal)
    operators = {symbol: sp.diag(a.kernels[symbol], b.kernels[symbol]) for symbol in sorted(a.kernels)}
    return alpha, beta, operators


@dataclass
class AccessResult:
    equivalent: bool
    witness: Word | None
    witness_probability0: Q | None
    witness_probability1: Q | None
    basis: sp.ImmutableMatrix
    basis_words: tuple[Word, ...]
    ambient_dimension: int
    witness_length_bound: int
    grammar_used: bool
    action_costs: dict[str, Q] = field(default_factory=dict)
    witness_action_cost: Q | None = None
    scope: str = "exact rational linear representations; physical validity must be checked separately"

    @property
    def span_dimension(self):
        return self.basis.cols

    def to_dict(self, include_basis=True):
        result = {"equivalent": self.equivalent,
                  "witness": None if self.witness is None else [list(x) for x in self.witness],
                  "witness_probability0": None if self.witness_probability0 is None else str(self.witness_probability0),
                  "witness_probability1": None if self.witness_probability1 is None else str(self.witness_probability1),
                  "span_dimension": self.span_dimension, "ambient_dimension": self.ambient_dimension,
                  "witness_length_bound": self.witness_length_bound, "grammar_used": self.grammar_used,
                  "witness_action_cost": None if self.witness_action_cost is None else str(self.witness_action_cost),
                  "scope": self.scope}
        if include_basis:
            result["basis_columns"] = [[str(self.basis[i, j]) for i in range(self.basis.rows)]
                                       for j in range(self.basis.cols)]
            result["basis_words"] = [[list(x) for x in w] for w in self.basis_words]
        return result


def compare_linear_representations(rep0: LinearRepresentation, rep1: LinearRepresentation,
                                   grammar: ActionDFA | None = None,
                                   action_costs: Mapping[str, object] | None = None) -> AccessResult:
    """Return an exact obstruction or minimum supplied additive-cost witness.

    Unit action costs give breadth-first witness search. Nonnegative additive
    costs change queue order, not the invariant-span certificate. This witness
    minimizes one transcript's action cost, not statistical sample cost.
    With a grammar the dimension bound uses the full DFA product dimension.
    """
    alpha, beta, operators = _pair(rep0, rep1, grammar)
    d = beta.rows
    actions = {a for a, y in operators}
    costs = {a: Q(1) for a in actions} if action_costs is None else {a: rational(v) for a, v in action_costs.items()}
    if set(costs) != actions or any(v < 0 for v in costs.values()):
        raise ValueError("Supply one nonnegative exact cost for every action")
    basis = sp.zeros(d, 0)
    basis_words = []
    queue, serial = [], count()
    heappush(queue, (Q(0), 0, next(serial), (), beta))
    while queue:
        cost, length, _, word, vector = heappop(queue)
        enlarged = basis.row_join(vector)
        if enlarged.rank() == basis.cols:
            continue
        basis = enlarged
        basis_words.append(word)
        contrast = (alpha*vector)[0]
        if contrast != 0:
            assert length <= d-1
            assert grammar is None or grammar.accepts([a for a, y in word])
            return AccessResult(False, word, rep0.word_probability(word), rep1.word_probability(word),
                                sp.ImmutableMatrix(basis), tuple(basis_words), d, max(0, d-1),
                                grammar is not None, costs, cost)
        for symbol, operator in operators.items():
            heappush(queue, (cost+costs[symbol[0]], length+1, next(serial),
                            (symbol,)+word, operator*vector))
    return AccessResult(True, None, None, None, sp.ImmutableMatrix(basis), tuple(basis_words),
                        d, max(0, d-1), grammar is not None, costs)


def compare_models(model0: RationalInstrument, model1: RationalInstrument, grammar=None, action_costs=None):
    result = compare_linear_representations(model0.representation, model1.representation, grammar, action_costs)
    result.scope = "exact rational normalized classical instruments; action-only grammar; assumed calibration"
    return result


def verify_access_result(rep0, rep1, result: AccessResult, grammar=None) -> bool:
    """Check invariant-span obstruction or witness evidence without the search.

    For a witness, this verifies its probabilities and feasibility. Minimum
    cost additionally relies on the priority-span theorem and search, rather
    than being independently certified by the single returned transcript.
    """
    if isinstance(rep0, RationalInstrument):
        rep0 = rep0.representation
    if isinstance(rep1, RationalInstrument):
        rep1 = rep1.representation
    alpha, beta, operators = _pair(rep0, rep1, grammar)
    if not result.equivalent:
        w = result.witness
        return (w is not None and len(w) <= beta.rows-1
                and (grammar is None or grammar.accepts([a for a, y in w]))
                and rep0.word_probability(w) == result.witness_probability0
                and rep1.word_probability(w) == result.witness_probability1
                and result.witness_probability0 != result.witness_probability1)
    b = result.basis
    if b.rows != beta.rows or b.rank() != b.cols or b.row_join(beta).rank() != b.cols:
        return False
    if alpha*b != sp.zeros(1, b.cols):
        return False
    return all(b.row_join(k*b).rank() == b.cols for k in operators.values())


def minimum_cost_witness(model0, model1, action_costs, grammar=None):
    """Convenience entry point for a minimum nonnegative additive-cost witness.

    Costs are fixed per action, including every possible outcome. Conditional,
    history-dependent, or multiobjective costs need a different construction.
    """
    if isinstance(model0, RationalInstrument) and isinstance(model1, RationalInstrument):
        return compare_models(model0, model1, grammar, action_costs)
    if isinstance(model0, LinearRepresentation) and isinstance(model1, LinearRepresentation):
        return compare_linear_representations(model0, model1, grammar, action_costs)
    raise TypeError("Both models must use the same public representation class")


def experiment_law(model, actions: Sequence[str], grammar=None, max_transcripts=100000) -> dict[tuple[str, ...], Q]:
    """Exact normalized outcome law for a fixed open-loop action protocol."""
    rep = model.representation if isinstance(model, RationalInstrument) else model
    if not isinstance(rep, LinearRepresentation):
        raise TypeError("Expected RationalInstrument or LinearRepresentation")
    actions = tuple(actions)
    if grammar is not None and not grammar.accepts(actions):
        raise ValueError("The complete action protocol is not accepted by the grammar")
    alphabets = []
    for action in actions:
        outcomes = sorted(y for a, y in rep.kernels if a == action)
        if not outcomes:
            raise ValueError("Unknown action: "+str(action))
        alphabets.append(outcomes)
    if prod(map(len, alphabets)) > max_transcripts:
        raise ValueError("Explicit transcript count exceeds max_transcripts; increase deliberately or use a compressed method")
    law = {ys: rep.word_probability(tuple(zip(actions, ys))) for ys in product(*alphabets)}
    if any(p < 0 for p in law.values()) or sum(law.values()) != 1:
        raise ValueError("This experiment's linear representation does not produce a normalized nonnegative law")
    return law


def affinity_upper(law0: Mapping, law1: Mapping, bits=160) -> Q:
    """Outward dyadic upper bound on sum sqrt(p*q), using integer arithmetic."""
    if set(law0) != set(law1) or not isinstance(bits, int) or bits < 8:
        raise ValueError("Common support keys and at least 8 precision bits are required")
    if any(rational(p) < 0 for p in list(law0.values())+list(law1.values())) or sum(map(rational, law0.values())) != 1 or sum(map(rational, law1.values())) != 1:
        raise ValueError("Affinity requires normalized nonnegative probability laws")
    scale = 1 << bits
    total = 0
    for key in law0:
        z = rational(law0[key])*rational(law1[key])
        if z < 0:
            raise ValueError("Probabilities must be nonnegative")
        root = isqrt(z.numerator*scale*scale//z.denominator)
        if root*root*z.denominator < z.numerator*scale*scale:
            root += 1
        total += root
    return Q(total, scale)


def iid_error_upper(rho_upper, records: int, bits=160) -> Q:
    """Outward bound rho^M/2 with a rational contraction fallback.

    Upward rounded squaring can have a near-one fixed point at coarse
    precision. Independently, rho^M <= 1/(1+M*(1-rho)) for 0<=rho<1.
    Taking the smaller bound prevents a rounding fixed point from stalling
    the count search. Neither bound is claimed to give the minimum count.
    """
    rho = rational(rho_upper)
    if not isinstance(records, int) or isinstance(records, bool) or records < 0 or rho < 0:
        raise ValueError("Nonnegative integer record count and affinity bound required")
    if records == 0 or rho >= 1:
        return Q(1, 2)
    scale = 1 << bits
    base = (rho.numerator*scale+rho.denominator-1)//rho.denominator
    answer = scale
    exponent = records
    while exponent:
        if exponent & 1:
            answer = (answer*base+scale-1)//scale
        base = (base*base+scale-1)//scale
        exponent //= 2
    contraction_bound = Q(1, 2)/(1+records*(1-rho))
    return min(Q(1, 2), Q(answer, 2*scale), contraction_bound)


def sufficient_iid_records(rho_upper, target_error=Q(1, 10), bits=160):
    """Find a certified sufficient count; does not assert the optimal count."""
    rho, target = rational(rho_upper), rational(target_error)
    if not 0 < target < Q(1, 2):
        raise ValueError("Target average error must lie strictly between 0 and 1/2")
    if rho >= 1:
        return None, None
    high = 1
    while iid_error_upper(rho, high, bits) > target:
        high *= 2
    low = 0
    while high-low > 1:
        middle = (high+low)//2
        if iid_error_upper(rho, middle, bits) <= target:
            high = middle
        else:
            low = middle
    bound = iid_error_upper(rho, high, bits)
    assert bound <= target
    return high, bound


def _resource_totals(actions, action_bounds, overhead):
    supplied = action_bounds is not None
    action_bounds = {} if action_bounds is None else action_bounds
    totals = {k: rational(v) for k, v in (overhead or {}).items()}
    dimensions = set().union(*(set(v) for v in action_bounds.values())) if action_bounds else set()
    for name in dimensions:
        totals.setdefault(name, Q(0))
    if any(v < 0 for v in totals.values()):
        raise ValueError("Resource bounds must be nonnegative")
    for action in actions:
        if supplied and (action not in action_bounds or set(action_bounds[action]) != dimensions):
            raise ValueError("Every used action needs bounds for all declared action-resource dimensions")
        for name, value in action_bounds.get(action, {}).items():
            value = rational(value)
            if value < 0:
                raise ValueError("Resource bounds must be nonnegative")
            totals[name] = totals.get(name, Q(0))+value
    return totals


@dataclass
class ExperimentResult:
    actions: tuple[str, ...]
    law0: dict[tuple[str, ...], Q]
    law1: dict[tuple[str, ...], Q]
    total_variation: Q
    single_record_bayes_error: Q
    affinity_upper: Q
    precision_bits: int
    sufficient_records: int | None
    certified_average_error_upper: Q | None
    target_error: Q
    per_record_resource_upper_bounds: dict[str, Q]
    certified_batch_resource_upper_bounds: dict[str, Q]
    affordable_records: int | None
    budget_average_error_upper: Q | None
    error_certificate_method: str = "Bhattacharyya with outward rational rounding"
    assumptions: tuple[str, ...] = (
        "iid records with the supplied initial preparations",
        "same fixed action protocol and known models under both hypotheses",
        "equal prior probabilities; optimal iid likelihood classification",
        "resource bounds are supplied assumptions, not inferred thermodynamics")

    def to_dict(self):
        return {"actions": list(self.actions),
                "transcripts": [{"outcomes": list(k), "probability0": str(self.law0[k]),
                                 "probability1": str(self.law1[k])} for k in self.law0],
                "total_variation": str(self.total_variation),
                "single_record_bayes_error": str(self.single_record_bayes_error),
                "affinity_upper": str(self.affinity_upper), "precision_bits": self.precision_bits,
                "sufficient_records": self.sufficient_records,
                "certified_average_error_upper": None if self.certified_average_error_upper is None else str(self.certified_average_error_upper),
                "target_error": str(self.target_error),
                "per_record_resource_upper_bounds": {k: str(v) for k, v in self.per_record_resource_upper_bounds.items()},
                "certified_batch_resource_upper_bounds": {k: str(v) for k, v in self.certified_batch_resource_upper_bounds.items()},
                "affordable_records": self.affordable_records,
                "budget_average_error_upper": None if self.budget_average_error_upper is None else str(self.budget_average_error_upper),
                "error_certificate_method": self.error_certificate_method,
                "assumptions": list(self.assumptions)}


def evaluate_experiment(model0, model1, actions, grammar=None,
                        action_resource_upper_bounds=None, per_record_resource_upper_bounds=None,
                        target_error=Q(1, 10), max_records=None, total_resource_budgets=None,
                        precision_bits=160, max_precision_bits=2560, max_transcripts=100000):
    """Evaluate one finite protocol exactly, then enclose its iid error bound.

    A nonzero contrast with an unresolved affinity enclosure remains
    distinguishable; it is never mislabeled equivalent. Finite budgets affect
    affordable iid repetitions, not access equivalence. This is not an optimizer.
    """
    actions = tuple(actions)
    if not isinstance(precision_bits, int) or precision_bits < 8 or max_precision_bits < precision_bits:
        raise ValueError("Use integer precision_bits>=8 and max_precision_bits>=precision_bits")
    law0 = experiment_law(model0, actions, grammar, max_transcripts)
    law1 = experiment_law(model1, actions, grammar, max_transcripts)
    if set(law0) != set(law1):
        raise ValueError("Models need the same explicit outcome alphabet for this protocol")
    target = rational(target_error)
    if not 0 < target < Q(1, 2):
        raise ValueError("Target error must lie strictly between 0 and 1/2")
    tv = sum(abs(law0[k]-law1[k]) for k in law0)/2
    bayes = sum(min(law0[k], law1[k]) for k in law0)/2
    bits = precision_bits
    if tv == 0:
        rho = Q(1)
    else:
        rho = affinity_upper(law0, law1, bits)
        while rho >= 1 and bits*2 <= max_precision_bits:
            bits *= 2
            rho = affinity_upper(law0, law1, bits)
    records, bound = sufficient_iid_records(rho, target, bits)
    method = "Bhattacharyya with outward rational rounding"
    if tv == 0:
        method = "exactly identical fixed-protocol laws"
    elif rho >= 1:
        method = "affinity enclosure unresolved at precision cap; laws differ exactly"
    if bayes <= target:
        records, bound = 1, bayes
        method = "exact single-record Bayes error"
    resources = _resource_totals(actions, action_resource_upper_bounds, per_record_resource_upper_bounds)
    affordable = max_records
    if affordable is not None and (not isinstance(affordable, int) or isinstance(affordable, bool) or affordable < 0):
        raise ValueError("max_records must be a nonnegative integer")
    for name, value in (total_resource_budgets or {}).items():
        budget = rational(value)
        if budget < 0 or name not in resources:
            raise ValueError("Every nonnegative budget requires a declared per-record resource bound")
        cost = resources[name]
        if cost:
            cap = budget//cost
            affordable = cap if affordable is None else min(affordable, cap)
    budget_error = None if affordable is None else iid_error_upper(rho, affordable, bits)
    if affordable is not None and affordable >= 1:
        budget_error = min(budget_error, bayes)
    return ExperimentResult(actions, law0, law1, tv, bayes, rho, bits, records, bound, target,
                            resources, {} if records is None else {k: records*v for k, v in resources.items()},
                            affordable, budget_error, method)


def enumerate_experiments(model0, model1, max_actions: int, grammar=None,
                          per_record_resource_limits=None, **evaluation_options):
    """Exhaust a specified finite open-loop menu, with explicit resource limits.

    The number of protocols grows exponentially in max_actions. The returned
    menu is not a solution to unrestricted adaptive experimental design.
    """
    if not isinstance(max_actions, int) or isinstance(max_actions, bool) or max_actions < 0:
        raise ValueError("max_actions must be a nonnegative integer")
    rep0 = model0.representation if isinstance(model0, RationalInstrument) else model0
    rep1 = model1.representation if isinstance(model1, RationalInstrument) else model1
    if set(rep0.kernels) != set(rep1.kernels):
        raise ValueError("Models require the same action/outcome alphabet")
    alphabet = sorted({a for a, y in rep0.kernels})
    output = []
    for length in range(max_actions+1):
        for actions in product(alphabet, repeat=length):
            if grammar is not None and not grammar.accepts(actions):
                continue
            resources = _resource_totals(actions, evaluation_options.get("action_resource_upper_bounds"),
                                          evaluation_options.get("per_record_resource_upper_bounds"))
            allowed = True
            for name, limit in (per_record_resource_limits or {}).items():
                if name not in resources or rational(limit) < 0:
                    raise ValueError("Limits require named nonnegative resource bounds")
                if resources[name] > rational(limit):
                    allowed = False
            if allowed:
                output.append(evaluate_experiment(model0, model1, actions, grammar, **evaluation_options))
    return output
