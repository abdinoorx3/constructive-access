"""Finite-menu, target-specific experimental suites with exact certificates.

This module uses complete finite record laws, not latent state reconstruction.
Optimality refers only to the sum of the supplied charges for selecting a suite
once from this finite menu; repetitions and apparatus costs are separate.
"""
from __future__ import annotations
from fractions import Fraction
from itertools import combinations
from math import ceil


class SuiteError(ValueError):
    """A malformed or computationally unsupported finite design contract."""


def _q(value):
    if isinstance(value, bool) or not isinstance(value, (int, str, Fraction)):
        raise SuiteError("Exact entries must be integers, rational strings, or Fractions.")
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise SuiteError("Invalid rational entry.") from exc


def _s(value):
    return str(value)


def _validate(targets, menu):
    if not isinstance(targets, dict) or len(targets) < 2:
        raise SuiteError("At least two named candidate models are required.")
    if any(not isinstance(k, str) or not k or not isinstance(v, str) or not v
           for k, v in targets.items()):
        raise SuiteError("Model names and target labels must be nonempty strings.")
    if not isinstance(menu, list) or not menu:
        raise SuiteError("A nonempty finite experiment menu is required.")
    names = set()
    checked = []
    for experiment in menu:
        if not isinstance(experiment, dict) or set(experiment) != {"name", "charge", "laws", "tv_radii"}:
            raise SuiteError("Each experiment needs exactly name, charge, laws, tv_radii.")
        name = experiment["name"]
        if not isinstance(name, str) or not name or name in names:
            raise SuiteError("Experiment names must be distinct nonempty strings.")
        names.add(name)
        charge = _q(experiment["charge"])
        if charge < 0:
            raise SuiteError("Selection charges must be nonnegative.")
        laws, radii = experiment["laws"], experiment["tv_radii"]
        if not isinstance(laws, dict) or set(laws) != set(targets):
            raise SuiteError("Each experiment must supply every candidate law.")
        if not isinstance(radii, dict) or set(radii) != set(targets):
            raise SuiteError("Each experiment must supply every whole-record TV radius.")
        first = laws[next(iter(targets))]
        if not isinstance(first, dict) or not first:
            raise SuiteError("A finite nonempty record alphabet is required.")
        alphabet = set(first)
        if any(not isinstance(y, str) or not y for y in alphabet):
            raise SuiteError("Recorded outcomes must have nonempty string labels.")
        fixed = {}
        for h, law in laws.items():
            if not isinstance(law, dict) or set(law) != alphabet:
                raise SuiteError("The record alphabet must be common to all candidates.")
            fixed[h] = {y: _q(p) for y, p in law.items()}
            if any(p < 0 for p in fixed[h].values()) or sum(fixed[h].values()) != 1:
                raise SuiteError("Candidate record laws must be normalized and nonnegative.")
        eps = {h: _q(v) for h, v in radii.items()}
        if any(not 0 <= v <= 1 for v in eps.values()):
            raise SuiteError("Whole-record TV radii must lie in [0,1].")
        checked.append({"name": name, "charge": charge, "laws": fixed, "tv_radii": eps})
    return checked


def design_suite(targets, menu, delta="1/20", *, max_pairs=20, max_states=100000):
    """Select a minimum-charge separating suite and give an iid event test.

    The finite candidate with the true target must include the actual system,
    up to the declared whole-record TV radii. Records for any one experiment
    must be iid under that fixed system. No independence across experiments
    is needed for the simultaneous confidence bound. Radii must be justified
    externally; a joint calibration failure probability is added separately.
    """
    experiments = _validate(targets, menu)
    delta = _q(delta)
    if not 0 < delta < 1:
        raise SuiteError("The sampling failure budget must lie strictly between zero and one.")
    if isinstance(max_pairs, bool) or not isinstance(max_pairs, int) or max_pairs < 0:
        raise SuiteError("max_pairs must be a nonnegative integer.")
    if isinstance(max_states, bool) or not isinstance(max_states, int) or max_states < 1:
        raise SuiteError("max_states must be a positive integer.")
    pairs = [(i, j) for i, j in combinations(sorted(targets), 2) if targets[i] != targets[j]]
    if len(pairs) > max_pairs:
        raise SuiteError("Exact finite-menu cover exceeds the declared pair cap.")
    base = {"schema": "constructive-access-suite-v1", "targets": dict(targets),
            "sampling_failure_budget": _s(delta), "pair_count": len(pairs),
            "scope": "Known finite candidate family; fixed iid law within each selected experiment; externally justified whole-record TV radii.",
            "objective": "Minimum sum of one-use selection charges within the supplied finite menu; not minimum repeated-record charge or total energy."}
    if not pairs:
        return dict(base, status="target_constant", target=next(iter(targets.values())),
                    selected=[], selection_charge="0", test_events=[], total_records=0,
                    repeated_charge="0", assumptions=["The true target is represented by the declared family."])
    masks, witnesses = [], []
    union = 0
    for experiment in experiments:
        mask = 0
        events = {}
        for k, (i, j) in enumerate(pairs):
            p, q = experiment["laws"][i], experiment["laws"][j]
            event = tuple(sorted(y for y in p if p[y] > q[y]))
            tv = sum(p[y] - q[y] for y in event)
            gap = tv - experiment["tv_radii"][i] - experiment["tv_radii"][j]
            if gap > 0:
                mask |= 1 << k
                events[k] = (event, gap)
        masks.append(mask)
        witnesses.append(events)
        union |= mask
    full = (1 << len(pairs)) - 1
    if union != full:
        return dict(base, status="insufficient_robust_menu", selected=[],
                    unresolved_pairs=[list(pair) for k, pair in enumerate(pairs) if not (union >> k & 1)],
                    interpretation="No event with positive TV-ball separation exists for these pairs in this menu. This does not certify equivalence under other experiments or under more structured calibration sets.")
    # Exact weighted set cover: masks summarize which cross-target pairs are separated.
    # A snapshot per experiment permits using each experiment at most once.
    dp = {0: (Fraction(0), ())}
    for index, experiment in enumerate(experiments):
        for mask, (cost, chosen) in list(dp.items()):
            new_mask = mask | masks[index]
            candidate = (cost + experiment["charge"], chosen + (index,))
            old = dp.get(new_mask)
            if old is None or (candidate[0], len(candidate[1]), candidate[1]) < (old[0], len(old[1]), old[1]):
                dp[new_mask] = candidate
            if len(dp) > max_states:
                raise SuiteError("Exact finite-menu cover exceeds the declared state cap.")
    cost, chosen = dp[full]
    # Assign each required pair its widest available gap among the chosen experiments.
    assigned = {index: {} for index in chosen}
    pair_certificates = []
    for k, pair in enumerate(pairs):
        choices = [(witnesses[index][k][1], -index, index) for index in chosen if k in witnesses[index]]
        gap, _, index = max(choices)
        event, _ = witnesses[index][k]
        previous = assigned[index].get(event)
        assigned[index][event] = gap if previous is None else min(previous, gap)
        pair_certificates.append({"models": list(pair), "experiment": experiments[index]["name"],
                                  "event": list(event), "robust_gap": _s(gap)})
    event_count = sum(len(v) for v in assigned.values())
    # e > 2 implies 2 L exp(-2 n r²) <= 2 L / 2^k when 2 n r² >= k.
    exponent = 0
    while Fraction(2 * event_count, 2 ** exponent) > delta:
        exponent += 1
    test_events, selected = [], []
    total_records, repeated_charge = 0, Fraction(0)
    for index in chosen:
        experiment = experiments[index]
        if not assigned[index]:
            raise SuiteError("Internal error: selected redundant experiment.")
        radius = min(assigned[index].values()) / 4
        records = ceil(Fraction(exponent, 1) / (2 * radius ** 2))
        selected.append({"name": experiment["name"], "selection_charge": _s(experiment["charge"]),
                         "records": records, "sampling_radius": _s(radius),
                         "alphabet": sorted(next(iter(experiment["laws"].values())))})
        total_records += records
        repeated_charge += records * experiment["charge"]
        for event, _ in sorted(assigned[index].items()):
            test_events.append({"experiment": experiment["name"], "event": list(event),
                                "nominal": {h: _s(sum(law[y] for y in event)) for h, law in experiment["laws"].items()},
                                "tv_radii": {h: _s(v) for h, v in experiment["tv_radii"].items()},
                                "sampling_radius": _s(radius)})
    return dict(base, status="certified_suite", selected=selected,
                selection_charge=_s(cost), repeated_charge=_s(repeated_charge),
                total_records=total_records, test_events=test_events,
                pair_certificates=pair_certificates, event_count=event_count,
                union_bound_exponent=exponent,
                sampling_failure_upper=_s(Fraction(2 * event_count, 2 ** exponent)),
                cover_states=len(dp),
                assumptions=["One fixed true candidate target has every selected actual law within its declared TV ball.",
                             "Each experiment uses the stated number of fresh independent records with a fixed law.",
                             "Preparation, controls, readout, nuisance parameters and reset are included in the record laws.",
                             "A joint probability of calibration failure must be added to the sampling failure bound."])


def verify_suite(targets, menu, report, *, expected_delta="1/20", max_pairs=20, max_states=100000):
    """Replay the complete exact finite-menu design bound to independently supplied inputs.

    This is a recomputation check, not a distinct independent certificate algorithm.
    """
    if not isinstance(report, dict) or "sampling_failure_budget" not in report:
        raise SuiteError("Malformed suite report.")
    if _q(report["sampling_failure_budget"]) != _q(expected_delta):
        raise SuiteError("Report sampling budget differs from the independently requested budget.")
    expected = design_suite(targets, menu, expected_delta,
                            max_pairs=max_pairs, max_states=max_states)
    if report != expected:
        raise SuiteError("Suite report does not match the supplied design contract.")
    return {"valid": True, "method": "Exact full recomputation from supplied inputs."}


def infer_target(targets, menu, report, counts, *, expected_delta="1/20", max_pairs=20, max_states=100000):
    """Check the design, retain compatible models, and commit only to one target.

    counts[experiment][outcome] contains nonnegative integer frequencies for
    every declared outcome, including zero counts. Empty or ambiguous target
    sets are returned explicitly. Nonrejection never proves the family true.
    """
    verify_suite(targets, menu, report, expected_delta=expected_delta, max_pairs=max_pairs, max_states=max_states)
    if report["status"] == "target_constant":
        if counts != {}:
            raise SuiteError("No data are needed for a target-constant family.")
        return {"status": "commit", "target": report["target"], "retained_models": sorted(targets),
                "condition": "Valid only if the true target belongs to the declared family."}
    if report["status"] != "certified_suite":
        return {"status": "abstain_insufficient_access", "retained_models": sorted(targets)}
    if not isinstance(counts, dict) or set(counts) != {x["name"] for x in report["selected"]}:
        raise SuiteError("Supply exactly the selected experiments' outcome counts.")
    for experiment in report["selected"]:
        observed = counts[experiment["name"]]
        if not isinstance(observed, dict) or set(observed) != set(experiment["alphabet"]):
            raise SuiteError("All declared outcomes, including zero counts, must be retained.")
        if any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in observed.values()):
            raise SuiteError("Outcome counts must be nonnegative integers.")
        if sum(observed.values()) != experiment["records"]:
            raise SuiteError("Record count differs from the certified fixed sample size.")
    retained = set(targets)
    sizes = {x["name"]: x["records"] for x in report["selected"]}
    for test in report["test_events"]:
        observed = counts[test["experiment"]]
        frequency = Fraction(sum(observed[y] for y in test["event"]), sizes[test["experiment"]])
        for h in list(retained):
            if abs(frequency - _q(test["nominal"][h])) > _q(test["tv_radii"][h]) + _q(test["sampling_radius"]):
                retained.remove(h)
    values = sorted({targets[h] for h in retained})
    if not values:
        status = "abstain_all_rejected"
    elif len(values) == 1:
        status = "commit"
    else:
        status = "abstain_ambiguous"
    result = {"status": status, "retained_models": sorted(retained), "retained_targets": values,
              "sampling_failure_upper": report["sampling_failure_upper"],
              "condition": "Requires the stated candidate-coverage, TV-calibration and iid assumptions."}
    if len(values) == 1:
        result["target"] = values[0]
    return result
