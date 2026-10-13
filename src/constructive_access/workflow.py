"""End-to-end, model-bound exact access and finite experiment certificates."""
from __future__ import annotations

from fractions import Fraction as Q
from math import prod

from .contract import ContractError, canonical_json, load_contract
from ._vendor import accesskit as ak
from ._vendor import cost_certificate as cc

REPORT_SCHEMA = "constructive-access-report-v1"


class VerificationError(ValueError):
    """A report fails verification against the independently supplied contract."""


def _portable_representation(model):
    rep = model.representation
    return {"initial": list(map(str, rep.initial)), "terminal": list(map(str, rep.terminal)),
            "kernels": [{"action": a, "outcome": y,
                         "matrix": [list(map(str, row)) for row in matrix.tolist()]}
                        for (a, y), matrix in sorted(rep.kernels.items())]}


def _paired_model(contract):
    data = contract.to_dict()
    models = contract.instruments()
    pair = cc.pair_model(*[_portable_representation(m) for m in models],
                         data["resources"]["action_costs"], data.get("grammar"))
    return models, pair


def _log_upper(x, terms=48):
    """Exact positive-series upper bound on log(x), x >= 1."""
    x = Q(x)
    if x < 1:
        raise ValueError("Logarithm bound requires x >= 1")
    power = 0
    while x >= 2:
        x /= 2
        power += 1

    def upper(y):
        z = (y-1)/(y+1)
        partial = 2*sum((z**(2*k+1)/(2*k+1) for k in range(terms)), Q(0))
        tail = 2*z**(2*terms+1)/((2*terms+1)*(1-z*z))
        return partial+tail

    return upper(x)+power*upper(Q(2))


def _ceil(x):
    return -(-x.numerator//x.denominator)


def _robust_experiment(data, experiment, charge):
    calibration = data.get("calibration")
    if calibration is None:
        return {"status": "not_supplied", "scope": "Nominal-model results do not include calibration error"}
    rows = experiment["transcripts"]
    event = [row for row in rows if Q(row["probability0"]) > Q(row["probability1"])]
    p0 = sum((Q(row["probability0"]) for row in event), Q(0))
    p1 = sum((Q(row["probability1"]) for row in event), Q(0))
    radius0, radius1 = map(Q, calibration["record_tv_radii"])
    lo0, hi1 = max(Q(0), p0-radius0), min(Q(1), p1+radius1)
    gap = lo0-hi1
    error_budget = Q(data["design"]["target_average_error"])
    failure = Q(calibration["joint_failure_probability"])
    conditional_error = error_budget-failure
    out = {"status": "pending", "event": [row["outcomes"] for row in event],
           "nominal_event_probabilities": [str(p0), str(p1)],
           "actual_model0_event_probability_lower": str(lo0),
           "actual_model1_event_probability_upper": str(hi1),
           "robust_gap_lower": str(gap),
           "joint_calibration_failure_probability": str(failure),
           "assumptions": ["Whole-record TV bounds hold jointly with the declared confidence",
                           "Conditional on pre-acquisition calibration information, each fixed actual hypothesis generates iid classification records; the calibration event certifies these conditional whole-record laws",
                           "The event and threshold are fixed before classification records are collected; classification records are not reused to select or calibrate them",
                           "TV radii and calibration confidence are supplied assumptions, not inferred by this toolkit"]}
    if not data["preparation"]["iid_records"]:
        out["status"] = "iid_assumption_not_declared"
    elif conditional_error <= 0:
        out["status"] = "calibration_failure_exhausts_error_budget"
    elif gap <= 0:
        out["status"] = "calibration_gap_not_certified"
    else:
        log_upper = _log_upper(1/conditional_error)
        count = _ceil(2*log_upper/gap**2)
        out.update({"status": "certified_under_declared_calibration", "records": count,
                    "event_frequency_threshold": str((lo0+hi1)/2),
                    "decision": "Choose target.labels[0] if event frequency exceeds threshold; otherwise choose target.labels[1]",
                    "conditional_each_hypothesis_error_upper": str(conditional_error),
                    "unconditional_each_hypothesis_error_upper": str(error_budget),
                    "log_inverse_conditional_error_upper": str(log_upper),
                    "hoeffding_exponent": str(Q(count)*gap**2/2),
                    "batch_declared_charge": str(count*charge),
                    "method": "Fixed-event Hoeffding bound with exact rational logarithm enclosure; no sample-count optimality claim"})
    return out


def _derive_report(contract, bundle):
    """Verify supplied algebra and evaluate its experiment, without synthesis."""
    data = contract.to_dict()
    if not isinstance(bundle, dict) or set(bundle) != {"schema", "model", "model_sha256", "certificate"}:
        raise VerificationError("Certificate bundle has missing or unsupported fields")
    if canonical_json(bundle["model"]) != canonical_json(cc.canonical_model(bundle["model"])):
        raise VerificationError("Certificate model is not in the supported canonical format")
    cert = bundle["certificate"]
    if not isinstance(cert, dict) or set(cert) != {"mode", "witness", "basis", "closure", "beta_coefficients"}:
        raise VerificationError("Certificate has missing or unsupported fields")
    if cert["witness"] is not None and (not isinstance(cert["witness"], dict) or
                                        set(cert["witness"]) != {"word", "cost", "contrast"}):
        raise VerificationError("Witness has missing or unsupported fields")
    for entries, keys in ((cert["basis"], {"word", "cost", "effect"}),
                          (cert["closure"], {"source", "letter", "coefficients"})):
        if not isinstance(entries, list) or any(not isinstance(e, dict) or set(e) != keys for e in entries):
            raise VerificationError("Certificate identities have missing or unsupported fields")
    models, paired = _paired_model(contract)
    checked = cc.check_certificate(bundle, expected_model=paired)
    equivalent = checked["mode"] == "equivalent"
    report = {"schema": REPORT_SCHEMA, "toolkit_version": "0.1.0",
              "contract_sha256": contract.sha256, "contract_name": data["name"],
              "question": data["question"], "target": data["target"],
              "status": "equivalent_under_declared_access" if equivalent else "distinguishable_under_declared_access",
              "model_names": [m.name for m in models], "certificate": bundle,
              "certificate_verification": checked,
              "resources": data["resources"], "preparation": data["preparation"],
              "scope": ["Exactly two specified finite classical rational instruments",
                        "Exact equivalence or minimum additive action-charge nonzero witness under the declared action grammar",
                        "A finite model certificate does not establish physical calibration, model completeness, or historical novelty",
                        "Minimum witness charge does not imply minimum repeated-record charge or optimal adaptive design",
                        "Resource accounting covers only the explicitly included items"],
              "experiment": None, "robust_experiment": None}
    if equivalent:
        report["conclusion"] = "The two different target labels cannot be distinguished by any permitted finite protocol in these supplied models"
        report["scope"].append("No amount of repeated sampling separates identical laws; this does not rule out changing the access or model assumptions")
        return report
    letters = {letter["id"]: letter for letter in paired["letters"]}
    word = bundle["certificate"]["witness"]["word"]
    symbols = [(letters[s]["action"], letters[s]["outcome"]) for s in word]
    actions = [a for a, _ in symbols]
    report["witness"] = {"symbols": [list(s) for s in symbols], "actions": actions,
                         "outcome_event": [y for _, y in symbols],
                         "probabilities": [str(m.word_probability(symbols)) for m in models],
                         "minimum_action_charge": checked["minimum_action_cost"],
                         "length": len(word),
                         "implementation": "Apply the entire fixed action sequence to every record and retain every outcome; no postselection"}
    transcript_count = prod(len(models[0].actions[a]) for a in actions)
    options = data["design"]
    if transcript_count > options["max_transcripts"]:
        report["experiment"] = {"status": "transcript_limit_exceeded", "transcript_count": transcript_count,
                                "configured_limit": options["max_transcripts"],
                                "conclusion": "Exact access certificate remains valid; full statistical evaluation was not performed"}
        return report
    result = ak.evaluate_experiment(*models, actions, grammar=contract.grammar(),
                                    target_error=options["target_average_error"],
                                    precision_bits=options["precision_bits"],
                                    max_precision_bits=options["max_precision_bits"],
                                    max_transcripts=options["max_transcripts"])
    experiment = result.to_dict()
    # The unified schema records resource semantics separately instead of using
    # legacy unconstrained dictionaries that can be mistaken for measured heat.
    for key in ("per_record_resource_upper_bounds", "certified_batch_resource_upper_bounds",
                "affordable_records", "budget_average_error_upper"):
        experiment.pop(key)
    experiment["status"] = "evaluated"
    experiment["classifier"] = "Equal-prior likelihood ratio on complete iid transcripts; ties may be assigned arbitrarily"
    charge = Q(checked["minimum_action_cost"])+Q(data["resources"]["per_record_overhead"])
    experiment["per_record_declared_charge"] = str(charge)
    experiment["nominal_batch_declared_charge"] = (None if result.sufficient_records is None else
                                                   str(result.sufficient_records*charge))
    if not data["preparation"]["iid_records"]:
        experiment["status"] = "single_record_only_iid_not_declared"
        for key in ("sufficient_records", "certified_average_error_upper", "nominal_batch_declared_charge"):
            experiment[key] = None
        experiment["assumptions"] = ["Known models and equal prior probabilities; no repeated-record guarantee is asserted"]
        experiment["classifier"] = "Single-record likelihood ratio only; repeated-record guarantee requires iid preparation"
    report["experiment"] = experiment
    report["robust_experiment"] = _robust_experiment(data, experiment, charge)
    report["conclusion"] = "A least-action-charge contrast exists; the complete action experiment and its explicitly scoped statistical guarantees are reported"
    return report


def analyze(source):
    """Synthesize, independently check, and evaluate a supported contract.

    This computes no parameter fit from observations and makes no empirical
    discovery claim. Finite transcript enumeration has an explicit safety cap.
    """
    contract = load_contract(source)
    _, paired = _paired_model(contract)
    bundle = cc.generate_certificate(paired)
    # Search diagnostics are not mathematical evidence and are not independently
    # checked by the portable verifier. Do not export them as certified data.
    bundle.pop("generator_statistics")
    return _derive_report(contract, bundle)


def verify_report(source, report):
    """Check algebra and every reported quantity against a caller-bound model.

    No witness search, rank calculation, or cost optimizer is invoked. The
    independently supplied contract must be trusted as the intended model; an
    internally consistent report and model are not a calibration certificate.
    """
    contract = load_contract(source)
    if not isinstance(report, dict) or report.get("schema") != REPORT_SCHEMA:
        raise VerificationError("Unrecognized report schema")
    if report.get("contract_sha256") != contract.sha256:
        raise VerificationError("Report is bound to a different supplied contract")
    try:
        rebuilt = _derive_report(contract, report.get("certificate"))
        if canonical_json(rebuilt) != canonical_json(report):
            raise VerificationError("Report fields differ from independently recomputed quantities")
    except (cc.CertificateError, ContractError, KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, VerificationError):
            raise
        raise VerificationError(str(exc)) from exc
    return {"verified": True, "contract_sha256": contract.sha256,
            "status": report["status"], "certificate_mode": report["certificate_verification"]["mode"],
            "synthesis_repeated": False,
            "scope": "Mathematical report integrity under the independently supplied contract; empirical calibration is not checked"}
