"""Replay one frozen simulated detector dataset through the public exact API.

No records are generated here. The input is the recorded first driven dataset
from the frozen virtual laboratory. It is a simulation, not a laboratory test.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path


def certification_module():
    """Prefer the installed package; source fallback uses only the copied tree."""
    name = "constructive_access.detector_calibration.certification"
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name not in ("constructive_access", "constructive_access.detector_calibration", name):
            raise
    root = Path(__file__).resolve().parents[1]
    candidates = (root/"code/certification.py", root/"detector_calibration/code/certification.py")
    source = next((p for p in candidates if p.is_file()), None)
    if source is None:
        raise ModuleNotFoundError("Install constructive-access with detector calibration, or retain its relative source tree")
    spec = importlib.util.spec_from_file_location("_calibration_tutorial_certification", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def default_input():
    root = Path(__file__).resolve().parents[1]
    candidates = (Path(__file__).with_name("calibration_counts.json"),
                  root/"examples/calibration_counts.json", root/"experiment/calibration_counts.json",
                  root/"detector_calibration/experiment/calibration_counts.json")
    return next((p for p in candidates if p.is_file()), candidates[0])


def endpoint_baseline(confidence, calibration, bounds, cert):
    """The same covered event gives a broader-class endpoint lower bound.

    This endpoint contrast does not impose the local Gamma(f) geometry identity.
    It still requires the declared stationary ordinary-reversal diffusion,
    detector contract and generator bounds for the finite exposure correction.
    """
    if calibration["status"] != "covered_region":
        return {"entropy_lower": F(0), "status": "calibration_abstention"}
    h, lam = F(confidence["h"]), F(confidence["lambda"])
    b = 1-lam*h
    if not 0 <= b <= 1:
        raise ValueError("This fixed tutorial endpoint baseline requires 0<=1-lambda*h<=1")
    interval = cert.normalized_interval(confidence["statistics"]["S0"]["interval"], 2, calibration["gain"])
    blur = cert.bias_bounds(0, h, lam, calibration["width"][1], bounds)["delta_N"]
    contrast = max(F(0), interval[0]-blur)
    lower = 2*h**3*contrast**2/(2*b+cert.sqrt_upper(F(2)))**2
    return {"entropy_lower": lower, "corrected_contrast_lower": contrast,
            "status": "positive_certificate" if lower else "no_positive_certificate",
            "model_class": "stationary ordinary-reversal diffusion without the pointwise Gamma identity",
            "confidence_event": "same simultaneous science and calibration event; no additional allocation"}


def combined_certificate(geometry, endpoint):
    """Take a maximum on the one shared confidence event and stricter model class."""
    lower_geometry, lower_endpoint = geometry["entropy_lower"], endpoint["entropy_lower"]
    if not isinstance(lower_geometry, F) or not isinstance(lower_endpoint, F):
        raise TypeError("Exact Fraction bounds from this tutorial's common-event pipeline are required")
    if min(lower_geometry, lower_endpoint) < 0:
        raise ValueError("Nonnegative covered lower bounds required")
    selected = "geometry" if lower_geometry >= lower_endpoint else "endpoint"
    lower = max(lower_geometry, lower_endpoint)
    return {"entropy_lower": lower, "selected_component": selected,
            "status": "positive_certificate" if lower else "no_positive_certificate",
            "failure_probability_upper": geometry["alpha"], "additional_failure_allocation": F(0),
            "confidence_event": "Both components use the same simultaneous science and reference-calibration event",
            "joint_model_scope": "Ordinary stationary compact diffusion with the exact pointwise identity Gamma(f)=gamma*(1-f^2), unknown constant gamma>0, and all declared detector and generator-bound premises",
            "scope_note": "The maximum inherits the stricter geometry contract; endpoint selection does not extend this combined procedure to the endpoint-only model class",
            "tie_rule": "Select geometry when the two lower bounds are equal"}


def validate_combination_fixture():
    """Synthetic algebra checks only; these are not additional simulated datasets."""
    for g, e, selected in ((F(1, 3), F(1, 2), "endpoint"),
                           (F(3, 4), F(1, 2), "geometry"),
                           (F(0), F(0), "geometry")):
        result = combined_certificate({"entropy_lower": g, "alpha": F(1, 20)}, {"entropy_lower": e})
        if result["entropy_lower"] != max(g, e) or result["selected_component"] != selected or result["failure_probability_upper"] != F(1, 20) or result["additional_failure_allocation"] != 0:
            raise AssertionError("Shared-event combination branch check failed")
    return {"status": "passed", "cases": 3,
            "scope": "Synthetic combination-algebra fixtures, not experimental evidence"}


def replay(fixture, cert):
    if fixture.get("data_origin") != "simulated":
        raise ValueError("This tutorial requires its declared frozen simulated-data fixture")
    contract, dataset = fixture["contract"], fixture["dataset"]
    design = contract
    order, records = design["order"], design["science_records"]
    counts = dataset["science_counts"]
    if len(counts) != len(cert.category_order(order)) or sum(counts) != records:
        raise ValueError("Science counts do not match the frozen design")
    summary = cert.summarize_counts(dict(zip(cert.category_order(order), counts)), order,
                                   design["h"], design["lambda_"], design["nominal_gain"])
    confidence = cert.confidence_statistics(summary, design["alpha_target"])
    nref = design["reference_records"]
    calibration = cert.calibrate_references(nref, dataset["gain_plus"], nref, dataset["step_plus"],
                                            design["step_time"], design["width_prior"], design["alpha_calibration"],
                                            gain_prior=design["gain_prior"])
    geometry = cert.certify(confidence, calibration, design["generator_bounds"])
    baseline = endpoint_baseline(confidence, calibration, design["generator_bounds"], cert)
    combined = combined_certificate(geometry, baseline)
    recorded = dataset["certificate"]
    if geometry["entropy_lower"] != F(recorded["entropy_lower"]) or baseline["entropy_lower"] != F(recorded["endpoint_lower"]) or geometry.get("order_selected") != recorded["order_selected"]:
        raise ValueError("Exact replay differs from the frozen benchmark record")
    target = next(case for case in contract["cases"] if case["name"] == dataset["case"])
    tv_charge = F(dataset["simulation_product_law_tv_upper"])
    return {"status": "replayed", "data_origin": "simulated",
            "description": "Exact replay of one frozen virtual-laboratory dataset; not physical measurements",
            "case": dataset["case"], "replicate": dataset["replicate"],
            "science_records": records, "science_readouts": records*(2*order+4),
            "reference_readouts": 2*nref, "design": design,
            "calibration": calibration, "geometry": geometry, "endpoint_baseline": baseline,
            "combined": combined,
            "recorded_results_match_exactly": True,
            "evaluation_only_simulator_entropy": F(target["omega"])**2/F(target["gamma"]),
            "simulation_product_law_tv_upper": tv_charge,
            "ideal_random_bit_simulator_failure_upper": min(F(1), geometry["alpha"]+tv_charge),
            "confidence_event_count": confidence["confidence_event_count"],
            "scope": {"geometry": "requires exact pointwise Gamma(f)=gamma*(1-f^2), with unknown positive constant gamma",
                      "baseline": "broader model class; lower numerical bound is not a matched-assumption efficiency claim",
                      "coverage": "finite-model confidence conditional on the stated physical premises; simulator approximation charge is reported in the frozen benchmark",
                      "cost": "replicas and calibration readings counted; preparation, clock, waveform and energy costs are additional"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = args.input if args.input is not None else default_input()
    cert = certification_module()
    fixture = json.loads(source.read_text())
    code_sha = hashlib.sha256(Path(cert.__file__).read_bytes()).hexdigest()
    if code_sha != fixture["contract"]["source_sha256"]["code/certification.py"]:
        raise ValueError("Certification source differs from the frozen dataset contract")
    result = replay(fixture, cert)
    result["combination_branch_checks"] = validate_combination_fixture()
    result["source_sha256"] = {"fixture": hashlib.sha256(source.read_bytes()).hexdigest(),
                               "certification.py": code_sha,
                               "calibration_tutorial.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    text = json.dumps(cert.jsonable(result), indent=2, sort_keys=True)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
