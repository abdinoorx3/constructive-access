"""Repair a two-lag noise obstruction by a fixed, longer correlation lag.

The new experimental choice is model-assisted and evaluated on fresh records.
No theorem says that an unknown spectrum must possess a useful negative lag.
"""
from fractions import Fraction as F
from pathlib import Path
import argparse
import hashlib
import json
import math

from constructive_access.frontier import energy_certificate as ec
from constructive_access.virtual.contrast import sqrt_upper
from constructive_access.robust.preparation import exp_negative_bounds
try:
    from . import joint_analog
    from .spectral_dynamics import from_gap
except ImportError:
    import joint_analog
    from spectral_dynamics import from_gap


SENSOR = [4096, 3896, 3314, 2408, 1266, 0, -1266, -2408, -3314, -3896,
          -4096, -3896, -3314, -2408, -1266, 0, 1266, 2408, 3314, 3896]


def certify_mean(records, mean, lag=F(1, 2), alpha=F(1, 20),
                 mean_error=0, conditional_radius=0, external_failure=0,
                 selection="predeclared", detector_contract="memoryless_independent_endpoint_noise",
                 record_contract="independent_stationary_pairs"):
    """Certify negative autocorrelation with one bounded analog pair score.

    Mean is the exact empirical mean of W=(1+Z0*Zlag)/2 in [0,1].
    Conditional radius must hold given the complete preceding record if an
    independent preparation contract is replaced by conditional mean control.
    """
    if type(records) is not int or records <= 0:
        raise ValueError("A positive fixed integer quota is required")
    if selection not in ("predeclared", "independent_pilot", "model_assisted_before_new_data"):
        raise ValueError("Choosing the lag on the same inference data is unsupported")
    if detector_contract != "memoryless_independent_endpoint_noise":
        raise ValueError("The physical detector must have memoryless independent endpoint noise")
    if record_contract not in ("independent_stationary_pairs", "bounded_conditional_mean_error"):
        raise ValueError("Independent pairs or a valid conditional-mean bound required")
    mean, lag, alpha, error, drift, eta = map(ec.exact,
        (mean, lag, alpha, mean_error, conditional_radius, external_failure))
    if lag <= 0 or not 0 < alpha < 1 or not 0 <= eta < 1 or alpha+eta >= 1:
        raise ValueError("Positive lag and valid failure allocation required")
    if error < 0 or drift < 0 or mean+error < 0 or mean-error > 1:
        raise ValueError("Invalid empirical mean or declared error radius")
    radius = sqrt_upper(ec.log_interval(1/alpha)[1]/(2*records))
    signed = F(1, 2)-mean-radius-error-drift
    gap = max(F(0), signed)
    bridge = from_gap(gap, lag_unit=lag/2, range_bound=2)
    return {"status": bridge["status"], "records": records, "mean": mean,
            "lag": lag, "sampling_radius_upper": radius, "mean_error": error,
            "conditional_mean_penalty": drift, "signed_gap_lower": signed,
            "negative_correlation_gap_lower": gap, "direct_bridge": bridge,
            "entropy_production_rate_lower": bridge["entropy_production_rate_lower"],
            "confidence_failure_upper": alpha+eta, "selection": selection,
            "acquisitions": records, "stationary_pair_preparations": records,
            "detector_readouts": 2*records, "declared_evolution_time": records*lag,
            "calibration_acquisitions": 0, "record_contract": record_contract,
            "detector_contract": detector_contract,
            "physical_contract": "Stationary homogeneous Markov model with ordinary even-variable reversal, valid stationary preparation, unchanged bounded noninvasive detector and calibrated clock. Covered overdamped diffusions also qualify; odd-variable reversal requires a different proof.",
            "resource_scope": "Preparation, reset, readout, processing time and energy overhead are unmeasured and not assigned zero."}


def certify_product_sum(records, product_sum, product_denominator, **kwargs):
    if type(records) is not int or records <= 0 or type(product_sum) is not int or type(product_denominator) is not int or product_denominator <= 0:
        raise ValueError("Exact integer sum, positive denominator and quota required")
    mean = F(records*product_denominator+product_sum, 2*records*product_denominator)
    return certify_mean(records, mean, **kwargs)


def correlation_interval(lag=F(1, 2), forward=F(20), backward=F(1), noisy=True, terms=80):
    """Exact rounded-ring correlation interval by rational uniformization."""
    lag, forward, backward = map(ec.exact, (lag, forward, backward))
    rate = forward+backward; x = rate*lag
    if lag <= 0 or forward <= 0 or backward <= 0 or type(terms) is not int or terms < 1 or x >= terms+2:
        raise ValueError("Positive rates/lag and adequate uniformization truncation required")
    f = [F(v, 8192 if noisy else 4096) for v in SENSOR]; states = len(f)
    A = sum((v*v for v in f), F(0))/states
    vector = list(f); weight = F(1); partial = A
    for k in range(1, terms+1):
        vector = [(forward*vector[(j+1) % states]+backward*vector[(j-1) % states])/rate for j in range(states)]
        weight *= x/k
        partial += weight*sum((a*b for a, b in zip(f, vector)), F(0))/states
    tail = weight*x/(terms+1)/(1-x/(terms+2))
    exp_bounds = exp_negative_bounds(x, terms)
    products = [e*c for e in exp_bounds for c in (partial-A*tail, partial+A*tail)]
    return [max(-A, min(products)), min(A, max(products))]


def population_audit():
    old = joint_analog.noisy_population_obstruction()
    interval = correlation_interval()
    gap = max(F(0), -interval[1]/2)
    return {"scope": "Exact model-assisted population redesign after the original failure; not a rule for all unknown spectra",
            "old_two_lag_family_gap_upper": old["max_gap_all_real_s_upper"],
            "new_lag": F(1, 2), "new_correlation_interval": interval,
            "population_negative_score_gap_lower": gap,
            "population_bridge": from_gap(gap, lag_unit=F(1, 4)),
            "actual_entropy_rate_interval": [19*x for x in ec.log_interval(F(20))],
            "detector_memory_countermodel": "An equilibrium one-state target plus an unobserved driven sensor ring generates exactly the same analog law. The full target-plus-sensor entropy is positive, but target-only entropy is zero. Assigning the inferred rate to the target requires the memoryless detector contract."}


def _plain(value):
    if isinstance(value, F): return str(value)
    if isinstance(value, dict): return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [_plain(v) for v in value]
    return value


def _read(states, rng, sensor, noisy):
    import numpy as np
    if noisy: return sensor[states]+4096*rng.choice([-1, 1], size=len(states)), 8192
    return sensor[states], 4096


def _pair(rng, count, lag, forward, backward, sensor, noisy):
    x = rng.integers(0, len(sensor), size=count)
    y = (x+rng.poisson(float(forward*lag), size=count)-rng.poisson(float(backward*lag), size=count)) % len(sensor)
    left, d0 = _read(x, rng, sensor, noisy); right, d1 = _read(y, rng, sensor, noisy)
    product = left*right
    return {"records": count, "product_sum": int(product.sum()), "product_denominator": d0*d1}, left, right


def analyze_statistics(stat, design):
    nc = stat["baseline_calibration"]["records"]; cal = stat["baseline_calibration"]
    B = F(cal["sum_squares"], nc*cal["denominator"]**2)
    means = [(1+B)/2]; quotas = [nc]
    for pair in stat["baseline_pairs"]:
        n = pair["records"]; d = pair["product_denominator"]
        means.append(F(n*d+pair["product_sum"], 2*n*d)); quotas.append(n)
    baseline = {s: joint_analog.certify_means(quotas, means, parameter=F(s),
                     lag_unit=F(design["baseline_lag_unit"]), alpha=F(design["alpha"]))
                for s in design["baseline_parameters"]}
    new = certify_product_sum(**stat["retimed_pairs"], lag=F(design["new_lag"]),
                               alpha=F(design["alpha"]), selection="model_assisted_before_new_data")
    return baseline, new


def run(root):
    import numpy as np
    from constructive_access.calibrated.analog_benchmark import wilson
    from constructive_access.virtual.thermodynamic_lab import deterministic_npz
    source = root/"contracts/retiming.json"; raw = source.read_bytes(); design = json.loads(raw)
    if not design["frozen_before_heldout_generation"]: raise ValueError("Freeze the redesign before its new data")
    sensor = np.asarray(SENSOR, dtype=np.int64); seed = design["seed"]; reps = design["replicates_per_case"]
    old_delta = F(design["baseline_lag_unit"]); tau = F(design["new_lag"])
    nc, old_n, new_n = design["baseline_calibration_records"], design["baseline_pairs_per_lag"], design["new_pair_records"]
    rows = []; summaries = []; archive = {}
    for ci, case in enumerate(design["cases"]):
        name = case["name"]; noisy = case["noisy"]
        forward, backward = F(case["forward_rate"]), F(case["backward_rate"])
        truth = float((forward-backward)*math.log(float(forward/backward)))
        oracle = correlation_interval(tau, forward, backward, noisy)
        case_rows = []
        for rep in range(reps):
            rng = np.random.default_rng(np.random.SeedSequence([seed, ci, rep, 0]))
            z, den = _read(rng.integers(0, len(sensor), size=nc), rng, sensor, noisy)
            stat = {"baseline_calibration": {"records": nc, "sum_squares": int((z*z).sum()), "denominator": den},
                    "baseline_pairs": []}
            if rep == 0: archive[name+"__baseline_calibration"] = z
            for k in (1, 2):
                rng = np.random.default_rng(np.random.SeedSequence([seed, ci, rep, k]))
                pair, left, right = _pair(rng, old_n, k*old_delta, forward, backward, sensor, noisy)
                stat["baseline_pairs"].append(pair)
                if rep == 0:
                    archive[name+f"__baseline_lag{k}_left"] = left
                    archive[name+f"__baseline_lag{k}_right"] = right
            rng = np.random.default_rng(np.random.SeedSequence([seed, ci, rep, 3]))
            pair, left, right = _pair(rng, new_n, tau, forward, backward, sensor, noisy)
            stat["retimed_pairs"] = pair
            if rep == 0:
                archive[name+"__retimed_left"] = left; archive[name+"__retimed_right"] = right
            baseline, new = analyze_statistics(stat, design)
            rates = {"baseline_s="+key: float(value["best_entropy_production_rate_lower"]) for key, value in baseline.items()}
            rates["retimed"] = float(new["entropy_production_rate_lower"])
            row = {"case": name, "replicate": rep, "exact_statistics": stat, "rate_lower": rates}
            if rep == 0: row["illustrative_exact_certificates"] = _plain({"baseline": baseline, "retimed": new})
            rows.append(row); case_rows.append(row)
            if (rep+1) % 100 == 0: print(f"{name}: {rep+1}/{reps}", flush=True)
        methods = {}
        for key in case_rows[0]["rate_lower"]:
            rate = np.asarray([r["rate_lower"][key] for r in case_rows]); positive = int(np.sum(rate > 0))
            methods[key] = {"positive_certificates": positive, "positive_fraction": positive/reps,
                            "wilson95": wilson(positive, reps), "mean_rate_lower": float(rate.mean()),
                            "median_rate_lower": float(np.median(rate)),
                            "empirical_upper_coverage_failures": int(np.sum(rate > truth+1e-12)),
                            "mean_fraction_of_true_entropy": None if truth == 0 else float(rate.mean()/truth)}
        summaries.append({"case": name, "replicates": reps, "actual_entropy_rate": truth,
                          "new_lag_correlation_interval": _plain(oracle), "methods": methods})
        print(json.dumps({"case": name, "detections": {k: v["positive_certificates"] for k, v in methods.items()}}), flush=True)
    target = root/"results/retiming_raw.npz"; deterministic_npz(target, archive)
    result = {"evidence": "Fresh held-out synthetic data after model-assisted retiming; not laboratory measurements",
              "contract_sha256": hashlib.sha256(raw).hexdigest(), "design": design,
              "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "baseline_implementation_sha256": hashlib.sha256(Path(joint_analog.__file__).read_bytes()).hexdigest(),
              "summaries": summaries, "replicate_results": rows, "population_audit": _plain(population_audit()),
              "raw_archive": target.name, "raw_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
              "scope": "900 evolution-time units for each design. Retimed design uses 3600 readings/1800 preparations; old design 16000 readings/10000 acquisitions. Unmeasured preparation and processing overhead prevents a claim of equal total time or calibrated joules. Separate 95% methods; their maximum is not a 95% certificate."}
    out = root/"results/retiming.json"; out.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n"); print(out, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(); run(args.root)
