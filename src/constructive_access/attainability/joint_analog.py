"""Joint reversible-moment inference from bounded analog measurements.

Calibration uses one reading Z, not a conditionally independent twin.  Its
square estimates B=E Z^2 >= E(E[Z|X]^2)=A, with the correct inequality sign.
The zero-lag coordinate contributes sampling uncertainty and acquisition cost
but is invariant under the stationary additive-reversible comparison.
"""
from fractions import Fraction as F
from pathlib import Path
import argparse
import hashlib
import json
import math

from constructive_access.frontier import energy_certificate as ec
from constructive_access.virtual.contrast import sqrt_upper
from constructive_access.calibrated import analog, calibration
try:
    from .spectral_dynamics import from_gap
except ImportError:
    from spectral_dynamics import from_gap


def support(parameter=F(1, 2)):
    s = ec.exact(parameter)
    if not 0 <= s <= 1:
        raise ValueError("Use an exact fixed parameter in [0,1]")
    return {"parameter": s, "coefficients": [-s*s, 2*s, F(-1)],
            "reversible_support": -(1-s)**2/2,
            "score_bounds": [(F(1, 2), F(1)), (F(0), F(1)), (F(0), F(1))]}


def certify_means(records, means, parameter=F(1, 2), lag_unit=F(1),
                  alpha=F(1, 20), mean_errors=None, conditional_radii=None,
                  external_failure=0, selection="predeclared",
                  record_contract="independent_stationary_acquisitions",
                  detector_contract="same_memoryless_noninvasive_bounded_channel"):
    """Exact fixed-contrast certificate at lags [0, Delta, 2 Delta].

    The first mean summarizes (1+Z^2)/2 from single stationary readings;
    the next two summarize (1+Z0*Zt)/2 from independent stationary pairs.
    Conditional detector noises at distinct endpoints must be independent.
    Conditional radii bound each acquisition's conditional mean discrepancy
    from its ideal stationary target, given the complete preceding record.
    The contract is declared, not verified by these statistics.
    """
    if selection not in ("predeclared", "independent_pilot"):
        raise ValueError("Same-data parameter tuning requires separate inference")
    if record_contract not in ("independent_stationary_acquisitions", "bounded_conditional_mean_error"):
        raise ValueError("Supply independent records or a conditional-mean contract")
    if detector_contract != "same_memoryless_noninvasive_bounded_channel":
        raise ValueError("The same bounded memoryless noninvasive detector is required")
    n = list(records); p = list(map(ec.exact, means))
    if len(n) != 3 or len(p) != 3 or any(type(x) is not int or x <= 0 for x in n):
        raise ValueError("Three positive integer quotas and exact means required")
    sup = support(parameter); a = sup["coefficients"]; s = sup["parameter"]
    err = [F(0)]*3 if mean_errors is None else list(map(ec.exact, mean_errors))
    drift = [F(0)]*3 if conditional_radii is None else list(map(ec.exact, conditional_radii))
    if len(err) != 3 or len(drift) != 3 or any(x < 0 for x in err+drift):
        raise ValueError("Three nonnegative numerical and conditional radii required")
    if record_contract == "bounded_conditional_mean_error" and conditional_radii is None:
        raise ValueError("Explicit conditional radii required")
    for m, e, bounds in zip(p, err, sup["score_bounds"]):
        if m+e < bounds[0] or m-e > bounds[1]:
            raise ValueError("Supplied empirical mean is outside its score range")
    delta, alpha, eta = map(ec.exact, (lag_unit, alpha, external_failure))
    if delta <= 0 or not 0 < alpha < 1 or not 0 <= eta < 1 or alpha+eta >= 1:
        raise ValueError("Positive lag and valid failure allocation required")
    proxy = s**4/(4*n[0]) + 4*s*s/n[1] + F(1, n[2])
    radius = sqrt_upper(ec.log_interval(1/alpha)[1]*proxy/2)
    observed = sum((x*y for x, y in zip(a, p)), F(0))
    numeric = sum((abs(x)*y for x, y in zip(a, err)), F(0))
    conditional = sum((abs(x)*y for x, y in zip(a, drift)), F(0))
    signed = observed-sup["reversible_support"]-radius-numeric-conditional
    gap = max(F(0), signed)
    denominator = (2*s*sqrt_upper(delta)+sqrt_upper(2*delta))**2
    direct = from_gap(gap, lag_unit=delta, range_bound=2)
    return {"status": "positive_certificate" if gap else "no_positive_certificate",
            "support": sup, "records": n, "means": p, "lag_unit": delta,
            "empirical_contrast": observed, "concentration_variance_proxy": proxy,
            "sampling_radius_upper": radius, "quantization_penalty": numeric,
            "conditional_mean_penalty": conditional, "signed_gap_lower": signed,
            "reversible_gap_lower": gap, "comparison_time_denominator_upper": denominator,
            "entropy_production_rate_lower": 8*gap*gap/denominator,
            "direct_bridge": direct,
            "best_entropy_production_rate_lower": max(8*gap*gap/denominator, direct["entropy_production_rate_lower"]),
            "confidence_failure_upper": alpha+eta, "external_failure_upper": eta,
            "acquisitions": sum(n), "detector_readouts": n[0]+2*n[1]+2*n[2],
            "declared_evolution_time": delta*n[1]+2*delta*n[2],
            "unmeasured_overhead": "Stationary preparation, resets and detector energy are not assigned zero cost",
            "selection": selection, "record_contract": record_contract,
            "detector_contract": detector_contract,
            "physical_contract": "Stationary even-state homogeneous CTMC or covered overdamped diffusion; ordinary reversal; noninvasive unchanged memoryless detector; independent endpoint noise; calibrated time. Entropy is in nats per time unit, not uncalibrated heat."}


def certify_sums(records, score_sums, **kwargs):
    n = list(records); sums = list(map(ec.exact, score_sums))
    if len(n) != 3 or len(sums) != 3 or any(type(x) is not int or x <= 0 for x in n):
        raise ValueError("Three positive integer quotas and exact sums required")
    return certify_means(n, [v/N for v, N in zip(sums, n)], **kwargs)


def quantized_square_summary(readings, denominator=2**24):
    """Exactly round (1+Z^2)/2 for stored digital readings, with error radius.

    Rounding error refers to stored readings, not unknown physical sensor error.
    """
    if type(denominator) is not int or denominator <= 0:
        raise ValueError("Positive integer quantizer required")
    count = total = squares = 0
    for raw in readings:
        z = analog._stored_number(raw)
        if not -1 <= z <= 1:
            raise ValueError("Readings must lie in [-1,1]")
        w = (1+z*z)/2
        q = (2*w.numerator*denominator+w.denominator)//(2*w.denominator)
        count += 1; total += q; squares += q*q
    if count == 0:
        raise ValueError("At least one reading required")
    mean = F(total, count*denominator)
    return {"records": count, "mean": mean, "score_sum": F(total, denominator),
            "mean_quantization_radius": F(1, 2*denominator),
            "quantized_empirical_variance": F(squares, count*denominator**2)-mean*mean,
            "quantizer_denominator": denominator, "status": "descriptive_only"}


def population_gap(B, correlation_one, correlation_two, parameter=F(1, 2)):
    B, C1, C2 = map(ec.exact, (B, correlation_one, correlation_two))
    s = support(parameter)["parameter"]
    if not 0 <= B <= 1 or any(abs(x) > B for x in (C1, C2)):
        raise ValueError("Bounded second moment and compatible correlations required")
    return (-s*s*B+2*s*C1-C2)/2


def noisy_population_obstruction(terms=48):
    """Exact retrospective population audit of the frozen noisy ring.

    Positive rational uniformization encloses both correlations. Completing
    the square then upper-bounds the gap for *all real s*, so failure of the
    entire two-lag family cannot be blamed only on finite sampling.
    This does not claim impossibility for different lags or other observables.
    """
    from constructive_access.robust.preparation import exp_negative_bounds
    if type(terms) is not int or terms < 4:
        raise ValueError("Use at least four terms for these fixed lags")
    sensor = [4096, 3896, 3314, 2408, 1266, 0, -1266, -2408, -3314, -3896,
              -4096, -3896, -3314, -2408, -1266, 0, 1266, 2408, 3314, 3896]
    f = [F(v, 8192) for v in sensor]; count = len(f)
    A = sum((v*v for v in f), F(0))/count; B = A+F(1, 4)
    intervals = []
    for lag in (F(1, 10), F(1, 5)):
        x = 21*lag; vector = list(f); weight = F(1); partial = A
        for k in range(1, terms+1):
            vector = [(20*vector[(j+1) % count]+vector[(j-1) % count])/21 for j in range(count)]
            weight *= x/k
            partial += weight*sum((a*b for a, b in zip(f, vector)), F(0))/count
        tail = weight*x/(terms+1)/(1-x/(terms+2))
        elo, ehi = exp_negative_bounds(x, terms)
        products = [e*c for e in (elo, ehi) for c in (partial-A*tail, partial+A*tail)]
        intervals.append([max(-A, min(products)), min(A, max(products))])
    c1lo, c1hi = intervals[0]; c2lo, c2hi = intervals[1]
    squared_upper = max(c1lo*c1lo, c1hi*c1hi)
    gap_upper = (squared_upper/B-c2lo)/2
    return {"scope": "Exact population obstruction for this fixed detector and two lags, not all accessible experiments",
            "A": A, "B": B, "correlation_intervals": intervals,
            "max_gap_all_real_s_upper": gap_upper,
            "no_population_witness_in_entire_quadratic_family": gap_upper < 0,
            "uniformization_terms": terms, "detector_integers": sensor,
            "interpretation": "Completing the square gives sup_s g_s=(C1^2/B-C2)/2. Negative certified upper means collecting more samples at these same settings does not create a positive population gap."}


def _plain(value):
    if isinstance(value, F): return str(value)
    if isinstance(value, dict): return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [_plain(v) for v in value]
    return value


def _read(states, rng, sensor, scale, noisy=False, zero_gain=False):
    import numpy as np
    if zero_gain: return np.zeros(len(states), dtype=np.int64), scale
    if noisy: return sensor[states]+scale*rng.choice([-1, 1], size=len(states)), 2*scale
    return sensor[states], scale


def analyze_statistics(stat, design):
    """Replay all six frozen analyses through the declared production APIs."""
    nc = stat["calibration_records"]; den = stat["calibration_denominator"]
    m = F(stat["calibration_sum"], nc*den)
    B = F(stat["calibration_sum_squares"], nc*den**2)
    cal = calibration.analog_calibration_summary(nc, m, B, F(design["split_alpha_calibration"]))
    n = [x["records"] for x in stat["science"]]
    means = [F(x["records"]*x["product_denominator"]+x["product_sum"],
               2*x["records"]*x["product_denominator"]) for x in stat["science"]]
    coins = [F(x["encoded_successes"], x["records"]) for x in stat["science"]]
    common = dict(records=[nc, *n], lag_unit=F(design["lag_unit"]), alpha=F(design["joint_alpha"]))
    base_s, tuned_s = map(F, design["parameters"])
    result = {"joint_analog": certify_means(means=[(1+B)/2, *means], parameter=base_s, **common),
              "joint_analog_tuned": certify_means(means=[(1+B)/2, *means], parameter=tuned_s, **common),
              "joint_coin_tuned": certify_means(means=[(1+B)/2, *coins], parameter=tuned_s, **common)}
    kwargs = dict(lags=[1, 2], records=n,
        durations=[F(design["lag_unit"]), 2*F(design["lag_unit"])], calibration_vertices=cal["vertices"],
        alpha=F(design["split_alpha_science"]), calibration_failure=F(design["split_alpha_calibration"]))
    result["split_analog"] = analog.certify_means(coefficients=[2*base_s, -1], means=means, **kwargs)
    result["split_analog_tuned"] = analog.certify_means(coefficients=[2*tuned_s, -1], means=means, **kwargs)
    result["split_coin"] = analog.certify_means(coefficients=[2*base_s, -1], means=coins, **kwargs)
    corrected = certify_means(means=[(1+B)/2, *means], parameter=tuned_s,
                             conditional_radii=[F(1, 2), F(0), F(0)],
                             record_contract="bounded_conditional_mean_error", **common)
    return result, cal, corrected


def run(root):
    import numpy as np
    from constructive_access.calibrated.analog_benchmark import population, wilson
    from constructive_access.virtual.thermodynamic_lab import deterministic_npz
    path = root/"contracts/joint_analog.json"; raw = path.read_bytes(); design = json.loads(raw)
    if not design["frozen_before_heldout_generation"]: raise ValueError("Freeze the design first")
    seed, reps = design["seed"], design["replicates_per_case"]
    scale, states = design["sensor_grid_denominator"], design["cycle_states"]
    nc, N = design["calibration_records"], design["science_records_per_lag"]
    delta = F(design["lag_unit"]); lags = [delta, 2*delta]
    sensor = np.rint(scale*np.cos(2*np.pi*np.arange(states)/states)).astype(np.int64)
    rows = []; summaries = []; raw_archive = {}
    for ci, case in enumerate(design["cases"]):
        name = case["name"]; noisy = case["detector"] == "half_cosine_plus_half_fair_noise"
        forward, backward = float(F(case["forward_rate"])), float(F(case["backward_rate"]))
        oracle = population(states, forward, backward, sensor, scale, lags,
                            "noisy_driven" if noisy else "driven")
        oracle["stationary_entropy_rate"] = (forward-backward)*math.log(forward/backward)
        oracle["true_joint_gaps"] = {str(s): float(population_gap(
            F(oracle["observed_second_moment"]), F(2*oracle["agreement_means"][0]-1),
            F(2*oracle["agreement_means"][1]-1), F(s))) for s in design["parameters"]}
        case_rows = []
        for rep in range(reps):
            rng = np.random.default_rng(np.random.SeedSequence([seed, ci, rep, 0]))
            st = rng.integers(0, states, size=nc)
            z, den = _read(st, rng, sensor, scale, noisy, case["calibration_detector"] == "zero_gain")
            stat = {"calibration_records": nc, "calibration_sum": int(np.sum(z)),
                    "calibration_sum_squares": int(np.sum(z*z)), "calibration_denominator": den, "science": []}
            if rep == 0: raw_archive[name+"__calibration"] = z
            for k, lag in enumerate(lags, 1):
                rng = np.random.default_rng(np.random.SeedSequence([seed, ci, rep, k]))
                x = rng.integers(0, states, size=N)
                y = (x+rng.poisson(forward*float(lag), size=N)-rng.poisson(backward*float(lag), size=N)) % states
                z0, d0 = _read(x, rng, sensor, scale, noisy); zt, dt = _read(y, rng, sensor, scale, noisy)
                prod = z0*zt; D = d0*dt
                bits = rng.random(N) < (D+prod)/(2*D)
                stat["science"].append({"records": N, "product_sum": int(np.sum(prod)),
                    "product_denominator": D, "encoded_successes": int(np.sum(bits))})
                if rep == 0:
                    raw_archive[name+f"__lag{k}_left"] = z0
                    raw_archive[name+f"__lag{k}_right"] = zt
                    raw_archive[name+f"__lag{k}_coins"] = bits.astype(np.uint8)
            results, cal, corrected = analyze_statistics(stat, design)
            rates = {key: float(value["entropy_production_rate_lower"]) for key, value in results.items()}
            direct_linear = {key: float(value["direct_bridge"]["linear_rate_lower"])
                             for key, value in results.items() if "direct_bridge" in value}
            direct_nonlinear = {key: float(value["direct_bridge"]["entropy_production_rate_lower"])
                                for key, value in results.items() if "direct_bridge" in value}
            row = {"case": name, "replicate": rep, "exact_statistics": stat, "rate_lower": rates,
                   "direct_linear_rate_lower": direct_linear, "direct_nonlinear_rate_lower": direct_nonlinear,
                   "conservative_calibration_mismatch_repair_rate": float(corrected["entropy_production_rate_lower"]),
                   "joint_analog_sampling_radius": float(results["joint_analog"]["sampling_radius_upper"]),
                   "joint_analog_tuned_sampling_radius": float(results["joint_analog_tuned"]["sampling_radius_upper"]),
                   "split_sampling_radius": float(results["split_analog"]["sampling_radius_upper"]),
                   "split_calibration_B_upper": float(cal["observed_second_moment_upper"])}
            if rep == 0: row["illustrative_exact_certificates"] = _plain(results)
            case_rows.append(row); rows.append(row)
            if (rep+1) % 100 == 0: print(f"{name}: {rep+1}/{reps}", flush=True)
        names = list(case_rows[0]["rate_lower"])
        methods = {}
        for key in names:
            rates = np.array([r["rate_lower"][key] for r in case_rows]); detections = int(np.sum(rates > 0))
            methods[key] = {"positive_certificates": detections, "positive_fraction": detections/reps,
                            "wilson95": wilson(detections, reps), "mean_rate_lower": float(rates.mean()),
                            "median_rate_lower": float(np.median(rates)), "maximum_rate_lower": float(rates.max())}
        bridges = {}
        for key in case_rows[0]["direct_linear_rate_lower"]:
            linear = np.array([r["direct_linear_rate_lower"][key] for r in case_rows])
            nonlinear = np.array([r["direct_nonlinear_rate_lower"][key] for r in case_rows])
            quadratic_mean = methods[key]["mean_rate_lower"]
            bridges[key] = {"mean_linear_rate_lower": float(linear.mean()),
                            "mean_nonlinear_rate_lower": float(nonlinear.mean()),
                            "mean_nonlinear_to_quadratic_ratio": None if quadratic_mean == 0 else float(nonlinear.mean()/quadratic_mean),
                            "mean_fraction_of_true_entropy": None if oracle["stationary_entropy_rate"] == 0 else float(nonlinear.mean()/oracle["stationary_entropy_rate"]),
                            "empirical_upper_coverage_failures": int(np.sum(nonlinear > oracle["stationary_entropy_rate"]+1e-12))}
        summaries.append({"case": name, "contract_valid": case["contract_valid"], "population": oracle,
                          "replicates": reps, "methods": methods, "direct_bridges": bridges,
                          "mismatch_repair_positives": sum(r["conservative_calibration_mismatch_repair_rate"] > 0 for r in case_rows)})
        print(json.dumps({"case": name, "detections": {k: v["positive_certificates"] for k, v in methods.items()}}), flush=True)
    raw_path = root/"results/joint_analog_raw.npz"; deterministic_npz(raw_path, raw_archive)
    result = {"evidence": "Synthetic frozen held-out stationary experiments; no new physical measurements",
              "design": design, "contract_sha256": hashlib.sha256(raw).hexdigest(),
              "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "summaries": summaries, "replicate_results": rows,
              "raw_archive": raw_path.name, "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
              "scope": "Matched raw data and resources; six predeclared statistical methods are reported individually. Their maximum is not a 95% certificate. The methods table keeps the earlier quadratic physical bridge to isolate statistical design; direct_bridges reuses each joint method's same confidence event for the stronger linear/nonlinear physical bridge. Wilson intervals summarize Monte Carlo detection counts, not physical evidence."}
    out = root/"results/joint_analog.json"; out.write_text(json.dumps(result, sort_keys=True, indent=2)+"\n")
    print(out, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--population-audit", action="store_true")
    args = parser.parse_args()
    if args.population_audit:
        target = args.root/"results/joint_analog_population_audit.json"
        target.write_text(json.dumps(_plain(noisy_population_obstruction()), sort_keys=True, indent=2)+"\n")
        print(target)
    else:
        run(args.root)
