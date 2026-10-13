"""Exploratory held-out diagnostics of published harmonic-trap measurements.

No independence-based confidence intervals or thermodynamic certificates are
computed. The lower-noise detector remains a measurement, not latent truth.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

if __package__:
    from .empirical_loader import load_measurement
else:
    from empirical_loader import load_measurement


TRAIN = 10000
BLOCKS = 8
LAGS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256)
FIT_LAGS = (1, 2, 4, 8, 16)
DT = 1e-5


def covariance(x, y, lag):
    """Ordered empirical covariance with whole-segment means and N-lag divisor."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim != 1 or x.shape != y.shape or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Matching finite one-dimensional records required")
    if not isinstance(lag, (int, np.integer)) or not 0 <= lag < x.size:
        raise ValueError("lag must be an integer smaller than the record")
    a, b = x - x.mean(), y - y.mean()
    return float(np.mean(a * b) if lag == 0 else np.mean(a[:-lag] * b[lag:]))


def transform_parameters(training_records):
    """Training-only channel centers and a common scale for the supplied group."""
    records = [np.asarray(x, float) for x in training_records]
    if not records or any(x.ndim != 1 or len(x) < 2 or not np.all(np.isfinite(x)) for x in records):
        raise ValueError("Finite training records required")
    scale = 2 * np.sqrt(np.mean([np.var(x) for x in records]))
    if scale <= 0:
        raise ValueError("Positive training variance required")
    return [float(x.mean()) for x in records], float(scale)


def bounded_transform(x, center, scale):
    if not np.isfinite(center) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("Finite center and positive scale required")
    return np.tanh((np.asarray(x, float) - center) / scale)


def moment_margins(g, variance):
    """Population-inequality margins; nonnegative is compatible, not sufficient."""
    if variance < 0 or not np.isfinite(variance):
        raise ValueError("Finite nonnegative variance required")
    rows = []
    for k in sorted(g):
        if k <= 0:
            continue
        rows.append({"inequality": "nonnegative_centered_covariance", "lags": [k], "margin": float(g[k])})
        if 2 * k in g:
            drop = g[k] - g[2 * k]
            rows.append({"inequality": "monotone_decay", "lags": [k, 2*k], "margin": float(drop)})
            rows.append({"inequality": "variance_upper_only_drop", "lags": [k, 2*k], "margin": float(variance / 4 - drop)})
        if 4 * k in g:
            # 2k=(2/3)k+(1/3)4k: Holder gives g(2k)^3<=g(k)^2g(4k).
            rows.append({"inequality": "weighted_centered_log_convexity", "lags": [k, 2*k, 4*k], "margin": float(g[k]**2 * g[4*k] - g[2*k]**3)})
    return rows


def segment_moments(z):
    mean = float(np.mean(z))
    variance = float(np.var(z))
    g = {k: covariance(z, z, k) for k in LAGS}
    return {
        "sample_mean": mean,
        "observed_second_moment": float(np.mean(z*z)),
        "observed_variance": variance,
        "covariances": [{"lag_samples": k, "covariance": g[k], "pair_count": len(z)-k} for k in LAGS],
        "inequality_margins": moment_margins(g, variance),
    }


def bounded_diagnostic(z):
    result = segment_moments(z)
    blocks = [segment_moments(b) for b in np.array_split(z, BLOCKS)]
    for j, row in enumerate(result["inequality_margins"]):
        values = [b["inequality_margins"][j]["margin"] for b in blocks]
        row["block_min"] = float(min(values))
        row["block_max"] = float(max(values))
        row["blocks_negative"] = int(sum(v < 0 for v in values))
    result["contiguous_block_means"] = [b["sample_mean"] for b in blocks]
    result["contiguous_block_variances"] = [b["observed_variance"] for b in blocks]
    result["interpretation"] = "Plug-in margins, not simultaneous confidence bounds; negative margins alone do not reject reversibility."
    return result


def detector_rows(x, y):
    r = x-y
    vr = covariance(r, r, 0)
    rows = []
    for k in LAGS:
        xx, yy = covariance(x, x, k), covariance(y, y, k)
        xy, yx = covariance(x, y, k), covariance(y, x, k)
        rr = covariance(r, r, k)
        rows.append({
            "lag_samples": k, "lag_seconds": k*DT, "pair_count": len(x)-k,
            "auto_n1_nm2": xx, "auto_reference_nm2": yy,
            "auto_reference_minus_n1_nm2": yy-xx,
            "cross_n1_to_reference_nm2": xy, "cross_reference_to_n1_nm2": yx,
            "symmetrized_cross_nm2": (xy+yx)/2,
            "cross_lag_asymmetry_nm2": xy-yx,
            "residual_covariance_nm2": rr,
            "residual_autocorrelation": rr/vr if vr > 0 else None,
        })
    return rows


def detector_diagnostic(x, y):
    rows = detector_rows(x, y)
    block_rows = [detector_rows(a,b) for a,b in zip(np.array_split(x,BLOCKS), np.array_split(y,BLOCKS))]
    for j, row in enumerate(rows):
        for metric in ("residual_autocorrelation", "cross_lag_asymmetry_nm2", "auto_reference_minus_n1_nm2"):
            v = [b[j][metric] for b in block_rows]
            row[metric+"_block_range"] = [float(min(v)),float(max(v))]
    latent = rows[0]["symmetrized_cross_nm2"]
    return {
        "covariance_rows": rows,
        "unit_gain_independent_noise_variance_decomposition": {
            "latent_variance_candidate_nm2": latent,
            "n1_noise_variance_candidate_nm2": rows[0]["auto_n1_nm2"] - latent,
            "reference_noise_variance_candidate_nm2": rows[0]["auto_reference_nm2"] - latent,
            "interpretation": "Conditional on unit gains and mutually independent additive detector errors independent of particle; not a calibrated noise estimate.",
        },
    }


def fit_ou_training(x, y):
    c = np.array([(covariance(x,y,k)+covariance(y,x,k))/2 for k in FIT_LAGS])
    if np.any(c <= 0):
        return {"status": "unavailable: nonpositive training cross covariance"}
    slope, intercept = np.polyfit(np.array(FIT_LAGS)*DT, np.log(c), 1)
    if slope >= 0:
        return {"status": "unavailable: nondecaying training fit"}
    return {"status": "fitted", "fit_lags_samples": list(FIT_LAGS),
            "amplitude_nm2": float(np.exp(intercept)), "relaxation_rate_per_second": float(-slope),
            "relaxation_time_seconds": float(-1/slope),
            "interpretation": "Naive single-exponential covariance fit on training data; not a mixing or thermodynamic calibration."}


def analyze(directory):
    names = ["harmwell_n1.h5", "harmwell_n1_groundtruth.h5", "harmwell_n2.h5"]
    records = {name: load_measurement(Path(directory), name) for name in names}
    x, y, z = [records[n][0] for n in names]
    centers, scale = transform_parameters([x[:TRAIN],y[:TRAIN]])
    zcenters, zscale = transform_parameters([z[:TRAIN]])
    parameters = [(centers[0],scale),(centers[1],scale),(zcenters[0],zscale)]
    transformed = {name: {
        "training_center_nm": center, "training_scale_nm": width,
        **bounded_diagnostic(bounded_transform(raw[TRAIN:],center,width))
    } for name,raw,(center,width) in zip(names,[x,y,z],parameters)}
    comparison = detector_diagnostic(x[TRAIN:],y[TRAIN:])
    fit = fit_ou_training(x[:TRAIN],y[:TRAIN])
    if fit["status"] == "fitted":
        latent = comparison["covariance_rows"][0]["symmetrized_cross_nm2"]
        fit["held_out_comparison"] = [{
            "lag_samples": row["lag_samples"],
            "predicted_covariance_nm2": float(fit["amplitude_nm2"] * np.exp(-fit["relaxation_rate_per_second"]*row["lag_seconds"])),
            "observed_symmetrized_cross_nm2": row["symmetrized_cross_nm2"],
            "error_normalized_by_zero_lag_cross": float((fit["amplitude_nm2"] * np.exp(-fit["relaxation_rate_per_second"]*row["lag_seconds"])-row["symmetrized_cross_nm2"])/latent),
        } for row in comparison["covariance_rows"] if row["lag_samples"] > 0]
    return {
        "analysis_status": "exploratory descriptive reanalysis of published experimental records",
        "source_doi": "10.5281/zenodo.6680673", "paper_doi": "10.1016/j.isci.2022.104731",
        "source_authors": ["J. S. Bryan IV", "P. Basak", "J. Bechhoefer", "S. Presse"], "source_license": "CC-BY-4.0",
        "training_samples": TRAIN, "evaluation_samples": len(x)-TRAIN,
        "lag_grid_samples": list(LAGS), "nominal_dt_seconds": DT,
        "evaluation_span_seconds_per_record": (len(x)-TRAIN-1)*DT,
        "training_span_seconds_per_record": (TRAIN-1)*DT,
        "eight_block_lengths": [len(b) for b in np.array_split(x[TRAIN:],BLOCKS)],
        "protocol_note": "Split and lag grid fixed before new correlation calculations; whole-file descriptive statistics and sources had already been inspected, so this is not prospectively blinded validation.",
        "exposure_note": "Simultaneous detectors share one physical exposure; overlapping lag pairs and adjacent blocks are dependent. Separate n2 record has undocumented alignment to n1.",
        "original_sha256": {n: records[n][1]["sha256"] for n in names},
        "detector_comparison": comparison, "training_ou_model": fit,
        "bounded_channels": transformed,
        "thermodynamic_certificate": {"status": "abstain", "alpha": None, "entropy_rate_lower_bound": None,
            "reasons": ["No independently established hidden-process dependence bound", "No uniform detector memory or conditional-independence calibration", "No actual timestamp jitter bound", "No finite-sample stationary mean/variance calibration", "Population Markov/reversal and physical entropy scope remain assumptions"]},
        "block_interpretation": "Ranges across eight contiguous, internally recentered blocks are descriptive sensitivity checks, not confidence intervals or independent replicates.",
    }


def main():
    base = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir",type=Path,default=base/"data/source_audit")
    p.add_argument("--output",type=Path,default=base/"results/empirical_analysis.json")
    args=p.parse_args()
    result=analyze(args.data_dir)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
