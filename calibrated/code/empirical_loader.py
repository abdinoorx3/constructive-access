"""Read immutable public measurements; produce a descriptive, non-certifying audit.

Requires h5py and numpy. No serialized source code from the data is executed.
Nominal units are taken from the cited authors' source, not inferred from values.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


FILES = {
    "harmwell_n1.h5": "ffa81dda2576934bd51cf544199a12da7d5568806883401a08848ad215c1286c",
    "harmwell_n1_groundtruth.h5": "ef390252ce9730b53c688958cad5fa4ff2379800f09ccc13ac2d6e90bb0e413b",
    "harmwell_n2.h5": "38efb704da6fb86a9da49938d3e84ff4c3d1e4c56588768cbdf37149f28b4a2e",
}
def load_measurement(directory: Path, name: str) -> tuple[np.ndarray, dict]:
    """Validate a selected file and load only its numerical position series."""
    if name not in FILES:
        raise ValueError("Only the three audited public measurement files are supported")
    path = Path(directory) / name
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != FILES[name]:
        raise ValueError(f"SHA-256 mismatch for {name}")
    key = "x_data" if "groundtruth" in name else "data"
    with h5py.File(path, "r") as h5:
        x = np.asarray(h5[key][()], dtype=float)
        if x.shape != (49999,) or not np.all(np.isfinite(x)):
            raise ValueError(f"Unexpected position schema in {name}")
        dt = None if "groundtruth" in name else float(h5["dt"][()])
        if dt is not None and dt != 10000.0:
            raise ValueError("Unexpected nominal sampling interval")
        schema = {k: {"shape": list(h5[k].shape), "dtype": str(h5[k].dtype)} for k in h5}
    return x, {
        "file": name, "sha256": digest, "schema": schema,
        "position_unit": "nm", "nominal_dt_seconds": 1e-5,
        "dt_provenance": "authors' code: dt in ns; stored scalar 10000; lower-noise file aligned to n1 by repository description",
        "actual_timestamps_present": False,
        "role": "simultaneous lower-noise measurement" if "groundtruth" in name else "experimental detector trajectory",
        "license": "CC-BY-4.0", "source_doi": "10.5281/zenodo.6680673",
    }


def descriptive(x: np.ndarray) -> dict:
    return {
        "count": int(x.size), "nominal_span_seconds": float((x.size - 1) * 1e-5),
        "mean_nm": float(np.mean(x)), "standard_deviation_nm": float(np.std(x)),
        "min_nm": float(np.min(x)), "max_nm": float(np.max(x)),
        "lag1_pearson_correlation": float(np.corrcoef(x[:-1], x[1:])[0, 1]),
        "ten_contiguous_block_means_nm": [float(v.mean()) for v in np.array_split(x, 10)],
    }


def audit(directory: Path) -> dict:
    loaded = {name: load_measurement(directory, name) for name in FILES}
    x = loaded["harmwell_n1.h5"][0]
    reference = loaded["harmwell_n1_groundtruth.h5"][0]
    residual = x - reference
    return {
        "status": "descriptive empirical audit; no entropy-production certificate",
        "simulation": False, "new_measurement": False,
        "source_doi": "10.5281/zenodo.6680673",
        "paper_doi": "10.1016/j.isci.2022.104731",
        "source_apparatus_statement": "Selected harmonic wells are described as optical tweezers without feedback; feedback produces other potentials in the study.",
        "records": [{**meta, **descriptive(values)} for values, meta in loaded.values()],
        "simultaneous_detector_comparison": {
            "aligned_samples": int(x.size),
            "pearson_correlation": float(np.corrcoef(x, reference)[0, 1]),
            "difference_mean_nm": float(np.mean(residual)),
            "difference_rmse_nm": float(np.sqrt(np.mean(residual ** 2))),
            "difference_lag1_correlation": float(np.corrcoef(residual[:-1], residual[1:])[0, 1]),
            "interpretation": "Difference between two measurements, not an error series against exact latent positions; does not establish uniform detector calibration.",
        },
        "certificate_eligibility": {
            "decision": "abstain",
            "unestablished_conditions": [
                "stationary initialization or a controlled nonstationary correction",
                "independently valid hidden-process dependence or mixing bound",
                "uniform detector error, conditional independence and memory contract",
                "actual clock jitter bound beyond nominal dt",
                "validated Markov state, time-reversal parity and physical entropy accounting",
            ],
            "prohibited_inferences": [
                "Observed lag correlation alone does not calibrate hidden-state mixing.",
                "A fitted force function in the reference HDF5 is not a measured force trajectory.",
                "Finite descriptive stationarity checks cannot prove stationarity.",
                "Particle bath dissipation does not include unmeasured laser, detector or controller expenditure.",
            ],
        },
    }


def main() -> None:
    base = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=base / "data/source_audit")
    parser.add_argument("--output", type=Path, default=base / "results/data_audit.json")
    args = parser.parse_args()
    result = audit(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
