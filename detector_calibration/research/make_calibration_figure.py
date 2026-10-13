"""Deterministic figures for the frozen detector benchmark and design pilot.

The two panels deliberately separate sampled finite-data certificates from
population plug-in design diagnostics. No laboratory data are represented.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


def number(value):
    return float(F(value)) if isinstance(value, (str, int)) else float(value)


def read_inputs(benchmark, records, pilot):
    report = json.loads(benchmark.read_text())
    if report.get("status") != "completed_frozen_virtual_experiment" or report.get("data_origin") != "simulated":
        raise ValueError("A completed frozen simulated benchmark is required")
    if hashlib.sha256(records.read_bytes()).hexdigest() != report["counts_sha256"]:
        raise ValueError("Raw record archive differs from the benchmark binding")
    raw = json.loads(gzip.decompress(records.read_bytes()))
    if raw["contract_sha256"] != report["contract_sha256"]:
        raise ValueError("Record and report contracts differ")
    if len(raw["records"]) != report["datasets"]:
        raise ValueError("Record count differs from the completed report")
    design = json.loads(pilot.read_text())
    if design.get("hardware_contract") != "initial" or design.get("status") != "unfrozen_exact_population_diagnostics_no_random_records":
        raise ValueError("Use the explicitly labeled initial-hardware population pilot")
    parameters = design["parameters"]
    expected = {"gamma": F(1), "omega": F(3), "gain": F(9, 10),
                "nominal_gain": F(9, 10), "lambda_": F(1),
                "target_records": F(10000000), "reference_records_each": F(100000),
                "joint_failure_budget": F(1, 20)}
    if any(F(parameters[key]) != value for key, value in expected.items()):
        raise ValueError("Pilot parameters differ from the actual-rate, resource or confidence labels")
    if parameters["width_prior"] != "[actual_tau/2,2*actual_tau]" or parameters["step_time"] != "actual_tau/5":
        raise ValueError("Pilot reference waveform or exposure prior differs from the plotted design")
    if [F(x) for x in parameters["caps"]] != [F(4), F(4), F(13), F(13)]:
        raise ValueError("Pilot hardware bounds differ from the plotted contract")
    # The retained pilot metadata gives the total budget, not both allocations.
    # Its audited source fixes each allocation at 1/40 in prospective(). Guard
    # that binding as well; the total budget alone does not establish the split.
    if design["source_sha256"]["projected_pilot.py"] != "c864479987f597daaec509666885d4cc27564d8de3fad7b1241ef885c7704a24":
        raise ValueError("The 0.025/0.025 label requires the audited pilot allocation source")
    return report, raw, design


def make_figure(report, raw, pilot):
    settings = {"font.family": "DejaVu Sans", "font.size": 10,
                "axes.titlesize": 12, "axes.labelsize": 10,
                "xtick.labelsize": 9, "ytick.labelsize": 9,
                "pdf.fonttype": 42, "ps.fonttype": 42,
                "axes.spines.top": False, "axes.spines.right": False,
                "savefig.facecolor": "white", "figure.facecolor": "white"}
    with plt.rc_context(settings):
        fig, (left, right) = plt.subplots(1, 2, figsize=(12.2, 5.9), gridspec_kw={"width_ratios": [1.28, 1]})
        fig.subplots_adjust(left=.075, right=.98, top=.85, bottom=.28, wspace=.29)
        blue, orange, green = "#1f628e", "#c97435", "#397757"
        label_map = {"driven": "Driven", "equilibrium": "Equilibrium",
                     "weak_driven": "Weak drive", "changed_calibration": "Changed\ncalibration"}
        cases = report["cases"]
        if any(case["datasets"] != 400 for case in cases):
            raise ValueError("This figure caption is tied to the frozen 400-dataset design")
        positions = np.arange(len(cases), dtype=float)
        geometry, baseline, truths, labels = [], [], [], []
        for case in cases:
            rows = [r for r in raw["records"] if r["case"] == case["name"]]
            if len(rows) != case["datasets"]:
                raise ValueError("Per-case record count differs from the frozen report")
            g = [number(row["certificate"]["entropy_lower"]) for row in rows]
            e = [number(row["certificate"]["endpoint_lower"]) for row in rows]
            if any(x < 0 for x in g+e):
                raise ValueError("A plotted lower certificate is negative")
            if sum(x > 0 for x in g) != case["positive"] or sum(x > 0 for x in e) != case["endpoint_positive"]:
                raise ValueError("Positive counts do not match the report")
            geometry.append(g)
            baseline.append(e)
            truths.append(number(case["simulator_entropy"]))
            labels.append(label_map.get(case["name"], case["name"])+f"\nG: {case['positive']}/{case['datasets']}\nE: {case['endpoint_positive']}/{case['datasets']}")
        for data, offset, color in ((geometry, -.15, blue), (baseline, .15, orange)):
            artists = left.boxplot(data, positions=positions+offset, widths=.23,
                                   whis=(2.5, 97.5), showfliers=False, patch_artist=True,
                                   manage_ticks=False, medianprops={"color": "black", "linewidth": 1.1},
                                   boxprops={"edgecolor": color, "linewidth": 1.1},
                                   whiskerprops={"color": color}, capprops={"color": color})
            for box in artists["boxes"]:
                box.set_facecolor(color)
                box.set_alpha(.68)
        left.scatter(positions, truths, marker="D", s=30, color="#252525", zorder=5)
        left.set_yscale("symlog", linthresh=1e-5, linscale=.55)
        left.set_ylim(-4e-6, max(30, max(truths)*1.5))
        left.set_yticks([0, 1e-4, 1e-2, 1, 20])
        left.set_yticklabels(["0", r"$10^{-4}$", r"$10^{-2}$", "1", "20"])
        left.set_xticks(positions, labels)
        left.set_ylabel("Entropy-production rate (per model time)")
        left.set_title("A  Frozen simulated datasets", loc="left", fontweight="bold")
        left.grid(axis="y", alpha=.18)
        left.legend(handles=[Patch(facecolor=blue, alpha=.68, label="G: Geometry lower bound"),
                             Patch(facecolor=orange, alpha=.68, label="E: Broader-class endpoint bound"),
                             Line2D([], [], linestyle="none", marker="D", color="#252525", label="Actual simulator rate")],
                    loc="center left", bbox_to_anchor=(.01, .54), fontsize=8.5, frameon=False)

        rows = sorted((r for r in pilot["cases"] if F(r["lag"]) == F(7, 200) and F(r["exposure"]) == F(1, 100000)), key=lambda row: row["maximum_order"])
        if [r["maximum_order"] for r in rows] != [0, 1, 2]:
            raise ValueError("Matched-readout pilot must include exactly orders 0,1,2 at the fixed hardware setting")
        replicas, prefix0, best, quotas = [], [], [], []
        for row in rows:
            matched = row["matched_readout_diagnostic"]
            result = matched["population_plugin"]
            if matched["budget_ceiling"] != 80000000 or not matched["target_readouts"] <= 80000000:
                raise ValueError("Unexpected readout budget in the design pilot")
            if row["center_replicas"] != 2*row["maximum_order"]+2 or matched["target_records"] != 80000000//(row["center_replicas"]+2) or matched["target_readouts"] != matched["target_records"]*(row["center_replicas"]+2):
                raise ValueError("Replica counts, record quotas and readout costs are inconsistent")
            replicas.append(row["center_replicas"])
            quotas.append(matched["target_records"])
            prefix0.append(number(result["order_results"][0]["entropy_lower"]))
            best.append(number(result["entropy_lower"]))
        right.plot(replicas, prefix0, "o--", color=blue, linewidth=1.5, label="Prefix 0, all available replicas")
        right.plot(replicas, best, "s-", color=green, linewidth=1.2, markerfacecolor="none", markersize=9,
                   label="Best jointly covered prefix")
        right.set_xticks(replicas, [f"{k}\n{n/1e6:.2f} M trials" for k, n in zip(replicas, quotas)])
        right.set_xlabel("Center replicas per trial")
        right.set_ylabel("Population plug-in diagnostic")
        right.set_ylim(0, max(.01, max(best+prefix0))*1.4)
        right.set_title("B  Matched-readout design calculation", loc="left", fontweight="bold")
        right.grid(axis="y", alpha=.18)
        right.legend(loc="best", fontsize=8.4, frameon=False)
        right.text(.02, .98, "≤80 M science readouts + 0.2 M reference readouts\nNo sampled records in this panel; actual rate = 9",
                   transform=right.transAxes, va="top", fontsize=8.3,
                   bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85})
        if best == prefix0:
            right.text(.02, .83, "Curves coincide: prefix 0 is selected at all three designs.",
                       transform=right.transAxes, fontsize=8.3)
        fig.suptitle("Detector calibration: finite-data behavior and design limits", x=.075, ha="left", fontsize=15, fontweight="bold")
        fig.text(.075, .13, "A: Boxes show 25–75% of 400 simulated datasets per case; whiskers show 2.5–97.5%. Positive counts are shown for each bound.\n"
                 "Geometry uses a pointwise diffusion identity; the endpoint comparison applies to a broader model class. The vertical scale is nonlinear.",
                 fontsize=8.7, va="top", linespacing=1.45)
        fig.text(.075, .065, "B: Expected sufficient statistics are inserted into the confidence formulas. These values are neither confidence certificates nor power estimates.\n"
                 "The pilot splits failure probability 0.025/0.025; the frozen benchmark uses 0.04/0.01. Preparation, calibration infrastructure and energy are additional costs.",
                 fontsize=8.7, va="top", linespacing=1.45)
        return fig


def main():
    base = Path(__file__).resolve().parents[1]/"experiment"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=base/"benchmark.json")
    parser.add_argument("--records", type=Path, default=base/"benchmark_counts.json.gz")
    parser.add_argument("--pilot", type=Path, default=base/"projected_pilot_initial.json")
    parser.add_argument("--output-prefix", type=Path, default=base/"detector_calibration")
    args = parser.parse_args()
    figure = make_figure(*read_inputs(args.benchmark, args.records, args.pilot))
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_prefix.with_suffix(".pdf"),
                   metadata={"Title": "Detector calibration and finite-data entropy inference", "Author": "Abdinoor / 2D Labs", "CreationDate": None, "ModDate": None})
    figure.savefig(args.output_prefix.with_suffix(".png"), dpi=180,
                   metadata={"Title": "Detector calibration and finite-data entropy inference", "Author": "Abdinoor / 2D Labs"})
    plt.close(figure)


if __name__ == "__main__":
    main()
