"""Complete target-specific finite-menu example. All counts below are synthetic."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
from constructive_access.suite import design_suite, infer_target, verify_suite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    targets = {"A": "even", "A_copy": "even", "B": "even", "C": "odd", "D": "odd"}
    ps = {
        "left": {"A": "1/4", "A_copy": "1/4", "B": "3/4", "C": "1/4", "D": "3/4"},
        "right": {"A": "1/4", "A_copy": "1/4", "B": "3/4", "C": "3/4", "D": "1/4"},
        "direct": {"A": "1/4", "A_copy": "1/4", "B": "1/4", "C": "3/4", "D": "3/4"}}
    menu = [{"name": name, "charge": "3" if name == "direct" else "1",
             "laws": {h: {"0": str(1-Fraction(p)), "1": p} for h, p in values.items()},
             "tv_radii": {h: "1/100" for h in targets}} for name, values in ps.items()]
    design = design_suite(targets, menu, delta="1/20")
    counts = {e["name"]: {"1": e["records"]//4, "0": e["records"]-e["records"]//4}
              for e in design["selected"]}
    result = {"evidence_type": "exact finite-law design and deliberately synthetic counts; no physical data",
              "targets": targets, "menu": menu, "design": design,
              "design_replay": verify_suite(targets, menu, design),
              "synthetic_counts": counts,
              "decision": infer_target(targets, menu, design, counts),
              "left_only": design_suite(targets, menu[:1]),
              "right_only": design_suite(targets, menu[1:2]),
              "direct_only": design_suite(targets, menu[2:])}
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(json.dumps({"status": design["status"], "experiments": [x["name"] for x in design["selected"]],
                      "selection_charge": design["selection_charge"], "total_records": design["total_records"],
                      "retained_models": result["decision"]["retained_models"],
                      "target": result["decision"].get("target")}))


if __name__ == "__main__":
    main()
