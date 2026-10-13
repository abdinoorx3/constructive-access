"""Evaluate analytical detector-access designs through the installed toolkit.

This example acquires no records. Numerical count displays evaluate exact
proved inequalities; they are not interval-certified or experimental outcomes.
"""
from pathlib import Path
import argparse
import json
from constructive_access.detector import counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    payload = counts.report()
    payload['usage'] = 'Analytical design example only. Physical calibration, phase preparation and resource assumptions are supplied, not inferred.'
    text = json.dumps(payload, indent=2, sort_keys=True)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
