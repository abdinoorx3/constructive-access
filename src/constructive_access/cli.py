"""Command line interface for contract analysis and independent checking."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .contract import ContractError, read_json
from .workflow import VerificationError, analyze, verify_report


def main(argv=None):
    parser = argparse.ArgumentParser(prog="constructive-access")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("analyze", help="Synthesize and check a declared two-model experiment")
    build.add_argument("contract", type=Path)
    build.add_argument("-o", "--output", type=Path)
    check = sub.add_parser("verify", help="Check a report against an independently supplied contract")
    check.add_argument("contract", type=Path)
    check.add_argument("report", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "analyze":
            report = analyze(args.contract)
            rendered = json.dumps(report, indent=2, sort_keys=True)+"\n"
            if args.output is None:
                print(rendered, end="")
            else:
                args.output.write_text(rendered)
                print(json.dumps({"status": report["status"], "report": str(args.output),
                                  "contract_sha256": report["contract_sha256"]}, sort_keys=True))
        else:
            print(json.dumps(verify_report(args.contract, read_json(args.report)), indent=2, sort_keys=True))
    except (ContractError, VerificationError, OSError, ValueError) as exc:
        print(json.dumps({"status": "error", "type": type(exc).__name__, "message": str(exc)}), file=sys.stderr)
        return 2
    return 0
