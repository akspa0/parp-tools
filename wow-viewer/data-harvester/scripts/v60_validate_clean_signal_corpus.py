#!/usr/bin/env python3
"""Validate a v60 clean-signal corpus and optionally write its JSON report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from harvester.v60.clean_signal_corpus import validate_clean_signal_corpus  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a v7 clean-signal v60 corpus")
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()

    try:
        report = validate_clean_signal_corpus(args.corpus)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"valid": False, "corpus_root": str(args.corpus), "failures": [str(exc)]}, indent=2))
        return 2

    if args.write_report:
        report_path = args.corpus / "validation.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["report_path"] = str(report_path)
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
