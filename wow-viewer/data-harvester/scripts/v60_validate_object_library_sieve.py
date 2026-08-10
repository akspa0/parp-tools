#!/usr/bin/env python3
"""Validate a v60 real-object-library-derived sieve corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from harvester.v60.object_library_sieve import validate_object_library_sieve_corpus  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate v60-object-library-sieve-v1")
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    report = validate_object_library_sieve_corpus(args.corpus)
    if args.write_report:
        (args.corpus / "object_library_sieve_validation.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
