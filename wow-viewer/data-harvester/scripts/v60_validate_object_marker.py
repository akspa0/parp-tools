#!/usr/bin/env python3
"""Validate a v60 footprint-guided object-marker corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from harvester.v60.object_marker import (  # noqa: E402
    ObjectMarkerError,
    validate_object_marker_corpus,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate v60-object-marker-v1")
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    try:
        report = validate_object_marker_corpus(args.corpus)
    except (FileNotFoundError, ObjectMarkerError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    if args.write_report:
        (args.corpus / "object_marker_validation.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
