#!/usr/bin/env python3
"""Dry-run-first builder for the Spec 139 synthetic clean-signal corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from harvester.v60.clean_signal_corpus import (  # noqa: E402
    build_clean_signal_corpus,
    clean_signal_build_plan,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a v7 clean-signal v60 corpus")
    parser.add_argument("--control-corpus", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--confidence", type=float, default=1.0)
    parser.add_argument("--confirm-build", action="store_true", help="publish the user-owned corpus build")
    args = parser.parse_args()

    try:
        plan = clean_signal_build_plan(args.control_corpus, confidence_value=args.confidence)
        if not args.confirm_build:
            print(json.dumps(plan, indent=2), flush=True)
            print("DRY RUN ONLY: add --confirm-build to publish the clean-signal corpus.", flush=True)
            return 0
        result = build_clean_signal_corpus(
            args.control_corpus,
            args.output,
            confidence_value=args.confidence,
        )
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
