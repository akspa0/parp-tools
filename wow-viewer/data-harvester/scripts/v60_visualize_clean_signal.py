#!/usr/bin/env python3
"""Render operator-facing visual review sheets for a clean-signal corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from harvester.v60.clean_signal_visual_review import render_clean_signal_review  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Render Spec 139 clean-signal visual review sheets")
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--rows-per-family", type=int, default=4)
    args = parser.parse_args()
    try:
        report = render_clean_signal_review(
            args.corpus,
            args.output_dir,
            rows_per_family=args.rows_per_family,
        )
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
