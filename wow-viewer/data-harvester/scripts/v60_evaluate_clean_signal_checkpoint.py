#!/usr/bin/env python3
"""Evaluate a clean-signal checkpoint on a prepared corpus without training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from harvester.v60.clean_signal_transfer import evaluate_clean_signal_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--source-kind", default=None)
    args = parser.parse_args()
    report = evaluate_clean_signal_checkpoint(
        args.checkpoint,
        args.corpus,
        args.output,
        batch_size=args.batch_size,
        device=args.device,
        source_kind=args.source_kind,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
