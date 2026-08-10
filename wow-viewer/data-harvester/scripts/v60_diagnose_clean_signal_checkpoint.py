#!/usr/bin/env python3
"""Run image-only prediction diagnostics for a trained Spec 139 checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from harvester.v60.clean_signal_diagnostics import diagnose_clean_signal_checkpoint  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose a Spec 139 clean-signal checkpoint")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    args = parser.parse_args()
    try:
        report = diagnose_clean_signal_checkpoint(
            args.checkpoint,
            args.corpus,
            args.output,
            batch_size=args.batch_size,
            device=args.device,
        )
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        raise SystemExit(str(exc)) from exc
    family_rows = sorted(
        (
            {
                "family": family,
                "final_height_mae": metrics["final_height_mae"],
                "improvement_vs_tile_mean": metrics["improvement_vs_tile_mean"],
            }
            for family, metrics in report["by_family"].items()
        ),
        key=lambda row: float(row["improvement_vs_tile_mean"]),
    )
    print(
        json.dumps(
            {
                "schema": report["schema"],
                "checkpoint": report["checkpoint"],
                "device": report["device"],
                "validation_row_count": report["validation_row_count"],
                "aggregate": report["aggregate"],
                "worst_families": family_rows[:8],
                "outputs": report["outputs"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
