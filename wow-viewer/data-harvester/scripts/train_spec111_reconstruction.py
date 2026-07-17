#!/usr/bin/env python3
"""Spec 111 US3 (T017): retrain the active reconstruction stage on lighting-rebalanced data.

The active, unblocked stage is Spec 108's ``WdlPriorNet`` (Spec 102's residual chain is BLOCKED on
its M0 target reharvest), so this script prepares and delegates to the existing
``scripts/train_spec103_wdl_prior.py`` trainer rather than introducing any new architecture
(contract: no DepthAnything-family/multi-head/shared-weight paths; extend the existing lineage).

EXECUTION GATE (contract, Training/evaluation execution contract item 1): running this script
without ``--confirm-run`` only validates the configuration and prints the exact delegated training
command -- it never starts a GPU run. ``--confirm-run`` exists to be passed by the user at the
moment they explicitly authorize spending the compute; completing Phases 1-2 or preparing this
script does not constitute that authorization.

Usage (validate only -- always safe):
    uv run --directory wow-viewer/data-harvester python scripts/train_spec111_reconstruction.py \
        --rebalanced-plan <plan.json from scripts/rebalance_lighting_variants.py> \
        --store <mixed paired store> --output <checkpoint.pt> \
        --baseline-checkpoint <currently deployed checkpoint.pt>

Usage (execute -- only with explicit user authorization at this moment):
    ... same arguments ... --confirm-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def build_delegated_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).parent / "train_spec103_wdl_prior.py"),
        "--store", str(args.store),
        "--output", str(args.output),
        "--epochs", str(args.epochs),
        "--batch", str(args.batch),
    ]


def validate(args: argparse.Namespace) -> dict:
    plan = json.loads(args.rebalanced_plan.read_text(encoding="utf-8"))
    lighting_times = plan.get("lighting_times")
    if not lighting_times:
        raise SystemExit(
            "Error: the rebalanced plan carries no lighting_times (every bucket was sparse). "
            "Re-run Phase 1 bucketing on a store with real matched coverage first."
        )
    if not all(isinstance(value, float) and 0.0 <= value < 1.0 for value in lighting_times):
        raise SystemExit("Error: lighting_times must be bare normalized floats in [0, 1).")
    if plan.get("leak_safety_tags_preserved") is not True:
        raise SystemExit("Error: plan does not assert leak-safety tag preservation; refusing.")
    if not args.store.exists():
        raise SystemExit(f"Error: paired store not found: {args.store}")
    if not args.baseline_checkpoint.exists():
        raise SystemExit(
            f"Error: baseline checkpoint not found: {args.baseline_checkpoint}. The comparison "
            "gate (T018) requires the currently deployed checkpoint to compare against; there is "
            "no committed default."
        )
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebalanced-plan", required=True, type=Path)
    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--baseline-checkpoint", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--confirm-run", action="store_true",
                        help="Actually start the GPU training run. Pass only with explicit "
                             "user authorization given at this moment.")
    args = parser.parse_args()

    plan = validate(args)
    command = build_delegated_command(args)

    print(f"Rebalanced plan: {args.rebalanced_plan}")
    print(f"  source_build_fingerprint: {plan.get('source_build_fingerprint')}")
    print(f"  lighting_times: {len(plan['lighting_times'])} variants across "
          f"{len(plan.get('bucket_weights', {}))} buckets "
          f"({len(plan.get('sparse_buckets', []))} sparse)")
    print(f"Baseline checkpoint: {args.baseline_checkpoint}")
    print(f"Delegated trainer command:\n  {' '.join(command)}")

    if not args.confirm_run:
        print("\nValidation complete. NOT starting training: pass --confirm-run only when the "
              "user has explicitly authorized this GPU run.")
        return 0

    print("\n--confirm-run given: starting the delegated training run.")
    return subprocess.call(command)


if __name__ == "__main__":
    sys.exit(main())
