#!/usr/bin/env python3
"""Dry-run-first architecture/loss matrix runner for Spec 139."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from harvester.v60.clean_signal_losses import (  # noqa: E402
    CLEAN_SIGNAL_LOSS_PROFILES,
    get_clean_signal_loss_config,
)
from harvester.v60.clean_signal_model import (  # noqa: E402
    CLEAN_SIGNAL_ARCHITECTURES,
    build_clean_signal_model,
)
from harvester.v60.clean_signal_train import (  # noqa: E402
    CleanSignalTrainConfig,
    build_clean_signal_split,
    load_clean_signal_rows,
    select_clean_signal_training_rows,
    train_clean_signal_model,
)


def _csv_values(value: str, *, name: str, choices: tuple[str, ...]) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError(f"{name} must contain unique non-empty values")
    unknown = sorted(set(values) - set(choices))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown {name} {unknown}; choose from {list(choices)}")
    return values


def _architectures(value: str) -> list[str]:
    return _csv_values(value, name="architectures", choices=CLEAN_SIGNAL_ARCHITECTURES)


def _loss_profiles(value: str) -> list[str]:
    return _csv_values(value, name="loss-profiles", choices=CLEAN_SIGNAL_LOSS_PROFILES)


def _require_new_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing training output: {path}")


def build_training_plan(args: argparse.Namespace) -> dict[str, Any]:
    """Build a JSON-safe no-write plan shared by dry-run and confirmed execution."""

    root, rows = load_clean_signal_rows(args.corpus)
    split = build_clean_signal_split(rows, mode=args.split, seed=args.seed)
    train_rows = select_clean_signal_training_rows(split.train_rows, count=args.train_size, seed=args.seed)
    architectures = list(args.architectures)
    loss_profiles = list(args.loss_profiles)
    architecture_plans: list[dict[str, Any]] = []
    for architecture in architectures:
        model, identity = build_clean_signal_model(architecture, profile=args.model_profile)
        architecture_plans.append(
            {
                "architecture": architecture,
                "model_profile": args.model_profile,
                "parameter_count": identity["parameter_count"],
                "model_identity": identity,
            }
        )
        del model
    split_payload = split.as_dict()
    split_payload["train_row_ids"] = [row.row_id for row in train_rows]
    split_payload["train_row_count"] = len(train_rows)
    return {
        "schema": "v7-clean-signal-training-plan-v1",
        "corpus_root": str(root.resolve()),
        "corpus_manifest": str((root / "clean_signal_manifest.json").resolve()),
        "architectures": architecture_plans,
        "loss_profiles": [get_clean_signal_loss_config(profile).as_dict() for profile in loss_profiles],
        "split": split_payload,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "patience": args.patience,
        "seed": args.seed,
        "device": args.device,
        "output": str(args.output.resolve()),
        "forbidden_signals_seen": [],
        "dry_run": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the v7 clean-signal v60 architecture/loss matrix")
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--architectures", type=_architectures, default=list(CLEAN_SIGNAL_ARCHITECTURES))
    parser.add_argument("--loss-profiles", type=_loss_profiles, default=list(CLEAN_SIGNAL_LOSS_PROFILES))
    parser.add_argument("--model-profile", choices=("tiny", "full"), default="tiny")
    parser.add_argument("--split", choices=("within_family", "complete_family"), default="within_family")
    parser.add_argument("--train-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7137)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--confirm-run", action="store_true", help="required to launch user-owned training")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.train_size < 1 or args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        raise SystemExit("train-size, epochs, batch-size, and patience must be positive")
    if args.learning_rate <= 0.0:
        raise SystemExit("learning-rate must be positive")
    try:
        plan = build_training_plan(args)
        if not args.confirm_run:
            print(json.dumps(plan, indent=2), flush=True)
            print("DRY RUN ONLY: add --confirm-run to launch user-owned training.", flush=True)
            return 0
        _require_new_output(args.output)
        args.output.mkdir(parents=True, exist_ok=True)
        split = build_clean_signal_split(
            load_clean_signal_rows(args.corpus)[1],
            mode=args.split,
            seed=args.seed,
        )
        train_rows = select_clean_signal_training_rows(split.train_rows, count=args.train_size, seed=args.seed)
        split_payload = split.as_dict()
        split_payload["train_row_ids"] = [row.row_id for row in train_rows]
        split_payload["train_row_count"] = len(train_rows)
        train_config = CleanSignalTrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            patience=args.patience,
            seed=args.seed,
            device=args.device,
        )
        results = []
        for architecture in args.architectures:
            for profile in args.loss_profiles:
                result = train_clean_signal_model(
                    train_rows,
                    split.validation_rows,
                    architecture=architecture,
                    profile=profile,
                    output=args.output / architecture / profile,
                    config=train_config,
                    split=split_payload,
                    model_profile=args.model_profile,
                )
                results.append(result)
        report = {"schema": "v7-clean-signal-training-matrix-v1", "plan": plan, "results": results}
        (args.output / "training_matrix_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
