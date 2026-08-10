#!/usr/bin/env python3
"""Run the bounded terrain-only v60 control experiment.

Without ``--confirm-run`` this validates the NPZ corpus, prints the fixed split
and baselines, and exits.  Only the user-owned confirmed run performs training.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from harvester.v50.height_relative_model import height_loss  # noqa: E402
from harvester.v60.control_experiment import (  # noqa: E402
    ControlDataset,
    fixed_validation_rows,
    load_control_rows,
    select_training_schedule,
    split_summary,
    tile_mean_baseline,
    write_json,
)
from harvester.v60.terrain_models import (  # noqa: E402
    TERRAIN_ARCHITECTURES,
    UNET_LITE_ID,
    build_terrain_model,
)


def _parse_sizes(value: str) -> list[int]:
    sizes = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not sizes or any(size < 1 for size in sizes) or len(set(sizes)) != len(sizes):
        raise argparse.ArgumentTypeError("train sizes must be unique positive integers")
    return sizes


def _parse_architectures(value: str) -> list[str]:
    architectures = [item.strip() for item in value.split(",") if item.strip()]
    if not architectures or len(set(architectures)) != len(architectures):
        raise argparse.ArgumentTypeError("architectures must be unique and non-empty")
    unknown = sorted(set(architectures) - set(TERRAIN_ARCHITECTURES))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown architectures {unknown}; choose from {list(TERRAIN_ARCHITECTURES)}"
        )
    return architectures


def _require_new_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise SystemExit(f"refusing to overwrite existing output: {path}")


def _evaluate(model: torch.nn.Module, rows, batch_size: int, device: torch.device) -> dict:
    model.eval()
    loader = DataLoader(ControlDataset(rows), batch_size=batch_size, shuffle=False, num_workers=0)
    errors: list[float] = []
    by_family: dict[str, list[float]] = {}
    by_variant: dict[str, list[float]] = {}
    with torch.no_grad():
        for shadow, target in loader:
            prediction = model(shadow.to(device)).cpu().numpy()
            target_np = target.numpy()
            for index in range(prediction.shape[0]):
                error = float(np.abs(prediction[index] - target_np[index]).mean())
                row = rows[len(errors)]
                errors.append(error)
                by_family.setdefault(row.control_family, []).append(error)
                by_variant.setdefault(str(row.variant), []).append(error)
    return {
        "mae": float(np.mean(errors)),
        "by_family": {key: float(np.mean(values)) for key, values in sorted(by_family.items())},
        "by_variant": {key: float(np.mean(values)) for key, values in sorted(by_variant.items())},
    }


def _train_one(
    architecture: str, train_rows, validation_rows, args, output: Path, seed: int
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, identity = build_terrain_model(architecture)
    model = model.to(device)
    train_loader = DataLoader(
        ControlDataset(train_rows),
        batch_size=min(args.batch_size, len(train_rows)),
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=max(1, args.epochs * len(train_loader)),
        pct_start=0.1,
    )
    best_mae = float("inf")
    best_epoch = -1
    best_metrics: dict | None = None
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        for shadow, target in train_loader:
            shadow = shadow.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            loss = height_loss(model(shadow), target)
            loss.backward()
            optimizer.step()
            scheduler.step()
        metrics = _evaluate(model, validation_rows, args.batch_size, device)
        if metrics["mae"] < best_mae:
            best_mae = metrics["mae"]
            best_epoch = epoch
            best_metrics = metrics
            output.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output / "checkpoint_best.pt")
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"[size {len(train_rows):3d} epoch {epoch:03d}] "
                f"val_mae={metrics['mae']:.6f}",
                flush=True,
            )
    output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output / "checkpoint_last.pt")
    return {
        "architecture": architecture,
        "model_identity": identity,
        "train_row_count": len(train_rows),
        "device": str(device),
        "best_epoch": best_epoch,
        "best_val_mae": best_mae,
        "best_metrics": best_metrics,
        "final_metrics": metrics,
        "seconds": time.time() - start,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Terrain-only v60 control experiment")
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--architectures",
        type=_parse_architectures,
        default=[UNET_LITE_ID],
        help=f"comma-separated candidates; choices: {', '.join(TERRAIN_ARCHITECTURES)}",
    )
    parser.add_argument("--train-sizes", type=_parse_sizes, default=[8, 16, 32])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=6001)
    parser.add_argument("--confirm-run", action="store_true", help="required to launch training")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        raise SystemExit("epochs, batch-size, and lr must be positive")

    root, rows = load_control_rows(args.corpus)
    validation_rows = fixed_validation_rows(rows)
    if any(size > sum(row.split == "train" for row in rows) for size in args.train_sizes):
        raise SystemExit("a requested training size exceeds the manifest train split")
    baseline = tile_mean_baseline(validation_rows)
    schedule = select_training_schedule(rows, args.train_sizes, args.seed)
    plans = []
    for size in args.train_sizes:
        train_rows = schedule[size]
        plans.append({"train_size": size, "split": split_summary(train_rows, validation_rows)})

    print(f"Corpus: {root}")
    print("Signal: terrain_shadow_256 -> height_257")
    print(f"Validation rows: {len(validation_rows)}; families: {len({row.control_family for row in validation_rows})}")
    print(f"Tile-mean validation MAE: {baseline['mae']:.6f}")
    for plan in plans:
        print(f"Train size {plan['train_size']}: {len(plan['split']['train_families'])} train families")
    print("Architectures:")
    for architecture in args.architectures:
        model, identity = build_terrain_model(architecture)
        print(
            f"  {architecture}: {identity['parameter_count']:,} parameters; "
            f"{identity['input_contract']} -> {identity['output_contract']}; "
            "random_init"
        )
        del model
    if not args.confirm_run:
        print("DRY RUN ONLY: add --confirm-run to launch user-owned terrain training.")
        return 0

    _require_new_output(args.output)
    results = []
    for architecture_index, architecture in enumerate(args.architectures):
        for plan in plans:
            train_rows = schedule[plan["train_size"]]
            results.append(
                _train_one(
                    architecture,
                    train_rows,
                    validation_rows,
                    args,
                    args.output / architecture / f"train-{plan['train_size']:03d}",
                    args.seed + (architecture_index * 1000) + plan["train_size"],
                )
            )
    metrics_by_family = {
        f"{result['architecture']}/train-{result['train_row_count']:03d}": result["best_metrics"]["by_family"]
        for result in results
    }
    write_json(
        args.output / "experiment-report.json",
        {
            "schema": "v60-architecture-bakeoff-report-v1",
            "data_stage": "control",
            "dataset_manifest": str(root / "control_manifest.json"),
            "model_version": "terrain_architecture_bakeoff_v1",
            "architectures": args.architectures,
            "training_row_count": max(args.train_sizes),
            "training_sizes": args.train_sizes,
            "seed": args.seed,
            "baseline": {"validation_tile_mean": baseline},
            "metrics_by_family": metrics_by_family,
            "split_plans": plans,
            "results": results,
            "object_signals_used": False,
            "ambiguity": {"baseline_ambiguous_rows": baseline["ambiguous_rows"]},
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
