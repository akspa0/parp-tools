"""Spec 117 US2: standalone WDL-lattice predictor trainer (USER runs CUDA).

Predicts the 545-point WDL-scale height lattice from minimap RGB alone. Deliberately the smallest
trainer in this project's v50 lane: no feature-store input, no liquid/brush/normal loss terms, no
AMP/OneCycle bells -- those all address dense 257x257 geometry prediction, not a 545-value vector
regression. Reuses (never reimplements) the already-validated curriculum/source/held-out-split
machinery from ``height_relative_train``/``direct_geometry_train``.

Unlike the existing coarse/detailer trainers, ``--held-out-split`` is REQUIRED with no
``--val-key``/``--val-value`` fallback: spec FR-004 says the standalone predictor "MUST refuse to
run against a leaky or unspecified split," not merely default away from one.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from harvester.spec117.lattice_contract import (
    architecture_identity,
    build_lattice_stage_run,
    identity_for_path,
)
from harvester.spec117.lattice_model import (
    LatticeNet,
    compute_lattice_tile_mean_baseline,
    encode_lattice_target,
    lattice_loss,
    select_lattice_rows,
)
from harvester.v50.contracts import release_identity, require_store_release, validate_release
from harvester.v50.direct_geometry_train import apply_held_out_split
from harvester.v50.height_relative_train import (
    SOURCE_CHOICES,
    TrainerContractError,
    curriculum_identity,
    require_new_output,
    select_training_rows,
    validate_curriculum_contract,
    validate_source_selection,
)
from harvester.v50.model_stage_contract import sha256_file

STAGE = "lattice_prior"
REQUIRED_WDL_ARRAYS = ("wdl_outer_17", "wdl_inner_16", "wdl_outer_present", "wdl_inner_present")
LR_SCHEDULES = frozenset({"constant", "onecycle"})


def require_wdl_arrays(group) -> None:
    """Fail closed when the store predates the Spec 117 catalog amendment (US1)."""
    missing = [name for name in REQUIRED_WDL_ARRAYS if name not in group]
    if missing:
        raise TrainerContractError(
            f"store is missing WDL lattice arrays {missing}; rebuild the store after the Spec 117 "
            "catalog amendment (docs/architecture/v50-clean-room-dataset-repo-audit-2026-07-15.md) "
            "before training the lattice predictor"
        )


def build_lattice_plan(
    *,
    architecture: dict,
    source: str,
    train_rows: int,
    val_rows: int,
    excluded_train: int,
    excluded_val: int,
    batch_size: int,
    epochs: int,
    seed: int,
    lr: float,
    lr_schedule: str,
) -> dict:
    """Machine-readable no-training preview printed before CUDA allocation."""
    if batch_size < 1 or epochs < 1:
        raise TrainerContractError("batch size and epochs must both be positive")
    if lr_schedule not in LR_SCHEDULES:
        raise TrainerContractError(f"lr schedule must be one of {sorted(LR_SCHEDULES)}")
    return {
        "schema": "v117-lattice-plan-v1",
        "stage": STAGE,
        "architecture": architecture,
        "source_filter": source,
        "split_counts": {"train": train_rows, "val": val_rows},
        "excluded_no_present_lattice": {"train": excluded_train, "val": excluded_val},
        "batch_size": batch_size,
        "epochs": epochs,
        "seed": seed,
        "optimizer": {"name": "AdamW", "learning_rate": lr, "weight_decay": 1e-4},
        "lr_schedule": lr_schedule,
        "train_steps_per_epoch": math.ceil(max(train_rows, 1) / batch_size),
        "deployment_inputs": ["minimap_rgb"],
        "training_target": "wdl_outer_17+wdl_inner_16 -> wdl_lattice_545",
        "no_gan_no_adversarial_no_generative_image": True,
    }


def main() -> int:
    import pyarrow.parquet as pq
    import torch
    import zarr
    from torch.utils.data import DataLoader, Dataset

    ap = argparse.ArgumentParser(description="Spec 117 standalone WDL-lattice trainer (USER runs CUDA)")
    ap.add_argument("--store", required=True, type=Path, help="v50 curriculum store carrying wdl_outer_17/wdl_inner_16")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--run-id", required=True, help="immutable run identity, e.g. lattice-authored-v1")
    ap.add_argument("--source", required=True, choices=sorted(SOURCE_CHOICES))
    ap.add_argument(
        "--held-out-split", required=True, type=Path,
        help="REQUIRED (FR-004): a v50-held-out-split-v1 directory (spec116_build_held_out_split.py "
             "or equivalent). Refuses a leaky split (verified_violation_count != 0). No --val-key/"
             "--val-value fallback exists for this trainer.",
    )
    ap.add_argument("--confirm-run", action="store_true",
                     help="launch CUDA training; without this flag only print the validated plan")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr-schedule", default="constant", choices=sorted(LR_SCHEDULES))
    ap.add_argument("--base", type=int, default=24, help="LatticeNet encoder width")
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=117)
    ap.add_argument("--release", default="v50.1", type=validate_release)
    ap.add_argument("--workers", type=int, choices=[0], default=0)
    args = ap.parse_args()

    group = zarr.open_group(str(args.store), mode="r")
    try:
        require_store_release(group, args.release, store=args.store)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    index = pq.read_table(args.store / "index.parquet").to_pylist()
    try:
        array_lengths = {name: int(group[name].shape[0]) for name in group.array_keys()}
        validate_curriculum_contract(attrs=dict(group.attrs), array_lengths=array_lengths, index_rows=index)
        validate_source_selection(attrs=dict(group.attrs), source=args.source)
        require_wdl_arrays(group)
        selected_rows = select_training_rows(index, args.source)
    except TrainerContractError as exc:
        raise SystemExit(str(exc)) from exc

    try:
        train_rows_all, val_rows_all, split_manifest = apply_held_out_split(
            index_rows=index, selected_rows=selected_rows, split_dir=args.held_out_split,
        )
    except TrainerContractError as exc:
        raise SystemExit(str(exc)) from exc

    train_rows, excluded_train = select_lattice_rows(group, train_rows_all)
    val_rows, excluded_val = select_lattice_rows(group, val_rows_all)
    if len(train_rows) < 32 or len(val_rows) < 8:
        raise SystemExit(
            f"insufficient rows with a present lattice sample after --source {args.source}: "
            f"train={len(train_rows)} val={len(val_rows)} "
            f"(excluded_no_present_lattice train={excluded_train} val={excluded_val})"
        )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = LatticeNet(base=args.base)
    identity = architecture_identity(
        model, architecture_id="lattice_net",
        config={"class": "LatticeNet", "base": args.base, "input": "3x256x256", "output": "545"},
    )
    plan = build_lattice_plan(
        architecture=identity, source=args.source,
        train_rows=len(train_rows), val_rows=len(val_rows),
        excluded_train=excluded_train, excluded_val=excluded_val,
        batch_size=args.batch, epochs=args.epochs, seed=args.seed,
        lr=args.lr, lr_schedule=args.lr_schedule,
    )
    plan["held_out_split"] = {
        "path": str((args.held_out_split / "split.json").resolve()),
        "sha256": sha256_file(args.held_out_split / "split.json"),
        "verified_violation_count": int(split_manifest["verified_violation_count"]),
        "absolute_comparison_to_prior_runs_invalid": True,
    }
    print(json.dumps(plan, indent=2), flush=True)
    if not args.confirm_run:
        print("DRY RUN ONLY: add --confirm-run to launch user-owned CUDA training.", flush=True)
        return 0
    try:
        require_new_output(args.output)
    except TrainerContractError as exc:
        raise SystemExit(str(exc)) from exc
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; user-run training refuses CPU.")
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    class RowDataset(Dataset):
        def __init__(self, rows: list[int]) -> None:
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, i: int):
            row = self.rows[i]
            rgb = np.asarray(group["minimap_rgb"][row], dtype=np.float32) / 255.0
            target, mask, _tile_min, _tile_max = encode_lattice_target(
                np.asarray(group["wdl_outer_17"][row]),
                np.asarray(group["wdl_inner_16"][row]),
                np.asarray(group["wdl_outer_present"][row]),
                np.asarray(group["wdl_inner_present"][row]),
            )
            return (
                torch.from_numpy(rgb).permute(2, 0, 1),
                torch.from_numpy(target),
                torch.from_numpy(mask),
            )

    val_targets_masks = []
    for row in val_rows:
        target, mask, _tile_min, _tile_max = encode_lattice_target(
            np.asarray(group["wdl_outer_17"][row]),
            np.asarray(group["wdl_inner_16"][row]),
            np.asarray(group["wdl_outer_present"][row]),
            np.asarray(group["wdl_inner_present"][row]),
        )
        val_targets_masks.append((target, mask))
    tile_mean_baseline = compute_lattice_tile_mean_baseline(val_targets_masks)

    device = torch.device("cuda")
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    train_loader = DataLoader(
        RowDataset(train_rows), batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=True, generator=train_generator,
    )
    val_loader = DataLoader(RowDataset(val_rows), batch_size=args.batch, num_workers=args.workers, pin_memory=True)
    scheduler = None
    if args.lr_schedule == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader)
        )

    args.output.mkdir(parents=True, exist_ok=True)
    curriculum_id = curriculum_identity(args.store)
    run_identity = {
        **release_identity(args.release),
        "stage": STAGE,
        "architecture": identity,
        # identity.config_sha256 hashes the config but does not carry it in plain form; lattice_bridge.py
        # needs the raw `base` back to reconstruct an architecturally-identical LatticeNet before
        # load_state_dict, so it travels alongside identity rather than only inside the hash.
        "lattice_config": {"base": args.base},
        "source_filter": args.source,
        "store": str(args.store.resolve()),
        "optimizer": plan["optimizer"],
        "schedule": {
            "max_epochs": args.epochs, "batch_size": args.batch,
            "patience": args.patience, "workers": args.workers, "seed": args.seed,
            "lr_schedule": args.lr_schedule,
        },
        "held_out_split": plan["held_out_split"],
    }
    (args.output / "training_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    (args.output / "run_identity.json").write_text(json.dumps(run_identity, indent=2), encoding="utf-8")

    per_epoch: list[dict] = []
    best = float("inf")
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for x, y, mask in train_loader:
            opt.zero_grad(set_to_none=True)
            predicted = model(x.to(device))
            loss = lattice_loss(predicted, y.to(device), mask.to(device))
            loss.backward()
            opt.step()
            if scheduler is not None:
                scheduler.step()
            train_losses.append(float(loss.detach().item()))
        model.eval()
        val_absolute_error = 0.0
        val_present = 0.0
        with torch.no_grad():
            for x, y, mask in val_loader:
                y_device = y.to(device)
                mask_device = mask.to(device)
                predicted = model(x.to(device))
                val_absolute_error += float((torch.abs(predicted - y_device) * mask_device).sum().item())
                val_present += float(mask_device.sum().item())
        val_mae = val_absolute_error / max(val_present, 1.0)
        train_loss = float(np.mean(train_losses))
        per_epoch.append({"epoch": epoch, "train_loss": train_loss, "val_mae": val_mae})
        checkpoint = {
            **run_identity, "model": model.state_dict(), "epoch": epoch,
            "val_mae": val_mae, "curriculum_identity": curriculum_id,
        }
        torch.save(checkpoint, args.output / "checkpoint_last.pt")
        if val_mae < best:
            best = val_mae
            stale = 0
            torch.save(checkpoint, args.output / "checkpoint_best.pt")
        else:
            stale += 1
        print(
            f"[epoch {epoch:03d}] train_loss={train_loss:.6f} val_mae={val_mae:.6f} "
            f"tile_mean={tile_mean_baseline:.6f} best={best:.6f} stale={stale}/{args.patience}",
            flush=True,
        )
        if args.patience > 0 and stale >= args.patience:
            print(f"[early-stop] no improvement for {stale} epochs", flush=True)
            break

    best_record = min(per_epoch, key=lambda e: e["val_mae"])
    best_checkpoint = torch.load(args.output / "checkpoint_best.pt", map_location=device, weights_only=False)
    beats_tile_mean = bool(best_record["val_mae"] < tile_mean_baseline)

    stage_run = build_lattice_stage_run(
        run_id=args.run_id,
        architecture=identity,
        curriculum=identity_for_path(args.store / "index.parquet", display_path=str(args.store.resolve())),
        checkpoint={
            **identity_for_path(args.output / "checkpoint_best.pt"),
            "best_epoch": int(best_checkpoint["epoch"]),
        },
        baselines={"tile_mean": {"val_mae": tile_mean_baseline}},
        metrics={
            "best_epoch": best_record["epoch"],
            "best_val_mae": best_record["val_mae"],
            "beats_tile_mean_baseline": beats_tile_mean,
            "held_out_split": plan["held_out_split"],
            "excluded_no_present_lattice": plan["excluded_no_present_lattice"],
            "structural_failure_epoch1_best": best_record["epoch"] == 1,
        },
    )
    (args.output / "model_stage_run.json").write_text(json.dumps(stage_run, indent=2), encoding="utf-8")
    (args.output / "training_summary.json").write_text(
        json.dumps({"per_epoch_metrics": per_epoch, "model_stage_run": stage_run["run_id"]}, indent=2),
        encoding="utf-8",
    )
    if stage_run["metrics"]["structural_failure_epoch1_best"]:
        print("STRUCTURAL FAILURE: best epoch is epoch 1; this run is not a success.", flush=True)
        return 1
    print(
        f"best_epoch={best_record['epoch']} best_val_mae={best_record['val_mae']:.6f} "
        f"tile_mean_baseline={tile_mean_baseline:.6f} beats_tile_mean_baseline={beats_tile_mean} "
        f"promotion=pending(user gate; US3 integration should not proceed without this being true, "
        f"per spec US2 acceptance 3, unless explicitly overridden)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
