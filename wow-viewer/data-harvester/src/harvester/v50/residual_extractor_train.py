"""User-run trainer for the residual-extractor (Spec 125 US7).

Consumes a paired store: each row pairs one minimap RGB tile (``minimap_rgb``, 256x256x3) with the
same tile's textureless terrain-shadow residual (``residual_256``, 256x256 single channel). The
synthesizer produces both for the same tile, so the pairs are exact and free.

The model learns to extract the residual from any minimap RGB tile. Once the residual is known,
subtracting it from the minimap strips the shading layer, revealing the albedo/texturing underneath.

Contract enforcement lives in small pure functions so the gates are CPU-testable without CUDA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from harvester.v50.contracts import release_identity, require_store_release, validate_release
from harvester.v50.residual_extractor_model import (
    TARGET_CONTRACT_VERSION,
    ResidualExtractorNet,
    residual_loss,
)

ALLOWED_MAPS = frozenset({"Kalimdor", "Azeroth"})
CURRICULUM_SCHEMA = "v125-residual-extractor-curriculum-v1"
REQUIRED_ARRAYS = frozenset({"minimap_rgb", "residual_256"})
REQUIRED_INDEX_FIELDS = frozenset({"map", "source_group_id", "split"})
ARCHITECTURE_ID = "residual_extractor_v125"


class TrainerContractError(ValueError):
    """Raised when a run would violate a Spec 125 contract gate."""


def validate_curriculum_contract(
    *,
    attrs: dict,
    array_lengths: dict[str, int],
    index_rows: list[dict],
) -> None:
    """Fail closed before CUDA allocation when the input is not the Spec 125 extractor curriculum."""
    if attrs.get("schema") != CURRICULUM_SCHEMA:
        raise TrainerContractError(
            f"curriculum schema must be {CURRICULUM_SCHEMA!r}, got {attrs.get('schema')!r}"
        )
    missing_arrays = sorted(REQUIRED_ARRAYS - array_lengths.keys())
    if missing_arrays:
        raise TrainerContractError(f"curriculum is missing required arrays {missing_arrays}")
    if not index_rows:
        raise TrainerContractError("curriculum index contains zero rows")
    misaligned = sorted(name for name, count in array_lengths.items() if count != len(index_rows))
    if misaligned:
        raise TrainerContractError(
            f"curriculum arrays are not row-aligned with index ({len(index_rows)} rows): {misaligned}"
        )
    missing_fields = sorted(
        {field for field in REQUIRED_INDEX_FIELDS if any(field not in row for row in index_rows)}
    )
    if missing_fields:
        raise TrainerContractError(f"curriculum index is missing required fields {missing_fields}")

    invalid_splits = sorted({str(row["split"]) for row in index_rows} - {"train", "val"})
    if invalid_splits:
        raise TrainerContractError(f"curriculum has invalid split values {invalid_splits}")

    splits_by_group: dict[str, set[str]] = {}
    for row in index_rows:
        splits_by_group.setdefault(str(row["source_group_id"]), set()).add(str(row["split"]))
    leaked = sorted(group_id for group_id, splits in splits_by_group.items() if len(splits) != 1)
    if leaked:
        raise TrainerContractError(
            f"curriculum leaks source groups across train/val; first groups: {leaked[:5]}"
        )

    validate_curriculum_maps(index_rows)


def validate_curriculum_maps(index_rows: list[dict]) -> None:
    """This lane trains and evaluates on Kalimdor and Azeroth only."""
    maps = {str(row.get("map", "")) for row in index_rows}
    out_of_scope = sorted(maps - ALLOWED_MAPS)
    if out_of_scope:
        raise TrainerContractError(
            f"curriculum contains out-of-scope maps {out_of_scope}; this lane is restricted to {sorted(ALLOWED_MAPS)}"
        )


def build_training_plan(
    *,
    index_rows: list[dict],
    selected_rows: list[int],
    batch_size: int,
    epochs: int,
    parameter_count: int,
    seed: int,
) -> dict:
    """Build the machine-readable no-training preview printed before CUDA allocation."""
    if batch_size < 1 or epochs < 1:
        raise TrainerContractError("batch size and epochs must both be positive")
    selected = [index_rows[i] for i in selected_rows]
    split_counts = {
        split: sum(str(row.get("split")) == split for row in selected)
        for split in ("train", "val")
    }
    map_counts: dict[str, int] = {}
    for row in selected:
        name = str(row.get("map", "unknown"))
        map_counts[name] = map_counts.get(name, 0) + 1
    return {
        "schema": "v125-residual-extractor-plan-v1",
        "architecture": ARCHITECTURE_ID,
        "parameter_count": parameter_count,
        "target_contract_version": TARGET_CONTRACT_VERSION,
        "selected_rows": len(selected_rows),
        "split_counts": split_counts,
        "map_counts": dict(sorted(map_counts.items())),
        "batch_size": batch_size,
        "epochs": epochs,
        "seed": seed,
        "train_steps_per_epoch": math.ceil(split_counts["train"] / batch_size),
        "deployment_inputs": ["minimap_rgb"],
        "training_target": "residual_256",
    }


def require_new_output(output: Path) -> None:
    """Never overwrite an existing run directory or partial checkpoint set."""
    if output.exists() and any(output.iterdir()):
        raise TrainerContractError(
            f"refusing to overwrite non-empty training output {output}; choose a new run path"
        )


def compute_flat_baseline(targets: list[np.ndarray]) -> float:
    """Mean absolute error of predicting each tile's own mean residual — the trivial baseline the
    model must beat to prove it extracts real shading structure."""
    if not targets:
        raise TrainerContractError("cannot compute a baseline over zero validation tiles")
    errors = [float(np.abs(t - t.mean()).mean()) for t in targets]
    return float(np.mean(errors))


def curriculum_identity(store_path: Path) -> str:
    """Content identity binding a run to the exact curriculum build."""
    digest = hashlib.sha256()
    digest.update((store_path / "index.parquet").read_bytes())
    summary = store_path / "summary.json"
    if summary.exists():
        digest.update(summary.read_bytes())
    return f"sha256:{digest.hexdigest()}"


def build_run_summary(
    *,
    identity: str,
    split_mode: str,
    per_epoch: list[dict],
    baseline_mae: float,
    train_rows: int,
    val_rows: int,
    architecture: str = ARCHITECTURE_ID,
) -> dict:
    best = min(per_epoch, key=lambda e: e["val_mae"]) if per_epoch else {"epoch": 0, "val_mae": float("inf")}
    return {
        "schema": "v125-residual-extractor-run-v1",
        "curriculum_identity": identity,
        "split_mode": split_mode,
        "target_contract_version": TARGET_CONTRACT_VERSION,
        "per_epoch_metrics": per_epoch,
        "flat_baseline": {"val_mae": baseline_mae},
        "best_epoch": best["epoch"],
        "best_val_mae": best["val_mae"],
        "beats_baseline": best["val_mae"] < baseline_mae,
        "structural_failure_epoch1_best": bool(per_epoch) and best["epoch"] == 1,
        "train_rows": train_rows,
        "val_rows": val_rows,
        "architecture": architecture,
    }


def main() -> int:
    import pyarrow.parquet as pq
    import torch
    import zarr
    from torch.utils.data import DataLoader, Dataset

    ap = argparse.ArgumentParser(description="v50 residual-extractor trainer (USER runs CUDA)")
    ap.add_argument("--store", required=True, type=Path, help="paired minimap_rgb + residual_256 store")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument(
        "--confirm-run",
        action="store_true",
        help="launch CUDA training; without this flag only print the validated plan",
    )
    ap.add_argument("--val-key", default="split")
    ap.add_argument("--val-value", default="val")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument(
        "--workers",
        type=int,
        choices=[0],
        default=0,
        help="must remain 0 for the current Windows-local Zarr dataset implementation",
    )
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=125)
    ap.add_argument("--release", default="v50.1", type=validate_release)
    args = ap.parse_args()

    group = zarr.open_group(str(args.store), mode="r")
    try:
        require_store_release(group, args.release, store=args.store)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    index = pq.read_table(args.store / "index.parquet").to_pylist()
    try:
        array_lengths = {name: int(group[name].shape[0]) for name in group.array_keys()}
        validate_curriculum_contract(
            attrs=dict(group.attrs), array_lengths=array_lengths, index_rows=index
        )
    except TrainerContractError as exc:
        raise SystemExit(str(exc)) from exc

    if any(args.val_key not in row for row in index):
        raise SystemExit(f"curriculum index does not contain --val-key {args.val_key!r}")

    selected_rows = list(range(len(index)))
    train_rows = [i for i in selected_rows if str(index[i].get(args.val_key)) != args.val_value]
    val_rows = [i for i in selected_rows if str(index[i].get(args.val_key)) == args.val_value]
    if len(train_rows) < 32 or len(val_rows) < 8:
        raise SystemExit(
            f"insufficient rows: train={len(train_rows)} val={len(val_rows)}"
        )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = ResidualExtractorNet()
    plan = build_training_plan(
        index_rows=index,
        selected_rows=selected_rows,
        batch_size=args.batch,
        epochs=args.epochs,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        seed=args.seed,
    )
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
            residual = np.asarray(group["residual_256"][row], dtype=np.float32) / 255.0
            if residual.ndim == 3:
                residual = residual[..., 0]
            return torch.from_numpy(rgb).permute(2, 0, 1), torch.from_numpy(residual)

    baseline_mae = compute_flat_baseline(
        [np.asarray(group["residual_256"][r], dtype=np.float32) / 255.0 for r in val_rows]
    )

    device = torch.device("cuda")
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    train_loader = DataLoader(
        RowDataset(train_rows),
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        generator=train_generator,
    )
    val_loader = DataLoader(RowDataset(val_rows), batch_size=args.batch, num_workers=args.workers, pin_memory=True)

    args.output.mkdir(parents=True, exist_ok=True)
    identity = curriculum_identity(args.store)
    run_identity = {
        **release_identity(args.release),
        "model_variant": ARCHITECTURE_ID,
        "parameter_count": plan["parameter_count"],
        "target_contract_version": TARGET_CONTRACT_VERSION,
        "store": str(args.store.resolve()),
        "optimizer": {"name": "AdamW", "learning_rate": args.lr, "weight_decay": 1e-4},
        "loss": {"point": "smooth_l1", "gradient_l1_weight": 0.25},
        "schedule": {
            "max_epochs": args.epochs,
            "batch_size": args.batch,
            "patience": args.patience,
            "workers": args.workers,
            "seed": args.seed,
        },
    }
    (args.output / "training_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    (args.output / "run_identity.json").write_text(json.dumps(run_identity, indent=2), encoding="utf-8")

    per_epoch: list[dict] = []
    best = float("inf")
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for x, y in train_loader:
            opt.zero_grad(set_to_none=True)
            loss = residual_loss(model(x.to(device)), y.to(device))
            loss.backward()
            opt.step()
            train_losses.append(float(loss.detach().item()))
        model.eval()
        val_absolute_error = 0.0
        val_elements = 0
        with torch.no_grad():
            for x, y in val_loader:
                y_device = y.to(device)
                absolute_error = torch.abs(model(x.to(device)) - y_device)
                val_absolute_error += float(absolute_error.sum().item())
                val_elements += absolute_error.numel()
        val_mae = val_absolute_error / val_elements
        train_loss = float(np.mean(train_losses))
        per_epoch.append({"epoch": epoch, "train_loss": train_loss, "val_mae": val_mae})
        checkpoint = {**run_identity, "model": model.state_dict(), "epoch": epoch, "val_mae": val_mae,
                      "curriculum_identity": identity}
        torch.save(checkpoint, args.output / "checkpoint_last.pt")
        if val_mae < best:
            best = val_mae
            stale = 0
            torch.save(checkpoint, args.output / "checkpoint_best.pt")
        else:
            stale += 1
        print(
            f"[epoch {epoch:03d}] train_loss={train_loss:.6f} val_mae={val_mae:.6f} "
            f"baseline={baseline_mae:.6f} best={best:.6f} stale={stale}/{args.patience}",
            flush=True,
        )
        if args.patience > 0 and stale >= args.patience:
            print(f"[early-stop] no improvement for {stale} epochs", flush=True)
            break

    summary = build_run_summary(
        identity=identity,
        split_mode=str(group.attrs.get("split_mode", f"{args.val_key}={args.val_value}")),
        per_epoch=per_epoch, baseline_mae=baseline_mae,
        train_rows=len(train_rows), val_rows=len(val_rows),
    )
    (args.output / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if summary["structural_failure_epoch1_best"]:
        print("STRUCTURAL FAILURE: best epoch is epoch 1; this run is not a success.", flush=True)
        return 1
    print(f"best_epoch={summary['best_epoch']} best_val_mae={summary['best_val_mae']:.6f} "
          f"baseline={baseline_mae:.6f} beats_baseline={summary['beats_baseline']}", flush=True)
    return 0
