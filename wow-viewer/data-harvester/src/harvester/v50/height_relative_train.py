"""User-run trainer for the v50 relative-height model (Spec 112 US3, T019).

Consumes the dual-source curriculum (``v50-mixed-curriculum-v1``): each row pairs one minimap image
(authored OR synthetic, recorded in ``minimap_source`` — the model never sees the label) with the
same tile's real height ground truth, encoded per the Relative-Height Target Contract.

Contract enforcement lives in small pure functions so the gates are CPU-testable without CUDA:
``validate_curriculum_maps`` (Kalimdor/Azeroth only, FR-011), ``compute_tile_mean_baseline``
(SC-004's in-run baseline), ``build_run_summary`` (FR-010's machine-readable record, incl. the
epoch-1-best structural-failure flag from the execution contract).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from harvester.v50.contracts import release_identity, require_store_release, validate_release
from harvester.v50.height_relative_model import (
    TARGET_CONTRACT_VERSION,
    HeightRelativeNet,
    encode_relative_height,
    height_loss,
)

ALLOWED_MAPS = frozenset({"Kalimdor", "Azeroth"})
CURRICULUM_SCHEMA = "v50-mixed-curriculum-v1"
REQUIRED_ARRAYS = frozenset({"minimap_rgb", "height_257"})
REQUIRED_INDEX_FIELDS = frozenset({"map", "source_group_id", "minimap_source", "split"})


class TrainerContractError(ValueError):
    """Raised when a run would violate a Spec 112 contract gate."""


def validate_curriculum_contract(
    *,
    attrs: dict,
    array_lengths: dict[str, int],
    index_rows: list[dict],
) -> None:
    """Fail closed before CUDA allocation when the input is not the Spec 112 curriculum.

    This is intentionally pure so the schema, row-alignment, and leak-safety gates are covered by
    CPU tests. Release identity is additionally checked by ``require_store_release`` in ``main``.
    """
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

    source_values = {str(row["minimap_source"]) for row in index_rows}
    invalid_sources = sorted(source_values - {"authored", "synthetic"})
    if invalid_sources:
        raise TrainerContractError(f"curriculum has invalid minimap_source values {invalid_sources}")
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
    """FR-011: this lane trains and evaluates on Kalimdor and Azeroth only."""
    maps = {str(row.get("map", "")) for row in index_rows}
    out_of_scope = sorted(maps - ALLOWED_MAPS)
    if out_of_scope:
        raise TrainerContractError(
            f"curriculum contains out-of-scope maps {out_of_scope}; this lane is restricted to {sorted(ALLOWED_MAPS)}"
        )


def compute_tile_mean_baseline(targets: list[np.ndarray]) -> float:
    """Mean absolute error of predicting each tile's own mean normalized height — the trivial
    baseline SC-004 requires the model to beat, computed in-run so the claim is self-contained."""
    if not targets:
        raise TrainerContractError("cannot compute a baseline over zero validation tiles")
    errors = [float(np.abs(t - t.mean()).mean()) for t in targets]
    return float(np.mean(errors))


def curriculum_identity(store_path: Path) -> str:
    """Content identity binding a run to the exact curriculum build (index bytes + store attrs)."""
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
    source_counts: dict[str, int],
) -> dict:
    best = min(per_epoch, key=lambda e: e["val_mae"]) if per_epoch else {"epoch": 0, "val_mae": float("inf")}
    return {
        "schema": "v112-height-run-v1",
        "curriculum_identity": identity,
        "split_mode": split_mode,
        "target_contract_version": TARGET_CONTRACT_VERSION,
        "per_epoch_metrics": per_epoch,
        "tile_mean_baseline": {"val_mae": baseline_mae},
        "best_epoch": best["epoch"],
        "best_val_mae": best["val_mae"],
        "beats_baseline": best["val_mae"] < baseline_mae,
        # The rejected lane's signature failure: an epoch-1 best means the model got WORSE with
        # training -- a structural problem, never reportable as success (execution contract §3).
        "structural_failure_epoch1_best": bool(per_epoch) and best["epoch"] == 1,
        "train_rows": train_rows,
        "val_rows": val_rows,
        "minimap_source_counts": source_counts,
    }


def main() -> int:
    import torch
    import pyarrow.parquet as pq
    import zarr
    from torch.utils.data import DataLoader, Dataset

    ap = argparse.ArgumentParser(description="v50 relative-height trainer (USER runs CUDA)")
    ap.add_argument("--store", required=True, type=Path, help="dual-source curriculum store")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--val-key", default="split"); ap.add_argument("--val-value", default="val")
    ap.add_argument("--epochs", type=int, default=100); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--release", default="v50.1", type=validate_release)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; user-run training refuses CPU.")
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

    train_rows = [i for i, r in enumerate(index) if str(r.get(args.val_key)) != args.val_value]
    val_rows = [i for i, r in enumerate(index) if str(r.get(args.val_key)) == args.val_value]
    if len(train_rows) < 32 or len(val_rows) < 8:
        raise SystemExit(f"insufficient rows: train={len(train_rows)} val={len(val_rows)}")
    source_counts: dict[str, int] = {}
    for r in index:
        source_counts[str(r.get("minimap_source", "unknown"))] = source_counts.get(str(r.get("minimap_source", "unknown")), 0) + 1

    class RowDataset(Dataset):
        def __init__(self, rows: list[int]) -> None:
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, i: int):
            row = self.rows[i]
            rgb = np.asarray(group["minimap_rgb"][row], dtype=np.float32) / 255.0
            target, _, _ = encode_relative_height(np.asarray(group["height_257"][row]))
            return torch.from_numpy(rgb).permute(2, 0, 1), torch.from_numpy(target)

    baseline_mae = compute_tile_mean_baseline(
        [encode_relative_height(np.asarray(group["height_257"][r]))[0] for r in val_rows]
    )

    device = torch.device("cuda")
    model = HeightRelativeNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    train_loader = DataLoader(RowDataset(train_rows), batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(RowDataset(val_rows), batch_size=args.batch, num_workers=args.workers, pin_memory=True)

    args.output.mkdir(parents=True, exist_ok=True)
    identity = curriculum_identity(args.store)
    run_identity = {**release_identity(args.release), "model_variant": "height-relative-v112",
                    "target_contract_version": TARGET_CONTRACT_VERSION, "store": str(args.store.resolve())}
    (args.output / "run_identity.json").write_text(json.dumps(run_identity, indent=2), encoding="utf-8")

    per_epoch: list[dict] = []
    best = float("inf"); stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in train_loader:
            opt.zero_grad(set_to_none=True)
            loss = height_loss(model(x.to(device)), y.to(device))
            loss.backward(); opt.step()
        model.eval(); maes = []
        with torch.no_grad():
            for x, y in val_loader:
                maes.append(float(torch.nn.functional.l1_loss(model(x.to(device)), y.to(device)).item()))
        val_mae = float(np.mean(maes))
        per_epoch.append({"epoch": epoch, "val_mae": val_mae})
        checkpoint = {**run_identity, "model": model.state_dict(), "epoch": epoch, "val_mae": val_mae,
                      "curriculum_identity": identity}
        torch.save(checkpoint, args.output / "checkpoint_last.pt")
        if val_mae < best:
            best = val_mae; stale = 0
            torch.save(checkpoint, args.output / "checkpoint_best.pt")
        else:
            stale += 1
        print(f"[epoch {epoch:03d}] val_mae={val_mae:.6f} baseline={baseline_mae:.6f} best={best:.6f} stale={stale}/{args.patience}", flush=True)
        if args.patience > 0 and stale >= args.patience:
            print(f"[early-stop] no improvement for {stale} epochs", flush=True)
            break

    summary = build_run_summary(
        identity=identity,
        split_mode=str(group.attrs.get("split_mode", f"{args.val_key}={args.val_value}")),
        per_epoch=per_epoch, baseline_mae=baseline_mae,
        train_rows=len(train_rows), val_rows=len(val_rows), source_counts=source_counts,
    )
    (args.output / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if summary["structural_failure_epoch1_best"]:
        print("STRUCTURAL FAILURE: best epoch is epoch 1 (the rejected lane's signature); this run is not a success.", flush=True)
        return 1
    print(f"best_epoch={summary['best_epoch']} best_val_mae={summary['best_val_mae']:.6f} "
          f"baseline={baseline_mae:.6f} beats_baseline={summary['beats_baseline']}", flush=True)
    return 0
