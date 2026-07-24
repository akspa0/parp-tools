"""Spec 121 diagnostic: WHY did a Stage A run flatline on held-out tiles?

Three architectures (LatticeNet v2/v5, MitB0LatticeNet) plateaued at ~0.21–0.24 val MAE against a
0.128 tile-mean baseline while train loss kept dropping. That wall across wildly different
capacities points at data/split/signal, not the model. This diagnostic discriminates the three
candidate causes on a finished OR in-progress run (reads a checkpoint copy; never writes to the
run dir):

- **memorization**: native masked MAE on TRAIN rows << tile-mean while val >> tile-mean.
- **zone shift**: per-map val MAE differs sharply (the Spec 116 split isolates whole regions, so
  a zone-specific color->height mapping cannot transfer).
- **signal absence / hedging**: prediction variance collapses relative to target variance AND
  train MAE is also poor.

Read-only. CPU by default so it never competes with a live training run for the GPU.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from harvester.spec117.lattice_model import (
    compute_lattice_tile_mean_baseline,
    encode_lattice_target,
    select_lattice_rows,
)
from harvester.spec121.lattice_backbone_model import (
    LATTICE_NET_ID,
    build_stage_a_model,
    config_from_payload,
)
from harvester.v50.direct_geometry_train import apply_held_out_split
from harvester.v50.height_relative_train import select_training_rows


def _rebuild_model(checkpoint: dict):
    backbone = checkpoint["backbone_config"]
    if backbone["architecture"] == LATTICE_NET_ID:
        model, _ = build_stage_a_model(LATTICE_NET_ID, base=int(backbone["base"]))
    else:
        model, _ = build_stage_a_model(
            backbone["architecture"],
            mit_config=config_from_payload(backbone["segformer_config"]),
        )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def _native_mae_per_row(model, group, rows: list[int], device) -> list[dict]:
    import torch

    results: list[dict] = []
    with torch.no_grad():
        for row in rows:
            rgb = np.asarray(group["minimap_rgb"][row], dtype=np.float32) / 255.0
            target, mask, _tmin, _tmax = encode_lattice_target(
                np.asarray(group["wdl_outer_17"][row]),
                np.asarray(group["wdl_inner_16"][row]),
                np.asarray(group["wdl_outer_present"][row]),
                np.asarray(group["wdl_inner_present"][row]),
            )
            rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
            pred = model(rgb_t).squeeze(0).cpu().numpy()
            present = mask > 0
            tile_mean = float(target[present].mean())
            results.append({
                "row": int(row),
                "model_mae": float(np.abs(pred[present] - target[present]).mean()),
                "tile_mean_mae": float(np.abs(target[present] - tile_mean).mean()),
                "pred_std": float(pred[present].std()),
                "target_std": float(target[present].std()),
            })
    return results


def _summarize(records: list[dict], label: str, index_rows=None) -> dict:
    maes = np.array([r["model_mae"] for r in records])
    tm = np.array([r["tile_mean_mae"] for r in records])
    summary = {
        "label": label,
        "n": len(records),
        "model_mae": float(maes.mean()),
        "tile_mean_mae": float(tm.mean()),
        "margin_vs_tile_mean": float((tm.mean() - maes.mean()) / max(tm.mean(), 1e-9)),
        "pct_rows_beating_tile_mean": float((maes < tm).mean()),
        "pred_std": float(np.mean([r["pred_std"] for r in records])),
        "target_std": float(np.mean([r["target_std"] for r in records])),
    }
    if index_rows is not None:
        per_map: dict[str, list[int]] = {}
        for i, r in enumerate(records):
            per_map.setdefault(str(index_rows[r["row"]].get("map", "?")), []).append(i)
        summary["per_map"] = {
            m: {
                "n": len(idxs),
                "model_mae": float(maes[idxs].mean()),
                "tile_mean_mae": float(tm[idxs].mean()),
                "pct_rows_beating_tile_mean": float((maes[idxs] < tm[idxs]).mean()),
            }
            for m, idxs in sorted(per_map.items())
        }
    return summary


def diagnose(
    *,
    run_dir: Path,
    store_path: Path,
    split_dir: Path,
    source: str,
    train_sample: int,
    val_sample: int,
    device_name: str,
) -> dict:
    import pyarrow.parquet as pq
    import torch
    import zarr

    # Copy the checkpoint first: a live run can replace it mid-read.
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_copy = Path(tmp) / "ckpt.pt"
        shutil.copy2(run_dir / "checkpoint_best.pt", ckpt_copy)
        checkpoint = torch.load(ckpt_copy, map_location="cpu", weights_only=False)
    model = _rebuild_model(checkpoint)
    device = torch.device(device_name)
    model = model.to(device)

    group = zarr.open_group(str(store_path), mode="r")
    index = pq.read_table(store_path / "index.parquet").to_pylist()
    selected = select_training_rows(index, source)
    train_rows_all, val_rows_all, _manifest = apply_held_out_split(
        index_rows=index, selected_rows=selected, split_dir=split_dir,
    )
    train_rows, _ = select_lattice_rows(group, train_rows_all)
    val_rows, _ = select_lattice_rows(group, val_rows_all)

    rng = np.random.default_rng(121)
    train_pick = sorted(rng.choice(train_rows, size=min(train_sample, len(train_rows)), replace=False).tolist())
    val_pick = sorted(rng.choice(val_rows, size=min(val_sample, len(val_rows)), replace=False).tolist())

    val_tm = compute_lattice_tile_mean_baseline(
        [encode_lattice_target(
            np.asarray(group["wdl_outer_17"][r]),
            np.asarray(group["wdl_inner_16"][r]),
            np.asarray(group["wdl_outer_present"][r]),
            np.asarray(group["wdl_inner_present"][r]),
        )[:2] for r in val_rows]
    )

    train_records = _native_mae_per_row(model, group, train_pick, device)
    val_records = _native_mae_per_row(model, group, val_pick, device)
    return {
        "schema": "v121-stage-a-diagnosis-v1",
        "run_dir": str(run_dir.resolve()),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_val_mae": float(checkpoint.get("val_mae", float("nan"))),
        "full_val_tile_mean_baseline": val_tm,
        "train_subset": _summarize(train_records, "train"),
        "val_subset": _summarize(val_records, "val", index_rows=index),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Spec 121 Stage A flatline diagnostic (read-only)")
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--store", required=True, type=Path)
    ap.add_argument("--held-out-split", required=True, type=Path)
    ap.add_argument("--source", default="authored")
    ap.add_argument("--train-sample", type=int, default=120)
    ap.add_argument("--val-sample", type=int, default=240)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="cpu (default) never competes with a live training run for the GPU")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    result = diagnose(
        run_dir=args.run_dir, store_path=args.store, split_dir=args.held_out_split,
        source=args.source, train_sample=args.train_sample, val_sample=args.val_sample,
        device_name=args.device,
    )
    text = json.dumps(result, indent=2)
    print(text, flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
