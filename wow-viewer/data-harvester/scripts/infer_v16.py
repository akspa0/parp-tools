"""Run deterministic V16 inference and emit patch-ready outputs.

This script does three things in one pass:
1. Reads V16 input tiles from `<build>.zarr`.
2. Runs V15/V16 terrain model inference.
3. Writes:
   - consolidated prediction store: `<build>.pred.zarr`
   - patch-ready per-tile summaries compatible with `terrain-patch-adt`

Usage:
    cd wow-viewer/data-harvester
    uv run python scripts/infer_v16.py \
        --build 3_3_5_12340 \
        --checkpoint ../models/v16/runs/<run>/checkpoints/v16_best.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import zarr
import zarr.codecs
import zarr.storage

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from harvester.v15_model import V15Model  # noqa: E402


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DATASET_ROOT = _PROJECT_ROOT / "output" / "datasets" / "v16"
_OUTPUT_ROOT = _PROJECT_ROOT / "output" / "datasets" / "v16_inference"

_HEIGHT_SIZE = 257
_ALPHA_SIZE = 256
_HOLES_SIZE = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V16 terrain inference and emit patch-ready outputs")
    parser.add_argument("--dataset-dir", type=Path, default=_DATASET_ROOT, help="Directory containing V16 <build>.zarr stores")
    parser.add_argument("--build", required=True, help="Build key (for example: 3_3_5_12340)")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to V16 checkpoint (.pt)")
    parser.add_argument("--output-root", type=Path, default=_OUTPUT_ROOT, help="Root output directory for prediction runs")
    parser.add_argument("--run-name", type=str, default=None, help="Inference run folder name (default: UTC timestamp)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--limit", type=int, default=None, help="Optional tile limit (for smoke runs)")
    parser.add_argument("--amp", action="store_true", help="Enable CUDA autocast during inference (off by default for stricter determinism)")
    parser.add_argument("--no-patch-ready", action="store_true", help="Skip per-tile patch-ready summary outputs")
    return parser.parse_args()


def _now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Favor deterministic replay over peak speed.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def _resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _normalize_state_dict_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    prefix = "_orig_mod."
    if any(key.startswith(prefix) for key in state_dict):
        return {
            (key[len(prefix):] if key.startswith(prefix) else key): value
            for key, value in state_dict.items()
        }
    return state_dict


def _load_checkpoint_model(checkpoint_path: Path, device: torch.device) -> tuple[V15Model, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise RuntimeError(f"Unsupported checkpoint payload type: {type(checkpoint)!r}")

    model = V15Model().to(device)
    model.load_state_dict(_normalize_state_dict_keys(state_dict))
    model.eval()
    return model, checkpoint if isinstance(checkpoint, dict) else {}


def _write_patch_ready_summary(
    patch_ready_root: Path,
    *,
    tile_name: str,
    predicted_height_257: np.ndarray,
    shard_tag: str,
) -> None:
    tile_dir = patch_ready_root / tile_name
    tile_dir.mkdir(parents=True, exist_ok=True)
    npy_path = tile_dir / "predicted_height_257.npy"
    np.save(npy_path, predicted_height_257.astype(np.float32))

    summary = {
        "tile_name": tile_name,
        "shard": shard_tag,
        "predicted_height_257_path": npy_path.name,
    }
    (tile_dir / "inference_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _as_rows(table: pa.Table) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_index in range(table.num_rows):
        row = {column: table.column(column)[row_index].as_py() for column in table.column_names}
        rows.append(row)
    return rows


def _ensure_required_arrays(root: zarr.Group) -> None:
    required = ["minimap_rgb"]
    missing = [name for name in required if name not in root]
    if missing:
        raise RuntimeError(f"Input store is missing required arrays: {', '.join(missing)}")


def main() -> None:
    args = parse_args()
    started_at = _now_iso_utc()
    _seed_everything(args.seed)

    dataset_dir = args.dataset_dir.resolve()
    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.exists():
        raise RuntimeError(f"Checkpoint not found: {checkpoint_path}")

    source_store_path = dataset_dir / f"{args.build}.zarr"
    if not source_store_path.exists():
        raise RuntimeError(f"Input store not found: {source_store_path}")

    source_index_path = source_store_path / "index.parquet"
    if not source_index_path.exists():
        raise RuntimeError(f"Input index missing: {source_index_path}")

    run_name = args.run_name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = (args.output_root.resolve() / run_name)
    pred_store_path = run_root / f"{args.build}.pred.zarr"
    patch_ready_root = run_root / "patch_ready"
    run_root.mkdir(parents=True, exist_ok=True)

    if pred_store_path.exists():
        raise RuntimeError(f"Prediction store already exists: {pred_store_path}")

    table = pq.read_table(str(source_index_path))
    rows = _as_rows(table)
    if args.limit is not None:
        if args.limit <= 0:
            raise RuntimeError("--limit must be > 0")
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError("No rows found in input index for inference.")

    device = _resolve_device(args.device)
    model, checkpoint_payload = _load_checkpoint_model(checkpoint_path, device)

    source_store = zarr.storage.LocalStore(str(source_store_path), read_only=True)
    source_root = zarr.open_group(store=source_store, mode="r")
    _ensure_required_arrays(source_root)

    total = len(rows)
    tile_ids = np.asarray([int(row["tile_id"]) for row in rows], dtype=np.int64)
    means = np.asarray([float(row["height_mean"]) for row in rows], dtype=np.float32)
    stds = np.asarray([float(row["height_std"]) for row in rows], dtype=np.float32)

    codec = zarr.codecs.BloscCodec(cname="lz4", clevel=1, shuffle="shuffle")
    pred_store = zarr.storage.LocalStore(str(pred_store_path), read_only=False)
    pred_root = zarr.open_group(store=pred_store, mode="w")

    batch_chunk = min(max(args.batch_size, 1), 64)
    pred_height = pred_root.create_array(
        "height_pred_257",
        shape=(_safe_int(total), _HEIGHT_SIZE, _HEIGHT_SIZE),
        chunks=(batch_chunk, _HEIGHT_SIZE, _HEIGHT_SIZE),
        dtype=np.float32,
        compressors=[codec],
    )
    pred_normals = pred_root.create_array(
        "normal_pred_xyz",
        shape=(_safe_int(total), _HEIGHT_SIZE, _HEIGHT_SIZE, 3),
        chunks=(batch_chunk, _HEIGHT_SIZE, _HEIGHT_SIZE, 3),
        dtype=np.float32,
        compressors=[codec],
    )
    pred_alpha = pred_root.create_array(
        "alpha_pred_256",
        shape=(_safe_int(total), _ALPHA_SIZE, _ALPHA_SIZE, 4),
        chunks=(batch_chunk, _ALPHA_SIZE, _ALPHA_SIZE, 4),
        dtype=np.float32,
        compressors=[codec],
    )
    pred_holes = pred_root.create_array(
        "holes_pred_16",
        shape=(_safe_int(total), _HOLES_SIZE, _HOLES_SIZE),
        chunks=(batch_chunk, _HOLES_SIZE, _HOLES_SIZE),
        dtype=np.float32,
        compressors=[codec],
    )
    pred_liquid = pred_root.create_array(
        "liquid_pred_mask_256",
        shape=(_safe_int(total), _ALPHA_SIZE, _ALPHA_SIZE),
        chunks=(batch_chunk, _ALPHA_SIZE, _ALPHA_SIZE),
        dtype=np.float32,
        compressors=[codec],
    )
    pred_mcly = pred_root.create_array(
        "mcly_pred_logits_16x16x4x16",
        shape=(_safe_int(total), _HOLES_SIZE, _HOLES_SIZE, 4, 16),
        chunks=(batch_chunk, _HOLES_SIZE, _HOLES_SIZE, 4, 16),
        dtype=np.float32,
        compressors=[codec],
    )

    if not args.no_patch_ready:
        patch_ready_root.mkdir(parents=True, exist_ok=True)

    print(f"Build: {args.build}")
    print(f"Tiles: {total}")
    print(f"Input: {source_store_path}")
    print(f"Output: {pred_store_path}")
    print(f"Patch-ready summaries: {'on' if not args.no_patch_ready else 'off'}")
    print(f"Device: {device}")
    print(f"AMP: {args.amp and device.type == 'cuda'}")

    with torch.no_grad():
        for start in range(0, total, args.batch_size):
            end = min(start + args.batch_size, total)
            batch_ids = tile_ids[start:end]
            batch_minimap = np.stack(
                [source_root["minimap_rgb"][int(tile_id)] for tile_id in batch_ids],
                axis=0,
            ).astype(np.float32) / 255.0

            inp = torch.from_numpy(batch_minimap).permute(0, 3, 1, 2).to(device)
            with torch.amp.autocast("cuda", enabled=(args.amp and device.type == "cuda")):
                out_height, out_normals, out_alpha, out_holes, out_liquid, out_mcly = model(inp)

            out_height_np = out_height.squeeze(1).cpu().numpy().astype(np.float32)
            out_normals_np = F.normalize(out_normals, dim=1).permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)
            out_alpha_np = out_alpha.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)
            out_holes_np = out_holes.squeeze(1).cpu().numpy().astype(np.float32)
            out_liquid_np = out_liquid.squeeze(1).cpu().numpy().astype(np.float32)
            out_mcly_np = out_mcly.view(-1, 4, 16, 16, 16).permute(0, 3, 4, 1, 2).cpu().numpy().astype(np.float32)

            batch_means = means[start:end][:, None, None]
            batch_stds = stds[start:end][:, None, None]
            out_height_world = (out_height_np * (batch_stds + 1e-8)) + batch_means

            out_alpha_np = np.clip(out_alpha_np, 0.0, 1.0)
            out_holes_np = np.clip(out_holes_np, 0.0, 1.0)
            out_liquid_np = np.clip(out_liquid_np, 0.0, 1.0)

            pred_height[start:end] = out_height_world
            pred_normals[start:end] = out_normals_np
            pred_alpha[start:end] = out_alpha_np
            pred_holes[start:end] = out_holes_np
            pred_liquid[start:end] = out_liquid_np
            pred_mcly[start:end] = out_mcly_np

            if not args.no_patch_ready:
                for batch_offset, row in enumerate(rows[start:end]):
                    tile_name = f"{row['map']}_{int(row['tile_x'])}_{int(row['tile_y'])}"
                    source_tile_id = int(row["tile_id"])
                    _write_patch_ready_summary(
                        patch_ready_root,
                        tile_name=tile_name,
                        predicted_height_257=out_height_world[batch_offset],
                        shard_tag=f"{args.build}.zarr:tile_id={source_tile_id}",
                    )

            print(f"Inference {end}/{total}")

    output_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        out_row = dict(row)
        out_row["tile_id"] = row_index
        out_row["source_tile_id"] = int(row["tile_id"])
        output_rows.append(out_row)

    output_table = pa.Table.from_pylist(output_rows)
    pq.write_table(output_table, str(pred_store_path / "index.parquet"))

    source_placements = source_store_path / "placements.parquet"
    if source_placements.exists():
        source_tile_id_set = {int(row["source_tile_id"]) for row in output_rows}
        placements_table = pq.read_table(str(source_placements))
        placement_rows = _as_rows(placements_table)
        placement_rows = [row for row in placement_rows if int(row.get("tile_id", -1)) in source_tile_id_set]
        if placement_rows:
            source_to_output = {int(row["source_tile_id"]): int(row["tile_id"]) for row in output_rows}
            for row in placement_rows:
                row["source_tile_id"] = int(row["tile_id"])
                row["tile_id"] = source_to_output[int(row["source_tile_id"])]
            pq.write_table(pa.Table.from_pylist(placement_rows), str(pred_store_path / "placements.parquet"))

    run_manifest = {
        "model_version": "v16",
        "build": args.build,
        "run_name": run_name,
        "source_store_path": str(source_store_path),
        "prediction_store_path": str(pred_store_path),
        "patch_ready_root": str(patch_ready_root) if not args.no_patch_ready else None,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "input_index_path": str(source_index_path),
        "input_index_sha256": _sha256_file(source_index_path),
        "seed": args.seed,
        "device": str(device),
        "amp_enabled": bool(args.amp and device.type == "cuda"),
        "compile_enabled": False,
        "source_tile_count": int(table.num_rows),
        "output_tile_count": total,
        "limited": args.limit is not None,
        "torch_version": torch.__version__,
        "started_at": started_at,
        "finished_at": _now_iso_utc(),
        "checkpoint_meta_epoch": checkpoint_payload.get("epoch"),
    }
    (run_root / "_inference_run.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")

    if not args.no_patch_ready:
        print("\nNext command (LK patch path):")
        print(
            "dotnet run --project ../tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -- "
            f"terrain-patch-adt --input-adt-dir <staged-map-root> --inference-dir \"{patch_ready_root}\" --output-dir <patched-output-root>"
        )
        print("\nOptional alpha route after patching:")
        print(
            "dotnet run --project ../tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -- "
            "convert-lk-to-alpha --input <patched-output-root> --output <patched-output.wdt>"
        )

    print(f"\nDone. Wrote {pred_store_path}")


def _safe_int(value: int) -> int:
    if value < 0:
        raise ValueError("Negative shape is not valid.")
    return int(value)


if __name__ == "__main__":
    main()
