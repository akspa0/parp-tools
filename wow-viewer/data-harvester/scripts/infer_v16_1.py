"""Run stitched V16.1 inference and write a combined prediction Zarr store."""

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
import zarr
import zarr.codecs
import zarr.storage

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from harvester.v16_1_models import (  # noqa: E402
    V161HeightModel,
    V161HolesModel,
    V161LiquidModel,
    V161NormalModel,
    V161TexcompModel,
    recompose_from_mcly_alpha,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DATASET_ROOT = _PROJECT_ROOT / "output" / "datasets" / "v16"
_OUTPUT_ROOT = _PROJECT_ROOT / "output" / "datasets" / "v16_1_inference"


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


def _resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA but it is unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _as_rows(table: pa.Table) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_index in range(table.num_rows):
        rows.append({column: table.column(column)[row_index].as_py() for column in table.column_names})
    return rows


def _safe_int(value: int) -> int:
    return int(value)


def _load_model(model_cls: type[torch.nn.Module], checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = model_cls().to(device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run stitched V16.1 inference and write combined prediction Zarr output")
    p.add_argument("--dataset-dir", type=Path, default=_DATASET_ROOT)
    p.add_argument("--build", required=True)
    p.add_argument("--output-root", type=Path, default=_OUTPUT_ROOT)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--tile-id", type=int, default=None)
    p.add_argument("--height-checkpoint", type=Path, default=None)
    p.add_argument("--normal-checkpoint", type=Path, default=None)
    p.add_argument("--holes-checkpoint", type=Path, default=None)
    p.add_argument("--liquid-checkpoint", type=Path, default=None)
    p.add_argument("--texcomp-checkpoint", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _seed_everything(args.seed)
    device = _resolve_device(args.device)

    source_store_path = args.dataset_dir.resolve() / f"{args.build}.zarr"
    if not source_store_path.exists():
        raise RuntimeError(f"Input store not found: {source_store_path}")
    index_path = source_store_path / "index.parquet"
    if not index_path.exists():
        raise RuntimeError(f"Missing index.parquet in {source_store_path}")

    source_store = zarr.storage.LocalStore(str(source_store_path), read_only=True)
    source_root = zarr.open_group(store=source_store, mode="r")
    table = pq.read_table(str(index_path))
    rows = _as_rows(table)
    if args.tile_id is not None:
        rows = [row for row in rows if int(row["tile_id"]) == int(args.tile_id)]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError("No rows selected for stitched inference.")

    models: dict[str, torch.nn.Module] = {}
    checkpoints: dict[str, str] = {}
    if args.height_checkpoint:
        models["height"] = _load_model(V161HeightModel, args.height_checkpoint.resolve(), device)
        checkpoints["height"] = str(args.height_checkpoint.resolve())
    if args.normal_checkpoint:
        models["normal"] = _load_model(V161NormalModel, args.normal_checkpoint.resolve(), device)
        checkpoints["normal"] = str(args.normal_checkpoint.resolve())
    if args.holes_checkpoint:
        models["holes"] = _load_model(V161HolesModel, args.holes_checkpoint.resolve(), device)
        checkpoints["holes"] = str(args.holes_checkpoint.resolve())
    if args.liquid_checkpoint:
        models["liquid"] = _load_model(V161LiquidModel, args.liquid_checkpoint.resolve(), device)
        checkpoints["liquid"] = str(args.liquid_checkpoint.resolve())
    if args.texcomp_checkpoint:
        models["texcomp"] = _load_model(V161TexcompModel, args.texcomp_checkpoint.resolve(), device)
        checkpoints["texcomp"] = str(args.texcomp_checkpoint.resolve())
    if not models:
        raise RuntimeError("Provide at least one checkpoint.")

    run_name = args.run_name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = args.output_root.resolve() / run_name
    pred_store_path = run_root / f"{args.build}.pred.zarr"
    run_root.mkdir(parents=True, exist_ok=True)
    if pred_store_path.exists():
        raise RuntimeError(f"Prediction store already exists: {pred_store_path}")

    total = len(rows)
    tile_ids = np.asarray([int(row["tile_id"]) for row in rows], dtype=np.int64)
    means = np.asarray([float(row["height_mean"]) for row in rows], dtype=np.float32)
    stds = np.asarray([float(row["height_std"]) for row in rows], dtype=np.float32)

    codec = zarr.codecs.BloscCodec(cname="lz4", clevel=1, shuffle="shuffle")
    pred_store = zarr.storage.LocalStore(str(pred_store_path), read_only=False)
    pred_root = zarr.open_group(store=pred_store, mode="w")
    batch_chunk = min(max(args.batch_size, 1), 64)

    arrays: dict[str, Any] = {}
    if "height" in models:
        arrays["height_pred_257"] = pred_root.create_array("height_pred_257", shape=(_safe_int(total), 257, 257), chunks=(batch_chunk, 257, 257), dtype=np.float32, compressors=[codec])
    if "normal" in models:
        arrays["normal_pred_xyz"] = pred_root.create_array("normal_pred_xyz", shape=(_safe_int(total), 257, 257, 3), chunks=(batch_chunk, 257, 257, 3), dtype=np.float32, compressors=[codec])
    if "holes" in models:
        arrays["holes_pred_16"] = pred_root.create_array("holes_pred_16", shape=(_safe_int(total), 16, 16), chunks=(batch_chunk, 16, 16), dtype=np.float32, compressors=[codec])
    if "liquid" in models:
        arrays["liquid_pred_mask_256"] = pred_root.create_array("liquid_pred_mask_256", shape=(_safe_int(total), 256, 256), chunks=(batch_chunk, 256, 256), dtype=np.float32, compressors=[codec])
        arrays["liquid_type_pred_16"] = pred_root.create_array("liquid_type_pred_16", shape=(_safe_int(total), 16, 16), chunks=(batch_chunk, 16, 16), dtype=np.int16, compressors=[codec])
        arrays["liquid_type_logits_5x16x16"] = pred_root.create_array("liquid_type_logits_5x16x16", shape=(_safe_int(total), 5, 16, 16), chunks=(batch_chunk, 5, 16, 16), dtype=np.float32, compressors=[codec])
    if "texcomp" in models:
        arrays["alpha_pred_256"] = pred_root.create_array("alpha_pred_256", shape=(_safe_int(total), 256, 256, 4), chunks=(batch_chunk, 256, 256, 4), dtype=np.float32, compressors=[codec])
        arrays["mcly_mask_pred_16x16x4"] = pred_root.create_array("mcly_mask_pred_16x16x4", shape=(_safe_int(total), 16, 16, 4), chunks=(batch_chunk, 16, 16, 4), dtype=np.float32, compressors=[codec])
        arrays["mcly_id_pred_16x16x4"] = pred_root.create_array("mcly_id_pred_16x16x4", shape=(_safe_int(total), 16, 16, 4), chunks=(batch_chunk, 16, 16, 4), dtype=np.int16, compressors=[codec])
        arrays["recomposed_pred_rgb_256"] = pred_root.create_array("recomposed_pred_rgb_256", shape=(_safe_int(total), 256, 256, 3), chunks=(batch_chunk, 256, 256, 3), dtype=np.float32, compressors=[codec])

    print(f"Build: {args.build}")
    print(f"Tiles: {total}")
    print(f"Device: {device}")
    print(f"Output: {pred_store_path}")
    print(f"Models: {', '.join(sorted(models.keys()))}")

    with torch.no_grad():
        for start in range(0, total, args.batch_size):
            end = min(start + args.batch_size, total)
            batch_ids = tile_ids[start:end]
            batch_minimap = np.stack([source_root["minimap_rgb"][int(tile_id)] for tile_id in batch_ids], axis=0).astype(np.float32) / 255.0
            inp = torch.from_numpy(batch_minimap).permute(0, 3, 1, 2).to(device)

            if "height" in models:
                pred_h = models["height"](inp).squeeze(1).cpu().numpy().astype(np.float32)
                pred_h_world = pred_h * (stds[start:end][:, None, None] + 1e-8) + means[start:end][:, None, None]
                arrays["height_pred_257"][start:end] = pred_h_world
            if "normal" in models:
                pred_n = models["normal"](inp)
                pred_n = torch.nn.functional.normalize(pred_n, dim=1).permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)
                arrays["normal_pred_xyz"][start:end] = pred_n
            if "holes" in models:
                pred_ho = models["holes"](inp).squeeze(1).cpu().numpy().astype(np.float32)
                arrays["holes_pred_16"][start:end] = pred_ho
            if "liquid" in models:
                pred_mask, pred_type_logits = models["liquid"](inp)
                pred_mask_np = pred_mask.squeeze(1).cpu().numpy().astype(np.float32)
                pred_type_logits_np = pred_type_logits.cpu().numpy().astype(np.float32)
                pred_type_np = pred_type_logits.argmax(dim=1).cpu().numpy().astype(np.int16)
                arrays["liquid_pred_mask_256"][start:end] = pred_mask_np
                arrays["liquid_type_logits_5x16x16"][start:end] = pred_type_logits_np
                arrays["liquid_type_pred_16"][start:end] = pred_type_np
            if "texcomp" in models:
                pred_alpha, pred_mcly_mask, pred_mcly_ids = models["texcomp"](inp)
                pred_recomp = recompose_from_mcly_alpha(pred_alpha, pred_mcly_ids, pred_mcly_mask)
                arrays["alpha_pred_256"][start:end] = pred_alpha.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)
                arrays["mcly_mask_pred_16x16x4"][start:end] = pred_mcly_mask.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)
                arrays["mcly_id_pred_16x16x4"][start:end] = pred_mcly_ids.argmax(dim=2).permute(0, 2, 3, 1).cpu().numpy().astype(np.int16)
                arrays["recomposed_pred_rgb_256"][start:end] = pred_recomp.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)
            print(f"Inference {end}/{total}")

    output_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        out_row = dict(row)
        out_row["tile_id"] = row_index
        out_row["source_tile_id"] = int(row["tile_id"])
        output_rows.append(out_row)
    pq.write_table(pa.Table.from_pylist(output_rows), str(pred_store_path / "index.parquet"))

    manifest = {
        "model_version": "v16.1",
        "build": args.build,
        "run_name": run_name,
        "prediction_store_path": str(pred_store_path),
        "source_store_path": str(source_store_path),
        "started_at": _now_iso_utc(),
        "seed": args.seed,
        "device": str(device),
        "checkpoints": checkpoints,
        "checkpoint_hashes": {name: _sha256_file(Path(path)) for name, path in checkpoints.items()},
        "tile_count": total,
        "tile_filter": {"tile_id": args.tile_id, "limit": args.limit},
        "arrays": sorted(arrays.keys()),
    }
    (run_root / "_v16_1_inference_run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Done. Run root: {run_root}")


if __name__ == "__main__":
    main()
