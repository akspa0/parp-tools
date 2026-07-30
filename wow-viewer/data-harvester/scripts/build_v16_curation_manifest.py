"""Build a V16-store curation manifest (bucket scores, kept/dropped tiles, review panels).

Spec 122 note (2026-07-30): this script's own real-caller search found it is still imported by
``harvester.test_v18_focus_masks``, ``scripts/run_v18_baseline_contract.py``, and
``scripts/build_v18_curation_manifest.py`` -- it is NOT purely historical and is not converted to a
shim. It remains the correct tool for V16/V18-shaped stores. For the v50 lane, the canonical
curation entrypoint is ``WowViewer.Tool.Harvest curate`` (see
``wow-viewer/src/core/WowViewer.Core.Curation`` and ``harvester.curation_store``).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import sys
from pathlib import Path
from typing import Any

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import numpy as np
from PIL import Image as _PILImage
from PIL import ImageDraw as _PILImageDraw
import pyarrow.parquet as pq
import zarr
import zarr.storage

from harvester.v16_curation import (
    DIFFICULTY_BUCKETS,
    alpha_painted,
    crop_257_to_256,
    dilate,
    edge_strength,
    f1,
    height_gradient_strength,
    iou,
    is_blank_what_plate,
    mcly_painted_coverage,
    load_curation_keys,
    minimap_grayscale,
    normal_edge_strength,
    normal_relief,
    write_rows_parquet,
)
from harvester.v16_1_dataset import compose_terrain_valid_mask_257

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DATASET_ROOT = _PROJECT_ROOT / "output" / "datasets" / "v16"
_OUTPUT_ROOT = _DATASET_ROOT / "curation"
_PANEL_SIZE = 256
_PROFILE_ALIASES = {
    "v18_focus_terrain_v1": "normal_terrain_v16_1_1",
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a reusable terrain dataset curation manifest.")
    p.add_argument("--dataset-dir", type=Path, default=_DATASET_ROOT)
    p.add_argument("--build", type=str)
    p.add_argument("--builds", nargs="+")
    p.add_argument(
        "--profile",
        choices=["basic_v1", "normal_terrain_v1", "normal_terrain_v16_1_1", "v18_focus_terrain_v1"],
        default="normal_terrain_v16_1_1",
    )
    p.add_argument("--sample-seed", type=int, default=42)
    p.add_argument("--max-tiles-per-build", type=int, default=0)
    p.add_argument("--min-minimap-gray-std", type=float, default=4.0)
    p.add_argument("--min-height-std", type=float, default=3.0)
    p.add_argument("--min-normal-coverage", type=float, default=0.25)
    p.add_argument("--normal-edge-threshold", type=float, default=0.08)
    p.add_argument("--minimap-edge-threshold", type=float, default=0.08)
    p.add_argument("--min-normal-edge-f1", type=float, default=0.10)
    p.add_argument("--min-edge-frac", type=float, default=0.01)
    p.add_argument(
        "--max-wmo-wipeout-trainable-cov",
        type=float,
        default=0.30,
        help="Reject a tile when WMO-driven loss gating leaves at most this much trainable terrain.",
    )
    p.add_argument(
        "--min-wmo-wipeout-modf-cov",
        type=float,
        default=0.25,
        help="Require at least this much WMO footprint coverage before the WMO wipeout filter can reject a tile.",
    )
    p.add_argument(
        "--min-wmo-wipeout-loss-gate-cov",
        type=float,
        default=0.35,
        help="Require at least this much loss-gate coverage before the WMO wipeout filter can reject a tile.",
    )
    p.add_argument(
        "--min-wmo-wipeout-share",
        type=float,
        default=0.75,
        help="Require WMOs to dominate the loss gate by at least this fraction before the wipeout filter can reject a tile.",
    )
    p.add_argument(
        "--min-trainable-cov",
        type=float,
        default=0.20,
        help="Reject a tile when the surviving terrain-valid training area falls below this coverage.",
    )
    p.add_argument("--dilate-radius", type=int, default=2)
    p.add_argument("--worst-k", type=int, default=12)
    p.add_argument(
        "--workers",
        type=int,
        default=-1,
        help="Worker count for tile auditing. Use -1 to auto-resolve a CPU-friendly default.",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=128,
        help="Tile rows per worker task.",
    )
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


def _open_root(zarr_path: Path):
    store = zarr.storage.LocalStore(str(zarr_path), read_only=True)
    root = zarr.open_group(store=store, mode="r")
    return store, root


def _sample_row_ids(table, max_tiles: int, seed: int) -> list[int]:
    ids = list(range(table.num_rows))
    if max_tiles <= 0 or len(ids) <= max_tiles:
        return ids
    rng = np.random.RandomState(seed)
    chosen = rng.choice(np.asarray(ids, dtype=np.int64), size=max_tiles, replace=False)
    return [int(v) for v in chosen.tolist()]


def _row_get(table, idx: int, key: str, default: Any = None) -> Any:
    if key not in table.column_names:
        return default
    return table.column(key)[idx].as_py()


def _resolve_workers(requested: int) -> int:
    if requested >= 0:
        return int(requested)
    cpu_count = os.cpu_count() or 4
    return max(2, min(16, cpu_count))


def _build_row_meta(table, row_idx: int) -> dict[str, Any]:
    return {
        "tile_id": int(_row_get(table, row_idx, "tile_id", row_idx)),
        "map": str(_row_get(table, row_idx, "map", "")),
        "tile_x": int(_row_get(table, row_idx, "tile_x", -1) or -1),
        "tile_y": int(_row_get(table, row_idx, "tile_y", -1) or -1),
        "has_normals": bool(_row_get(table, row_idx, "has_normal_xyz", False)),
        "has_alpha": bool(_row_get(table, row_idx, "has_alpha_256", False)),
        "has_liquid": bool(_row_get(table, row_idx, "has_liquid_mask", False)),
        "has_mcly": bool(_row_get(table, row_idx, "has_mcly_texture_ids", False)),
        "has_object_roof": bool(_row_get(table, row_idx, "has_object_roof_mask", False)),
        "height_std": float(_row_get(table, row_idx, "height_std", 0.0) or 0.0),
        "n_mddf": int(_row_get(table, row_idx, "n_mddf", 0) or 0),
        "n_modf": int(_row_get(table, row_idx, "n_modf", 0) or 0),
    }


def _chunk_rows(rows: list[dict[str, Any]], chunk_size: int) -> list[list[dict[str, Any]]]:
    size = max(1, int(chunk_size))
    return [rows[i:i + size] for i in range(0, len(rows), size)]


def _to_rgb_panel(arr: np.ndarray, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    x = arr.astype(np.float32)
    rng = hi - lo
    y = np.zeros_like(x, dtype=np.float32) if abs(rng) < 1e-8 else np.clip((x - lo) / rng, 0.0, 1.0)
    u8 = (y * 255.0).astype(np.uint8)
    if u8.ndim == 2:
        return np.repeat(u8[:, :, None], 3, axis=2)
    return u8


def _draw_label(img: np.ndarray, text: str) -> np.ndarray:
    pil = _PILImage.fromarray(img, "RGB")
    drw = _PILImageDraw.Draw(pil)
    drw.rectangle([(0, 0), (pil.width, 18)], fill=(10, 10, 10))
    drw.text((4, 3), text, fill=(245, 245, 245))
    return np.asarray(pil)


def _resize(img: np.ndarray, size: int = _PANEL_SIZE) -> np.ndarray:
    return np.asarray(_PILImage.fromarray(img).resize((size, size), _PILImage.Resampling.NEAREST))


def _overlay_edges(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    h, w = a.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[b, 1] = 255
    rgb[a, 0] = 255
    both = np.logical_and(a, b)
    rgb[both] = np.array([255, 255, 0], dtype=np.uint8)
    return rgb


def _crop_257_to_256(x: np.ndarray) -> np.ndarray:
    return x[:256, :256]


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, float(value))))


def _score_row_v16_1_1(row: dict[str, Any]) -> dict[str, float | str]:
    deformation_richness = _clamp01(
        (0.45 * min(row["terrain_detail_mean"] / 0.22, 1.5))
        + (0.35 * min(row["normal_relief_mean"] / 0.20, 1.5))
        + (0.20 * min(row["normal_edge_frac"] / 0.12, 1.5))
    )
    normal_coverage = _clamp01((row["normal_cov"] - 0.20) / 0.60)
    terrain_validity = _clamp01(
        (0.80 * min(row["terrain_valid_cov"] / 0.75, 1.25))
        + (0.20 * (1.0 - min((row["object_cov"] + row["roof_cov"] + (0.85 * row["liquid_cov"])) / 0.75, 1.0)))
    )
    painted_signal = _clamp01(max(row["alpha_cov"] / 0.60, row["mcly_cov"] / 0.60))
    minimap_target_usefulness = _clamp01(
        (0.55 * min(row["normal_edge_f1"] / 0.75, 1.25))
        + (0.25 * min(row["minimap_gray_std"] / 18.0, 1.25))
        + (0.20 * min(min(row["normal_edge_frac"], row["minimap_edge_frac"]) / 0.10, 1.25))
    )

    usefulness_score = _clamp01(
        (0.30 * deformation_richness)
        + (0.15 * normal_coverage)
        + (0.20 * terrain_validity)
        + (0.15 * painted_signal)
        + (0.20 * minimap_target_usefulness)
    )
    difficulty_score = _clamp01(
        (0.55 * deformation_richness)
        + (0.20 * painted_signal)
        + (0.15 * normal_coverage)
        + (0.10 * minimap_target_usefulness)
    )
    pathology_pressure = _clamp01(
        max(0.0, 0.40 - terrain_validity) * 1.6
        + max(0.0, 0.32 - minimap_target_usefulness) * 1.2
        + max(0.0, row["object_cov"] + row["roof_cov"] + row["liquid_cov"] - 0.55) * 1.5
    )

    if pathology_pressure >= 0.22 and difficulty_score >= 0.35:
        difficulty_bucket = "pathological"
    elif difficulty_score >= 0.62 and usefulness_score >= 0.42:
        difficulty_bucket = "hard"
    elif difficulty_score >= 0.34 or usefulness_score >= 0.38:
        difficulty_bucket = "medium"
    else:
        difficulty_bucket = "easy"

    return {
        "quality_score": usefulness_score,
        "usefulness_score": usefulness_score,
        "difficulty_score": difficulty_score,
        "difficulty_bucket": difficulty_bucket,
        "difficulty_rank": int(DIFFICULTY_BUCKETS.index(difficulty_bucket)),
        "score_deformation_richness": deformation_richness,
        "score_normal_coverage": normal_coverage,
        "score_terrain_validity": terrain_validity,
        "score_painted_signal": painted_signal,
        "score_minimap_target_usefulness": minimap_target_usefulness,
        "score_pathology_pressure": pathology_pressure,
    }


def _compute_row(
    *,
    build: str,
    row_meta: dict[str, Any],
    root,
    args_dict: dict[str, Any],
) -> dict[str, Any]:
    tile_id = int(row_meta["tile_id"])
    minimap = root["minimap_rgb"][tile_id].astype(np.uint8)
    height_257 = root["height_257"][tile_id].astype(np.float32)
    gray = minimap_grayscale(minimap)
    minimap_edge = edge_strength(gray) >= float(args_dict["minimap_edge_threshold"])
    minimap_edge_d = dilate(minimap_edge, int(args_dict["dilate_radius"]))

    normals = root["normal_xyz"][tile_id].astype(np.float32) if bool(row_meta["has_normals"]) else np.zeros((257, 257, 3), dtype=np.float32)
    normal_mask = root["normal_mask"][tile_id].astype(np.float32) if "normal_mask" in root and bool(row_meta["has_normals"]) else np.zeros((257, 257), dtype=np.float32)
    relief = normal_relief(normals, normal_mask)
    relief = _crop_257_to_256(relief)
    normal_mask_256 = _crop_257_to_256(normal_mask)
    normal_edge = _crop_257_to_256(normal_edge_strength(normals, normal_mask)) >= float(args_dict["normal_edge_threshold"])
    normal_edge_d = dilate(normal_edge, int(args_dict["dilate_radius"]))

    alpha_cov = 0.0
    if bool(row_meta["has_alpha"]) and "alpha_256" in root:
        alpha = root["alpha_256"][tile_id].astype(np.float32)
        alpha_cov = float((alpha_painted(alpha) >= 0.05).mean())

    mcly_cov = 0.0
    if bool(row_meta["has_mcly"]) and "mcly_layer_mask" in root:
        mcly_cov = mcly_painted_coverage(root["mcly_layer_mask"][tile_id].astype(np.float32))

    liquid_cov = 0.0
    liquid_mask_256 = np.zeros((256, 256), dtype=np.float32)
    if bool(row_meta["has_liquid"]) and "liquid_mask" in root:
        liquid_mask_256 = root["liquid_mask"][tile_id].astype(np.float32)
        liquid_cov = float(liquid_mask_256.mean())

    object_roof_mask_256 = np.zeros((256, 256), dtype=np.float32)
    roof_cov = 0.0
    if bool(row_meta["has_object_roof"]) and "object_roof_mask" in root:
        object_roof_mask_256 = root["object_roof_mask"][tile_id].astype(np.float32)
        object_roof_mask_256 = np.clip(object_roof_mask_256, 0.0, 1.0)
        roof_cov = float(object_roof_mask_256.mean())
    object_roof_weight_257 = np.pad(1.0 - object_roof_mask_256, ((0, 1), (0, 1)), mode="edge")

    mddf_mask = root["mddf_mask"][tile_id].astype(np.float32) if "mddf_mask" in root else np.zeros((257, 257), dtype=np.float32)
    modf_mask = root["modf_mask"][tile_id].astype(np.float32) if "modf_mask" in root else np.zeros((257, 257), dtype=np.float32)
    object_presence_257 = np.maximum(mddf_mask, modf_mask).astype(np.float32, copy=False)
    if "object_precise_mask" in root:
        loss_gate_mask_257 = root["object_precise_mask"][tile_id].astype(np.float32)
    elif "object_filtered_mask" in root:
        loss_gate_mask_257 = root["object_filtered_mask"][tile_id].astype(np.float32)
    elif "object_mask" in root:
        loss_gate_mask_257 = root["object_mask"][tile_id].astype(np.float32)
    else:
        loss_gate_mask_257 = np.zeros((257, 257), dtype=np.float32)
    object_cov = float(object_presence_257.mean())
    if object_cov <= 0.0:
        if "object_filtered_mask" in root:
            object_cov = float(root["object_filtered_mask"][tile_id].astype(np.float32).mean())
        elif "object_mask" in root:
            object_cov = float(root["object_mask"][tile_id].astype(np.float32).mean())
    mddf_cov = float(np.clip(mddf_mask, 0.0, 1.0).mean())
    modf_cov = float(np.clip(modf_mask, 0.0, 1.0).mean())
    loss_gate_cov = float(np.clip(loss_gate_mask_257, 0.0, 1.0).mean())

    height_grad = crop_257_to_256(height_gradient_strength(height_257))
    terrain_detail_mean = float((0.65 * height_grad + 0.35 * relief).mean())
    what_plate = is_blank_what_plate(
        height_257=height_257,
        alpha_cov=alpha_cov,
        mcly_cov=mcly_cov,
        liquid_cov=liquid_cov,
        object_cov=object_cov,
    )
    terrain_valid_257 = compose_terrain_valid_mask_257(
        normal_mask_257=normal_mask,
        object_presence_257=object_presence_257,
        liquid_mask_256=liquid_mask_256,
        object_roof_weight_257=object_roof_weight_257,
        what_plate=what_plate,
    )
    trainable_257 = compose_terrain_valid_mask_257(
        normal_mask_257=normal_mask * (1.0 - np.clip(loss_gate_mask_257, 0.0, 1.0)),
        object_presence_257=np.zeros_like(object_presence_257, dtype=np.float32),
        liquid_mask_256=liquid_mask_256,
        object_roof_weight_257=object_roof_weight_257,
        what_plate=what_plate,
    )
    terrain_valid_cov = float(_crop_257_to_256(terrain_valid_257).mean())
    trainable_cov = float(_crop_257_to_256(trainable_257).mean())
    painted_signal_cov = float(max(alpha_cov, mcly_cov))
    wmo_loss_share = float(modf_cov / max(loss_gate_cov, 1e-6)) if loss_gate_cov > 0.0 else 0.0

    metrics = {
        "build": build,
        "tile_id": tile_id,
        "map": row_meta["map"],
        "tile_x": int(row_meta["tile_x"]),
        "tile_y": int(row_meta["tile_y"]),
        "has_normals": bool(row_meta["has_normals"]),
        "minimap_std": float(minimap.astype(np.float32).std()),
        "minimap_gray_std": float(gray.std()),
        "height_std": float(row_meta["height_std"]),
        "normal_cov": float(normal_mask_256.mean()),
        "normal_relief_mean": float(relief.mean()),
        "normal_edge_frac": float(normal_edge.mean()),
        "minimap_edge_frac": float(minimap_edge.mean()),
        "normal_edge_f1": float(f1(normal_edge_d, minimap_edge_d)),
        "normal_edge_iou": float(iou(normal_edge_d, minimap_edge_d)),
        "alpha_cov": alpha_cov,
        "mcly_cov": mcly_cov,
        "liquid_cov": liquid_cov,
        "object_cov": object_cov,
        "roof_cov": roof_cov,
        "mddf_cov": mddf_cov,
        "modf_cov": modf_cov,
        "loss_gate_cov": loss_gate_cov,
        "wmo_loss_share": wmo_loss_share,
        "terrain_valid_cov": terrain_valid_cov,
        "trainable_cov": trainable_cov,
        "painted_signal_cov": painted_signal_cov,
        "terrain_detail_mean": terrain_detail_mean,
        "what_plate": bool(what_plate),
        "n_mddf": int(row_meta["n_mddf"]),
        "n_modf": int(row_meta["n_modf"]),
    }
    return metrics


def _evaluate_profile(row: dict[str, Any], args: argparse.Namespace) -> tuple[bool, dict[str, Any]]:
    if bool(row.get("what_plate", False)):
        return False, {"quality_score": 0.0, "reject_reason": "blank_what_plate_tile"}

    if row["minimap_gray_std"] < float(args.min_minimap_gray_std):
        if row["height_std"] < float(args.min_height_std) and row["alpha_cov"] < 0.01 and row["liquid_cov"] < 0.01 and row["normal_cov"] < float(args.min_normal_coverage):
            return False, {"quality_score": 0.0, "reject_reason": "blank_low_signal_tile"}

    if args.profile == "basic_v1":
        score = 0.1
        score += min(row["minimap_gray_std"] / 18.0, 2.0)
        score += min(row["height_std"] / 18.0, 2.0)
        score += min(row["normal_cov"], 1.0)
        score += min(row["normal_edge_f1"], 1.0)
        score += min(row["terrain_detail_mean"] * 10.0, 1.5)
        return True, {
            "quality_score": float(score),
            "usefulness_score": float(score),
            "difficulty_score": float(score),
            "difficulty_bucket": "medium",
            "difficulty_rank": int(DIFFICULTY_BUCKETS.index("medium")),
            "score_deformation_richness": 0.0,
            "score_normal_coverage": 0.0,
            "score_terrain_validity": 0.0,
            "score_painted_signal": 0.0,
            "score_minimap_target_usefulness": 0.0,
            "score_pathology_pressure": 0.0,
            "reject_reason": None,
        }

    if not row["has_normals"] or row["normal_cov"] < float(args.min_normal_coverage):
        return False, {"quality_score": 0.0, "reject_reason": "insufficient_normal_coverage"}

    if row["normal_relief_mean"] < 0.015 and row["minimap_gray_std"] < float(args.min_minimap_gray_std):
        return False, {"quality_score": 0.0, "reject_reason": "blank_minimap_blank_normals"}

    if row["normal_edge_frac"] >= float(args.min_edge_frac) and row["minimap_edge_frac"] >= float(args.min_edge_frac):
        if row["normal_edge_f1"] < float(args.min_normal_edge_f1):
            return False, {"quality_score": 0.0, "reject_reason": "normal_minimap_edge_mismatch"}

    if (
        row["modf_cov"] >= float(args.min_wmo_wipeout_modf_cov)
        and row["loss_gate_cov"] >= float(args.min_wmo_wipeout_loss_gate_cov)
        and row["wmo_loss_share"] >= float(args.min_wmo_wipeout_share)
        and row["trainable_cov"] <= float(args.max_wmo_wipeout_trainable_cov)
    ):
        return False, {"quality_score": 0.0, "reject_reason": "wmo_loss_wipeout_tile"}

    if row["trainable_cov"] < float(args.min_trainable_cov):
        return False, {"quality_score": 0.0, "reject_reason": "insufficient_trainable_terrain"}

    if args.profile in {"normal_terrain_v16_1_1", "v18_focus_terrain_v1"}:
        payload = _score_row_v16_1_1(row)
        payload["reject_reason"] = None
        return True, payload

    score = 0.1
    score += min(row["minimap_gray_std"] / 18.0, 2.0)
    score += min(row["height_std"] / 18.0, 2.0)
    score += min(row["normal_cov"] * 1.5, 1.5)
    score += min(row["normal_relief_mean"] * 6.0, 1.5)
    score += min(row["normal_edge_f1"] * 2.0, 2.0)
    score += min(row["terrain_detail_mean"] * 10.0, 1.5)
    return True, {
        "quality_score": float(score),
        "usefulness_score": float(score),
        "difficulty_score": float(score),
        "difficulty_bucket": "medium",
        "difficulty_rank": int(DIFFICULTY_BUCKETS.index("medium")),
        "score_deformation_richness": 0.0,
        "score_normal_coverage": 0.0,
        "score_terrain_validity": 0.0,
        "score_painted_signal": 0.0,
        "score_minimap_target_usefulness": 0.0,
        "score_pathology_pressure": 0.0,
        "reject_reason": None,
    }


def _process_chunk(
    *,
    build: str,
    zarr_path_str: str,
    row_chunk: list[dict[str, Any]],
    args_dict: dict[str, Any],
) -> list[dict[str, Any]]:
    store, root = _open_root(Path(zarr_path_str))
    try:
        out: list[dict[str, Any]] = []
        for row_meta in row_chunk:
            row = _compute_row(build=build, row_meta=row_meta, root=root, args_dict=args_dict)
            keep, evaluation = _evaluate_profile(row, argparse.Namespace(**args_dict))
            row["profile"] = str(args_dict.get("requested_profile", args_dict["profile"]))
            row["canonical_profile"] = str(args_dict["profile"])
            row["keep"] = bool(keep)
            row.update(evaluation)
            row["quality_score"] = float(row.get("quality_score", 0.0) or 0.0)
            row["usefulness_score"] = float(row.get("usefulness_score", row["quality_score"]) or 0.0)
            row["difficulty_score"] = float(row.get("difficulty_score", row["quality_score"]) or 0.0)
            if row.get("difficulty_bucket") is None:
                row["difficulty_bucket"] = "pathological" if keep else None
            if row.get("difficulty_bucket") is not None:
                row["difficulty_rank"] = int(DIFFICULTY_BUCKETS.index(str(row["difficulty_bucket"])))
            out.append(row)
        return out
    finally:
        store.close()


def main() -> None:
    args = _parse_args()
    canonical_profile = _PROFILE_ALIASES.get(str(args.profile), str(args.profile))
    builds = args.builds or ([args.build] if args.build else [])
    if not builds:
        raise SystemExit("Provide --build or --builds")

    run_name = args.run_name or f"{args.profile}"
    output_dir = args.output_dir or (_OUTPUT_ROOT / run_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_workers = _resolve_workers(int(args.workers))
    args_dict = vars(args).copy()
    args_dict["profile"] = canonical_profile
    args_dict["requested_profile"] = str(args.profile)
    print(
        f"Curation: profile={args.profile} canonical_profile={canonical_profile} "
        f"workers={resolved_workers} chunk_size={max(1, int(args.chunk_size))}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    for build_idx, build in enumerate(builds):
        zarr_path = args.dataset_dir / f"{build}.zarr"
        if not zarr_path.exists():
            print(f"SKIP {build}: missing {zarr_path}", flush=True)
            continue
        table = pq.read_table(str(zarr_path / "index.parquet"))
        sampled_row_ids = _sample_row_ids(table, int(args.max_tiles_per_build), int(args.sample_seed) + build_idx)
        row_metas = [_build_row_meta(table, row_idx) for row_idx in sampled_row_ids]
        row_chunks = _chunk_rows(row_metas, int(args.chunk_size))
        print(f"Build {build}: tiles={len(row_metas)} chunks={len(row_chunks)}", flush=True)
        if resolved_workers <= 1 or len(row_chunks) <= 1:
            build_rows: list[dict[str, Any]] = []
            for chunk_idx, row_chunk in enumerate(row_chunks, start=1):
                build_rows.extend(
                    _process_chunk(
                        build=build,
                        zarr_path_str=str(zarr_path),
                        row_chunk=row_chunk,
                        args_dict=args_dict,
                    )
                )
                print(f"  {build}: chunk {chunk_idx}/{len(row_chunks)}", flush=True)
            rows.extend(build_rows)
            continue

        completed = 0
        build_rows = []
        with ProcessPoolExecutor(max_workers=resolved_workers) as executor:
            futures = [
                executor.submit(
                    _process_chunk,
                    build=build,
                    zarr_path_str=str(zarr_path),
                    row_chunk=row_chunk,
                    args_dict=args_dict,
                )
                for row_chunk in row_chunks
            ]
            for fut in as_completed(futures):
                build_rows.extend(fut.result())
                completed += 1
                print(f"  {build}: chunk {completed}/{len(row_chunks)}", flush=True)
        rows.extend(build_rows)

    if not rows:
        raise SystemExit("No rows were audited.")

    kept = [row for row in rows if row["keep"]]
    rejected = [row for row in rows if not row["keep"]]
    rows.sort(key=lambda row: (bool(row["keep"]), row["normal_edge_f1"], row["minimap_gray_std"]))
    write_rows_parquet(output_dir / "tiles.parquet", rows)
    write_rows_parquet(output_dir / "kept_tiles.parquet", kept)
    (output_dir / "tiles.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    summary = {
        "profile": args.profile,
        "canonical_profile": canonical_profile,
        "builds": builds,
        "dataset_dir": str(args.dataset_dir),
        "sample_seed": int(args.sample_seed),
        "max_tiles_per_build": int(args.max_tiles_per_build),
        "tile_count": len(rows),
        "kept_tiles": len(kept),
        "rejected_tiles": len(rejected),
        "keep_ratio": float(len(kept) / max(1, len(rows))),
        "normal_edge_f1_mean_kept": float(np.mean([row["normal_edge_f1"] for row in kept])) if kept else 0.0,
        "normal_edge_f1_p10_kept": float(np.percentile([row["normal_edge_f1"] for row in kept], 10)) if kept else 0.0,
        "minimap_gray_std_mean_kept": float(np.mean([row["minimap_gray_std"] for row in kept])) if kept else 0.0,
        "quality_score_mean_kept": float(np.mean([row["quality_score"] for row in kept])) if kept else 0.0,
        "usefulness_score_mean_kept": float(np.mean([row["usefulness_score"] for row in kept])) if kept else 0.0,
        "trainable_cov_mean_kept": float(np.mean([row["trainable_cov"] for row in kept])) if kept else 0.0,
        "loss_gate_cov_mean_kept": float(np.mean([row["loss_gate_cov"] for row in kept])) if kept else 0.0,
        "modf_cov_mean_kept": float(np.mean([row["modf_cov"] for row in kept])) if kept else 0.0,
        "roof_cov_mean_kept": float(np.mean([row["roof_cov"] for row in kept])) if kept else 0.0,
        "reject_reason_counts": {},
        "difficulty_bucket_counts": {},
        "difficulty_bucket_examples": {},
        "scouting_pool_recipe": {
            "train_max_tiles": 400,
            "train_epoch_tiles": 128,
            "val_max_tiles": 48,
            "intent": "mixed-complexity scouting pool with bucket-aware epoch rotation",
        },
        "worst_rejected_examples": rejected[: int(args.worst_k)],
    }
    reason_counts: dict[str, int] = {}
    for row in rejected:
        reason = str(row.get("reject_reason") or "unknown")
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    summary["reject_reason_counts"] = dict(sorted(reason_counts.items()))
    bucket_counts: dict[str, int] = {bucket: 0 for bucket in DIFFICULTY_BUCKETS}
    bucket_examples: dict[str, list[dict[str, Any]]] = {bucket: [] for bucket in DIFFICULTY_BUCKETS}
    for row in sorted(kept, key=lambda entry: (-float(entry.get("quality_score", 0.0)), int(entry.get("tile_id", -1)))):
        bucket = str(row.get("difficulty_bucket") or "")
        if bucket not in bucket_counts:
            continue
        bucket_counts[bucket] += 1
        if len(bucket_examples[bucket]) < 3:
            bucket_examples[bucket].append(
                {
                    "build": row["build"],
                    "map": row["map"],
                    "tile_id": int(row["tile_id"]),
                    "quality_score": float(row["quality_score"]),
                    "usefulness_score": float(row["usefulness_score"]),
                    "difficulty_score": float(row["difficulty_score"]),
                }
            )
    summary["difficulty_bucket_counts"] = bucket_counts
    summary["difficulty_bucket_examples"] = bucket_examples
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    worst_rows = rejected[: int(args.worst_k)]
    if worst_rows:
        panels: list[np.ndarray] = []
        for row in worst_rows:
            zarr_path = args.dataset_dir / f"{row['build']}.zarr"
            store, root = _open_root(zarr_path)
            try:
                tile_id = int(row["tile_id"])
                minimap = root["minimap_rgb"][tile_id].astype(np.uint8)
                gray = minimap_grayscale(minimap)
                minimap_edge = dilate(edge_strength(gray) >= float(args.minimap_edge_threshold), int(args.dilate_radius))
                normals = root["normal_xyz"][tile_id].astype(np.float32) if "normal_xyz" in root else np.zeros((257, 257, 3), dtype=np.float32)
                normal_mask_full = root["normal_mask"][tile_id].astype(np.float32) if "normal_mask" in root else np.zeros((257, 257), dtype=np.float32)
                relief = _crop_257_to_256(normal_relief(normals, normal_mask_full))
                normal_mask = _crop_257_to_256(normal_mask_full)
                normal_edge_strength_256 = _crop_257_to_256(normal_edge_strength(normals, normal_mask_full))
                normal_edge = dilate(normal_edge_strength_256 >= float(args.normal_edge_threshold), int(args.dilate_radius))
                overlay = _overlay_edges(normal_edge, minimap_edge)
                strip = np.concatenate(
                    [
                        _draw_label(_resize(minimap), "minimap"),
                        _draw_label(_resize(_to_rgb_panel(relief, 0.0, 1.0)), "normal relief"),
                        _draw_label(_resize(_to_rgb_panel(normal_mask, 0.0, 1.0)), "normal mask"),
                        _draw_label(_resize(overlay), f"edge f1={row['normal_edge_f1']:.3f}"),
                    ],
                    axis=1,
                )
                title = f"{row['build']}:{row['map']}:{row['tile_id']} reject={row['reject_reason']}"
                panels.append(_draw_label(strip, title))
            finally:
                store.close()
        canvas = np.concatenate(panels, axis=0)
        _PILImage.fromarray(canvas).save(output_dir / "worst_cases.png")

    print(f"Wrote {output_dir / 'summary.json'}", flush=True)
    print(f"Wrote {output_dir / 'kept_tiles.parquet'}", flush=True)
    print(
        f"profile={args.profile} kept={len(kept)}/{len(rows)} "
        f"keep_ratio={summary['keep_ratio']:.3f} "
        f"buckets={summary['difficulty_bucket_counts']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
