"""Selection and diagnostics for paired authored/flat-synthetic v50 tiles.

The v50 mixed curriculum keeps two minimap representations for many of the same
terrain rows: an authored minimap and a synthesized minimap. This module treats
those rows as a validation pair only. The synthetic side is a legacy flat fake
maptexture, not a terrain-shadow target. Its absolute difference from authored
RGB is useful for shadow/calibration diagnostics. It never rewrites the source
Zarr store.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import zarr

PAIR_SCHEMA = "v60-real-synthetic-pair-v1"


@dataclass(frozen=True)
class RealSyntheticPair:
    authored_row_index: int
    synthetic_row_index: int
    map_name: str
    tile_x: int
    tile_y: int
    source_group_id: str
    split: str


def load_pair_rows(
    store: str | Path,
    *,
    split_policy: str,
    val_map: str,
    validation_limit: int = 0,
) -> tuple[list[RealSyntheticPair], dict[str, int]]:
    """Return complete authored/synthetic pairs with deterministic split assignment."""
    if split_policy not in {"manifest", "map_holdout"}:
        raise ValueError(f"unknown split policy {split_policy!r}")
    if validation_limit < 0:
        raise ValueError("validation_limit must be non-negative")

    rows = pq.read_table(Path(store) / "index.parquet").to_pylist()
    groups: dict[str, dict[str, list[tuple[int, dict[str, Any]]]]] = defaultdict(
        lambda: {"authored": [], "synthetic": []}
    )
    for row_index, row in enumerate(rows):
        source = str(row.get("minimap_source", ""))
        group_id = str(row.get("source_group_id", ""))
        if source not in {"authored", "synthetic"} or not group_id:
            continue
        groups[group_id][source].append((row_index, row))

    incomplete_groups = 0
    duplicate_groups = 0
    pairs: list[RealSyntheticPair] = []
    for group_id in sorted(groups):
        group = groups[group_id]
        if len(group["authored"]) != 1 or len(group["synthetic"]) != 1:
            incomplete_groups += 1
            continue
        authored_index, authored = group["authored"][0]
        synthetic_index, synthetic = group["synthetic"][0]
        if (
            str(authored.get("map", "")) != str(synthetic.get("map", ""))
            or int(authored.get("tile_x", -1)) != int(synthetic.get("tile_x", -1))
            or int(authored.get("tile_y", -1)) != int(synthetic.get("tile_y", -1))
        ):
            raise ValueError(f"source group {group_id!r} has mismatched tile identity")

        if split_policy == "map_holdout":
            split = "val" if str(authored.get("map", "")) == val_map else "train"
        else:
            authored_split = str(authored.get("split", ""))
            synthetic_split = str(synthetic.get("split", ""))
            if authored_split not in {"train", "val"} or synthetic_split not in {"train", "val"}:
                continue
            if authored_split != synthetic_split:
                duplicate_groups += 1
                raise ValueError(
                    f"source group {group_id!r} crosses train/validation split: "
                    f"{authored_split!r} vs {synthetic_split!r}"
                )
            split = authored_split

        pairs.append(
            RealSyntheticPair(
                authored_row_index=authored_index,
                synthetic_row_index=synthetic_index,
                map_name=str(authored.get("map", "")),
                tile_x=int(authored.get("tile_x", -1)),
                tile_y=int(authored.get("tile_y", -1)),
                source_group_id=group_id,
                split=split,
            )
        )

    validation_pairs = [pair for pair in pairs if pair.split == "val"]
    if validation_limit:
        validation_pairs = validation_pairs[:validation_limit]
    selected_groups = {pair.source_group_id for pair in pairs if pair.split == "train"}
    selected_groups.update(pair.source_group_id for pair in validation_pairs)
    selected = [
        pair for pair in pairs if pair.split == "train" or pair.source_group_id in selected_groups
    ]
    if not any(pair.split == "train" for pair in selected) or not validation_pairs:
        raise ValueError("paired selection must contain train pairs and validation pairs")
    return selected, {
        "candidate_groups": len(groups),
        "incomplete_groups": incomplete_groups,
        "duplicate_groups": duplicate_groups,
        "complete_groups": len(pairs),
        "validation_pairs_before_limit": len([pair for pair in pairs if pair.split == "val"]),
        "validation_pairs_after_limit": len(validation_pairs),
    }


def pair_validation_rows(
    pairs: list[RealSyntheticPair],
) -> list[RealSyntheticPair]:
    """Return the validation subset, preserving deterministic manifest order."""
    return [pair for pair in pairs if pair.split == "val"]


def pair_domain_report(
    group: zarr.Group,
    pairs: list[RealSyntheticPair],
    shadow_npz_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Compare authored RGB to flat synthetic RGB and optional fixed shadow output."""
    if "minimap_rgb" not in group:
        raise ValueError("paired store is missing minimap_rgb")
    metrics: list[dict[str, Any]] = []
    for pair in pair_validation_rows(pairs):
        authored = np.asarray(group["minimap_rgb"][pair.authored_row_index], dtype=np.float32) / 255.0
        synthetic = np.asarray(group["minimap_rgb"][pair.synthetic_row_index], dtype=np.float32) / 255.0
        if authored.shape != (256, 256, 3) or synthetic.shape != (256, 256, 3):
            raise ValueError(f"paired RGB shape mismatch for {pair.source_group_id}")
        difference = np.abs(authored - synthetic)
        item: dict[str, Any] = {
                "source_group_id": pair.source_group_id,
                "map": pair.map_name,
                "tile_x": pair.tile_x,
                "tile_y": pair.tile_y,
                "authored_mean": float(authored.mean()),
                "synthetic_mean": float(synthetic.mean()),
                "authored_std": float(authored.std()),
                "synthetic_std": float(synthetic.std()),
                "mae": float(difference.mean()),
                "rmse": float(np.sqrt(np.mean(np.square(authored - synthetic)))),
                "fraction_difference_gt_0_10": float((difference > 0.10).mean()),
                "fraction_difference_gt_0_25": float((difference > 0.25).mean()),
            }
        fixed_shadow = _load_fixed_shadow(shadow_npz_dir, pair) if shadow_npz_dir is not None else None
        if fixed_shadow is not None:
            diff_luma = difference.mean(axis=2)
            item["fixed_shadow_mean"] = float(fixed_shadow.mean())
            item["fixed_shadow_std"] = float(fixed_shadow.std())
            item["fixed_shadow_vs_abs_diff_luma_correlation"] = _pearson(fixed_shadow, diff_luma)
            item["fixed_shadow_vs_inverse_abs_diff_luma_correlation"] = _pearson(
                fixed_shadow, 1.0 - diff_luma
            )
        metrics.append(item)
    if not metrics:
        raise ValueError("paired validation selection is empty")
    shadow_corr = [
        item["fixed_shadow_vs_abs_diff_luma_correlation"]
        for item in metrics
        if item.get("fixed_shadow_vs_abs_diff_luma_correlation") is not None
    ]
    inverse_shadow_corr = [
        item["fixed_shadow_vs_inverse_abs_diff_luma_correlation"]
        for item in metrics
        if item.get("fixed_shadow_vs_inverse_abs_diff_luma_correlation") is not None
    ]
    return {
        "schema": PAIR_SCHEMA,
        "diagnostic_role": "flat_synthetic_absdiff_and_optional_fixed_shadow_calibration",
        "pair_count": len(metrics),
        "mean_mae": float(np.mean([item["mae"] for item in metrics])),
        "mean_rmse": float(np.mean([item["rmse"] for item in metrics])),
        "mean_fraction_difference_gt_0_10": float(
            np.mean([item["fraction_difference_gt_0_10"] for item in metrics])
        ),
        "mean_fraction_difference_gt_0_25": float(
            np.mean([item["fraction_difference_gt_0_25"] for item in metrics])
        ),
        "fixed_shadow_pair_count": len(shadow_corr),
        "mean_fixed_shadow_vs_abs_diff_luma_correlation": (
            float(np.mean(shadow_corr)) if shadow_corr else None
        ),
        "mean_fixed_shadow_vs_inverse_abs_diff_luma_correlation": (
            float(np.mean(inverse_shadow_corr)) if inverse_shadow_corr else None
        ),
        "rows": metrics,
    }


def _fixed_shadow_path(shadow_npz_dir: str | Path, pair: RealSyntheticPair) -> Path:
    return Path(shadow_npz_dir) / f"{pair.map_name}_{pair.tile_x}_{pair.tile_y}_harvest.npz"


def _load_fixed_shadow(
    shadow_npz_dir: str | Path,
    pair: RealSyntheticPair,
) -> np.ndarray:
    path = _fixed_shadow_path(shadow_npz_dir, pair)
    if not path.is_file():
        raise FileNotFoundError(f"fixed shadow NPZ missing for {pair.source_group_id}: {path}")
    with np.load(path, allow_pickle=False) as payload:
        required = ("minimap_rgb_256", "terrain_shadow_256")
        missing = [name for name in required if name not in payload]
        if missing:
            raise ValueError(f"{path} is missing post-fix signals: {missing}")
        shadow = np.asarray(payload["terrain_shadow_256"], dtype=np.float32)
    if shadow.shape != (256, 256) or not np.isfinite(shadow).all():
        raise ValueError(f"{path}: terrain_shadow_256 must be finite 256x256")
    if shadow.min() < -1e-5 or shadow.max() > 1.00001:
        raise ValueError(f"{path}: terrain_shadow_256 is outside [0,1]")
    return np.clip(shadow, 0.0, 1.0)


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    left_flat = np.asarray(left, dtype=np.float64).reshape(-1)
    right_flat = np.asarray(right, dtype=np.float64).reshape(-1)
    left_centered = left_flat - left_flat.mean()
    right_centered = right_flat - right_flat.mean()
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denominator <= 1e-12:
        return None
    return float(np.dot(left_centered, right_centered) / denominator)


__all__ = [
    "PAIR_SCHEMA",
    "RealSyntheticPair",
    "load_pair_rows",
    "pair_domain_report",
    "pair_validation_rows",
]
