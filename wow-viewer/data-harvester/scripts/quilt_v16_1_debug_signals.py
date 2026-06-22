"""Build stitched quilts for V16.1 normal debug signals.

This script reproduces the same per-tile `hard_region` and `transition`
signals shown in `train_v16_1_common.py` normal validation previews, then
stitches them into map-level quilt PNGs.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from harvester.v16_1_dataset import V161Dataset  # noqa: E402


def _masked_mean(loss_map: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (loss_map * mask).sum() / mask.sum().clamp_min(1e-8)


def _resize_weight(weight: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if tuple(weight.shape[-2:]) == tuple(size):
        return weight
    return F.interpolate(weight, size=size, mode="bilinear", align_corners=False)


def _gradient_magnitude_257(x: torch.Tensor) -> torch.Tensor:
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    return torch.sqrt((dx * dx) + (dy * dy) + 1e-8)


def _hard_region_weight_from_targets(
    height_raw: torch.Tensor,
    target_normals: torch.Tensor,
    alpha_painted_256: torch.Tensor,
    mcly_any_16: torch.Tensor,
    terrain_valid_mask: torch.Tensor,
    base_mask: torch.Tensor,
    detail_boost: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    height_grad = _gradient_magnitude_257(height_raw)
    normal_grad = _gradient_magnitude_257(target_normals)
    normal_grad = normal_grad.mean(dim=1, keepdim=True)
    alpha_painted_257 = _resize_weight(alpha_painted_256, target_normals.shape[-2:])
    alpha_grad = _gradient_magnitude_257(alpha_painted_257)
    mcly_any_257 = _resize_weight(mcly_any_16, target_normals.shape[-2:])
    mcly_grad = _gradient_magnitude_257(mcly_any_257)

    valid_mean_height = _masked_mean(height_grad, base_mask)
    valid_mean_normal = _masked_mean(normal_grad, base_mask)
    valid_mean_alpha = _masked_mean(alpha_grad, base_mask)
    valid_mean_mcly = _masked_mean(mcly_grad, base_mask)
    height_grad_n = (height_grad / valid_mean_height.clamp_min(1e-6)).clamp(0.0, 4.0)
    normal_grad_n = (normal_grad / valid_mean_normal.clamp_min(1e-6)).clamp(0.0, 4.0)
    alpha_grad_n = (alpha_grad / valid_mean_alpha.clamp_min(1e-6)).clamp(0.0, 4.0)
    mcly_grad_n = (mcly_grad / valid_mean_mcly.clamp_min(1e-6)).clamp(0.0, 4.0)

    transition_signal = torch.maximum(alpha_grad_n, mcly_grad_n)
    hard_region_signal = ((0.50 * height_grad_n) + (0.25 * normal_grad_n) + (0.25 * transition_signal)).clamp(0.0, 4.0)
    hard_region_signal = hard_region_signal * terrain_valid_mask
    hard_region_weight = 1.0 + (float(detail_boost) * hard_region_signal)
    return hard_region_weight, {
        "hard_region_signal": hard_region_signal,
        "transition_signal": transition_signal,
    }


def _ensure_bchw(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 4:
        return x
    if x.ndim == 3:
        return x.unsqueeze(0)
    if x.ndim == 2:
        return x.unsqueeze(0).unsqueeze(0)
    raise RuntimeError(f"Expected tensor rank 2/3/4, got shape={tuple(x.shape)}")


def _safe_key(value: str) -> str:
    text = str(value).strip()
    if not text:
        return "unknown"
    out = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def _resolve_map_name(entry: dict[str, object]) -> str:
    for key in ("map", "meta_map", "world", "map_name"):
        val = entry.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()
    return "unknown_map"


@dataclass(frozen=True)
class TileRef:
    dataset_index: int
    build: str
    map_name: str
    tile_x: int
    tile_y: int


@dataclass
class GroupAccumulator:
    build: str
    map_name: str
    min_x: int
    max_x: int
    min_y: int
    max_y: int
    tile_size: int

    def __post_init__(self) -> None:
        h = (self.max_y - self.min_y + 1) * self.tile_size
        w = (self.max_x - self.min_x + 1) * self.tile_size
        self.hard_region_raw = np.full((h, w), np.nan, dtype=np.float32)
        self.transition_raw = np.full((h, w), np.nan, dtype=np.float32)
        self.hard_region_local = np.full((h, w), np.nan, dtype=np.float32)
        self.transition_local = np.full((h, w), np.nan, dtype=np.float32)
        self.train_mask_local = np.full((h, w), np.nan, dtype=np.float32)
        self.tiles_written = 0

    def write_tile(
        self,
        tile_x: int,
        tile_y: int,
        hard_region_raw: np.ndarray,
        transition_raw: np.ndarray,
        train_mask_raw: np.ndarray,
    ) -> None:
        x0 = (tile_x - self.min_x) * self.tile_size
        y0 = (tile_y - self.min_y) * self.tile_size
        x1 = x0 + self.tile_size
        y1 = y0 + self.tile_size

        hard_local = hard_region_raw / max(float(np.max(hard_region_raw)), 1e-6)
        trans_local = transition_raw / max(float(np.max(transition_raw)), 1e-6)
        train_local = train_mask_raw / max(float(np.max(train_mask_raw)), 1e-6)

        self.hard_region_raw[y0:y1, x0:x1] = hard_region_raw
        self.transition_raw[y0:y1, x0:x1] = transition_raw
        self.hard_region_local[y0:y1, x0:x1] = hard_local
        self.transition_local[y0:y1, x0:x1] = trans_local
        self.train_mask_local[y0:y1, x0:x1] = train_local
        self.tiles_written += 1


def _to_u8_from_nan01(arr: np.ndarray) -> np.ndarray:
    valid = np.isfinite(arr)
    out = np.zeros(arr.shape, dtype=np.uint8)
    if valid.any():
        clipped = np.clip(arr[valid], 0.0, 1.0)
        out[valid] = (clipped * 255.0).astype(np.uint8)
    return out


def _save_global_png(arr: np.ndarray, out_path: Path) -> dict[str, float]:
    valid = np.isfinite(arr)
    if not valid.any():
        Image.fromarray(np.zeros(arr.shape, dtype=np.uint8), mode="L").save(out_path)
        return {"valid_pixels": 0, "max_raw": 0.0, "mean_raw": 0.0}

    max_raw = float(np.nanmax(arr))
    mean_raw = float(np.nanmean(arr))
    norm = arr / max(max_raw, 1e-6)
    Image.fromarray(_to_u8_from_nan01(norm), mode="L").save(out_path)
    return {"valid_pixels": int(valid.sum()), "max_raw": max_raw, "mean_raw": mean_raw}


def _save_local_png(arr: np.ndarray, out_path: Path) -> dict[str, float]:
    valid = np.isfinite(arr)
    Image.fromarray(_to_u8_from_nan01(arr), mode="L").save(out_path)
    return {"valid_pixels": int(valid.sum())}


def _resize_tile(signal_257: torch.Tensor, tile_size: int) -> np.ndarray:
    resized = F.interpolate(signal_257.unsqueeze(0).unsqueeze(0), size=(tile_size, tile_size), mode="bilinear", align_corners=False)
    return resized.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32, copy=False)


def _collect_tile_refs(
    dataset: V161Dataset,
    include_maps: set[str] | None,
    max_tiles: int | None,
    seed: int,
    skip_unknown_map: bool,
) -> list[TileRef]:
    refs: list[TileRef] = []
    for i, entry_idx in enumerate(dataset._indices):
        entry = dataset._index_entries[entry_idx]
        map_name = _resolve_map_name(entry)
        if skip_unknown_map and map_name == "unknown_map":
            continue
        if include_maps is not None and map_name not in include_maps:
            continue
        tile_x = int(entry.get("tile_x", -1) if entry.get("tile_x") is not None else -1)
        tile_y = int(entry.get("tile_y", -1) if entry.get("tile_y") is not None else -1)
        if tile_x < 0 or tile_y < 0:
            continue
        refs.append(
            TileRef(
                dataset_index=i,
                build=str(entry.get("build") or entry.get("_build") or ""),
                map_name=map_name,
                tile_x=tile_x,
                tile_y=tile_y,
            )
        )

    if max_tiles is not None and max_tiles > 0 and len(refs) > max_tiles:
        rng = np.random.RandomState(seed)
        chosen = np.sort(rng.choice(len(refs), size=max_tiles, replace=False))
        refs = [refs[int(idx)] for idx in chosen]

    return refs


def _group_accumulators(tile_refs: list[TileRef], tile_size: int) -> dict[tuple[str, str], GroupAccumulator]:
    groups: dict[tuple[str, str], list[TileRef]] = {}
    for ref in tile_refs:
        groups.setdefault((ref.build, ref.map_name), []).append(ref)

    accs: dict[tuple[str, str], GroupAccumulator] = {}
    for key, refs in groups.items():
        xs = [r.tile_x for r in refs]
        ys = [r.tile_y for r in refs]
        accs[key] = GroupAccumulator(
            build=key[0],
            map_name=key[1],
            min_x=min(xs),
            max_x=max(xs),
            min_y=min(ys),
            max_y=max(ys),
            tile_size=tile_size,
        )
    return accs


def _compute_debug_signals(sample: dict[str, torch.Tensor], detail_boost: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_n = F.normalize(_ensure_bchw(sample["normals"]).float(), dim=1, eps=1e-6)
    height_raw = _ensure_bchw(sample["height_raw"]).float()
    normal_mask = _ensure_bchw(sample["normal_mask"]).float()
    terrain_valid_mask = _ensure_bchw(sample["terrain_valid_mask_257"]).float()
    object_weight = _ensure_bchw(sample["weight_257"]).float()
    mddf_mask = _ensure_bchw(sample["mddf_mask"]).float()
    modf_mask = _ensure_bchw(sample["modf_mask"]).float()
    liquid_mask = _ensure_bchw(sample["liquid_mask"]).float()
    alpha_painted_256 = _ensure_bchw(sample["alpha_painted_256"]).float()
    mcly_any_16 = _ensure_bchw(sample["mcly_any_16"]).float()
    what_plate_flag = float(sample["what_plate_flag"].item())

    liquid_mask_257 = _resize_weight(liquid_mask, target_n.shape[-2:])
    object_presence = torch.maximum(mddf_mask, modf_mask)
    liquid_weight = 1.0 - (0.85 * liquid_mask_257)
    instance_weight = 1.0 - (0.75 * object_presence)
    base_mask = normal_mask * terrain_valid_mask * object_weight * liquid_weight * instance_weight
    if what_plate_flag > 0.5:
        base_mask = torch.zeros_like(base_mask)

    hard_region_weight, hard_debug = _hard_region_weight_from_targets(
        height_raw=height_raw,
        target_normals=target_n,
        alpha_painted_256=alpha_painted_256,
        mcly_any_16=mcly_any_16,
        terrain_valid_mask=terrain_valid_mask,
        base_mask=base_mask,
        detail_boost=detail_boost,
    )
    train_mask = base_mask * hard_region_weight

    hard_region = hard_debug["hard_region_signal"][0, 0]
    transition = hard_debug["transition_signal"][0, 0]
    train_mask = train_mask[0, 0]
    return (
        hard_region.detach().cpu().numpy().astype(np.float32, copy=False),
        transition.detach().cpu().numpy().astype(np.float32, copy=False),
        train_mask.detach().cpu().numpy().astype(np.float32, copy=False),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Stitch V16.1 hard-region and transition quilts")
    parser.add_argument("--dataset-dir", type=str, default="../output/datasets/v16", help="Dataset directory containing <build>.zarr stores")
    parser.add_argument("--builds", nargs="*", default=None, help="Optional build list (default: all found)")
    parser.add_argument("--maps", nargs="*", default=None, help="Optional map filter (exact map names)")
    parser.add_argument("--include-unknown-map", action="store_true", help="Include rows where map name is missing/unknown")
    parser.add_argument("--split", choices=["all", "train", "val"], default="all", help="Which split to read")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Validation fraction used for split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split/capped sampling")
    parser.add_argument("--max-tiles", type=int, default=0, help="Optional cap after filters (0=unlimited)")
    parser.add_argument("--curation-manifest", type=str, default=None, help="Optional curation manifest root/file")
    parser.add_argument("--curation-min-terrain-validity", type=float, default=0.0, help="Optional curation terrain validity threshold")
    parser.add_argument("--curation-min-minimap-usefulness", type=float, default=0.0, help="Optional curation minimap usefulness threshold")
    parser.add_argument("--curation-reject-what-plate", action="store_true", help="Drop whiteplate tiles at dataset-filter stage")
    parser.add_argument("--normal-detail-boost", type=float, default=1.5, help="Detail boost used in hard-region weighting")
    parser.add_argument("--quilt-tile-size", type=int, default=64, help="Output tile size per ADT tile in quilt (e.g., 64/128/256)")
    parser.add_argument("--output-dir", type=str, default="../output/validation/quilt_debug", help="Output directory for stitched PNGs")
    args = parser.parse_args()

    split = "train" if args.split == "all" else args.split
    val_fraction = 0.0 if args.split == "all" else float(args.val_fraction)

    dataset = V161Dataset(
        dataset_dir=args.dataset_dir,
        builds=args.builds,
        split=split,
        val_fraction=val_fraction,
        seed=int(args.seed),
        augment=False,
        curation_manifest=args.curation_manifest,
        height_channel=False,
        curation_min_terrain_validity=float(args.curation_min_terrain_validity),
        curation_min_minimap_usefulness=float(args.curation_min_minimap_usefulness),
        curation_reject_what_plate=bool(args.curation_reject_what_plate),
    )

    include_maps = set(args.maps) if args.maps else None
    max_tiles = int(args.max_tiles) if int(args.max_tiles) > 0 else None
    tile_refs = _collect_tile_refs(
        dataset,
        include_maps=include_maps,
        max_tiles=max_tiles,
        seed=int(args.seed),
        skip_unknown_map=not bool(args.include_unknown_map),
    )
    if not tile_refs:
        raise RuntimeError("No tiles matched filters; nothing to stitch.")

    accs = _group_accumulators(tile_refs, tile_size=int(args.quilt_tile_size))
    print(f"Selected tiles: {len(tile_refs)}")
    print(f"Groups: {len(accs)}")

    for i, ref in enumerate(tile_refs, start=1):
        sample = dataset[ref.dataset_index]
        hard_raw_257, trans_raw_257, train_mask_257 = _compute_debug_signals(sample, detail_boost=float(args.normal_detail_boost))

        hard_raw = _resize_tile(torch.from_numpy(hard_raw_257), int(args.quilt_tile_size))
        trans_raw = _resize_tile(torch.from_numpy(trans_raw_257), int(args.quilt_tile_size))
        train_raw = _resize_tile(torch.from_numpy(train_mask_257), int(args.quilt_tile_size))

        acc = accs[(ref.build, ref.map_name)]
        acc.write_tile(
            tile_x=ref.tile_x,
            tile_y=ref.tile_y,
            hard_region_raw=hard_raw,
            transition_raw=trans_raw,
            train_mask_raw=train_raw,
        )
        if i % 200 == 0 or i == len(tile_refs):
            print(f"  processed {i}/{len(tile_refs)}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict[str, object]] = {}

    for (build, map_name), acc in sorted(accs.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        safe_build = _safe_key(build)
        safe_map = _safe_key(map_name)
        group_dir = out_dir / safe_build / safe_map
        group_dir.mkdir(parents=True, exist_ok=True)

        hr_local_path = group_dir / "quilt_hard_region_local.png"
        tr_local_path = group_dir / "quilt_transition_local.png"
        tm_local_path = group_dir / "quilt_train_mask_local.png"
        hr_global_path = group_dir / "quilt_hard_region_global.png"
        tr_global_path = group_dir / "quilt_transition_global.png"

        hr_local_stats = _save_local_png(acc.hard_region_local, hr_local_path)
        tr_local_stats = _save_local_png(acc.transition_local, tr_local_path)
        tm_local_stats = _save_local_png(acc.train_mask_local, tm_local_path)
        hr_global_stats = _save_global_png(acc.hard_region_raw, hr_global_path)
        tr_global_stats = _save_global_png(acc.transition_raw, tr_global_path)

        key = f"{build}/{map_name}"
        summary[key] = {
            "build": build,
            "map": map_name,
            "tiles_written": acc.tiles_written,
            "tile_bounds": {
                "min_x": acc.min_x,
                "max_x": acc.max_x,
                "min_y": acc.min_y,
                "max_y": acc.max_y,
            },
            "quilt_tile_size": acc.tile_size,
            "outputs": {
                "hard_region_local": str(hr_local_path.relative_to(out_dir)).replace("\\", "/"),
                "transition_local": str(tr_local_path.relative_to(out_dir)).replace("\\", "/"),
                "train_mask_local": str(tm_local_path.relative_to(out_dir)).replace("\\", "/"),
                "hard_region_global": str(hr_global_path.relative_to(out_dir)).replace("\\", "/"),
                "transition_global": str(tr_global_path.relative_to(out_dir)).replace("\\", "/"),
            },
            "stats": {
                "hard_region_local": hr_local_stats,
                "transition_local": tr_local_stats,
                "train_mask_local": tm_local_stats,
                "hard_region_global": hr_global_stats,
                "transition_global": tr_global_stats,
            },
        }
        print(f"Wrote quilts: {safe_build}/{safe_map} (tiles={acc.tiles_written})")

    summary_path = out_dir / "quilt_debug_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
