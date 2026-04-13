#!/usr/bin/env python3
"""Audit V7 dataset signal coverage and effective activation.

This script inspects the real ML corpus used by ``train_v7.py`` and reports:
- declared signal coverage per dataset root
- effective activation for liquid/object channels
- coarse image statistics for minimap and normal inputs
- the exact signal encoding assumptions the current V7 trainer uses
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from PIL import Image, ImageStat


INPUT_SIZE = 512
TILE_SIZE = 533.33333


@dataclass
class ImageStatsAccumulator:
    count: int = 0
    channel_mean_sum: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    channel_std_sum: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    luma_mean_sum: float = 0.0
    luma_std_sum: float = 0.0
    luma_p10_sum: float = 0.0
    luma_p50_sum: float = 0.0
    luma_p90_sum: float = 0.0

    def add(self, image_path: Path) -> None:
        image = Image.open(image_path).convert("RGB")
        stat = ImageStat.Stat(image)
        means = [value / 255.0 for value in stat.mean[:3]]
        stddevs = [value / 255.0 for value in stat.stddev[:3]]
        luma = image.convert("L")
        luma_stat = ImageStat.Stat(luma)
        histogram = luma.histogram()

        def percentile_from_histogram(target_percentile: float) -> float:
            cutoff = self._histogram_total(histogram) * target_percentile
            running = 0
            for value, count in enumerate(histogram):
                running += count
                if running >= cutoff:
                    return value / 255.0
            return 1.0

        self.count += 1
        for index in range(3):
            self.channel_mean_sum[index] += means[index]
            self.channel_std_sum[index] += stddevs[index]
        self.luma_mean_sum += luma_stat.mean[0] / 255.0
        self.luma_std_sum += luma_stat.stddev[0] / 255.0
        self.luma_p10_sum += percentile_from_histogram(0.10)
        self.luma_p50_sum += percentile_from_histogram(0.50)
        self.luma_p90_sum += percentile_from_histogram(0.90)

    @staticmethod
    def _histogram_total(histogram: List[int]) -> int:
        return sum(histogram)

    def format_summary(self) -> str:
        if self.count == 0:
            return "n/a"

        means = [value / self.count for value in self.channel_mean_sum]
        stddevs = [value / self.count for value in self.channel_std_sum]
        luma_mean = self.luma_mean_sum / self.count
        luma_std = self.luma_std_sum / self.count
        luma_p10 = self.luma_p10_sum / self.count
        luma_p50 = self.luma_p50_sum / self.count
        luma_p90 = self.luma_p90_sum / self.count
        mean_text = ", ".join(f"{value:.3f}" for value in means)
        std_text = ", ".join(f"{value:.3f}" for value in stddevs)
        return (
            f"mean=[{mean_text}] std=[{std_text}] "
            f"luma_mean={luma_mean:.3f} luma_std={luma_std:.3f} "
            f"luma_p10={luma_p10:.3f} luma_p50={luma_p50:.3f} luma_p90={luma_p90:.3f}"
        )


@dataclass
class DatasetSignalStats:
    dataset_root: Path
    tile_count: int = 0
    minimap_present: int = 0
    terrain_only_minimap_present: int = 0
    normalmap_present: int = 0
    heightmap_local_present: int = 0
    heightmap_global_present: int = 0
    mccv_map_present: int = 0
    shadow_maps_present: int = 0
    alpha_masks_present: int = 0
    chunk_layers_present: int = 0
    wdl_present: int = 0
    bounds_present: int = 0
    nontrivial_height_range: int = 0
    holes_present: int = 0
    liquids_declared: int = 0
    liquid_mask_path_present: int = 0
    liquid_mask_file_present: int = 0
    liquid_mask_nonzero: int = 0
    liquid_height_file_present: int = 0
    no_liquid_minimap_present: int = 0
    objects_declared: int = 0
    object_bounds_present: int = 0
    nonzero_object_mask_tiles: int = 0
    total_object_count: int = 0
    minimap_stats: ImageStatsAccumulator = field(default_factory=ImageStatsAccumulator)
    normalmap_stats: ImageStatsAccumulator = field(default_factory=ImageStatsAccumulator)


def resolve_dataset_path(dataset_root: Path, relative_path: Optional[str]) -> Optional[Path]:
    if not relative_path:
        return None
    path = Path(relative_path)
    return path if path.is_absolute() else dataset_root / path


def safe_load_json(path: Path) -> Optional[Dict[str, object]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def count_nonzero_mask_pixels(path: Path) -> int:
    image = Image.open(path).convert("L")
    histogram = image.histogram()
    return sum(histogram[1:])


def build_object_mask_pixels(objects: Optional[Sequence[Dict[str, object]]]) -> int:
    if not objects:
        return 0

    mask = Image.new("L", (INPUT_SIZE, INPUT_SIZE), 0)
    pixels = mask.load()

    for obj in objects:
        pos_x = float(obj.get("x", obj.get("pos_x", 0.0)))
        pos_y = float(obj.get("y", obj.get("pos_y", 0.0)))
        scale = float(obj.get("scale", 1.0))

        bounds_min = obj.get("bounds_min")
        bounds_max = obj.get("bounds_max")
        if isinstance(bounds_min, list) and isinstance(bounds_max, list) and len(bounds_min) >= 2 and len(bounds_max) >= 2:
            half_width = abs(float(bounds_max[0]) - float(bounds_min[0])) * 0.5 * scale
            half_depth = abs(float(bounds_max[1]) - float(bounds_min[1])) * 0.5 * scale
            pixels_per_unit = INPUT_SIZE / TILE_SIZE
            radius_x = max(1, int(half_width * pixels_per_unit))
            radius_y = max(1, int(half_depth * pixels_per_unit))
        else:
            radius_x = max(1, int(5 * scale))
            radius_y = radius_x

        if abs(pos_x) < 2 and abs(pos_y) < 2:
            normalized_x = int((pos_x + 1) * 0.5 * INPUT_SIZE)
            normalized_y = int((pos_y + 1) * 0.5 * INPUT_SIZE)
        else:
            normalized_x = int((pos_x / TILE_SIZE) * INPUT_SIZE) % INPUT_SIZE
            normalized_y = int((pos_y / TILE_SIZE) * INPUT_SIZE) % INPUT_SIZE

        x1 = max(0, normalized_x - radius_x)
        y1 = max(0, normalized_y - radius_y)
        x2 = min(INPUT_SIZE, normalized_x + radius_x)
        y2 = min(INPUT_SIZE, normalized_y + radius_y)
        for y in range(y1, y2):
            for x in range(x1, x2):
                pixels[x, y] = 255

    return count_nonzero_mask_pixels_from_image(mask)


def count_nonzero_mask_pixels_from_image(image: Image.Image) -> int:
    histogram = image.histogram()
    return sum(histogram[1:])


def audit_dataset_root(dataset_root: Path, image_sample_limit: int) -> DatasetSignalStats:
    stats = DatasetSignalStats(dataset_root=dataset_root)
    dataset_dir = dataset_root / "dataset"
    json_files = sorted(dataset_dir.glob("*.json"))

    for json_path in json_files:
        payload = safe_load_json(json_path)
        if payload is None:
            continue

        terrain = payload.get("terrain_data") or {}
        if not isinstance(terrain, dict):
            continue

        stats.tile_count += 1

        tile_name = str(terrain.get("adt_tile") or json_path.stem)
        terrain_only_minimap_path = resolve_dataset_path(dataset_root, terrain.get("terrain_only_minimap"))
        no_object_minimap_path = resolve_dataset_path(dataset_root, terrain.get("no_object_minimap"))
        no_mccv_minimap_path = resolve_dataset_path(dataset_root, terrain.get("no_mccv_minimap"))
        minimap_path = terrain_only_minimap_path or no_object_minimap_path or no_mccv_minimap_path or dataset_root / "images" / f"{tile_name}.png"
        normalmap_path = resolve_dataset_path(dataset_root, terrain.get("normalmap"))
        heightmap_local_path = resolve_dataset_path(dataset_root, terrain.get("heightmap_local") or terrain.get("heightmap"))
        heightmap_global_path = resolve_dataset_path(dataset_root, terrain.get("heightmap_global") or terrain.get("heightmap"))
        mccv_map_path = resolve_dataset_path(dataset_root, terrain.get("mccv_map"))
        liquid_mask_path = resolve_dataset_path(dataset_root, terrain.get("liquid_mask"))
        liquid_height_path = resolve_dataset_path(dataset_root, terrain.get("liquid_height"))
        no_liquid_minimap_path = resolve_dataset_path(dataset_root, terrain.get("no_liquid_minimap"))

        if minimap_path.exists():
            stats.minimap_present += 1
            if stats.minimap_stats.count < image_sample_limit:
                stats.minimap_stats.add(minimap_path)

        if terrain_only_minimap_path and terrain_only_minimap_path.exists():
            stats.terrain_only_minimap_present += 1

        if normalmap_path and normalmap_path.exists():
            stats.normalmap_present += 1
            if stats.normalmap_stats.count < image_sample_limit:
                stats.normalmap_stats.add(normalmap_path)

        if heightmap_local_path and heightmap_local_path.exists():
            stats.heightmap_local_present += 1
        if heightmap_global_path and heightmap_global_path.exists():
            stats.heightmap_global_present += 1
        if mccv_map_path and mccv_map_path.exists():
            stats.mccv_map_present += 1
        if no_liquid_minimap_path and no_liquid_minimap_path.exists():
            stats.no_liquid_minimap_present += 1

        shadow_maps = terrain.get("shadow_maps") or []
        if isinstance(shadow_maps, list) and shadow_maps:
            stats.shadow_maps_present += 1

        alpha_masks = terrain.get("alpha_masks") or []
        if isinstance(alpha_masks, list) and alpha_masks:
            stats.alpha_masks_present += 1

        chunk_layers = terrain.get("chunk_layers") or []
        if isinstance(chunk_layers, list) and chunk_layers:
            stats.chunk_layers_present += 1

        wdl_heights = terrain.get("wdl_heights") or {}
        if isinstance(wdl_heights, dict) and isinstance(wdl_heights.get("outer_17"), list) and len(wdl_heights["outer_17"]) == 289:
            stats.wdl_present += 1

        if "height_min" in terrain and "height_max" in terrain:
            stats.bounds_present += 1
            try:
                height_min = float(terrain.get("height_min", 0.0))
                height_max = float(terrain.get("height_max", 0.0))
                if abs(height_max - height_min) > 1e-4:
                    stats.nontrivial_height_range += 1
            except Exception:
                pass

        holes = terrain.get("holes") or []
        if isinstance(holes, list) and any(int(value) != 0 for value in holes):
            stats.holes_present += 1

        liquids = terrain.get("liquids") or []
        if isinstance(liquids, list) and liquids:
            stats.liquids_declared += 1

        if terrain.get("liquid_mask"):
            stats.liquid_mask_path_present += 1
        if liquid_mask_path and liquid_mask_path.exists():
            stats.liquid_mask_file_present += 1
            if count_nonzero_mask_pixels(liquid_mask_path) > 0:
                stats.liquid_mask_nonzero += 1
        if liquid_height_path and liquid_height_path.exists():
            stats.liquid_height_file_present += 1

        objects = terrain.get("objects") or []
        if isinstance(objects, list) and objects:
            stats.objects_declared += 1
            stats.total_object_count += len(objects)
            if any(obj.get("bounds_min") or obj.get("bounds_max") for obj in objects if isinstance(obj, dict)):
                stats.object_bounds_present += 1
            if build_object_mask_pixels([obj for obj in objects if isinstance(obj, dict)]) > 0:
                stats.nonzero_object_mask_tiles += 1

    return stats


def pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0/0 (0.0%)"
    return f"{numerator}/{denominator} ({(numerator / denominator) * 100:.1f}%)"


def print_dataset_summary(stats: DatasetSignalStats) -> None:
    print(f"\nDataset root: {stats.dataset_root}")
    print(f"Tiles: {stats.tile_count}")
    print("Core signals:")
    print(f"  minimap_rgb:         {pct(stats.minimap_present, stats.tile_count)}")
    print(f"  normal_rgb:          {pct(stats.normalmap_present, stats.tile_count)}")
    print(f"  wdl_prior:           {pct(stats.wdl_present, stats.tile_count)}")
    print(f"  bounds_hints:        {pct(stats.bounds_present, stats.tile_count)}")
    print(f"  nontrivial_range:    {pct(stats.nontrivial_height_range, stats.tile_count)}")
    print(f"  heightmap_local:     {pct(stats.heightmap_local_present, stats.tile_count)}")
    print(f"  heightmap_global:    {pct(stats.heightmap_global_present, stats.tile_count)}")
    print("Auxiliary signals:")
    print(f"  terrain_only_minimap:{pct(stats.terrain_only_minimap_present, stats.tile_count)}")
    print(f"  liquid_records:      {pct(stats.liquids_declared, stats.tile_count)}")
    print(f"  liquid_mask_field:   {pct(stats.liquid_mask_path_present, stats.tile_count)}")
    print(f"  liquid_mask_file:    {pct(stats.liquid_mask_file_present, stats.tile_count)}")
    print(f"  liquid_mask_nonzero: {pct(stats.liquid_mask_nonzero, stats.tile_count)}")
    print(f"  liquid_height_file:  {pct(stats.liquid_height_file_present, stats.tile_count)}")
    print(f"  no_liquid_minimap:   {pct(stats.no_liquid_minimap_present, stats.tile_count)}")
    print(f"  objects_declared:    {pct(stats.objects_declared, stats.tile_count)}")
    print(f"  object_bounds:       {pct(stats.object_bounds_present, stats.tile_count)}")
    print(f"  object_mask_nonzero: {pct(stats.nonzero_object_mask_tiles, stats.tile_count)}")
    print(f"  total_objects:       {stats.total_object_count}")
    print("Other available signals:")
    print(f"  mccv_map:            {pct(stats.mccv_map_present, stats.tile_count)}")
    print(f"  alpha_masks:         {pct(stats.alpha_masks_present, stats.tile_count)}")
    print(f"  shadow_maps:         {pct(stats.shadow_maps_present, stats.tile_count)}")
    print(f"  chunk_layers:        {pct(stats.chunk_layers_present, stats.tile_count)}")
    print(f"  holes_nonzero:       {pct(stats.holes_present, stats.tile_count)}")
    print("Input image statistics:")
    print(f"  minimap_rgb:         {stats.minimap_stats.format_summary()}")
    print(f"  normal_rgb:          {stats.normalmap_stats.format_summary()}")


def print_encoding_notes() -> None:
    print("\nCurrent V7 signal encodings in train_v7.py:")
    print("  minimap_rgb: prefers terrain_only_minimap, then no_object_minimap, then no_mccv_minimap, then raw image; bilinear resize to 512, Gaussian blur on minimap only, ImageNet normalization")
    print("  brightness audit: use luma_mean/std/p10/p50/p90 as a dataset cleanliness diagnostic, not as a direct height prior")
    print("  normal_rgb: RGB image, bilinear resize to 512, ImageNet normalization")
    print("  wdl_prior: outer_17 grid only, per-tile min/max normalization, bilinear upsample to 512")
    print("  bounds_hints: two constant full-frame masks from height_min and height_max normalized against global range")
    print("  liquid_mask: binary mask thresholded from stitched liquid-mask PNG, nearest resize")
    print("  liquid_height_prior: normalized stitched liquid-height raster masked to liquid coverage; WL-derived heights can feed the same channel later")
    print("  object_mask: binary rectangle footprints rendered from objects list using bounds when present, otherwise scale fallback")
    print("  raw auxiliary assets still available beyond the cleaned RGB surface: mccv_map, alpha_masks, alpha_atlas, shadow_maps, chunk_layers, holes, no_liquid_minimap")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit V7 dataset signal coverage and activation.")
    parser.add_argument("--dataset-root", action="append", required=True, help="Path to a V7 dataset root (repeatable).")
    parser.add_argument("--image-sample-limit", type=int, default=32, help="How many minimap/normal images to sample per root for summary stats.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for root_str in args.dataset_root:
        stats = audit_dataset_root(Path(root_str), args.image_sample_limit)
        print_dataset_summary(stats)
    print_encoding_notes()


if __name__ == "__main__":
    main()