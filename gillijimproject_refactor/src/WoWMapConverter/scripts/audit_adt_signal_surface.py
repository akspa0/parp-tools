#!/usr/bin/env python3
"""Audit ADT-derived dataset signal coverage.

This script is narrower than ``audit_v7_signals.py``. It answers a different
question: which raw ADT-side signals are already exported into the dataset,
which ones are dense enough to matter, and which chunk families appear to be
missing from the current exporter surface entirely.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


MCLY_FLAG_ANIMATION_ENABLED = 0x40
MCLY_FLAG_OVERBRIGHT = 0x80
MCLY_FLAG_USE_ALPHA_MAP = 0x100
MCLY_FLAG_ALPHA_COMPRESSED = 0x200
MCLY_FLAG_CUBE_MAP_REFLECTION = 0x400
MCLY_FLAG_UNKNOWN_0X800 = 0x800
MCLY_FLAG_UNKNOWN_0X1000 = 0x1000


@dataclass
class SignalSurfaceStats:
    dataset_root: Path
    tile_count: int = 0
    tiles_with_chunk_layers: int = 0
    tiles_with_wdl: int = 0
    tiles_with_mccv_map: int = 0
    tiles_with_alpha_masks: int = 0
    tiles_with_alpha_atlas: int = 0
    tiles_with_shadow_maps: int = 0
    tiles_with_shadow_bits: int = 0
    tiles_with_shadow_analysis: int = 0
    tiles_with_holes: int = 0
    tiles_with_liquids: int = 0
    tiles_with_objects: int = 0

    total_chunks: int = 0
    chunks_with_normals: int = 0
    chunks_with_mccv_colors: int = 0
    chunks_with_nonzero_area_id: int = 0
    chunks_with_nonzero_flags: int = 0

    total_layers: int = 0
    layers_with_effect_id: int = 0
    layers_with_ground_effects: int = 0
    layers_with_alpha_bits: int = 0
    layers_with_alpha_path: int = 0
    layers_with_nonzero_flags: int = 0
    layers_with_animation: int = 0
    layers_with_overbright: int = 0
    layers_with_alpha_map: int = 0
    layers_with_compressed_alpha: int = 0
    layers_with_cube_map: int = 0
    layers_with_flag_0x800: int = 0
    layers_with_flag_0x1000: int = 0

    total_liquids: int = 0
    liquids_with_heights: int = 0
    liquids_with_exists_bitmap: int = 0
    liquids_with_partial_rect: int = 0

    total_objects: int = 0
    objects_with_bounds: int = 0
    objects_with_model_path: int = 0
    m2_objects: int = 0
    wmo_objects: int = 0

    chunk_flag_counter: Counter[str] = field(default_factory=Counter)
    layer_flag_counter: Counter[str] = field(default_factory=Counter)
    liquid_type_counter: Counter[str] = field(default_factory=Counter)
    object_category_counter: Counter[str] = field(default_factory=Counter)
    effect_id_counter: Counter[str] = field(default_factory=Counter)
    area_id_counter: Counter[str] = field(default_factory=Counter)


def safe_load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def format_ratio(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "0/0 (0.0%)"
    return f"{numerator}/{denominator} ({(numerator / denominator) * 100.0:.1f}%)"


def summarize_counter(counter: Counter[str], limit: int = 8) -> str:
    if not counter:
        return "n/a"
    return ", ".join(f"{key}={value}" for key, value in counter.most_common(limit))


def iter_dataset_jsons(dataset_root: Path) -> Iterable[Path]:
    dataset_dir = dataset_root / "dataset"
    if not dataset_dir.exists():
        return []
    return sorted(dataset_dir.glob("*.json"))


def audit_dataset_root(dataset_root: Path) -> SignalSurfaceStats:
    stats = SignalSurfaceStats(dataset_root=dataset_root)

    for json_path in iter_dataset_jsons(dataset_root):
        payload = safe_load_json(json_path)
        if payload is None:
            continue

        terrain = payload.get("terrain_data")
        if not isinstance(terrain, dict):
            continue

        stats.tile_count += 1

        chunk_layers = terrain.get("chunk_layers")
        if isinstance(chunk_layers, list) and chunk_layers:
            stats.tiles_with_chunk_layers += 1

        if isinstance(terrain.get("wdl_heights"), dict):
            stats.tiles_with_wdl += 1
        if terrain.get("mccv_map"):
            stats.tiles_with_mccv_map += 1
        if isinstance(terrain.get("alpha_masks"), list) and terrain.get("alpha_masks"):
            stats.tiles_with_alpha_masks += 1
        if terrain.get("alpha_atlas"):
            stats.tiles_with_alpha_atlas += 1
        if isinstance(terrain.get("shadow_maps"), list) and terrain.get("shadow_maps"):
            stats.tiles_with_shadow_maps += 1
        if isinstance(terrain.get("shadow_bits"), list) and terrain.get("shadow_bits"):
            stats.tiles_with_shadow_bits += 1
        if isinstance(terrain.get("shadow_analysis"), list) and terrain.get("shadow_analysis"):
            stats.tiles_with_shadow_analysis += 1
        holes = terrain.get("holes")
        if isinstance(holes, list) and any(coerce_int(value) != 0 for value in holes):
            stats.tiles_with_holes += 1
        liquids = terrain.get("liquids")
        if isinstance(liquids, list) and liquids:
            stats.tiles_with_liquids += 1
        objects = terrain.get("objects")
        if isinstance(objects, list) and objects:
            stats.tiles_with_objects += 1

        if isinstance(chunk_layers, list):
            for chunk in chunk_layers:
                if not isinstance(chunk, dict):
                    continue

                stats.total_chunks += 1

                normals = chunk.get("normals")
                if isinstance(normals, list) and normals:
                    stats.chunks_with_normals += 1

                mccv_colors = chunk.get("mccv_colors")
                if isinstance(mccv_colors, list) and mccv_colors:
                    stats.chunks_with_mccv_colors += 1

                area_id = coerce_int(chunk.get("area_id"), 0)
                if area_id != 0:
                    stats.chunks_with_nonzero_area_id += 1
                    stats.area_id_counter[str(area_id)] += 1

                chunk_flags = coerce_int(chunk.get("flags"), 0)
                if chunk_flags != 0:
                    stats.chunks_with_nonzero_flags += 1
                    stats.chunk_flag_counter[f"0x{chunk_flags:08x}"] += 1

                layers = chunk.get("layers")
                if not isinstance(layers, list):
                    continue

                for layer in layers:
                    if not isinstance(layer, dict):
                        continue

                    stats.total_layers += 1
                    layer_flags = coerce_int(layer.get("flags"), 0)
                    if layer_flags != 0:
                        stats.layers_with_nonzero_flags += 1
                        stats.layer_flag_counter[f"0x{layer_flags:08x}"] += 1
                    if layer_flags & MCLY_FLAG_ANIMATION_ENABLED:
                        stats.layers_with_animation += 1
                    if layer_flags & MCLY_FLAG_OVERBRIGHT:
                        stats.layers_with_overbright += 1
                    if layer_flags & MCLY_FLAG_USE_ALPHA_MAP:
                        stats.layers_with_alpha_map += 1
                    if layer_flags & MCLY_FLAG_ALPHA_COMPRESSED:
                        stats.layers_with_compressed_alpha += 1
                    if layer_flags & MCLY_FLAG_CUBE_MAP_REFLECTION:
                        stats.layers_with_cube_map += 1
                    if layer_flags & MCLY_FLAG_UNKNOWN_0X800:
                        stats.layers_with_flag_0x800 += 1
                    if layer_flags & MCLY_FLAG_UNKNOWN_0X1000:
                        stats.layers_with_flag_0x1000 += 1

                    effect_id = coerce_int(layer.get("effect_id"), 0)
                    if effect_id > 0:
                        stats.layers_with_effect_id += 1
                        stats.effect_id_counter[str(effect_id)] += 1

                    ground_effects = layer.get("ground_effects")
                    if isinstance(ground_effects, list) and ground_effects:
                        stats.layers_with_ground_effects += 1

                    alpha_bits = layer.get("alpha_bits")
                    if isinstance(alpha_bits, str) and alpha_bits.strip():
                        stats.layers_with_alpha_bits += 1

                    alpha_path = layer.get("alpha_path")
                    if isinstance(alpha_path, str) and alpha_path.strip():
                        stats.layers_with_alpha_path += 1

        if isinstance(liquids, list):
            for liquid in liquids:
                if not isinstance(liquid, dict):
                    continue

                stats.total_liquids += 1
                liquid_type = coerce_int(liquid.get("type"), -1)
                stats.liquid_type_counter[str(liquid_type)] += 1

                heights = liquid.get("heights")
                if isinstance(heights, list) and heights:
                    stats.liquids_with_heights += 1

                exists_bitmap = liquid.get("exists_bitmap")
                if isinstance(exists_bitmap, str) and exists_bitmap.strip():
                    stats.liquids_with_exists_bitmap += 1

                x_offset = coerce_int(liquid.get("x_offset"), 0)
                y_offset = coerce_int(liquid.get("y_offset"), 0)
                width = coerce_int(liquid.get("width"), 8)
                height = coerce_int(liquid.get("height"), 8)
                if x_offset != 0 or y_offset != 0 or width != 8 or height != 8:
                    stats.liquids_with_partial_rect += 1

        if isinstance(objects, list):
            for obj in objects:
                if not isinstance(obj, dict):
                    continue

                stats.total_objects += 1
                category = str(obj.get("category") or "unknown").lower()
                stats.object_category_counter[category] += 1
                if category == "m2":
                    stats.m2_objects += 1
                elif category == "wmo":
                    stats.wmo_objects += 1

                bounds_min = obj.get("bounds_min")
                bounds_max = obj.get("bounds_max")
                if isinstance(bounds_min, list) and isinstance(bounds_max, list) and bounds_min and bounds_max:
                    stats.objects_with_bounds += 1

                model_path = obj.get("model_path")
                if isinstance(model_path, str) and model_path.strip():
                    stats.objects_with_model_path += 1

    return stats


def print_summary(stats: SignalSurfaceStats) -> None:
    print(f"\nDataset root: {stats.dataset_root}")
    print(f"Tiles: {stats.tile_count}")
    print("Tile-level exported ADT surfaces:")
    print(f"  chunk_layers:        {format_ratio(stats.tiles_with_chunk_layers, stats.tile_count)}")
    print(f"  wdl_heights:         {format_ratio(stats.tiles_with_wdl, stats.tile_count)}")
    print(f"  mccv_map:            {format_ratio(stats.tiles_with_mccv_map, stats.tile_count)}")
    print(f"  alpha_masks:         {format_ratio(stats.tiles_with_alpha_masks, stats.tile_count)}")
    print(f"  alpha_atlas:         {format_ratio(stats.tiles_with_alpha_atlas, stats.tile_count)}")
    print(f"  shadow_maps:         {format_ratio(stats.tiles_with_shadow_maps, stats.tile_count)}")
    print(f"  shadow_bits:         {format_ratio(stats.tiles_with_shadow_bits, stats.tile_count)}")
    print(f"  shadow_analysis:     {format_ratio(stats.tiles_with_shadow_analysis, stats.tile_count)}")
    print(f"  holes_nonzero:       {format_ratio(stats.tiles_with_holes, stats.tile_count)}")
    print(f"  liquids:             {format_ratio(stats.tiles_with_liquids, stats.tile_count)}")
    print(f"  objects:             {format_ratio(stats.tiles_with_objects, stats.tile_count)}")

    print("Chunk-level ADT surfaces:")
    print(f"  chunks_total:        {stats.total_chunks}")
    print(f"  normals:             {format_ratio(stats.chunks_with_normals, stats.total_chunks)}")
    print(f"  mccv_colors:         {format_ratio(stats.chunks_with_mccv_colors, stats.total_chunks)}")
    print(f"  nonzero_area_id:     {format_ratio(stats.chunks_with_nonzero_area_id, stats.total_chunks)}")
    print(f"  nonzero_chunk_flags: {format_ratio(stats.chunks_with_nonzero_flags, stats.total_chunks)}")
    print(f"  top_chunk_flags:     {summarize_counter(stats.chunk_flag_counter)}")
    print(f"  top_area_ids:        {summarize_counter(stats.area_id_counter)}")

    print("Layer-level ADT surfaces:")
    print(f"  layers_total:        {stats.total_layers}")
    print(f"  effect_id:           {format_ratio(stats.layers_with_effect_id, stats.total_layers)}")
    print(f"  ground_effects:      {format_ratio(stats.layers_with_ground_effects, stats.total_layers)}")
    print(f"  alpha_bits:          {format_ratio(stats.layers_with_alpha_bits, stats.total_layers)}")
    print(f"  alpha_path:          {format_ratio(stats.layers_with_alpha_path, stats.total_layers)}")
    print(f"  nonzero_flags:       {format_ratio(stats.layers_with_nonzero_flags, stats.total_layers)}")
    print(f"  animation_enabled:   {format_ratio(stats.layers_with_animation, stats.total_layers)}")
    print(f"  overbright:          {format_ratio(stats.layers_with_overbright, stats.total_layers)}")
    print(f"  use_alpha_map:       {format_ratio(stats.layers_with_alpha_map, stats.total_layers)}")
    print(f"  compressed_alpha:    {format_ratio(stats.layers_with_compressed_alpha, stats.total_layers)}")
    print(f"  cube_map_reflection: {format_ratio(stats.layers_with_cube_map, stats.total_layers)}")
    print(f"  flag_0x800:          {format_ratio(stats.layers_with_flag_0x800, stats.total_layers)}")
    print(f"  flag_0x1000:         {format_ratio(stats.layers_with_flag_0x1000, stats.total_layers)}")
    print(f"  top_layer_flags:     {summarize_counter(stats.layer_flag_counter)}")
    print(f"  top_effect_ids:      {summarize_counter(stats.effect_id_counter)}")

    print("Liquid ADT surfaces:")
    print(f"  liquid_records:      {stats.total_liquids}")
    print(f"  with_heights:        {format_ratio(stats.liquids_with_heights, stats.total_liquids)}")
    print(f"  exists_bitmap:       {format_ratio(stats.liquids_with_exists_bitmap, stats.total_liquids)}")
    print(f"  partial_rect:        {format_ratio(stats.liquids_with_partial_rect, stats.total_liquids)}")
    print(f"  top_liquid_types:    {summarize_counter(stats.liquid_type_counter)}")

    print("Object ADT surfaces:")
    print(f"  object_records:      {stats.total_objects}")
    print(f"  with_bounds:         {format_ratio(stats.objects_with_bounds, stats.total_objects)}")
    print(f"  with_model_path:     {format_ratio(stats.objects_with_model_path, stats.total_objects)}")
    print(f"  m2_objects:          {format_ratio(stats.m2_objects, stats.total_objects)}")
    print(f"  wmo_objects:         {format_ratio(stats.wmo_objects, stats.total_objects)}")
    print(f"  categories:          {summarize_counter(stats.object_category_counter)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit exported ADT-side signal coverage inside dataset JSON roots.")
    parser.add_argument(
        "dataset_roots",
        nargs="+",
        help="One or more harvested dataset roots, for example datasets/3_3_5_12340/Azeroth",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for root_text in args.dataset_roots:
        dataset_root = Path(root_text)
        print_summary(audit_dataset_root(dataset_root))


if __name__ == "__main__":
    main()