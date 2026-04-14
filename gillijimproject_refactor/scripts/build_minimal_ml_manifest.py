#!/usr/bin/env python3
"""Build a deduplicated minimal ML dataset manifest from harvested map manifests.

This script scans datasets/<client>/<map>/ml_dataset_manifest.json, groups tiles by
stable visual signatures, keeps the highest-quality tile per group, and writes:
1) A tile-level minimal manifest
2) A map-level export plan that can be reused as a compact corpus recipe
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class TileCandidate:
    group_key: str
    score: float
    client_label: str
    map_name: str
    tile_name: str
    dataset_map_root: str
    tile_json_path: str
    source_minimap_path: Optional[str]
    completeness_class: str
    source_sha256: Optional[str]
    alpha_sha256: Optional[str]
    shadow_sha256: Optional[str]
    existing_shadow_map_count: int
    existing_alpha_mask_count: int
    object_count: int
    chunk_layer_count: int
    heightmap_local_exists: bool
    heightmap_global_exists: bool
    source_minimap_exists: bool


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def safe_bool(value: Any) -> bool:
    return bool(value)


def extract_sha256(obj: Any) -> Optional[str]:
    if not isinstance(obj, dict):
        return None
    value = obj.get("sha256")
    if not isinstance(value, str):
        return None
    value = value.strip().lower()
    return value if value else None


def tile_quality_score(tile: Dict[str, Any]) -> float:
    score = 0.0

    source_exists = safe_bool(tile.get("source_minimap_exists"))
    local_exists = safe_bool(tile.get("heightmap_local_exists"))
    global_exists = safe_bool(tile.get("heightmap_global_exists"))
    alpha_atlas_exists = safe_bool(tile.get("alpha_atlas_exists"))

    if source_exists:
        score += 100.0
    if global_exists:
        score += 90.0
    if local_exists:
        score += 80.0
    if alpha_atlas_exists:
        score += 16.0

    shadow_count = safe_int(tile.get("existing_shadow_map_count"))
    alpha_count = safe_int(tile.get("existing_alpha_mask_count"))
    object_count = safe_int(tile.get("object_count"))
    chunk_layer_count = safe_int(tile.get("chunk_layer_count"))

    score += min(shadow_count, 8) * 3.0
    score += min(alpha_count, 8) * 3.0
    score += min(object_count, 64) * 0.25
    score += min(chunk_layer_count, 256) / 256.0 * 12.0

    completeness = str(tile.get("completeness_class") or "").strip().lower()
    if completeness == "full":
        score += 20.0
    elif completeness == "partial":
        score += 8.0
    elif completeness:
        score += 2.0

    return score


def build_group_key(tile: Dict[str, Any], map_name: str, tile_name: str) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    source_sha = extract_sha256(tile.get("source_minimap_signature"))
    alpha_sha = extract_sha256(tile.get("alpha_atlas_signature"))
    shadow_sha = extract_sha256(tile.get("shadow_map_signature"))

    if source_sha:
        parts = [source_sha, alpha_sha or "-", shadow_sha or "-"]
        return "sig:" + "|".join(parts), source_sha, alpha_sha, shadow_sha

    fallback = f"fallback:{map_name.lower()}:{tile_name.lower()}"
    return fallback, source_sha, alpha_sha, shadow_sha


def discover_manifest_paths(datasets_root: Path) -> List[Path]:
    paths = [
        path
        for path in datasets_root.glob("*/*/ml_dataset_manifest.json")
        if path.is_file()
    ]
    return sorted(paths)


def load_candidates(datasets_root: Path) -> Tuple[List[TileCandidate], List[str], int]:
    candidates: List[TileCandidate] = []
    failures: List[str] = []
    scanned_manifests = 0

    for manifest_path in discover_manifest_paths(datasets_root):
        scanned_manifests += 1
        dataset_map_root = manifest_path.parent
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            failures.append(f"{manifest_path}: {exc}")
            continue

        client_label = dataset_map_root.parent.name
        map_name_from_path = dataset_map_root.name

        tiles = payload.get("tiles")
        if not isinstance(tiles, list):
            continue

        for tile in tiles:
            if not isinstance(tile, dict):
                continue

            tile_name = str(tile.get("tile_name") or "").strip()
            map_name = str(tile.get("map_name") or map_name_from_path).strip()
            if not tile_name or not map_name:
                continue

            group_key, source_sha, alpha_sha, shadow_sha = build_group_key(tile, map_name, tile_name)
            score = tile_quality_score(tile)

            tile_json_path = str(tile.get("tile_json_path") or "").strip()
            source_minimap_path = tile.get("source_minimap_path")
            if source_minimap_path is not None:
                source_minimap_path = str(source_minimap_path)

            candidates.append(
                TileCandidate(
                    group_key=group_key,
                    score=score,
                    client_label=client_label,
                    map_name=map_name,
                    tile_name=tile_name,
                    dataset_map_root=dataset_map_root.relative_to(datasets_root).as_posix(),
                    tile_json_path=tile_json_path,
                    source_minimap_path=source_minimap_path,
                    completeness_class=str(tile.get("completeness_class") or "partial"),
                    source_sha256=source_sha,
                    alpha_sha256=alpha_sha,
                    shadow_sha256=shadow_sha,
                    existing_shadow_map_count=safe_int(tile.get("existing_shadow_map_count")),
                    existing_alpha_mask_count=safe_int(tile.get("existing_alpha_mask_count")),
                    object_count=safe_int(tile.get("object_count")),
                    chunk_layer_count=safe_int(tile.get("chunk_layer_count")),
                    heightmap_local_exists=safe_bool(tile.get("heightmap_local_exists")),
                    heightmap_global_exists=safe_bool(tile.get("heightmap_global_exists")),
                    source_minimap_exists=safe_bool(tile.get("source_minimap_exists")),
                )
            )

    return candidates, failures, scanned_manifests


def select_best_by_group(candidates: Iterable[TileCandidate]) -> Tuple[Dict[str, TileCandidate], Dict[str, List[TileCandidate]]]:
    selected: Dict[str, TileCandidate] = {}
    grouped: Dict[str, List[TileCandidate]] = {}

    def rank_key(item: TileCandidate) -> Tuple[float, int, int, int, int, str, str, str]:
        return (
            item.score,
            1 if item.heightmap_global_exists else 0,
            1 if item.heightmap_local_exists else 0,
            item.object_count,
            item.chunk_layer_count,
            item.client_label,
            item.map_name,
            item.tile_name,
        )

    for item in candidates:
        grouped.setdefault(item.group_key, []).append(item)
        current = selected.get(item.group_key)
        if current is None or rank_key(item) > rank_key(current):
            selected[item.group_key] = item

    return selected, grouped


def build_map_plan(selected_items: Iterable[TileCandidate]) -> Dict[str, List[str]]:
    plan: Dict[str, set[str]] = {}
    for item in selected_items:
        plan.setdefault(item.client_label, set()).add(item.map_name)
    return {label: sorted(maps) for label, maps in sorted(plan.items())}


def default_output_path(datasets_root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return datasets_root.parent / "output" / "build-validation" / "ml-audit" / f"minimal_dataset_manifest_{stamp}.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a deduplicated minimal ML dataset manifest.")
    parser.add_argument("--datasets-root", default="i:/parp/parp-tools/datasets", help="Root folder containing client/map datasets.")
    parser.add_argument("--output", default=None, help="Output manifest JSON path.")
    parser.add_argument("--plan-output", default=None, help="Optional output path for map-level minimal export plan JSON.")
    parser.add_argument("--top-duplicate-groups", type=int, default=100, help="How many duplicate-group summaries to include.")
    args = parser.parse_args()

    datasets_root = Path(args.datasets_root).resolve()
    if not datasets_root.exists() or not datasets_root.is_dir():
        raise SystemExit(f"Datasets root not found: {datasets_root}")

    output_path = Path(args.output).resolve() if args.output else default_output_path(datasets_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plan_output_path = Path(args.plan_output).resolve() if args.plan_output else output_path.with_name(output_path.stem + "_plan.json")
    plan_output_path.parent.mkdir(parents=True, exist_ok=True)

    candidates, failures, scanned_manifests = load_candidates(datasets_root)
    selected_by_group, grouped = select_best_by_group(candidates)
    selected_items = list(selected_by_group.values())
    selected_items.sort(key=lambda item: (item.client_label.lower(), item.map_name.lower(), item.tile_name.lower()))

    duplicate_group_rows = []
    duplicate_groups = [items for items in grouped.values() if len(items) > 1]
    duplicate_groups.sort(key=len, reverse=True)
    for items in duplicate_groups[: max(args.top_duplicate_groups, 0)]:
        chosen = selected_by_group[items[0].group_key]
        duplicate_group_rows.append(
            {
                "group_key": items[0].group_key,
                "count": len(items),
                "chosen": {
                    "client_label": chosen.client_label,
                    "map_name": chosen.map_name,
                    "tile_name": chosen.tile_name,
                    "dataset_map_root": chosen.dataset_map_root,
                    "score": round(chosen.score, 4),
                },
                "dropped": [
                    {
                        "client_label": item.client_label,
                        "map_name": item.map_name,
                        "tile_name": item.tile_name,
                        "dataset_map_root": item.dataset_map_root,
                        "score": round(item.score, 4),
                    }
                    for item in sorted(items, key=lambda row: row.score, reverse=True)
                    if item != chosen
                ],
            }
        )

    map_plan = build_map_plan(selected_items)

    manifest = {
        "schema_version": "ml-minimal-manifest.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "datasets_root": datasets_root.as_posix(),
        "manifests_scanned": scanned_manifests,
        "tiles_scanned": len(candidates),
        "unique_groups": len(selected_by_group),
        "duplicates_removed": len(candidates) - len(selected_by_group),
        "selection_policy": {
            "group_key": "source_minimap_signature.sha256 + alpha_atlas_signature.sha256 + shadow_map_signature.sha256",
            "fallback_group_key": "map_name + tile_name when source signature is unavailable",
            "score_dimensions": [
                "source_minimap_exists",
                "heightmap_global_exists",
                "heightmap_local_exists",
                "alpha_atlas_exists",
                "existing_shadow_map_count",
                "existing_alpha_mask_count",
                "object_count",
                "chunk_layer_count",
                "completeness_class",
            ],
        },
        "selected_tiles": [
            {
                "client_label": item.client_label,
                "map_name": item.map_name,
                "tile_name": item.tile_name,
                "dataset_map_root": item.dataset_map_root,
                "tile_json_path": item.tile_json_path,
                "source_minimap_path": item.source_minimap_path,
                "score": round(item.score, 4),
                "completeness_class": item.completeness_class,
                "source_sha256": item.source_sha256,
                "alpha_sha256": item.alpha_sha256,
                "shadow_sha256": item.shadow_sha256,
            }
            for item in selected_items
        ],
        "duplicate_group_samples": duplicate_group_rows,
        "failures": failures,
    }

    export_plan = {
        "schema_version": "ml-minimal-export-plan.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "datasets_root": datasets_root.as_posix(),
        "notes": "Client map lists derived from selected minimal tile set. Populate client paths per machine before use.",
        "clients": [
            {
                "label": label,
                "maps": maps,
            }
            for label, maps in map_plan.items()
        ],
    }

    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    plan_output_path.write_text(json.dumps(export_plan, indent=2), encoding="utf-8")

    print(f"Scanned manifests: {scanned_manifests}")
    print(f"Tiles scanned: {len(candidates)}")
    print(f"Unique groups: {len(selected_by_group)}")
    print(f"Duplicates removed: {len(candidates) - len(selected_by_group)}")
    print(f"Minimal manifest: {output_path.as_posix()}")
    print(f"Minimal export plan: {plan_output_path.as_posix()}")

    if failures:
        print(f"Warnings: {len(failures)} manifests failed to parse")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
