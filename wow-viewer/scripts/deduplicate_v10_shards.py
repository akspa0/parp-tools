#!/usr/bin/env python3
"""
deduplicate_v10_shards.py — Pattern-fingerprint deduplication for v10 training shards.

Uses the Wave 2 pattern dictionaries (MCLY, height profiles, MCAL compositions,
prefab cells, MCAL brushes) to build composite fingerprints for each tile, then
groups tiles by fingerprint and keeps only the best representative per group.

Usage:
    python wow-viewer/scripts/deduplicate_v10_shards.py \
        output/ml-training/v10_curated/v10_full_corpus_slim_pattern_manifest.json \
        --output output/ml-training/v10_curated/v10_deduplicated_manifest.json

    python wow-viewer/scripts/deduplicate_v10_shards.py \
        output/ml-training/v10_curated/v10_full_corpus_compact_pattern_manifest.json \
        --output output/ml-training/v10_curated/v10_deduplicated_compact_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


# Default Wave 2 pattern dictionary paths
DEFAULT_PATTERN_DICTIONARIES = (
    Path("output/build-validation/v10-wave2-mcly-dictionary/mclay_dictionary.json"),
    Path("output/build-validation/v10-wave2-mcly-dictionary/mcly_dictionary.json"),
    Path("output/build-validation/v10-wave2-height-profiles/height_profile_dictionary.json"),
    Path("output/build-validation/v10-wave2-mcal-compositions/mcal_composition_dictionary.json"),
    Path("output/build-validation/v10-wave2-mcal-brushes/mcal_brush_dictionary.json"),
    Path("output/build-validation/v10-wave2-prefab-cells-large/8x8/prefab_cell_dictionary.json"),
    Path("output/build-validation/v10-wave2-prefab-cells-large/12x12/prefab_cell_dictionary.json"),
    Path("output/build-validation/v10-wave2-prefab-cells-large/16x16/prefab_cell_dictionary.json"),
)

# Number of quantisation buckets for continuous stats used in v9 fingerprints
STAT_BUCKETS = 20


@dataclass(slots=True)
class PatternIndex:
    """Reverse index: tile_key -> pattern categories and their identifiers."""

    mcly_hashes: set[str] = field(default_factory=set)
    mcly_combination_keys: set[str] = field(default_factory=set)
    height_profile_ids: set[int] = field(default_factory=set)
    mcal_composition_hashes: set[str] = field(default_factory=set)
    mcal_brush_ids: set[int] = field(default_factory=set)
    prefab_cell_ids: set[int] = field(default_factory=set)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deduplicate v10 training shards using Wave 2 pattern fingerprints."
    )
    parser.add_argument(
        "input",
        help="Curated manifest JSON file (v10-training-curated-manifest.v1) to deduplicate.",
    )
    parser.add_argument("--output", required=True, help="Output deduplicated manifest path.")
    parser.add_argument(
        "--pattern-dictionary",
        action="append",
        default=[],
        help="Additional pattern dictionary JSON paths. May be repeated.",
    )
    parser.add_argument(
        "--no-default-pattern-dictionaries",
        action="store_true",
        help="Do not load the default Wave 2 pattern dictionary paths.",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=2,
        help="Minimum cluster size to report as a duplicate group (default: 2).",
    )
    parser.add_argument(
        "--max-per-fingerprint",
        type=int,
        default=1,
        help="Maximum tiles to keep per fingerprint group (default: 1 = one per group).",
    )
    parser.add_argument(
        "--native-v10-priority",
        action="store_true",
        default=True,
        help="When a fingerprint group mixes native v10 and legacy tiles, prefer native v10 (default: true).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for tie-breaking (default: 1337).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    # 1. Load the curated manifest
    manifest = json.loads(input_path.read_text(encoding="utf-8"))
    entries: list[dict[str, Any]] = manifest.get("entries", [])
    if not entries:
        print("ERROR: No entries found in manifest.")
        sys.exit(1)

    print(f"Loaded {len(entries)} entries from {input_path.name}")

    # 2. Load pattern dictionaries and build tile-to-pattern index
    pattern_sources = _resolve_pattern_sources(args)
    tile_patterns, pattern_stats = _build_pattern_index(pattern_sources)
    print(f"Loaded {len(pattern_sources)} pattern dictionaries")
    print(f"  Pattern-indexed tiles: {len(tile_patterns)}")
    for category, count in sorted(pattern_stats.items()):
        print(f"    {category}: {count} entries")

    # 3. Compute fingerprints and group entries
    fingerprint_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    native_v10_count = 0
    legacy_count = 0

    for entry in entries:
        tile_key = _tile_key(entry)
        patterns = tile_patterns.get(tile_key, None)
        fingerprint = _compute_fingerprint(entry, patterns)
        fingerprint_groups[fingerprint].append(entry)
        if patterns is not None:
            native_v10_count += 1
        else:
            legacy_count += 1

    print(f"\nNative v10 tiles (with pattern data): {native_v10_count}")
    print(f"Legacy v9 tiles (without pattern data): {legacy_count}")
    print(f"Total fingerprint groups: {len(fingerprint_groups)}")
    print(f"Unique groups (size 1): {sum(1 for g in fingerprint_groups.values() if len(g) == 1)}")
    print(f"Duplicate groups (size >= {args.min_cluster_size}): {sum(1 for g in fingerprint_groups.values() if len(g) >= args.min_cluster_size)}")

    # 4. Select representatives from each group
    selected: list[dict[str, Any]] = []
    removed_count = 0
    duplicate_report: list[dict[str, Any]] = []

    for fingerprint, group in fingerprint_groups.items():
        if len(group) <= args.max_per_fingerprint:
            selected.extend(group)
            continue

        # Sort group: native v10 first (richer signals), then by quality_score descending
        def _sort_key(item: dict[str, Any]) -> tuple:
            has_patterns = 1 if tile_patterns.get(_tile_key(item)) is not None else 0
            quality = float(item.get("quality_score", 0.0))
            return (-has_patterns, -quality, item.get("tile_name", ""))

        group.sort(key=_sort_key)
        keep = group[:args.max_per_fingerprint]
        removed = group[args.max_per_fingerprint:]

        selected.extend(keep)
        removed_count += len(removed)

        duplicate_report.append({
            "fingerprint": fingerprint,
            "group_size": len(group),
            "kept": [{"tile_name": e.get("tile_name", ""), "dataset_key": e.get("dataset_key", ""),
                       "quality_score": e.get("quality_score", 0), "has_pattern_data": tile_patterns.get(_tile_key(e)) is not None}
                      for e in keep],
            "removed": [{"tile_name": e.get("tile_name", ""), "dataset_key": e.get("dataset_key", ""),
                         "quality_score": e.get("quality_score", 0), "has_pattern_data": tile_patterns.get(_tile_key(e)) is not None}
                        for e in removed],
        })

    # 5. Sort selected entries for consistent output
    selected.sort(key=lambda item: (item.get("dataset_key", ""), item.get("tile_name", "")))

    # 6. Write deduplicated manifest
    dedup_manifest = dict(manifest)  # shallow copy
    dedup_manifest["schema_version"] = "v10-training-deduplicated-manifest.v1"
    dedup_manifest["entries"] = selected
    dedup_manifest["deduplication"] = {
        "input_entry_count": len(entries),
        "output_entry_count": len(selected),
        "removed_count": removed_count,
        "fingerprint_group_count": len(fingerprint_groups),
        "unique_group_count": sum(1 for g in fingerprint_groups.values() if len(g) == 1),
        "duplicate_group_count": sum(1 for g in fingerprint_groups.values() if len(g) >= args.min_cluster_size),
        "native_v10_count": native_v10_count,
        "legacy_count": legacy_count,
        "pattern_dictionary_sources": [str(p) for p in pattern_sources],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(dedup_manifest, indent=2), encoding="utf-8")

    # 7. Write duplicate report
    report_path = output_path.with_name(output_path.stem + "_duplicate_report.json")
    report = {
        "schema_version": "v10-deduplication-report.v1",
        "input_entry_count": len(entries),
        "output_entry_count": len(selected),
        "removed_count": removed_count,
        "fingerprint_group_count": len(fingerprint_groups),
        "duplicate_groups": duplicate_report,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nDeduplication complete:")
    print(f"  Input:  {len(entries)} entries")
    print(f"  Output: {len(selected)} entries")
    print(f"  Removed: {removed_count} duplicates")
    print(f"  Manifest: {output_path}")
    print(f"  Report: {report_path}")


def _resolve_pattern_sources(args: argparse.Namespace) -> list[Path]:
    """Resolve the list of pattern dictionary paths to load."""
    sources: list[Path] = []
    if not args.no_default_pattern_dictionaries:
        for path in DEFAULT_PATTERN_DICTIONARIES:
            resolved = Path(path).resolve()
            if resolved.is_file():
                sources.append(resolved)
    for path_str in args.pattern_dictionary:
        resolved = Path(path_str).resolve()
        if resolved.is_file():
            sources.append(resolved)
    return sources


def _build_pattern_index(
    pattern_sources: list[Path],
) -> tuple[dict[str, PatternIndex], Counter[str]]:
    """Build a reverse index mapping tile keys to their pattern memberships."""
    index: dict[str, PatternIndex] = defaultdict(PatternIndex)
    stats: Counter[str] = Counter()

    for path in pattern_sources:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"  WARNING: Could not read {path.name}: {exc}")
            continue

        schema = str(payload.get("schema_version", "")).lower()
        name = path.name.lower()

        if "mcly" in schema or "mclay" in name:
            _index_mcly(payload, index)
            stats["mcly"] += 1
        elif "height-profile" in schema or "height_profile" in name:
            _index_height_profiles(payload, index)
            stats["height_profile"] += 1
        elif "mcal-composition" in schema or "mcal_composition" in name:
            _index_mcal_compositions(payload, index)
            stats["mcal_composition"] += 1
        elif "mcal-brush" in schema or "mcal_brush" in name:
            _index_mcal_brushes(payload, index)
            stats["mcal_brush"] += 1
        elif "prefab-cell" in schema or "prefab_cell" in name:
            _index_prefab_cells(payload, index)
            stats["prefab_cell"] += 1
        elif "brush_dictionary" in name:
            _index_anchor_brushes(payload, index)
            stats["anchor_brush"] += 1
        else:
            # Try generic extraction
            _index_generic(payload, index)
            stats["generic"] += 1

    return dict(index), stats


def _index_mcly(payload: dict[str, Any], index: dict[str, PatternIndex]) -> None:
    """Index MCLY dictionary entries by tile name."""
    dictionary = payload.get("dictionary", [])
    if not isinstance(dictionary, list):
        return
    for entry in dictionary:
        if not isinstance(entry, dict):
            continue
        combo_hash = str(entry.get("combination_hash", ""))
        combo_key = str(entry.get("combination_key", ""))
        example_chunks = entry.get("example_chunks", [])
        if isinstance(example_chunks, list):
            for chunk in example_chunks:
                tile_name = _normalize_tile_name(chunk.get("tile_name", ""))
                if tile_name:
                    if combo_hash:
                        index[tile_name].mcly_hashes.add(combo_hash)
                    if combo_key:
                        index[tile_name].mcly_combination_keys.add(combo_key)


def _index_height_profiles(payload: dict[str, Any], index: dict[str, PatternIndex]) -> None:
    """Index height profile dictionary entries by tile name."""
    dictionary = payload.get("dictionary", [])
    if not isinstance(dictionary, list):
        return
    for entry in dictionary:
        if not isinstance(entry, dict):
            continue
        profile_id = entry.get("profile_id")
        if profile_id is None:
            continue
        examples = entry.get("examples", [])
        if isinstance(examples, list):
            for example in examples:
                tile_name = _normalize_tile_name(example.get("TileName", ""))
                if tile_name:
                    index[tile_name].height_profile_ids.add(int(profile_id))


def _index_mcal_compositions(payload: dict[str, Any], index: dict[str, PatternIndex]) -> None:
    """Index MCAL composition dictionary entries by tile name."""
    dictionary = payload.get("dictionary", [])
    if not isinstance(dictionary, list):
        return
    for entry in dictionary:
        if not isinstance(entry, dict):
            continue
        comp_hash = str(entry.get("composition_hash", ""))
        example_chunks = entry.get("example_chunks", [])
        if isinstance(example_chunks, list):
            for chunk in example_chunks:
                tile_name = _normalize_tile_name(chunk.get("tile_name", ""))
                if tile_name and comp_hash:
                    index[tile_name].mcal_composition_hashes.add(comp_hash)


def _index_mcal_brushes(payload: dict[str, Any], index: dict[str, PatternIndex]) -> None:
    """Index MCAL brush dictionary entries by tile name."""
    dictionary = payload.get("dictionary", [])
    if not isinstance(dictionary, list):
        return
    for entry in dictionary:
        if not isinstance(entry, dict):
            continue
        brush_id = entry.get("brush_id")
        if brush_id is None:
            continue
        example_chunks = entry.get("example_chunks", [])
        if isinstance(example_chunks, list):
            for chunk in example_chunks:
                tile_name = _normalize_tile_name(chunk.get("tile_name", ""))
                if tile_name:
                    index[tile_name].mcal_brush_ids.add(int(brush_id))


def _index_prefab_cells(payload: dict[str, Any], index: dict[str, PatternIndex]) -> None:
    """Index prefab cell dictionary entries by tile name."""
    dictionary = payload.get("dictionary", [])
    if not isinstance(dictionary, list):
        return
    for entry in dictionary:
        if not isinstance(entry, dict):
            continue
        cell_id = entry.get("cell_id")
        if cell_id is None:
            continue
        examples = entry.get("examples", [])
        if isinstance(examples, list):
            for example in examples:
                tile_name = _normalize_tile_name(example.get("tile_name", ""))
                if tile_name:
                    index[tile_name].prefab_cell_ids.add(int(cell_id))


def _index_anchor_brushes(payload: dict[str, Any], index: dict[str, PatternIndex]) -> None:
    """Index anchor brush dictionary entries by tile name."""
    dictionary = payload.get("dictionary", [])
    if not isinstance(dictionary, list):
        return
    for entry in dictionary:
        if not isinstance(entry, dict):
            continue
        examples = entry.get("examples", [])
        if isinstance(examples, list):
            for example in examples:
                tile_name = _normalize_tile_name(
                    example.get("tile_name", "") or example.get("TileName", "")
                )
                if tile_name:
                    # Anchor brushes don't have a separate ID field we need,
                    # but we record the tile as having pattern data
                    pass


def _index_generic(payload: dict[str, Any], index: dict[str, PatternIndex]) -> None:
    """Generic tile name extraction for unknown dictionary formats."""
    for key in ("tile_name", "TileName"):
        value = payload.get(key)
        if isinstance(value, str):
            tile_name = _normalize_tile_name(value)
            if tile_name:
                index[tile_name]  # ensure entry exists


def _normalize_tile_name(value: str) -> str:
    """Normalize a tile name for consistent lookup."""
    result = value.strip().replace("\\", "/").split("/")[-1].lower()
    if result.endswith(".npz"):
        result = result[:-4]
    if result.endswith("_v10"):
        result = result[:-4]
    return result


def _tile_key(entry: dict[str, Any]) -> str:
    """Extract a normalized tile key from a manifest entry."""
    tile_name = str(entry.get("tile_name", ""))
    shard_path = str(entry.get("shard_path", ""))
    # Use the tile_name first, fall back to extracting from shard_path
    key = _normalize_tile_name(tile_name) if tile_name else ""
    if not key and shard_path:
        key = _normalize_tile_name(shard_path)
    return key


def _compute_fingerprint(
    entry: dict[str, Any],
    patterns: PatternIndex | None,
) -> str:
    """
    Compute a composite fingerprint for a manifest entry.

    For tiles with Wave 2 pattern data (native v10), the fingerprint captures
    the exact set of patterns present. For legacy tiles without pattern data,
    the fingerprint uses era + dataset + quantised stats.
    """
    if patterns is not None and (
        patterns.mcly_hashes
        or patterns.height_profile_ids
        or patterns.mcal_composition_hashes
        or patterns.prefab_cell_ids
        or patterns.mcal_brush_ids
    ):
        return _rich_fingerprint(entry, patterns)
    else:
        return _stat_fingerprint(entry)


def _rich_fingerprint(entry: dict[str, Any], patterns: PatternIndex) -> str:
    """Compute a fingerprint from Wave 2 pattern data."""
    h = hashlib.sha256()

    # MCLY combination hashes (sorted for consistency)
    for item in sorted(patterns.mcly_hashes):
        h.update(f"mcly:{item}\n".encode())

    # Height profile IDs (sorted)
    for item in sorted(patterns.height_profile_ids):
        h.update(f"hp:{item}\n".encode())

    # MCAL composition hashes (sorted)
    for item in sorted(patterns.mcal_composition_hashes):
        h.update(f"mcal_comp:{item}\n".encode())

    # Prefab cell IDs (sorted)
    for item in sorted(patterns.prefab_cell_ids):
        h.update(f"prefab:{item}\n".encode())

    # MCAL brush IDs (sorted)
    for item in sorted(patterns.mcal_brush_ids):
        h.update(f"mcal_brush:{item}\n".encode())

    # Include era and dataset for extra disambiguation
    era = str(entry.get("era_tag", "unknown"))
    dataset = str(entry.get("dataset_key", "unknown"))
    h.update(f"era:{era}\n".encode())
    h.update(f"dataset:{dataset}\n".encode())

    return h.hexdigest()


def _stat_fingerprint(entry: dict[str, Any]) -> str:
    """
    Compute a fingerprint from quantised statistics for legacy tiles
    that lack Wave 2 pattern data.
    """
    h = hashlib.sha256()

    era = str(entry.get("era_tag", "unknown"))
    dataset = str(entry.get("dataset_key", "unknown"))
    h.update(f"era:{era}\n".encode())
    h.update(f"dataset:{dataset}\n".encode())

    # Quantise continuous stats into buckets
    height_range = float(entry.get("height_range", 0.0))
    height_bucket = _quantise(height_range, 0.0, 1000.0, STAT_BUCKETS)
    h.update(f"height_bucket:{height_bucket}\n".encode())

    minimap_var = float(entry.get("minimap_variance", 0.0))
    var_bucket = _quantise(minimap_var, 0.0, 0.1, STAT_BUCKETS)
    h.update(f"var_bucket:{var_bucket}\n".encode())

    minimap_grad = float(entry.get("minimap_gradient", 0.0))
    grad_bucket = _quantise(minimap_grad, 0.0, 0.5, STAT_BUCKETS)
    h.update(f"grad_bucket:{grad_bucket}\n".encode())

    # Available signals as a coarse differentiator
    signals = sorted(entry.get("available_signals", []))
    for sig in signals:
        h.update(f"sig:{sig}\n".encode())

    return h.hexdigest()


def _quantise(value: float, lo: float, hi: float, buckets: int) -> int:
    """Quantise a float value into a bucket index in [lo, hi)."""
    if hi <= lo:
        return 0
    normalised = max(0.0, min(1.0, (value - lo) / (hi - lo)))
    return min(buckets - 1, int(math.floor(normalised * buckets)))


if __name__ == "__main__":
    main()