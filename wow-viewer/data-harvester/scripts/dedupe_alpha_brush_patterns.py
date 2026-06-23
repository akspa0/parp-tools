"""Deduplicate alpha-brush scar patterns and rank near-duplicates."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import zarr

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATASET_DIR = _PROJECT_ROOT / "output" / "datasets" / "v18"
_DEFAULT_CATALOG_DIR = _PROJECT_ROOT / "output" / "analysis" / "alpha-brush-library" / "two-build-full"


@dataclass(slots=True)
class PatternAccumulator:
    exact_hash: str
    shape_hw: tuple[int, int]
    member_count: int = 0
    embedding_sum: np.ndarray | None = None
    build_counts: dict[str, int] = field(default_factory=dict)
    map_counts: dict[str, int] = field(default_factory=dict)
    layer_counts: dict[int, int] = field(default_factory=dict)
    cluster_counts: dict[int, int] = field(default_factory=dict)
    area_min: int = 0
    area_max: int = 0
    area_sum: int = 0
    examples: list[dict] = field(default_factory=list)


class ZarrCache:
    def __init__(self, dataset_dir: Path) -> None:
        self.dataset_dir = dataset_dir
        self._stores: dict[str, zarr.storage.LocalStore] = {}
        self._roots: dict[str, zarr.Group] = {}
        self._tile_cache: dict[tuple[str, int], np.ndarray] = {}
        self._tile_order: list[tuple[str, int]] = []
        self.max_tiles = 16

    def alpha(self, build: str):
        if build not in self._roots:
            zarr_path = self.dataset_dir / f"{build}.zarr"
            if not zarr_path.exists():
                raise FileNotFoundError(f"Missing Zarr store: {zarr_path}")
            store = zarr.storage.LocalStore(str(zarr_path), read_only=True)
            self._stores[build] = store
            self._roots[build] = zarr.open_group(store=store, mode="r")
        return self._roots[build]["alpha_256"]

    def tile(self, build: str, tile_id: int) -> np.ndarray:
        key = (build, int(tile_id))
        if key in self._tile_cache:
            return self._tile_cache[key]
        tile = np.asarray(self.alpha(build)[int(tile_id)], dtype=np.float32)
        self._tile_cache[key] = tile
        self._tile_order.append(key)
        while len(self._tile_order) > self.max_tiles:
            old = self._tile_order.pop(0)
            self._tile_cache.pop(old, None)
        return tile

    def close(self) -> None:
        for store in self._stores.values():
            store.close()
        self._stores.clear()
        self._roots.clear()
        self._tile_cache.clear()
        self._tile_order.clear()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deduplicate exact alpha scar patterns, then rank near-duplicates by embedding similarity."
    )
    parser.add_argument("--catalog-dir", type=Path, default=_DEFAULT_CATALOG_DIR)
    parser.add_argument("--dataset-dir", type=Path, default=_DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--hash-mode", choices=["binary", "u8"], default="binary")
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--examples-per-pattern", type=int, default=8)
    parser.add_argument("--min-members", type=int, default=1)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _dominant(counts: dict) -> object | None:
    if not counts:
        return None
    return sorted(counts.items(), key=lambda item: (-item[1], str(item[0])))[0][0]


def _component_mask_hash(cache: ZarrCache, row: dict, hash_mode: str) -> tuple[str, tuple[int, int]]:
    build = str(row.get("build", ""))
    tile_id = int(row.get("tile_id", -1))
    layer_idx = int(row.get("layer_idx", 0))
    threshold = float(row.get("threshold", 0.05))
    x, y, w, h = [int(value) for value in row.get("bbox_xywh", [0, 0, 1, 1])]

    tile = cache.tile(build, tile_id)
    layer = np.clip(tile[:, :, layer_idx], 0.0, 1.0)
    crop = layer[max(0, y) : min(layer.shape[0], y + h), max(0, x) : min(layer.shape[1], x + w)]
    if crop.size == 0:
        crop = np.zeros((1, 1), dtype=np.float32)
    if hash_mode == "binary":
        payload_arr = np.packbits((crop > threshold).astype(np.uint8, copy=False))
    else:
        payload_arr = np.clip(np.rint(crop * 255.0), 0, 255).astype(np.uint8, copy=False)
    shape_hw = (int(crop.shape[0]), int(crop.shape[1]))
    header = f"{hash_mode}|{shape_hw[0]}x{shape_hw[1]}|{threshold:.5f}|".encode("ascii")
    digest = hashlib.sha256(header + payload_arr.tobytes()).hexdigest()
    return digest, shape_hw


def _add_count(counts: dict, key: object) -> None:
    counts[key] = int(counts.get(key, 0) + 1)


def _add_component(
    patterns: dict[str, PatternAccumulator],
    exact_hash: str,
    shape_hw: tuple[int, int],
    row: dict,
    examples_per_pattern: int,
) -> None:
    embedding = np.asarray(row.get("embedding", []), dtype=np.float32)
    if embedding.size == 0:
        raise ValueError(f"Component {row.get('component_id')} has no embedding")

    pattern = patterns.get(exact_hash)
    if pattern is None:
        pattern = PatternAccumulator(exact_hash=exact_hash, shape_hw=shape_hw)
        patterns[exact_hash] = pattern

    pattern.member_count += 1
    pattern.embedding_sum = embedding.copy() if pattern.embedding_sum is None else pattern.embedding_sum + embedding
    build = str(row.get("build", ""))
    map_name = str(row.get("map_name", row.get("map", "")))
    layer_idx = int(row.get("layer_idx", -1))
    cluster_id = int(row.get("cluster_id", -1))
    area = int(row.get("area", 0))
    _add_count(pattern.build_counts, build)
    _add_count(pattern.map_counts, map_name)
    _add_count(pattern.layer_counts, layer_idx)
    _add_count(pattern.cluster_counts, cluster_id)
    pattern.area_sum += area
    pattern.area_min = area if pattern.member_count == 1 else min(pattern.area_min, area)
    pattern.area_max = area if pattern.member_count == 1 else max(pattern.area_max, area)
    if len(pattern.examples) < int(examples_per_pattern):
        pattern.examples.append(
            {
                "component_id": row.get("component_id"),
                "build": build,
                "map_name": map_name,
                "tile_id": int(row.get("tile_id", -1)),
                "tile_x": int(row.get("tile_x", -1)),
                "tile_y": int(row.get("tile_y", -1)),
                "layer_idx": layer_idx,
                "bbox_xywh": row.get("bbox_xywh", [0, 0, 1, 1]),
                "area": area,
                "threshold": float(row.get("threshold", 0.05)),
                "cluster_id": cluster_id,
            }
        )


def _pattern_id(exact_hash: str) -> str:
    return f"scar_{exact_hash[:16]}"


def _centroid(pattern: PatternAccumulator) -> np.ndarray:
    if pattern.embedding_sum is None:
        raise ValueError(f"Pattern {pattern.exact_hash} has no embeddings")
    vec = pattern.embedding_sum / max(1, pattern.member_count)
    norm = float(np.linalg.norm(vec))
    return vec / max(norm, 1e-8)


def _pattern_row(pattern: PatternAccumulator) -> dict:
    dominant_cluster = _dominant(pattern.cluster_counts)
    return {
        "pattern_id": _pattern_id(pattern.exact_hash),
        "exact_hash": pattern.exact_hash,
        "member_count": int(pattern.member_count),
        "shape_hw": [int(pattern.shape_hw[0]), int(pattern.shape_hw[1])],
        "dominant_cluster_id": int(dominant_cluster) if dominant_cluster is not None else None,
        "dominant_build": _dominant(pattern.build_counts),
        "dominant_map": _dominant(pattern.map_counts),
        "dominant_layer": _dominant(pattern.layer_counts),
        "area_min": int(pattern.area_min),
        "area_max": int(pattern.area_max),
        "area_mean": float(pattern.area_sum / max(1, pattern.member_count)),
        "build_counts": dict(sorted(pattern.build_counts.items())),
        "map_counts": dict(sorted(pattern.map_counts.items())),
        "layer_counts": {str(key): value for key, value in sorted(pattern.layer_counts.items())},
        "cluster_counts": {str(key): value for key, value in sorted(pattern.cluster_counts.items())},
        "examples": pattern.examples,
    }


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _neighbor_rows(patterns: list[PatternAccumulator], neighbors: int) -> list[dict]:
    by_cluster: dict[int, list[PatternAccumulator]] = {}
    for pattern in patterns:
        dominant_cluster = _dominant(pattern.cluster_counts)
        if dominant_cluster is None:
            continue
        by_cluster.setdefault(int(dominant_cluster), []).append(pattern)

    rows: list[dict] = []
    for cluster_id, cluster_patterns in sorted(by_cluster.items()):
        if len(cluster_patterns) <= 1:
            continue
        vectors = np.stack([_centroid(pattern) for pattern in cluster_patterns], axis=0).astype(np.float32)
        sims = vectors @ vectors.T
        np.fill_diagonal(sims, -1.0)
        for idx, pattern in enumerate(cluster_patterns):
            order = np.argsort(-sims[idx])[: int(neighbors)]
            for rank, neighbor_idx in enumerate(order, start=1):
                score = float(sims[idx, neighbor_idx])
                if score < -0.5:
                    continue
                neighbor = cluster_patterns[int(neighbor_idx)]
                rows.append(
                    {
                        "pattern_id": _pattern_id(pattern.exact_hash),
                        "neighbor_pattern_id": _pattern_id(neighbor.exact_hash),
                        "cluster_id": int(cluster_id),
                        "rank": int(rank),
                        "cosine_similarity": score,
                        "pattern_members": int(pattern.member_count),
                        "neighbor_members": int(neighbor.member_count),
                    }
                )
    return rows


def main() -> None:
    args = _parse_args()
    catalog_dir = Path(args.catalog_dir)
    output_dir = Path(args.output_dir) if args.output_dir is not None else catalog_dir / "dedupe"
    output_dir.mkdir(parents=True, exist_ok=True)

    patterns: dict[str, PatternAccumulator] = {}
    cache = ZarrCache(Path(args.dataset_dir))
    row_count = 0
    try:
        for row in _read_jsonl(catalog_dir / "components.jsonl"):
            exact_hash, shape_hw = _component_mask_hash(cache, row, str(args.hash_mode))
            _add_component(patterns, exact_hash, shape_hw, row, int(args.examples_per_pattern))
            row_count += 1
            if row_count == 1 or row_count % 10000 == 0:
                print(f"Processed {row_count} components; exact_patterns={len(patterns)}", flush=True)
            if args.max_rows is not None and row_count >= int(args.max_rows):
                break
    finally:
        cache.close()

    kept = [pattern for pattern in patterns.values() if pattern.member_count >= int(args.min_members)]
    kept.sort(key=lambda pattern: (-pattern.member_count, str(_dominant(pattern.map_counts)), pattern.exact_hash))
    pattern_rows = [_pattern_row(pattern) for pattern in kept]
    neighbor_rows = _neighbor_rows(kept, int(args.neighbors))

    _write_jsonl(output_dir / "exact_patterns.jsonl", pattern_rows)
    _write_jsonl(output_dir / "pattern_neighbors.jsonl", neighbor_rows)
    summary = {
        "catalog_dir": str(catalog_dir),
        "dataset_dir": str(args.dataset_dir),
        "output_dir": str(output_dir),
        "hash_mode": str(args.hash_mode),
        "component_rows_processed": int(row_count),
        "exact_pattern_count": int(len(patterns)),
        "kept_pattern_count": int(len(kept)),
        "neighbor_rows": int(len(neighbor_rows)),
        "largest_exact_pattern_members": int(max((pattern.member_count for pattern in kept), default=0)),
    }
    (output_dir / "dedupe_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
