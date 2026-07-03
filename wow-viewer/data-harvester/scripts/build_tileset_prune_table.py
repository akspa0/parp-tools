"""Build a V23 tileset prune table from one or more V22 stores."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import zarr
import zarr.storage


def _default_output_path(dataset_dir: Path, builds: list[str] | None) -> Path:
    if builds and len(builds) == 1:
        suffix = builds[0]
    else:
        suffix = "v23_union"
    return dataset_dir / f"tileset_prune_{suffix}.json"


def _iter_build_names(dataset_dir: Path, builds: list[str] | None) -> list[str]:
    if builds:
        return builds
    return sorted(path.name.removesuffix(".zarr") for path in dataset_dir.glob("*.zarr") if path.is_dir())


def _count_tileset_ids(store_path: Path) -> Counter[int]:
    store = zarr.storage.LocalStore(str(store_path), read_only=True)
    root = zarr.open_group(store=store, mode="r")
    if "mcly_tileset_ids" not in root:
        raise ValueError(f"{store_path} is missing mcly_tileset_ids")
    if "tilesets" not in root or "tileset_paths" not in root["tilesets"]:
        raise ValueError(f"{store_path} is missing tilesets/tileset_paths")

    counts: Counter[int] = Counter()
    mcly_tileset_ids = root["mcly_tileset_ids"]
    for tile_idx in range(int(mcly_tileset_ids.shape[0])):
        tile_ids = np.asarray(mcly_tileset_ids[tile_idx], dtype=np.int32)
        valid_ids = tile_ids[tile_ids >= 0]
        counts.update(int(value) for value in valid_ids.reshape(-1))
    return counts


def build_prune_table(dataset_dir: Path, builds: list[str] | None, top_k: int) -> dict[str, object]:
    counts: Counter[int] = Counter()
    resolved_builds = _iter_build_names(dataset_dir, builds)
    for build in resolved_builds:
        counts.update(_count_tileset_ids(dataset_dir / f"{build}.zarr"))

    retained = [tileset_id for tileset_id, _ in counts.most_common(top_k)]
    mapping = {int(tileset_id): index for index, tileset_id in enumerate(retained)}
    return {
        "builds": resolved_builds,
        "top_k": int(top_k),
        "oov_index": int(top_k),
        "tileset_id_to_index": mapping,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--builds", nargs="*", default=None)
    parser.add_argument("--top-k", type=int, default=256)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    payload = build_prune_table(args.dataset_dir, args.builds, args.top_k)
    output_path = args.output or _default_output_path(args.dataset_dir, args.builds)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
