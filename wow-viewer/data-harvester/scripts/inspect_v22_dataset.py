"""Inspect a built V22 Zarr dataset.

Usage::

    uv run python scripts/inspect_v22_dataset.py summary \
        --store ../output/datasets/v22/3_3_5_12340_smoke.zarr

    uv run python scripts/inspect_v22_dataset.py tile \
        --store ../output/datasets/v22/3_3_5_12340_smoke.zarr \
        --tile-index 0 \
        --output-json ../output/tmp/v22_tile_0.json

This is the human-readable inspection surface for Spec 086. It reads the
built Zarr store and emits store-level summaries plus single-tile JSON with
metadata and per-array stats.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import numpy as np
import zarr
import zarr.storage

DATA_HARVESTER_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DATA_HARVESTER_ROOT / "src"))

from harvester.v22_zarr_io import (  # noqa: E402
    V22Dataset,
    V22_FLAT_SPECS,
    V22_PER_TILE_SPECS,
)


SAMPLE_TILE_LIMIT: Final[int] = 5


@dataclass(frozen=True, slots=True)
class ArrayLayout:
    """Store-level shape and dtype for a named array."""

    name: str
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True, slots=True)
class ArrayStats:
    """Human-readable summary for one tile array."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    nonzero_count: int
    min_value: float | int | None
    max_value: float | int | None
    mean_value: float | None


@dataclass(frozen=True, slots=True)
class TileRef:
    """Minimal tile locator emitted in store summaries."""

    tile_id: int
    build: str
    map: str
    tile_x: int
    tile_y: int


@dataclass(frozen=True, slots=True)
class StoreSummary:
    """Top-level inspection summary for a V22 store."""

    path: str
    tile_count: int
    builds: tuple[str, ...]
    root_arrays: tuple[ArrayLayout, ...]
    flat_arrays: tuple[ArrayLayout, ...]
    model_count: int
    tileset_count: int
    sample_tiles: tuple[TileRef, ...]


@dataclass(frozen=True, slots=True)
class TileSummary:
    """Human-readable tile metadata and array stats."""

    tile_index: int
    tile_id: int
    build: str
    map: str
    tile_x: int
    tile_y: int
    mddf_count: int
    modf_count: int
    mtex_texture_paths: tuple[str, ...]
    placement_mddf_asset_paths: tuple[str, ...]
    placement_modf_asset_paths: tuple[str, ...]
    arrays: tuple[ArrayStats, ...]


def summarize_store(store_path: Path) -> StoreSummary:
    """Return a human-readable summary of a V22 Zarr store."""
    root = zarr.open_group(zarr.storage.LocalStore(str(store_path), read_only=True), mode="r")
    tile_index = list(root.attrs.get("tile_index", []))

    root_arrays = tuple(
        ArrayLayout(name=spec.name, shape=tuple(root[spec.name].shape), dtype=str(root[spec.name].dtype))
        for spec in V22_PER_TILE_SPECS
        if spec.name in root
    )
    flat_arrays = tuple(
        ArrayLayout(name=spec.name, shape=tuple(root[spec.name].shape), dtype=str(root[spec.name].dtype))
        for spec in V22_FLAT_SPECS
        if spec.name in root
    )

    sample_tiles = tuple(
        TileRef(
            tile_id=int(row.get("tile_id", index)),
            build=str(row.get("build", "")),
            map=str(row.get("map", "")),
            tile_x=int(row.get("tile_x", 0)),
            tile_y=int(row.get("tile_y", 0)),
        )
        for index, row in enumerate(tile_index[:SAMPLE_TILE_LIMIT])
    )

    return StoreSummary(
        path=str(store_path),
        tile_count=int(root.attrs.get("tile_count", 0)),
        builds=tuple(str(value) for value in root.attrs.get("builds", [])),
        root_arrays=root_arrays,
        flat_arrays=flat_arrays,
        model_count=int(root["models/model_paths"].shape[0]) if "models" in root else 0,
        tileset_count=int(root["tilesets/tileset_paths"].shape[0]) if "tilesets" in root else 0,
        sample_tiles=sample_tiles,
    )


def summarize_tile(store_path: Path, *, tile_index: int | None, tile_id: int | None) -> TileSummary:
    """Return human-readable metadata and array stats for one V22 tile."""
    dataset = V22Dataset(store_path)
    resolved_index = _resolve_tile_index(dataset, tile_index=tile_index, tile_id=tile_id)
    sample = dataset[resolved_index]
    arrays = tuple(_summarize_array(name, sample[name]) for name in _iter_array_names(sample))

    return TileSummary(
        tile_index=resolved_index,
        tile_id=int(np.asarray(sample["tile_id"]).item()),
        build=str(sample["build"]),
        map=str(sample["map"]),
        tile_x=int(np.asarray(sample["tile_x"]).item()),
        tile_y=int(np.asarray(sample["tile_y"]).item()),
        mddf_count=int(np.asarray(sample["mddf_count"]).reshape(-1)[0]),
        modf_count=int(np.asarray(sample["modf_count"]).reshape(-1)[0]),
        mtex_texture_paths=tuple(str(value) for value in sample.get("mtex_texture_paths", [])),
        placement_mddf_asset_paths=tuple(str(value) for value in sample.get("placement_mddf_asset_paths", [])),
        placement_modf_asset_paths=tuple(str(value) for value in sample.get("placement_modf_asset_paths", [])),
        arrays=arrays,
    )


def _resolve_tile_index(dataset: V22Dataset, *, tile_index: int | None, tile_id: int | None) -> int:
    if tile_index is not None:
        return tile_index
    if tile_id is None:
        raise ValueError("either tile_index or tile_id is required")

    tile_ids = dataset.tile_ids()
    matches = np.nonzero(tile_ids == tile_id)[0]
    if matches.size == 0:
        raise ValueError(f"tile_id {tile_id} not found")
    return int(matches[0])


def _iter_array_names(sample: dict[str, np.ndarray | str | list[str]]) -> tuple[str, ...]:
    names: list[str] = []
    for key, value in sample.items():
        if isinstance(value, np.ndarray):
            names.append(key)
    return tuple(sorted(names, key=str.casefold))


def _summarize_array(name: str, array: np.ndarray) -> ArrayStats:
    nonzero_count = int(np.count_nonzero(array))
    min_value, max_value, mean_value = _numeric_stats(array)
    return ArrayStats(
        name=name,
        shape=tuple(int(dim) for dim in array.shape),
        dtype=str(array.dtype),
        nonzero_count=nonzero_count,
        min_value=min_value,
        max_value=max_value,
        mean_value=mean_value,
    )


def _numeric_stats(array: np.ndarray) -> tuple[float | int | None, float | int | None, float | None]:
    if array.size == 0:
        return None, None, None

    if np.issubdtype(array.dtype, np.bool_):
        true_count = int(np.count_nonzero(array))
        return 0, 1, float(true_count / array.size)

    if np.issubdtype(array.dtype, np.integer):
        return int(array.min()), int(array.max()), float(array.mean())

    if np.issubdtype(array.dtype, np.floating):
        return float(array.min()), float(array.max()), float(array.mean())

    return None, None, None


def _write_or_print(payload: StoreSummary | TileSummary, output_json: Path | None) -> int:
    text = json.dumps(asdict(payload), indent=2)
    if output_json is None:
        print(text)
        return 0

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(text + "\n", encoding="utf-8")
    print(f"wrote {output_json}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a V22 Zarr store")
    sub = parser.add_subparsers(dest="command", required=True)

    summary = sub.add_parser("summary", help="Print a store-level JSON summary")
    summary.add_argument("--store", required=True, type=Path)
    summary.add_argument("--output-json", type=Path)

    tile = sub.add_parser("tile", help="Print one tile's metadata and array stats")
    tile.add_argument("--store", required=True, type=Path)
    tile.add_argument("--tile-index", type=int)
    tile.add_argument("--tile-id", type=int)
    tile.add_argument("--output-json", type=Path)

    args = parser.parse_args()
    match args.command:
        case "summary":
            return _write_or_print(summarize_store(args.store), args.output_json)
        case "tile":
            return _write_or_print(
                summarize_tile(args.store, tile_index=args.tile_index, tile_id=args.tile_id),
                args.output_json,
            )
        case unreachable:
            raise ValueError(f"unsupported command: {unreachable}")


if __name__ == "__main__":
    raise SystemExit(main())
