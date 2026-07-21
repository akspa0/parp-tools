"""Shared fixtures for Spec 116 tests.

Builds a tiny in-memory v50-style curriculum store (Zarr) plus an ``index.parquet`` and a
texture-name dump, so every story test can exercise the real read path without a client.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import zarr

CHUNKS = 16
MAX_LAYERS = 4


def write_texture_name_dump(path: Path, map_name: str, tiles: list[dict]) -> None:
    """Write a ``terrain-texture-name-dump-v1`` JSON file (the Spec 115 dump contract)."""
    payload = {
        "Format": "terrain-texture-name-dump-v1",
        "Map": map_name,
        "Tiles": tiles,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def build_store(
    store: Path,
    *,
    rows: list[dict],
) -> None:
    """Build a minimal v50 curriculum store with the arrays Spec 116 reads.

    Each row dict:
        map, tile_x, tile_y, split, source,
        texture_names: list[str]            # per-tile MTEX table (order is the contract)
        mcly_texture_ids: (16,16,4) int    # local MTEX index per chunk/slot
        mcly_layer_mask: (16,16,4) float    # coverage per chunk/slot
        alpha_256: (256,256,4) float        # MCAL alpha (optional; defaults zeros)
        height_257: (257,257) float         # (optional; defaults zeros)
    """
    n = len(rows)
    group = zarr.open_group(str(store), mode="w")
    group.attrs["schema"] = "v50-mixed-curriculum-v1"

    mcly_ids = np.full((n, CHUNKS, CHUNKS, MAX_LAYERS), -1, dtype=np.int32)
    mcly_mask = np.zeros((n, CHUNKS, CHUNKS, MAX_LAYERS), dtype=np.float32)
    alpha = np.zeros((n, 256, 256, MAX_LAYERS), dtype=np.float32)
    height = np.zeros((n, 257, 257), dtype=np.float32)
    minimap = np.zeros((n, 256, 256, 3), dtype=np.uint8)
    flags = np.zeros((n, CHUNKS, CHUNKS), dtype=np.int32)

    for i, row in enumerate(rows):
        mcly_ids[i] = np.asarray(row["mcly_texture_ids"], dtype=np.int32)
        mcly_mask[i] = np.asarray(row["mcly_layer_mask"], dtype=np.float32)
        if row.get("alpha_256") is not None:
            alpha[i] = np.asarray(row["alpha_256"], dtype=np.float32)
        if row.get("height_257") is not None:
            height[i] = np.asarray(row["height_257"], dtype=np.float32)
        minimap[i] = np.asarray(row.get("minimap_rgb", np.zeros((256, 256, 3), dtype=np.uint8)), dtype=np.uint8)

    group.create_array("mcly_texture_ids", data=mcly_ids)
    group.create_array("mcly_layer_mask", data=mcly_mask)
    group.create_array("alpha_256", data=alpha)
    group.create_array("height_257", data=height)
    group.create_array("minimap_rgb", data=minimap)
    group.create_array("mcnk_flags_16", data=flags)

    index_rows = [
        {
            "map": r["map"],
            "tile_x": r["tile_x"],
            "tile_y": r["tile_y"],
            "split": r.get("split", "train"),
            "source": r.get("source", "authored"),
        }
        for r in rows
    ]
    pq.write_table(pa.Table.from_pylist(index_rows), store / "index.parquet")


def _ids(layer0: int, layer1: int = -1, layer2: int = -1, layer3: int = -1) -> np.ndarray:
    ids = np.full((CHUNKS, CHUNKS, MAX_LAYERS), -1, dtype=np.int32)
    ids[:, :, 0] = layer0
    if layer1 >= 0:
        ids[:, :, 1] = layer1
    if layer2 >= 0:
        ids[:, :, 2] = layer2
    if layer3 >= 0:
        ids[:, :, 3] = layer3
    return ids


def _mask(slot1: bool = False, slot2: bool = False, slot3: bool = False) -> np.ndarray:
    m = np.zeros((CHUNKS, CHUNKS, MAX_LAYERS), dtype=np.float32)
    m[:, :, 0] = 1.0  # base always opaque
    if slot1:
        m[:, :, 1] = 1.0
    if slot2:
        m[:, :, 2] = 1.0
    if slot3:
        m[:, :, 3] = 1.0
    return m


@pytest.fixture
def consistent_store(tmp_path: Path) -> dict:
    """A store where one family always lands in one slot -> US1 should recommend slot_keyed.

    Tile A: slot0=grass(terrain), slot1=road.  Tile B: slot0=dirt(terrain), slot1=road.
    The 'road' family is always in slot 1; 'terrain' always in slot 0.
    """
    store = tmp_path / "consistent.zarr"
    store.mkdir()
    rows = [
        {
            "map": "Kalimdor", "tile_x": 24, "tile_y": 40, "split": "train", "source": "authored",
            "texture_names": [r"Tileset\X\XGrass.blp", r"Tileset\X\XRoad.blp"],
            "mcly_texture_ids": _ids(0, 1), "mcly_layer_mask": _mask(slot1=True),
        },
        {
            "map": "Kalimdor", "tile_x": 25, "tile_y": 40, "split": "train", "source": "authored",
            "texture_names": [r"Tileset\X\XDirt.blp", r"Tileset\X\XCobblestone.blp"],
            "mcly_texture_ids": _ids(0, 1), "mcly_layer_mask": _mask(slot1=True),
        },
    ]
    build_store(store, rows=rows)
    dump = tmp_path / "names.json"
    write_texture_name_dump(
        dump, "Kalimdor",
        [
            {"TileX": 24, "TileY": 40, "TextureNames": [r"Tileset\X\XGrass.blp", r"Tileset\X\XRoad.blp"]},
            {"TileX": 25, "TileY": 40, "TextureNames": [r"Tileset\X\XDirt.blp", r"Tileset\X\XCobblestone.blp"]},
        ],
    )
    return {"store": store, "dumps": [dump], "rows": rows}


@pytest.fixture
def spread_store(tmp_path: Path) -> dict:
    """A store where the same family spreads across all slots -> US1 should recommend family_keyed.

    One tile uses road in slot0, another uses road in slot1, another in slot2 -> 'road' has no
    consistent slot.
    """
    store = tmp_path / "spread.zarr"
    store.mkdir()
    rows = [
        {
            "map": "Azeroth", "tile_x": 10, "tile_y": 10, "split": "train", "source": "authored",
            "texture_names": [r"Tileset\X\XRoad.blp", r"Tileset\X\XGrass.blp"],
            "mcly_texture_ids": _ids(0, 1), "mcly_layer_mask": _mask(slot1=True),
        },
        {
            "map": "Azeroth", "tile_x": 11, "tile_y": 10, "split": "train", "source": "authored",
            "texture_names": [r"Tileset\X\XGrass.blp", r"Tileset\X\XRoad.blp"],
            "mcly_texture_ids": _ids(0, 1), "mcly_layer_mask": _mask(slot1=True),
        },
        {
            "map": "Azeroth", "tile_x": 12, "tile_y": 10, "split": "train", "source": "authored",
            "texture_names": [r"Tileset\X\XGrass.blp", r"Tileset\X\XDirt.blp", r"Tileset\X\XRoad.blp"],
            "mcly_texture_ids": _ids(0, 1, 2), "mcly_layer_mask": _mask(slot1=True, slot2=True),
        },
    ]
    build_store(store, rows=rows)
    dump = tmp_path / "names.json"
    write_texture_name_dump(
        dump, "Azeroth",
        [
            {"TileX": 10, "TileY": 10, "TextureNames": [r"Tileset\X\XRoad.blp", r"Tileset\X\XGrass.blp"]},
            {"TileX": 11, "TileY": 10, "TextureNames": [r"Tileset\X\XGrass.blp", r"Tileset\X\XRoad.blp"]},
            {"TileX": 12, "TileY": 10, "TextureNames": [r"Tileset\X\XGrass.blp", r"Tileset\X\XDirt.blp", r"Tileset\X\XRoad.blp"]},
        ],
    )
    return {"store": store, "dumps": [dump], "rows": rows}
