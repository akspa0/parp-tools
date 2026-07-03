"""Pytest coverage for the V22 Zarr writer/reader contract.

Uses V22ZarrWriter.add_from_v18() with synthetic V18 stores + enrichment
streams. No game client access required.
"""

from __future__ import annotations

import json
import struct
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import zarr
import zarr.storage

from harvester.v22_zarr_io import (
    V22ZarrWriter,
    V22Dataset,
    V22_ROOT_ARRAYS,
    V22_PER_TILE_SPECS,
)


# ── Synthetic data builders ─────────────────────────────────────────────


def _make_v18_store(path: Path, n_tiles: int = 2) -> Path:
    """Create a minimal V18 Zarr store with the required arrays, index,
    and placements."""
    store = zarr.storage.LocalStore(str(path), read_only=False)
    root = zarr.group(store=store, attributes={"v18_dataset_version": "v18"})

    # Per-tile arrays
    for spec in V22_PER_TILE_SPECS:
        if spec.shape == (1,):
            arr = np.zeros((n_tiles, *spec.shape), dtype=spec.dtype)
        else:
            arr = np.zeros((n_tiles, *spec.shape), dtype=spec.dtype)
        root.create_array(spec.name, data=arr)

    # Populate height_257 with tile-varying data
    for i in range(n_tiles):
        root["height_257"][i] = np.full((257, 257), float(i + 10), dtype=np.float32)
        root["minimap_rgb"][i] = np.ones((256, 256, 3), dtype=np.uint8) * (i + 10)
        root["object_filtered_mask"][i] = np.zeros((257, 257), dtype=np.float32)
        root["mcly_texture_ids"][i] = np.full((16, 16, 4), i, dtype=np.int32)
        root["mcly_layer_mask"][i] = np.ones((16, 16, 4), dtype=np.float32)
        root["liquid_basic_type_257"][i] = np.zeros((257, 257), dtype=np.uint8)
        root["normal_xyz"][i] = np.zeros((257, 257, 3), dtype=np.float32)
        root["normal_mask"][i] = np.ones((257, 257), dtype=bool)
        root["alpha_256"][i] = np.full((256, 256, 4), 0.25, dtype=np.float32)
        root["holes_16"][i] = np.zeros((16, 16), dtype=bool)
        root["liquid_mask"][i] = np.zeros((256, 256), dtype=np.float32)
        root["liquid_height"][i] = np.zeros((256, 256), dtype=np.float32)
        root["object_mask"][i] = np.zeros((257, 257), dtype=bool)
        root["object_precise_mask"][i] = np.zeros((257, 257), dtype=np.float32)
        root["object_instance_mask"][i] = np.full((257, 257), i, dtype=np.int32)
        root["mcnk_flags_16"][i] = np.zeros((16, 16), dtype=np.int32)
        root["mddf_mask"][i] = np.zeros((257, 257), dtype=np.float32)
        root["modf_mask"][i] = np.zeros((257, 257), dtype=np.float32)
        root["object_roof_mask"][i] = np.zeros((256, 256), dtype=np.float32)
        root["object_roof_confidence"][i] = np.zeros((256, 256), dtype=np.float32)
        root["shadow_mask"][i] = np.zeros((256, 256), dtype=np.float32)
        root["mcnr_mask_257"][i] = np.ones((257, 257), dtype=bool)

    # Index
    index_table = pa.table({
        "tile_id": pa.array(list(range(n_tiles)), type=pa.int64()),
        "build": pa.array(["3_3_5_12340"] * n_tiles, type=pa.string()),
        "map": pa.array(["Azeroth"] * n_tiles, type=pa.string()),
        "tile_x": pa.array([30] * n_tiles, type=pa.int32()),
        "tile_y": pa.array([50] * n_tiles, type=pa.int32()),
        "height_mean": pa.array([10.0, 11.0], type=pa.float32()),
        "height_std": pa.array([0.0, 0.0], type=pa.float32()),
    })
    pq.write_table(index_table, str(path / "index.parquet"))

    # Placements
    placements_table = pa.table({
        "tile_id": pa.array([0], type=pa.int64()),
        "instance_type": pa.array(["mddf"], type=pa.string()),
        "nameId": pa.array([1], type=pa.int32()),
        "uniqueId": pa.array([100], type=pa.int32()),
        "posX": pa.array([10.0], type=pa.float64()),
        "posY": pa.array([10.0], type=pa.float64()),
        "posZ": pa.array([100.0], type=pa.float64()),
        "rotX": pa.array([0.0], type=pa.float64()),
        "rotY": pa.array([0.0], type=pa.float64()),
        "rotZ": pa.array([0.0], type=pa.float64()),
        "scale": pa.array([1.0], type=pa.float64()),
        "asset_path": pa.array(["World/M2/Peasant.m2"], type=pa.string()),
    })
    pq.write_table(placements_table, str(path / "placements.parquet"))

    return path


def _make_enrichment_stream(path: Path) -> Path:
    """Write a minimal enrichment stream with one M2 entry."""
    with open(path, "wb") as f:
        # Header
        f.write(b"V22E")
        f.write(struct.pack("<I", 1))  # version

        # One M2 entry for World/M2/Peasant.m2
        f.write(b"ENTRY")
        m2_path = "World/M2/Peasant.m2"
        f.write(struct.pack("<I", len(m2_path)))
        f.write(m2_path.encode("utf-8"))
        f.write(struct.pack("<B", 1))  # kind = M2
        f.write(struct.pack("<B", 0))  # load_error = 0

        # 2 arrays: vertices (8, 3) float32, bounds (2, 3) float32
        f.write(struct.pack("<I", 2))

        # Array 1: vertices
        vert_name = "vertices"
        f.write(struct.pack("<I", len(vert_name)))
        f.write(vert_name.encode("utf-8"))
        f.write(struct.pack("<I", 2))  # ndim
        f.write(struct.pack("<II", 8, 3))  # shape
        f.write(b"<f4" + b"\x00" * 5)  # dtype
        vert_data = np.zeros(8 * 3, dtype=np.float32).tobytes()
        f.write(struct.pack("<q", len(vert_data)))
        f.write(vert_data)

        # Array 2: bounds
        bnd_name = "bounds"
        f.write(struct.pack("<I", len(bnd_name)))
        f.write(bnd_name.encode("utf-8"))
        f.write(struct.pack("<I", 2))
        f.write(struct.pack("<II", 2, 3))
        f.write(b"<f4" + b"\x00" * 5)
        bnd_data = np.array([-1, -1, -1, 1, 1, 1], dtype=np.float32).tobytes()
        f.write(struct.pack("<q", len(bnd_data)))
        f.write(bnd_data)

        # Terminator
        f.write(b"ENDS")

    return path


# ── Tests ──────────────────────────────────────────────────────────────


class TestV22ZarrWriterFromV18:
    """Test the new add_from_v18 API with synthetic stores."""

    def test_basic_round_trip(self, tmp_path: Path):
        v18 = _make_v18_store(tmp_path / "v18", n_tiles=2)
        enrich = _make_enrichment_stream(tmp_path / "enrich.bin")

        writer = V22ZarrWriter(tmp_path / "v22", overwrite=True)
        writer.add_from_v18(str(v18), str(enrich))
        writer.finalize()

        # Read back via V22Dataset
        ds = V22Dataset(tmp_path / "v22")
        assert len(ds) == 2

        tile = ds[0]
        assert tile["tile_id"] == np.int64(0)
        assert tile["height_257"].shape == (257, 257)
        assert tile["minimap_rgb"].shape == (256, 256, 3)
        assert tile["model_focus_mask"].shape == (257, 257)
        assert tile["mcnr_mask_257"].shape == (257, 257)
        assert tile["liquid_type_256"].shape == (256, 256)
        assert tile["ground_intent_height_257"].shape == (257, 257)
        assert tile["mddf_placement_data"].shape[0] == 1  # one placement

    def test_model_library_populated(self, tmp_path: Path):
        v18 = _make_v18_store(tmp_path / "v18", n_tiles=1)
        enrich = _make_enrichment_stream(tmp_path / "enrich.bin")

        writer = V22ZarrWriter(tmp_path / "v22", overwrite=True)
        writer.add_from_v18(str(v18), str(enrich))
        writer.finalize()

        ds = V22Dataset(tmp_path / "v22")
        root_path = tmp_path / "v22"
        import zarr as z
        grp = z.open_group(z.storage.LocalStore(str(root_path), read_only=True), mode="r")
        assert "models" in grp
        assert grp["models/model_paths"].shape[0] == 1

    def test_empty_enrichment_stream_no_crash(self, tmp_path: Path):
        v18 = _make_v18_store(tmp_path / "v18", n_tiles=1)
        empty_enrich = tmp_path / "empty_enrich.bin"
        empty_enrich.write_bytes(b"V22E" + struct.pack("<I", 1) + b"ENDS")

        writer = V22ZarrWriter(tmp_path / "v22", overwrite=True)
        writer.add_from_v18(str(v18), str(empty_enrich))
        writer.finalize()

        ds = V22Dataset(tmp_path / "v22")
        assert len(ds) == 1

    def test_empty_v18_missing_store(self, tmp_path: Path):
        v18 = tmp_path / "v18_missing"
        enrich = _make_enrichment_stream(tmp_path / "enrich.bin")

        with pytest.raises((RuntimeError, FileNotFoundError, ValueError)):
            writer = V22ZarrWriter(tmp_path / "v22", overwrite=True)
            writer.add_from_v18(str(v18), str(enrich))
            writer.finalize()


class TestV22DatasetFixedKey:
    """The V22Dataset reader contract is unchanged by the refactor."""

    def test_fixed_keys_present(self, tmp_path: Path):
        v18 = _make_v18_store(tmp_path / "v18", n_tiles=2)
        enrich = _make_enrichment_stream(tmp_path / "enrich.bin")

        writer = V22ZarrWriter(tmp_path / "v22", overwrite=True)
        writer.add_from_v18(str(v18), str(enrich))
        writer.finalize()

        ds = V22Dataset(tmp_path / "v22")
        tile = ds[0]

        # All documented root arrays should be present
        expected_keys = {
            "height_257", "normal_xyz", "normal_mask", "alpha_256", "holes_16",
            "liquid_mask", "liquid_height", "object_mask", "object_precise_mask",
            "object_instance_mask", "mcnk_flags_16", "mddf_mask", "modf_mask",
            "object_filtered_mask", "model_focus_mask", "model_above_terrain_mask",
            "object_roof_mask", "object_roof_confidence", "minimap_rgb", "shadow_mask",
            "mcly_texture_ids", "mcly_layer_mask", "mcnr_mask_257", "liquid_type_256",
            "ground_intent_height_257",
        }
        for key in expected_keys:
            assert key in tile, f"missing key: {key}"
            assert isinstance(tile[key], np.ndarray), f"non-array key: {key}"

    def test_placement_arrays_present(self, tmp_path: Path):
        v18 = _make_v18_store(tmp_path / "v18", n_tiles=2)
        enrich = _make_enrichment_stream(tmp_path / "enrich.bin")

        writer = V22ZarrWriter(tmp_path / "v22", overwrite=True)
        writer.add_from_v18(str(v18), str(enrich))
        writer.finalize()

        ds = V22Dataset(tmp_path / "v22")
        tile = ds[0]

        assert "mddf_placement_data" in tile
        assert "modf_placement_data" in tile
        assert "mddf_unique_ids" in tile
        assert "modf_unique_ids" in tile
        assert "mddf_model_ids" in tile
        assert "modf_model_ids" in tile
        assert "mddf_count" in tile
        assert "modf_count" in tile

    def test_empty_tile_no_placements(self, tmp_path: Path):
        v18 = _make_v18_store(tmp_path / "v18", n_tiles=2)
        enrich = _make_enrichment_stream(tmp_path / "enrich.bin")

        writer = V22ZarrWriter(tmp_path / "v22", overwrite=True)
        writer.add_from_v18(str(v18), str(enrich))
        writer.finalize()

        ds = V22Dataset(tmp_path / "v22")
        tile = ds[1]  # tile 1 has no placements

        assert tile["mddf_placement_data"].shape == (0, 9)
        assert tile["modf_placement_data"].shape == (0, 17)


class TestV22DatasetMetadataKeys:
    def test_tile_metadata(self, tmp_path: Path):
        v18 = _make_v18_store(tmp_path / "v18", n_tiles=1)
        enrich = _make_enrichment_stream(tmp_path / "enrich.bin")

        writer = V22ZarrWriter(tmp_path / "v22", overwrite=True)
        writer.add_from_v18(str(v18), str(enrich))
        writer.finalize()

        ds = V22Dataset(tmp_path / "v22")
        tile = ds[0]

        assert "tile_id" in tile
        assert "build" in tile
        assert "map" in tile
        assert "tile_x" in tile
        assert "tile_y" in tile