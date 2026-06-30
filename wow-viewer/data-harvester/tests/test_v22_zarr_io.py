"""Pytest coverage for the V22 Zarr writer/reader contract.

This test does not require a game client. It feeds a small synthetic V22
tile record into ``V22ZarrWriter``, finalizes a Zarr store on disk, then
loads it with ``V22Dataset`` and asserts the canonical fixed-key contract.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr
import zarr.storage

from harvester.v22_zarr_io import (
    V22ZarrWriter,
    V22Dataset,
    V22TileRecord,
    V22_ROOT_ARRAYS,
    V22_PER_TILE_SPECS,
)


def _synthetic_record(tile_id: int, build: str, map_name: str, x: int, y: int) -> V22TileRecord:
    per_tile: dict[str, np.ndarray] = {}
    per_tile["height_257"] = np.full((257, 257), float(tile_id), dtype=np.float32)
    per_tile["normal_xyz"] = np.zeros((257, 257, 3), dtype=np.float32)
    per_tile["normal_mask"] = np.zeros((257, 257), dtype=bool)
    per_tile["alpha_256"] = np.full((256, 256, 4), 1.0 / 4.0, dtype=np.float32)
    per_tile["holes_16"] = np.zeros((16, 16), dtype=bool)
    per_tile["liquid_mask"] = np.zeros((256, 256), dtype=np.float32)
    per_tile["liquid_height"] = np.zeros((256, 256), dtype=np.float32)
    per_tile["object_mask"] = np.zeros((257, 257), dtype=bool)
    per_tile["object_precise_mask"] = np.zeros((257, 257), dtype=np.float32)
    per_tile["object_instance_mask"] = np.full((257, 257), tile_id, dtype=np.int32)
    per_tile["mcnk_flags_16"] = np.zeros((16, 16), dtype=np.int32)
    per_tile["mddf_mask"] = np.zeros((257, 257), dtype=np.float32)
    per_tile["modf_mask"] = np.zeros((257, 257), dtype=np.float32)
    per_tile["object_filtered_mask"] = np.zeros((257, 257), dtype=np.float32)
    per_tile["model_focus_mask"] = np.zeros((257, 257), dtype=np.float32)
    per_tile["model_above_terrain_mask"] = np.full((257, 257), 1.0, dtype=np.float32)
    per_tile["object_roof_mask"] = np.zeros((256, 256), dtype=np.float32)
    per_tile["object_roof_confidence"] = np.zeros((256, 256), dtype=np.float32)
    per_tile["minimap_rgb"] = np.zeros((256, 256, 3), dtype=np.uint8)
    per_tile["shadow_mask"] = np.zeros((256, 256), dtype=np.float32)
    per_tile["mcly_texture_ids"] = np.zeros((16, 16, 4), dtype=np.int32)
    per_tile["mcly_layer_mask"] = np.ones((16, 16, 4), dtype=np.float32)
    per_tile["mcnr_mask_257"] = np.zeros((257, 257), dtype=bool)
    per_tile["liquid_type_256"] = np.zeros((256, 256), dtype=np.uint8)
    per_tile["ground_intent_height_257"] = np.full((257, 257), float(tile_id), dtype=np.float32)
    per_tile["mddf_count"] = np.asarray([2 if tile_id == 0 else 0], dtype=np.int32)
    per_tile["modf_count"] = np.asarray([1 if tile_id == 0 else 0], dtype=np.int32)
    per_tile["mcly_tileset_ids"] = np.zeros((16, 16, 4), dtype=np.int32)

    mddf = np.asarray(
        [
            [1.0, 100.0, 10, 20, 30, 0, 0, 0, 1.0],
            [2.0, 101.0, 11, 21, 31, 0, 0, 0, 1.5],
        ],
        dtype=np.float32,
    ) if tile_id == 0 else None

    modf = np.asarray(
        [[3.0, 200.0, 12, 22, 32, 0, 0, 0, 0, 0, 0, -1, -1, -1, 1, 1, 1]],
        dtype=np.float32,
    ) if tile_id == 0 else None

    return V22TileRecord(
        tile_id=tile_id,
        build=build,
        map=map_name,
        tile_x=x,
        tile_y=y,
        per_tile=per_tile,
        placement_mddf=mddf,
        placement_modf=modf,
        mddf_asset_paths=("World/M2/A.m2", "World/M2/B.m2") if tile_id == 0 else (),
        modf_asset_paths=("World/WMO/C.wmo",) if tile_id == 0 else (),
        mtex_texture_paths=("Tileset/X.blp", "Tileset/Y.blp"),
    )


def test_v22_writer_writes_root_arrays_and_reader_returns_fixed_keys(tmp_path: Path) -> None:
    store_path = tmp_path / "v22_test.zarr"
    writer = V22ZarrWriter(store_path)

    writer.add_tile(_synthetic_record(0, "3_3_5_12340", "Azeroth", 32, 32))
    writer.add_tile(_synthetic_record(1, "3_3_5_12340", "Azeroth", 32, 33))
    writer.add_tile(_synthetic_record(2, "3_3_5_12340", "Azeroth", 33, 32))

    writer.add_model(
        "World/M2/A.m2",
        {
            "vertices": np.zeros((8, 3), dtype=np.float32),
            "triangles": np.zeros((4, 3), dtype=np.int32),
            "normals": np.zeros((8, 3), dtype=np.float32),
            "bounds": np.zeros((2, 3), dtype=np.float32),
            "kind": np.asarray([1], dtype=np.uint8),
        },
    )
    writer.add_tileset(
        "Tileset/X.blp",
        {
            "texture_rgb": np.zeros((4, 4, 3), dtype=np.uint8),
            "texture_shape": np.asarray([4, 4], dtype=np.int32),
        },
    )

    writer.finalize()

    assert store_path.exists()
    grp = zarr.open_group(zarr.storage.LocalStore(str(store_path), read_only=True), mode="r")
    for spec in V22_PER_TILE_SPECS:
        if spec.name in {"mddf_count", "modf_count"}:
            assert grp[spec.name].shape == (3, 1)
        else:
            assert grp[spec.name].shape == (3, *spec.shape)
    assert grp["mddf_placement_data"].shape == (2, 9)
    assert grp["modf_placement_data"].shape == (1, 17)
    assert grp["models/model_paths"][:].tolist() == ["World/M2/A.m2"]
    assert grp["tilesets/tileset_paths"][:].tolist() == ["Tileset/X.blp"]
    assert grp.attrs["tile_count"] == 3

    dataset = V22Dataset(store_path)
    assert len(dataset) == 3
    sample = dataset[0]
    for spec in V22_PER_TILE_SPECS:
        assert spec.name in sample, f"missing key {spec.name}"
    assert sample["height_257"].shape == (257, 257)
    assert sample["height_257"].dtype == np.float32
    assert sample["height_257"][0, 0] == 0.0
    assert sample["mddf_placement_data"].shape == (2, 9)
    assert sample["modf_placement_data"].shape == (1, 17)
    assert sample["mddf_unique_ids"].tolist() == [100, 101]
    assert sample["modf_unique_ids"].tolist() == [200]
    assert sample["mddf_model_ids"].tolist() == [1, 2]
    assert sample["modf_model_ids"].tolist() == [3]
    assert sample["placement_mddf_asset_paths"] == ["World/M2/A.m2", "World/M2/B.m2"]
    assert sample["placement_modf_asset_paths"] == ["World/WMO/C.wmo"]
    assert sample["mtex_texture_paths"] == ["Tileset/X.blp", "Tileset/Y.blp"]

    # Empty tiles must still return the full key set with documented shapes.
    empty_sample = dataset[1]
    assert set(empty_sample.keys()) >= set(V22_ROOT_ARRAYS)
    assert empty_sample["mddf_placement_data"].shape == (0, 9)
    assert empty_sample["modf_placement_data"].shape == (0, 17)
    assert empty_sample["mddf_count"][0] == 0
    assert empty_sample["modf_count"][0] == 0
