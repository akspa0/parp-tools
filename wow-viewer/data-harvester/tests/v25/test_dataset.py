"""Tests for the lean V25 dataset builder, tile source, and PM4 record sidecar."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import zarr

from harvester.v25.dataset import (
    V25_PER_TILE_SPECS,
    V25TileSource,
    attach_holes_bits,
    attach_pm4_segments,
    build_v25_dataset,
    load_pm4_segment_records,
    map_tileset_ids,
    object_mask_256_from_precise,
    select_rows,
    wdl_height_33_from_257,
    write_prediction_store,
)

N_TILES = 3
VOCAB_SIZE = 8


def _make_v18_fixture(path: Path, n: int = N_TILES, build: str = "3_3_5_12340") -> None:
    root = zarr.open_group(str(path), mode="w")
    rng = np.random.default_rng(102)

    minimap = rng.integers(0, 255, size=(n, 256, 256, 3), dtype=np.uint8)
    height = rng.normal(50.0, 10.0, size=(n, 257, 257)).astype(np.float32)
    alpha = rng.integers(0, 255, size=(n, 256, 256, 4)).astype(np.float32)
    precise = np.zeros((n, 257, 257), dtype=np.float32)
    precise[:, 10:30, 10:30] = 1.0
    mcly_mask = np.zeros((n, 16, 16, 4), dtype=np.float32)
    mcly_mask[:, :, :, 0] = 1.0
    mcly_mask[:, :8, :, 1] = 1.0
    liquid_mask = np.zeros((n, 256, 256), dtype=np.float32)
    liquid_mask[:, 200:230, 200:230] = 1.0
    liquid_type = np.zeros((n, 256, 256), dtype=np.uint8)
    liquid_type[:, 200:230, 200:230] = 2
    liquid_height = np.zeros((n, 256, 256), dtype=np.float32)
    liquid_height[:, 200:230, 200:230] = 42.5
    mcnk_flags = np.full((n, 16, 16), 5, dtype=np.int32)
    normal_xyz = np.zeros((n, 257, 257, 3), dtype=np.float32)
    normal_xyz[..., 2] = 1.0
    shadow = np.zeros((n, 256, 256), dtype=np.float32)
    shadow[:, 50:60, 50:60] = 1.0
    visibility = np.zeros((n, 256, 256), dtype=np.float32)
    visibility[:, 12:28, 12:28] = 1.0
    ground_intent = height + 3.0
    instance = np.full((n, 257, 257), -1, dtype=np.int32)
    instance[:, 10:30, 10:30] = 900
    # A signal V25 must NOT carry over (corrupt at source per the V24 audit):
    holes = np.ones((n, 16, 16), dtype=bool)

    for name, data in [
        ("minimap_rgb", minimap),
        ("height_257", height),
        ("alpha_256", alpha),
        ("object_precise_mask", precise),
        ("mcly_layer_mask", mcly_mask),
        ("liquid_mask", liquid_mask),
        ("liquid_type_256", liquid_type),
        ("liquid_height", liquid_height),
        ("mcnk_flags_16", mcnk_flags),
        ("normal_xyz", normal_xyz),
        ("shadow_mask", shadow),
        ("object_visibility_mask", visibility),
        ("ground_intent_height_257", ground_intent),
        ("object_instance_mask", instance),
        ("holes_16", holes),
    ]:
        root.create_array(name, shape=data.shape, chunks=(1, *data.shape[1:]), dtype=data.dtype)
        root[name][:] = data

    index = pa.Table.from_pylist(
        [
            {
                "tile_id": i,
                "build": build,
                "map": "Azeroth" if i < 2 else "Northrend",
                "tile_x": 30 + i,
                "tile_y": 40,
                "height_mean": 50.0,
                "height_std": 10.0,
            }
            for i in range(n)
        ]
    )
    pq.write_table(index, path / "index.parquet")


def _make_v22_fixture(
    path: Path,
    n: int = N_TILES,
    dominant_tid: int = 7,
    secondary_tid: int = 3,
    tileset_paths: list[str] | None = None,
) -> None:
    root = zarr.open_group(str(path), mode="w")
    tileset_ids = np.full((n, 16, 16, 4), -1, dtype=np.int32)
    tileset_ids[:, :, :, 0] = dominant_tid
    tileset_ids[:, :8, :, 1] = secondary_tid
    root.create_array("mcly_tileset_ids", shape=tileset_ids.shape, chunks=(1, 16, 16, 4), dtype=tileset_ids.dtype)
    root["mcly_tileset_ids"][:] = tileset_ids

    # One M2 placement on tile 0, one WMO on tile 1, none on tile 2.
    mddf_data = np.array([[5.0, 900.0, -8300.0, 2100.0, 130.0, 0.0, 90.0, 0.0, 1.5]], dtype=np.float32)
    modf_data = np.array(
        [[2.0, 901.0, -8000.0, 2000.0, 100.0, 0.0, 45.0, 0.0, 0, 0, 0, -1, -1, -1, 1, 1, 1]],
        dtype=np.float32,
    )
    layout = {
        "mddf_placement_offset": (np.array([0, 1, 1], dtype=np.int64), (3,)),
        "mddf_count": (np.array([[1], [0], [0]], dtype=np.int32), (3, 1)),
        "mddf_placement_data": (mddf_data, mddf_data.shape),
        "mddf_model_ids": (np.array([0], dtype=np.int32), (1,)),
        "modf_placement_offset": (np.array([0, 0, 1], dtype=np.int64), (3,)),
        "modf_count": (np.array([[0], [1], [0]], dtype=np.int32), (3, 1)),
        "modf_placement_data": (modf_data, modf_data.shape),
        "modf_model_ids": (np.array([1], dtype=np.int32), (1,)),
    }
    for name, (data, shape) in layout.items():
        root.create_array(name, shape=shape, chunks=shape, dtype=data.dtype)
        root[name][:] = data

    models = root.create_group("models")
    paths = np.asarray(["world/tree.m2", "world/keep.wmo"], dtype="<U32")
    models.create_array("model_paths", data=paths, chunks=paths.shape)

    tilesets = root.create_group("tilesets")
    if tileset_paths is None:
        tileset_paths = [f"tileset/t{i}.blp" for i in range(10)]
    tpaths = np.asarray(tileset_paths, dtype="<U48")
    tilesets.create_array("tileset_paths", data=tpaths, chunks=tpaths.shape)


@pytest.fixture()
def built_store(tmp_path: Path) -> Path:
    v18 = tmp_path / "v18.zarr"
    v22 = tmp_path / "v22.zarr"
    out = tmp_path / "v25.zarr"
    _make_v18_fixture(v18)
    _make_v22_fixture(v22)
    build_v25_dataset(v18_store=v18, output=out, v22_store=v22, vocab_size=VOCAB_SIZE)
    return out


def test_build_round_trip_shapes_and_math(built_store: Path):
    root = zarr.open_group(str(built_store), mode="r")

    for spec in V25_PER_TILE_SPECS:
        assert spec.name in root, f"missing array {spec.name}"
        arr = root[spec.name]
        assert arr.shape == (N_TILES, *spec.shape)
        assert arr.dtype == spec.dtype

    # WDL prior must equal the stride-8 downsample of the stored height.
    h = np.asarray(root["height_257"][0])
    wdl = np.asarray(root["wdl_height_33"][0])
    assert np.array_equal(wdl, h[::8, ::8])

    # The canonical precise mask survives intact; 256 is only its projection.
    precise = np.asarray(root["object_precise_mask_257"][0])
    assert precise.shape == (257, 257)
    mask = np.asarray(root["object_mask_256"][0])
    assert np.array_equal(mask, object_mask_256_from_precise(precise))
    assert mask.shape == (256, 256)
    assert mask[9:29, 9:29].max() == 1.0
    assert mask[100:, 100:].max() == 0.0

    assert root.attrs["v25_dataset_version"].startswith("v25")
    assert root.attrs["tile_count"] == N_TILES


def test_lean_signals_only(built_store: Path):
    """Signals that do not feed the V25 model must not be carried over."""
    root = zarr.open_group(str(built_store), mode="r")
    present = set(root.array_keys())
    expected = {s.name for s in V25_PER_TILE_SPECS}
    assert present == expected, f"unexpected arrays: {present - expected}"
    assert "holes_16" not in present  # corrupt at source — never carried
    assert list(root.group_keys()) == []


def test_full_signal_round_trip(built_store: Path):
    """Normals, shadows, visibility, ground-intent, and instance ids ride along."""
    root = zarr.open_group(str(built_store), mode="r")
    nrm = np.asarray(root["normal_xyz_257"][0])
    assert nrm.dtype == np.int8
    assert nrm[0, 0, 2] == 127 and nrm[0, 0, 0] == 0  # unit +Z quantized
    sh = np.asarray(root["shadow_mask_256"][0])
    assert sh[55, 55] == 255 and sh[0, 0] == 0
    vis = np.asarray(root["object_visibility_256"][0])
    assert vis[20, 20] == 255 and vis[0, 0] == 0
    gih = np.asarray(root["ground_intent_height_257"][0])
    h = np.asarray(root["height_257"][0])
    assert np.allclose(gih, h + 3.0)
    inst = np.asarray(root["object_instance_mask"][0])
    assert inst[15, 15] == 900 and inst[0, 0] == -1


def test_curation_metadata_baked_into_index(tmp_path: Path):
    v18 = tmp_path / "v18.zarr"
    _make_v18_fixture(v18)
    manifest = pa.Table.from_pylist(
        [
            {"build": "3_3_5_12340", "tile_id": 0, "keep": True,
             "difficulty_bucket": "hard", "quality_score": 0.9, "reject_reason": None},
            {"build": "3_3_5_12340", "tile_id": 1, "keep": True,
             "difficulty_bucket": "pathological", "quality_score": 0.2, "reject_reason": None},
            {"build": "3_3_5_12340", "tile_id": 2, "keep": True,
             "difficulty_bucket": "hard", "quality_score": 0.7, "reject_reason": None},
        ]
    )
    manifest_path = tmp_path / "kept.parquet"
    pq.write_table(manifest, manifest_path)

    out = tmp_path / "v25_curated.zarr"
    build_v25_dataset(
        v18_store=v18, output=out, curation_manifest=manifest_path, vocab_size=VOCAB_SIZE
    )

    index = pq.read_table(out / "index.parquet").to_pylist()
    assert index[0]["difficulty_bucket"] == "hard"
    assert index[1]["difficulty_bucket"] == "pathological"
    assert index[0]["quality_score"] == pytest.approx(0.9)
    root = zarr.open_group(str(out), mode="r")
    assert "difficulty_bucket" in root.attrs["curation_columns"]

    source = V25TileSource(out)
    assert source.rows_for_buckets(None) == [0, 1, 2]
    assert source.rows_for_buckets(["hard"]) == [0, 2]
    assert source.rows_for_buckets(["pathological"]) == [1]


def test_height_repair_store_sparse_overlay(tmp_path: Path):
    """The repair store is NaN except repaired tiles — merge, never replace."""
    v18 = tmp_path / "v18.zarr"
    _make_v18_fixture(v18)
    raw_heights = np.asarray(zarr.open_group(str(v18), mode="r")["height_257"][:])

    repair_root = tmp_path / "repair.zarr"
    repair_grp = zarr.open_group(str(repair_root / "3_3_5_12340"), mode="w")
    corrected = np.full_like(raw_heights, np.nan)          # sparse overlay
    corrected[0] = raw_heights[0] + 5.0                    # only tile 0 repaired
    corrected[1, 100, 100] = raw_heights[1, 100, 100] - 2.0  # partial repair on tile 1
    repair_grp.create_array(
        "height_corrected_257", shape=corrected.shape, chunks=(1, 257, 257), dtype=corrected.dtype
    )
    repair_grp["height_corrected_257"][:] = corrected

    out = tmp_path / "v25_repaired.zarr"
    build_v25_dataset(
        v18_store=v18, output=out, vocab_size=VOCAB_SIZE, height_repair_root=repair_root
    )

    root = zarr.open_group(str(out), mode="r")
    h0 = np.asarray(root["height_257"][0])
    assert np.allclose(h0, raw_heights[0] + 5.0)           # fully repaired tile
    h1 = np.asarray(root["height_257"][1])
    assert h1[100, 100] == pytest.approx(raw_heights[1, 100, 100] - 2.0)
    assert h1[0, 0] == pytest.approx(raw_heights[1, 0, 0])  # untouched cells stay raw
    h2 = np.asarray(root["height_257"][2])
    assert np.array_equal(h2, raw_heights[2])              # unrepaired tile = raw
    assert np.isfinite(np.asarray(root["height_257"][:])).all()  # NaN never leaks

    wdl = np.asarray(root["wdl_height_33"][0])
    assert np.array_equal(wdl, h0[::8, ::8])               # prior derives from merged
    assert root.attrs["height_repaired_builds"] == ["3_3_5_12340"]
    assert root.attrs["nonfinite_height_tiles"] == 0
    index = pq.read_table(out / "index.parquet").to_pylist()
    assert index[0]["height_repaired"] is True
    assert index[1]["height_repaired"] is True
    assert index[2]["height_repaired"] is False


def test_liquid_and_flags_signals(built_store: Path):
    """Liquid mask/type/height and MCNK flags ride along as loss signals."""
    root = zarr.open_group(str(built_store), mode="r")
    lm = np.asarray(root["liquid_mask_256"][0])
    assert lm.dtype == np.uint8
    assert lm[210, 210] == 255 and lm[10, 10] == 0
    lt = np.asarray(root["liquid_type_256"][0])
    assert lt[210, 210] == 2 and lt[10, 10] == 0
    lh = np.asarray(root["liquid_height_256"][0])
    assert lh[210, 210] == pytest.approx(42.5) and lh[10, 10] == 0.0
    flags = np.asarray(root["mcnk_flags_16"][0])
    assert flags.dtype == np.int32 and flags[0, 0] == 5


def test_vocab_and_mcly_mapping(built_store: Path):
    root = zarr.open_group(str(built_store), mode="r")
    vocab = pq.read_table(built_store / "tileset_vocab.parquet").to_pylist()
    by_key = {(r["build"], r["tileset_path"]): r["vocab_id"] for r in vocab}
    # tileset 7 dominates (all 16x16 layer-0 cells) so its key ranks first.
    assert by_key[("3_3_5_12340", "tileset/t7.blp")] == 0
    assert by_key[("3_3_5_12340", "tileset/t3.blp")] == 1
    assert ("", "<oov>") in by_key  # OOV bucket present

    ids = np.asarray(root["mcly_vocab_ids"][0])
    assert ids[0, 0, 0] == 0        # tileset 7 -> vocab 0
    assert ids[0, 0, 1] == 1        # tileset 3 -> vocab 1
    assert ids[15, 0, 1] == -1      # inactive layer stays -1
    assert ids[0, 0, 2] == -1       # never-active layer


def test_multi_build_era_scoped_vocab(tmp_path: Path):
    """The same tileset path in two eras gets two distinct vocab entries —
    tileset content changed over time even when the names did not."""
    v18_a = tmp_path / "v18_a.zarr"
    v22_a = tmp_path / "v22_a.zarr"
    v18_b = tmp_path / "v18_b.zarr"
    v22_b = tmp_path / "v22_b.zarr"
    out = tmp_path / "v25_multi.zarr"

    _make_v18_fixture(v18_a, build="3_3_5_12340")
    # Build A: local id 7 -> grass, 3 -> rock.
    paths_a = [f"tileset/t{i}.blp" for i in range(10)]
    paths_a[7] = "Tileset/Grass.blp"
    paths_a[3] = "tileset/rock.blp"
    _make_v22_fixture(v22_a, dominant_tid=7, secondary_tid=3, tileset_paths=paths_a)

    _make_v18_fixture(v18_b, build="0_5_3_3368")
    # Build B: DIFFERENT local ids, same tileset paths (case/slash variance).
    paths_b = [f"tileset/other{i}.blp" for i in range(6)]
    paths_b[2] = "TILESET\\GRASS.BLP"
    paths_b[5] = "tileset/rock.blp"
    _make_v22_fixture(v22_b, dominant_tid=2, secondary_tid=5, tileset_paths=paths_b)

    build_v25_dataset(
        v18_store=[v18_a, v18_b],
        output=out,
        v22_store=[v22_a, v22_b],
        v24_store=None,
        vocab_size=VOCAB_SIZE,
    )

    root = zarr.open_group(str(out), mode="r")
    assert root.attrs["tile_count"] == 2 * N_TILES
    assert sorted(root.attrs["builds"]) == ["0_5_3_3368", "3_3_5_12340"]

    index = pq.read_table(out / "index.parquet").to_pylist()
    assert [r["build"] for r in index] == ["3_3_5_12340"] * N_TILES + ["0_5_3_3368"] * N_TILES

    # Grass dominates both builds, but each era keeps ITS OWN vocab id.
    ids_a = np.asarray(root["mcly_vocab_ids"][0])
    ids_b = np.asarray(root["mcly_vocab_ids"][N_TILES])
    assert ids_a[0, 0, 0] != ids_b[0, 0, 0]
    assert ids_a[0, 0, 1] != ids_b[0, 0, 1]

    vocab = pq.read_table(out / "tileset_vocab.parquet").to_pylist()
    by_key = {(r["build"], r["tileset_path"]): r["vocab_id"] for r in vocab}
    # Same normalized path, two eras, two entries.
    assert by_key[("3_3_5_12340", "tileset/grass.blp")] != by_key[("0_5_3_3368", "tileset/grass.blp")]
    assert by_key[("3_3_5_12340", "tileset/rock.blp")] != by_key[("0_5_3_3368", "tileset/rock.blp")]
    assert ids_a[0, 0, 0] == by_key[("3_3_5_12340", "tileset/grass.blp")]
    assert ids_b[0, 0, 0] == by_key[("0_5_3_3368", "tileset/grass.blp")]

    # Placements from both builds land on the right combined rows.
    placements = pq.read_table(out / "placements.parquet").to_pylist()
    rows = sorted(p["row"] for p in placements)
    assert rows == [0, 1, N_TILES + 0, N_TILES + 1]


def test_attach_tileset_images(built_store: Path, tmp_path: Path):
    """Era-scoped tileset images attach into a vocab-aligned tilesets group."""
    from PIL import Image
    from harvester.v25.dataset import attach_tileset_images

    png_dir = tmp_path / "tilesets_3_3_5"
    png_dir.mkdir()
    grass = np.zeros((256, 256, 3), dtype=np.uint8)
    grass[..., 1] = 200  # green
    Image.fromarray(grass).save(png_dir / "t0000.png")

    manifest = {
        "client_root": "output/tmp/wowarchive-clients/3_3_5_12340/World of Warcraft",
        "tilesets": [
            {"path": "Tileset\\t7.blp", "file": "t0000.png"},  # raw path normalizes
        ],
    }
    manifest_path = png_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    stats = attach_tileset_images(built_store, [manifest_path])
    assert stats["matched"] == 1

    root = zarr.open_group(str(built_store), mode="r")
    vocab = pq.read_table(built_store / "tileset_vocab.parquet").to_pylist()
    vid = next(r["vocab_id"] for r in vocab if r["tileset_path"] == "tileset/t7.blp")
    present = np.asarray(root["tilesets"]["tileset_present"][:])
    assert present[vid] == 1
    assert present.sum() == 1  # t3 had no image -> absent
    rgb = np.asarray(root["tilesets"]["tileset_rgb_256"][vid])
    assert rgb[128, 128, 1] == 200 and rgb[128, 128, 0] == 0

    source = V25TileSource(built_store)
    assert source.load(0)["mcly_vocab_ids"].max() >= 0  # store still loads fine


def test_placements_table(built_store: Path):
    table = pq.read_table(built_store / "placements.parquet").to_pylist()
    assert len(table) == 2
    m2 = next(r for r in table if r["kind"] == "m2")
    wmo = next(r for r in table if r["kind"] == "wmo")
    assert m2["row"] == 0
    assert m2["asset_path"] == "world/tree.m2"
    assert m2["pos_x"] == pytest.approx(-8300.0)
    assert m2["scale"] == pytest.approx(1.5)
    assert wmo["row"] == 1
    assert wmo["asset_path"] == "world/keep.wmo"
    assert wmo["rot_y"] == pytest.approx(45.0)


def test_select_rows_maps_and_manifest(tmp_path: Path):
    v18 = tmp_path / "v18.zarr"
    _make_v18_fixture(v18)
    index = pq.read_table(v18 / "index.parquet")

    assert select_rows(index, maps=["Azeroth"]) == [0, 1]
    assert select_rows(index, maps=["northrend"]) == [2]
    assert select_rows(index, limit=1) == [0]

    manifest = pa.Table.from_pylist(
        [
            {"build": "3_3_5_12340", "tile_id": 0, "keep": True, "difficulty_bucket": "hard"},
            {"build": "3_3_5_12340", "tile_id": 1, "keep": False, "difficulty_bucket": "hard"},
            {"build": "3_3_5_12340", "tile_id": 2, "keep": True, "difficulty_bucket": "easy"},
        ]
    )
    manifest_path = tmp_path / "kept.parquet"
    pq.write_table(manifest, manifest_path)
    assert select_rows(index, curation_manifest=manifest_path) == [0, 2]
    assert select_rows(index, curation_manifest=manifest_path, difficulty_bucket="hard") == [0]


def test_tile_source_preload_matches_random_access(built_store: Path):
    source = V25TileSource(built_store)
    assert len(source) == N_TILES

    cold = source.load(1)
    source.preload(list(range(N_TILES)))
    warm = source.load(1)

    for key in ("minimap", "clean_minimap", "object_mask", "height_257", "wdl_height_33", "alpha"):
        assert np.array_equal(cold[key], warm[key]), key
    assert cold["placements"] == warm["placements"]
    assert warm["minimap"].dtype == np.float32
    assert warm["minimap"].max() <= 1.0
    assert warm["mcly_vocab_ids"].dtype == np.int64


def test_pm4_segment_records_round_trip(built_store: Path, tmp_path: Path):
    export = {
        "Segments": [
            {
                "SegmentId": "seg_001",
                "Bounds": {
                    "Min": {"X": -8310.0, "Y": 2090.0, "Z": 120.0},
                    "Max": {"X": -8290.0, "Y": 2110.0, "Z": 140.0},
                },
                "FootprintHull": [{"X": -8310.0, "Y": 2090.0}, {"X": -8290.0, "Y": 2110.0}],
                "HeightStats": {
                    "MinimumPlaneDistance": 0.5,
                    "MaximumPlaneDistance": 20.0,
                    "AveragePlaneDistance": 10.0,
                },
                "SurfaceFamilyHistogram": {"floor": 12},
                "TopologyStats": {
                    "SurfaceCount": 3,
                    "TotalIndexCount": 96,
                    "AnchorPointCount": 4,
                    "AnchorNormalCount": 4,
                },
                "AnchorSignals": {
                    "LinkedPositionRefCount": 1,
                    "NormalHeadingCount": 2,
                    "TerminatorCount": 0,
                    "FloorMinimum": 1,
                    "FloorMaximum": 2,
                },
                "TileCoordinates": ["30_40"],
            }
        ]
    }
    export_path = tmp_path / "segments.json"
    export_path.write_text(json.dumps(export), encoding="utf-8")

    n = attach_pm4_segments(built_store, export_path)
    assert n == 1

    records = load_pm4_segment_records(built_store)
    assert len(records) == 1
    rec = records[0]
    assert rec.segment_id == "seg_001"
    assert rec.bounds is not None
    assert rec.bounds.min == (-8310.0, 2090.0, 120.0)
    assert rec.height_stats.average_plane_distance == 10.0
    assert rec.topology_stats.surface_count == 3
    assert rec.tile_coordinates == ["30_40"]

    assert load_pm4_segment_records(built_store, tile_coordinate="30_40") != []
    assert load_pm4_segment_records(built_store, tile_coordinate="0_0") == []


def test_attach_holes_bits(built_store: Path, tmp_path: Path):
    """True MCNK hole bitmasks join by (build, map, x, y); unknown tiles stay -1."""
    export = {
        "build_version": "3.3.5.12340",
        "client_root": "output/tmp/wowarchive-clients/3_3_5_12340/World of Warcraft",
        "hole_field": "mcnk_holes_uint16_row_major_yx",
        "maps": {
            # Fixture tiles: Azeroth (30,40), (31,40); Northrend (32,40).
            "Azeroth": [
                {"x": 30, "y": 40, "holes": [0x0003] + [0] * 255},
                {"x": 31, "y": 40, "holes": [0] * 256},
            ],
        },
    }
    export_path = tmp_path / "holes.json"
    export_path.write_text(json.dumps(export), encoding="utf-8")

    stats = attach_holes_bits(built_store, [export_path])
    assert stats == {"rows": N_TILES, "matched": 2, "holed": 1}

    root = zarr.open_group(str(built_store), mode="r")
    bits = np.asarray(root["holes_bits_16"][:])
    assert bits[0, 0, 0] == 0x0003        # holed chunk on tile 0
    assert bits[0, 0, 1] == 0
    assert bits[1].max() == 0             # matched, hole-free tile
    assert bits[2].min() == -1            # Northrend tile absent from export -> unknown
    assert root.attrs["holes_bits_matched"] == 2

    source = V25TileSource(built_store)
    rec = source.load(0)
    assert rec["holes_bits"][0, 0] == 0x0003
    rec2 = source.load(2)
    assert rec2["holes_bits"][0, 0] == -1


def test_write_prediction_store(tmp_path: Path):
    out = tmp_path / "pred.zarr"
    write_prediction_store(
        out,
        predictions={
            "height_257": np.zeros((1, 257, 257), dtype=np.float32),
            "wdl_height_33": np.zeros((1, 33, 33), dtype=np.float32),
        },
        placements=[{"kind": "m2", "pos_x": 1.0, "pos_y": 2.0, "pos_z": 3.0}],
        attrs={"source": "test"},
    )
    root = zarr.open_group(str(out), mode="r")
    assert root["height_257"].shape == (1, 257, 257)
    assert root.attrs["v25_prediction_store"] is True
    assert root.attrs["source"] == "test"
    table = pq.read_table(out / "placements.parquet")
    assert table.num_rows == 1


def test_derived_signal_helpers():
    precise = np.zeros((257, 257), dtype=np.float32)
    precise[0, 0] = 0.75
    mask = object_mask_256_from_precise(precise)
    assert mask.shape == (256, 256)
    assert mask[0, 0] == 0.75

    h = np.arange(257 * 257, dtype=np.float32).reshape(257, 257)
    wdl = wdl_height_33_from_257(h)
    assert wdl.shape == (33, 33)
    assert wdl[0, 1] == h[0, 8]

    ids = np.array([[[[7, 3, 99, -1]]]], dtype=np.int32).reshape(1, 1, 4)
    layer = np.array([[[1, 1, 1, 0]]], dtype=np.uint8).reshape(1, 1, 4)
    mapped = map_tileset_ids(ids, layer, {7: 0, 3: 1}, vocab_size=8)
    assert mapped[0, 0, 0] == 0
    assert mapped[0, 0, 1] == 1
    assert mapped[0, 0, 2] == 7   # OOV bucket
    assert mapped[0, 0, 3] == -1  # inactive layer
