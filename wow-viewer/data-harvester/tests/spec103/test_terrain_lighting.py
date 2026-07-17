"""CPU-only proof for Spec 103 authored lighting and grouped synthetic variants."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pyarrow.parquet as pq
import pytest
import zarr
from PIL import Image

from harvester.spec103.prefab_curation import validate_source_group_split
from harvester.spec103.terrain_lighting import (
    AUTHORED_MCSH_EVIDENCE_STATE,
    AUTHORED_MCSH_MODEL,
    DBC_PROFILE_SCHEMA,
    EVIDENCE_STATE,
    GRID_TO_RENDERER_NORMAL_TRANSFORM,
    LIT_PROFILE_SCHEMA,
    PROFILE_REVISION,
    _terrain_solar_direction,
    evaluate_authored_day_night,
    grid_normals_to_renderer,
    load_lighting_profile_artifact,
    shade_terrain,
    synthesize_authored_height_shadow,
)


def _load_builder() -> ModuleType:
    path = Path(__file__).parents[2] / "scripts" / "spec103_build_synthetic_store.py"
    name = "spec103_build_synthetic_store_tests"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_manifest(
    tmp_path: Path,
    *,
    license_name: str = "CC0-1.0",
    rights_assertion: str = "operator_authored_and_licensed",
    corrupt_hash: bool = False,
) -> Path:
    yy, xx = np.mgrid[0:257, 0:257].astype(np.float32)
    height = (xx * 0.35) + (yy * 0.08)
    height_path = tmp_path / "known.npy"
    np.save(height_path, height)
    digest = hashlib.sha256(height_path.read_bytes()).hexdigest()
    if corrupt_hash:
        digest = "0" * 64
    manifest = {
        "schema": "spec103-synthetic-manifest-v1",
        "tiles": [
            {
                "tile_name": "synth103_30_30",
                "map": "synth103",
                "tile_x": 30,
                "tile_y": 30,
                "pattern": "asymmetric_ramp",
                "amplitude": 100.0,
                "height_npy": str(height_path),
                "height_sha256": digest,
                "terrain_source_origin": "analytic_generated",
                "terrain_source_license": license_name,
                "terrain_source_rights_assertion": rights_assertion,
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _write_lit_profile(tmp_path: Path) -> Path:
    def track(index: int, rgb: tuple[float, float, float]) -> dict[str, object]:
        return {
            "track_id": index,
            "present": True,
            "rgb": {"r": rgb[0], "g": rgb[1], "b": rgb[2]},
        }

    payload = {
        "schema": LIT_PROFILE_SCHEMA,
        "source": {
            "kind": "archive_virtual_path",
            "label": "World/Maps/Azeroth/lights.lit",
            "virtual_path": "World/Maps/Azeroth/lights.lit",
            "sha256": "1" * 64,
        },
        "lit": {"version": 0x80000004, "track_count": 18, "group_stride": 0x1550},
        "selection": {
            "light_index": 0,
            "light_name": "Default",
            "group_index": 0,
            "group_kind": "Clear",
            "contributing_track_ids": [0, 1, 7],
        },
        "samples": [
            {
                "normalized_time": 0.25,
                "direct": track(0, (0.8, 0.6, 0.4)),
                "ambient": track(1, (0.2, 0.3, 0.4)),
                "fog": track(7, (0.1, 0.15, 0.2)),
            },
            {
                "normalized_time": 0.5,
                "direct": track(0, (1.0, 0.8, 0.6)),
                "ambient": track(1, (0.4, 0.5, 0.6)),
                "fog": track(7, (0.3, 0.4, 0.5)),
            },
        ],
    }
    path = tmp_path / "lit_profile.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_dbc_profile(tmp_path: Path) -> Path:
    names = [
        "Direct", "Ambient", "SkyTop", "SkyMiddle", "SkyMiddleToHorizon",
        "SkyAboveHorizon", "SkyHorizon", "Fog", "Unknown8", "Sun", "SunHalo",
        "Unknown11", "Cloud", "Unknown13", "Unknown14", "GroundShadow",
        "WaterLight", "WaterDark",
    ]
    tables = ["Light", "LightParams", "LightIntBand", "LightFloatBand", "LightSkybox"]
    payload = {
        "schema": DBC_PROFILE_SCHEMA,
        "source": {
            "exact_build": "1.12.1.5875",
            "tables": [
                {"table": name, "dbc_sha256": "2" * 64, "dbd_sha256": "3" * 64}
                for name in tables
            ],
        },
        "query": {
            "map_id": 0,
            "coordinate": {"world_position": {"x": 0.0, "y": 0.0, "z": 0.0}},
        },
        "samples": [
            {
                "evaluated_normalized0_to1": 0.5,
                "requested_time": {"input": "normalized:0.5"},
                "spatial_blend": {"global": {"weight": 1.0}, "local": None},
                "color_bands": [
                    {
                        "index": index,
                        "name": name,
                        "rgb": {
                            "r": 1.0 if index == 0 else 0.25,
                            "g": 0.5 if index == 0 else 0.25,
                            "b": 0.1 if index == 7 else 0.25,
                        },
                    }
                    for index, name in enumerate(names)
                ],
                "float_bands": [
                    {"index": index, "name": f"Float{index}", "value": float(index)}
                    for index in range(6)
                ],
                "primary_light_params": {"record_id": 12},
                "primary_skybox": None,
            }
        ],
    }
    path = tmp_path / "dbc_profile.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_profile_wraps_and_mcsh_only_modulates_clamped_directional_term() -> None:
    sample = evaluate_authored_day_night(1.5)
    assert sample.game_time == pytest.approx(0.5)
    assert sample.profile_revision == PROFILE_REVISION
    assert sample.evidence_state == EVIDENCE_STATE

    normals = np.stack(
        [
            sample.light_direction,
            sample.light_direction,
            -sample.light_direction,
            -sample.light_direction,
        ]
    )
    shadow = np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    lit = shade_terrain(np.ones(3, dtype=np.float32), normals, shadow, sample)
    ambient = sample.ambient_color * sample.ambient_intensity
    directional = sample.directional_color * sample.directional_intensity
    np.testing.assert_allclose(lit[0], ambient + directional, atol=1e-6)
    np.testing.assert_allclose(
        lit[1], ambient + directional * (1.0 - sample.mcsh_shadow_strength), atol=1e-6
    )
    np.testing.assert_allclose(lit[2], ambient, atol=1e-6)
    np.testing.assert_allclose(lit[3], ambient, atol=1e-6)


def test_solar_direction_keeps_a_fixed_north_west_bearing_instead_of_going_vertical_at_noon() -> None:
    # Spec 111: this module's direction formula had drifted from the corrected C#
    # TerrainSolarDirection (neither the north/south sign fix nor the fixed-bearing fix had been
    # ported). Assert both corrections hold here too: X and Y stay positive (north-west) at every
    # sampled hour, and noon does not collapse to a vertical, shadow-less sun.
    noon = _terrain_solar_direction(0.5)
    dawn = _terrain_solar_direction(11.0 / 24.0)
    dusk = _terrain_solar_direction(13.0 / 24.0)

    for direction in (noon, dawn, dusk):
        assert direction[0] > 0.0
        assert direction[1] > 0.0

    assert noon[0] == pytest.approx(noon[1])
    assert noon[0] / noon[1] == pytest.approx(dawn[0] / dawn[1], rel=1e-4)
    assert noon[0] / noon[1] == pytest.approx(dusk[0] / dusk[1], rel=1e-4)

    midnight = _terrain_solar_direction(0.0)
    mid_morning = _terrain_solar_direction(0.4)
    assert noon[2] > mid_morning[2] > midnight[2]


def test_grid_normal_transform_is_asymmetric_and_not_an_axis_noop() -> None:
    grid = np.asarray([[0.25, -0.75, 0.5]], dtype=np.float32)
    renderer = grid_normals_to_renderer(grid)
    np.testing.assert_array_equal(renderer, [[0.75, -0.25, 0.5]])
    assert GRID_TO_RENDERER_NORMAL_TRANSFORM == "grid_xyz_to_renderer_neg_y_neg_x_z_v1"

    builder = _load_builder()
    _yy, xx = np.mgrid[0:257, 0:257].astype(np.float32)
    grid_ramp_normal = builder.normals_from_height(xx)[128, 128]
    renderer_ramp_normal = grid_normals_to_renderer(grid_ramp_normal)
    assert grid_ramp_normal[0] < 0.0 and grid_ramp_normal[1] == pytest.approx(0.0)
    assert renderer_ramp_normal[0] == pytest.approx(0.0)
    assert renderer_ramp_normal[1] > 0.0


def test_authored_height_ray_shadow_is_deterministic_and_explicitly_not_client_exact() -> None:
    flat = np.zeros((257, 257), dtype=np.float32)
    assert not synthesize_authored_height_shadow(flat).any()

    step = flat.copy()
    step[:, :128] = 100.0
    first = synthesize_authored_height_shadow(step)
    second = synthesize_authored_height_shadow(step)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (256, 256)
    assert first.any()
    assert AUTHORED_MCSH_MODEL == "authored_height_ray_shadow_256_v1"
    assert AUTHORED_MCSH_EVIDENCE_STATE.endswith("not_client_exact")


def test_builder_writes_grouped_time_variants_and_clean_rights_contract(
    tmp_path: Path,
) -> None:
    builder = _load_builder()
    manifest = _write_manifest(tmp_path)
    output = tmp_path / "variants.zarr"
    contract = builder.build_synthetic_store(
        manifest_path=manifest,
        output_path=output,
        lighting_times=[0.25, 0.5],
        synthesize_mcsh=True,
        require_licensed_synthetic=True,
    )

    assert contract["tile_count"] == 2
    assert contract["source_tile_count"] == 1
    assert contract["rights_class"] == "clean_synthetic"
    assert contract["contains_raw_game_client_files"] is False
    assert contract["contains_client_derived_training_data"] is False
    assert contract["distribution_policy"] == "operator_declared_license_only"
    assert contract["source_license_summary"] == ["CC0-1.0"]

    rows = pq.read_table(output / "index.parquet").to_pylist()
    assert len(rows) == 2
    assert len({row["source_group_id"] for row in rows}) == 1
    assert [row["source_tile_id"] for row in rows] == [0, 0]
    assert [row["game_time"] for row in rows] == pytest.approx([0.25, 0.5])
    assert all(row["lighting_profile_revision"] == PROFILE_REVISION for row in rows)
    assert all(row["lighting_evidence_state"] == EVIDENCE_STATE for row in rows)
    assert all(row["lighting_normal_transform"] == GRID_TO_RENDERER_NORMAL_TRANSFORM for row in rows)
    assert all(row["shadow_evidence_state"] == AUTHORED_MCSH_EVIDENCE_STATE for row in rows)
    assert all(row["shadow_model"] == AUTHORED_MCSH_MODEL for row in rows)
    assert all(row["minimap_source"] == "synthesized_authored_lighting" for row in rows)

    root = zarr.open_group(str(output), mode="r")
    assert root["minimap_rgb"].shape == (2, 256, 256, 3)
    assert not np.array_equal(root["minimap_rgb"][0], root["minimap_rgb"][1])


def test_source_group_variants_cannot_cross_train_and_validation() -> None:
    index = [
        {"source_group_id": "source:a"},
        {"source_group_id": "source:a"},
        {"source_group_id": "source:b"},
    ]

    with pytest.raises(ValueError, match="Source-group partition leakage"):
        validate_source_group_split(index, [0, 2], [1])

    validate_source_group_split(index, [0, 1], [2])


def test_lit_profile_variants_are_hash_bound_and_private_byod(tmp_path: Path) -> None:
    builder = _load_builder()
    manifest = _write_manifest(tmp_path)
    profile = _write_lit_profile(tmp_path)
    output = tmp_path / "lit_variants.zarr"

    contract = builder.build_synthetic_store(
        manifest_path=manifest,
        output_path=output,
        lighting_profile=profile,
    )

    assert contract["rights_class"] == "private_byod"
    assert contract["contains_client_derived_training_data"] is True
    assert contract["lighting_color_source_kinds"] == ["client_lit_profile_artifact"]
    assert contract["tile_count"] == 2
    rows = pq.read_table(output / "index.parquet").to_pylist()
    assert len({row["source_group_id"] for row in rows}) == 1
    assert all(row["lighting_profile_artifact_sha256"] for row in rows)
    assert rows[0]["direction_evidence_state"].startswith("authored_")

    with pytest.raises(ValueError, match="rejects client-derived"):
        builder.build_synthetic_store(
            manifest_path=manifest,
            output_path=tmp_path / "illegal_clean.zarr",
            lighting_profile=profile,
            require_licensed_synthetic=True,
        )


def test_dbc_profile_loader_preserves_exact_colors_and_source_hashes(tmp_path: Path) -> None:
    profile = _write_dbc_profile(tmp_path)

    samples = load_lighting_profile_artifact(profile)

    assert len(samples) == 1
    sample = samples[0]
    assert sample.profile_revision == "light-dbc-exact-build-colors-v1"
    assert sample.color_source_kind == "client_light_dbc_profile_artifact"
    assert len(sample.color_source_sha256) == 64
    np.testing.assert_allclose(sample.directional_color, [1.0, 0.5, 0.25])
    np.testing.assert_allclose(sample.ambient_color, [0.25, 0.25, 0.25])
    np.testing.assert_allclose(sample.fog_color, [0.25, 0.25, 0.1])


def test_licensed_gate_rejects_unspecified_rights_hash_mismatch_and_capture_path(
    tmp_path: Path,
) -> None:
    builder = _load_builder()
    unspecified = _write_manifest(tmp_path, license_name="UNSPECIFIED")
    with pytest.raises(ValueError, match="terrain_source_license"):
        builder.build_synthetic_store(
            manifest_path=unspecified,
            output_path=tmp_path / "unspecified.zarr",
            lighting_times=[0.5],
            require_licensed_synthetic=True,
        )

    mismatch_dir = tmp_path / "mismatch"
    mismatch_dir.mkdir()
    mismatched = _write_manifest(mismatch_dir, corrupt_hash=True)
    with pytest.raises(ValueError, match="height_sha256 mismatch"):
        builder.build_synthetic_store(
            manifest_path=mismatched,
            output_path=tmp_path / "mismatch.zarr",
            lighting_times=[0.5],
            require_licensed_synthetic=True,
        )

    valid_dir = tmp_path / "valid"
    valid_dir.mkdir()
    valid = _write_manifest(valid_dir)
    with pytest.raises(ValueError, match="never from captured PNGs"):
        builder.build_synthetic_store(
            manifest_path=valid,
            output_path=tmp_path / "captured.zarr",
            minimap_dir=tmp_path / "captures",
            lighting_times=[0.5],
            require_licensed_synthetic=True,
        )


def test_captured_lit_rgb_requires_hash_bound_sidecar_and_stays_private_byod(
    tmp_path: Path,
) -> None:
    builder = _load_builder()
    manifest = _write_manifest(tmp_path)
    captures = tmp_path / "captures"
    captures.mkdir()
    png = captures / "synth103_30_30.png"
    Image.fromarray(np.full((256, 256, 3), 127, dtype=np.uint8)).save(png)
    png_sha256 = hashlib.sha256(png.read_bytes()).hexdigest()
    capture_root = tmp_path / "capture-root"
    adt = capture_root / "World" / "Maps" / "synth103" / "synth103_30_30.adt"
    adt.parent.mkdir(parents=True)
    adt.write_bytes(b"synthetic-adt-evidence")
    adt_sha256 = hashlib.sha256(adt.read_bytes()).hexdigest()
    sidecar = {
        "schema": "wowviewer-terrain-capture-lighting-v2",
        "renderer_contract": "mcnr_lambert_plus_mcsh_directional_v1",
        "lighting_source_kind": "client_lit_global_clear",
        "lighting_profile_revision": "lit-global-clear-colors-v1",
        "lighting_evidence_state": (
            "client_lit_colors_authored_direction_and_mcsh_strength"
        ),
        "tile": {"name": "synth103_30_30", "map": "synth103", "x": 30, "y": 30},
        "input": {
            "client_root": str(capture_root),
            "adt_path": "World/Maps/synth103/synth103_30_30.adt",
            "adt_sha256": adt_sha256,
        },
        "output": {"png_sha256": png_sha256, "width": 256, "height": 256},
        "camera": {
            "mode": "top_down_orthographic_one_adt_tile_v1",
            "position": [800.0, 900.0, 1000.0],
            "far_plane": 2000.0,
            "terrain_min_height": 0.0,
            "terrain_max_height": 100.0,
            "image_axis_contract": "right=adt_tile_x_positive;down=adt_tile_y_positive",
        },
        "lighting_source": {
            "identifier": "World/Maps/Azeroth/lights.lit",
            "sha256": "a" * 64,
            "lit_version": "0x80000004",
            "lit_light_index": 0,
            "lit_group_index": 0,
            "lit_time": 1008.0,
            "contributing_track_ids": [0, 1, 7],
            "direction_evidence_state": "authored_solar_direction_not_lit_data",
            "mcsh_evidence_state": "authored_mcsh_strength_not_client_exact",
        },
        "lighting": {
            "game_time": 0.35,
            "light_direction": [0.1, 0.2, 0.97],
            "directional_color": [0.5, 0.4, 0.3],
            "directional_intensity": 1.0,
            "ambient_color": [0.2, 0.2, 0.25],
            "ambient_intensity": 1.0,
            "fog_color": [0.3, 0.2, 0.2],
            "mcsh_shadow_strength": 0.6,
        },
    }
    Path(f"{png}.lighting.json").write_text(json.dumps(sidecar), encoding="utf-8")

    output = tmp_path / "captured.zarr"
    contract = builder.build_synthetic_store(
        manifest_path=manifest,
        output_path=output,
        minimap_dir=captures,
    )

    assert contract["rights_class"] == "private_byod"
    assert contract["contains_client_derived_training_data"] is True
    assert contract["capture_lighting_sidecar_count"] == 1
    row = pq.read_table(output / "index.parquet").to_pylist()[0]
    assert row["capture_lighting_source_sha256"] == "a" * 64
    assert row["capture_lit_track_ids"] == [0, 1, 7]
    assert row["lighting_profile_revision"] == "lit-global-clear-colors-v1"
    assert row["capture_adt_sha256"] == adt_sha256
    assert row["capture_camera_mode"] == "top_down_orthographic_one_adt_tile_v1"
    assert row["capture_image_axis_contract"] == (
        "right=adt_tile_x_positive;down=adt_tile_y_positive"
    )
    assert len(row["capture_lighting_source_identity_sha256"]) == 64

    sidecar_path = Path(f"{png}.lighting.json")
    bad_sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    bad_sidecar["input"]["adt_sha256"] = "0" * 64
    sidecar_path.write_text(json.dumps(bad_sidecar), encoding="utf-8")
    with pytest.raises(ValueError, match="capture ADT hash"):
        builder.build_synthetic_store(
            manifest_path=manifest,
            output_path=tmp_path / "bad-adt-hash.zarr",
            minimap_dir=captures,
        )

    bad_sidecar = dict(sidecar)
    bad_sidecar["camera"] = {
        **sidecar["camera"],
        "image_axis_contract": "right=unknown;down=unknown",
    }
    sidecar_path.write_text(json.dumps(bad_sidecar), encoding="utf-8")
    with pytest.raises(ValueError, match="axis/orientation"):
        builder.build_synthetic_store(
            manifest_path=manifest,
            output_path=tmp_path / "bad-orientation.zarr",
            minimap_dir=captures,
        )

    bad_sidecar = dict(sidecar)
    bad_sidecar["lighting_source"] = {
        **sidecar["lighting_source"],
        "sha256": "not-a-digest",
    }
    sidecar_path.write_text(json.dumps(bad_sidecar), encoding="utf-8")
    with pytest.raises(ValueError, match="lighting_source.sha256"):
        builder.build_synthetic_store(
            manifest_path=manifest,
            output_path=tmp_path / "bad-light-source.zarr",
            minimap_dir=captures,
        )

    bad_sidecar = dict(sidecar)
    bad_sidecar["lighting_source"] = {
        **sidecar["lighting_source"],
        "contributing_track_ids": [0, 1],
    }
    sidecar_path.write_text(json.dumps(bad_sidecar), encoding="utf-8")
    with pytest.raises(ValueError, match="direct, ambient, and fog tracks"):
        builder.build_synthetic_store(
            manifest_path=manifest,
            output_path=tmp_path / "bad-light-tracks.zarr",
            minimap_dir=captures,
        )

    sidecar_path.unlink()
    with pytest.raises(ValueError, match="missing required lighting sidecar"):
        builder.build_synthetic_store(
            manifest_path=manifest,
            output_path=tmp_path / "missing-sidecar.zarr",
            minimap_dir=captures,
        )
