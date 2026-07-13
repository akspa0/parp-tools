from __future__ import annotations

import json

import numpy as np
import pytest
import zarr

from harvester.spec102.m0 import STRICT_OBJECT_TARGET_KEY
from harvester.spec102.m0_scope import M0_BUILD_LOCAL_SCHEMA, STRICT_LIQUID_EVIDENCE_RULE
from harvester.spec102.signal_audit import (
    AUDIT_SCHEMA,
    M0_AUDITED_SIGNAL_KEYS,
    current_audited_signal_fingerprint,
    project_placement_to_terrain,
    render_signal_panel,
    sha256_file,
    validate_m0_training_audit,
)
from harvester.spec102.strict_target_contract import REQUIRED_STRICT_OBJECT_TARGET_VERSION


def test_placement_projection_reports_below_terrain_clearance() -> None:
    terrain = np.full((257, 257), 200.0, dtype=np.float32)
    placement = {
        "posX": 10.0, "posY": 10.0, "posZ": 100.0,
        "bbMinX": 0.0, "bbMinY": 0.0, "bbMinZ": 90.0,
        "bbMaxX": 20.0, "bbMaxY": 20.0, "bbMaxZ": 110.0,
    }
    relation = project_placement_to_terrain(placement, tile_x=0, tile_y=0, terrain_height_257=terrain)
    assert relation is not None
    assert relation.top_source in {"bounds_y", "bounds_z"}
    assert relation.clearance < -0.5


def test_signal_panel_contains_every_numeric_signal_column() -> None:
    sample = {
        "metadata": {"row": 1, "build": "b", "map": "m", "tile_x": 2, "tile_y": 3, "liquid_source": "mh2o"},
        "minimap_rgb": np.zeros((256, 256, 3), dtype=np.uint8),
        STRICT_OBJECT_TARGET_KEY: np.zeros((257, 257), dtype=np.float32),
        "liquid_mask_256": np.zeros((256, 256), dtype=np.uint8),
        "liquid_height_256": np.zeros((256, 256), dtype=np.float32),
        "mcnk_flags_16": np.zeros((16, 16), dtype=np.int32),
        "normal_xyz_257": np.zeros((257, 257, 3), dtype=np.int8),
        "height_257": np.zeros((257, 257), dtype=np.float32),
    }
    panel = render_signal_panel([sample], split="validation_map", source_label="test")
    assert panel.size == (2048, 362)


def test_m0_training_audit_requires_exact_build_local_input_hashes(tmp_path) -> None:
    store = tmp_path / "numeric.zarr"
    group = zarr.open_group(str(store), mode="w")
    arrays = {
        "minimap_rgb": np.zeros((1, 256, 256, 3), dtype=np.uint8),
        STRICT_OBJECT_TARGET_KEY: np.zeros((1, 257, 257), dtype=np.float32),
        "object_geometry_visible_top_elevation_257": np.zeros((1, 257, 257), dtype=np.float32),
        "object_geometry_visible_terrain_elevation_257": np.zeros((1, 257, 257), dtype=np.float32),
        "object_geometry_visible_source_257": np.zeros((1, 257, 257), dtype=np.uint8),
        "liquid_mask_256": np.zeros((1, 256, 256), dtype=np.uint8),
        "liquid_height_256": np.zeros((1, 256, 256), dtype=np.float32),
        "mcnk_flags_16": np.ones((1, 16, 16), dtype=np.int32),
        "normal_xyz_257": np.zeros((1, 257, 257, 3), dtype=np.int8),
        "height_257": np.zeros((1, 257, 257), dtype=np.float32),
    }
    for name, value in arrays.items():
        group.create_array(name, data=value)
    (store / "contract.json").write_text("{}", encoding="utf-8")
    (store / "index.parquet").write_bytes(b"index-v1")
    split = tmp_path / "m0-split.json"
    split.write_text("{}", encoding="utf-8")
    scope = {
        "schema": M0_BUILD_LOCAL_SCHEMA,
        "kind": "build_local",
        "allowed_builds": ["3_3_5_12340"],
        "required_splits": ["train", "validation_map", "test_build_local"],
        "cross_era_claim": False,
        "target_quality_basis": "strict_transformed_geometry_terrain_visible",
        "target_version": REQUIRED_STRICT_OBJECT_TARGET_VERSION,
        "terrain_visibility_proof": "per_fragment_transformed_geometry_vs_raw_mcvt_z",
        "liquid_evidence_rule": STRICT_LIQUID_EVIDENCE_RULE,
        "validation_map": "Northrend",
        "test_map": "Kalimdor",
    }
    report = {
        "schema": AUDIT_SCHEMA,
        "store": str(store.resolve()),
        "split_manifest": str(split.resolve()),
        "store_contract_sha256": sha256_file(store / "contract.json"),
        "store_index_sha256": sha256_file(store / "index.parquet"),
        "split_manifest_sha256": sha256_file(split),
        "scoped_signal_fingerprint": current_audited_signal_fingerprint(store, scoped_rows=[0]),
        "safe_for_m0_build_local_training": True,
        "m0_training_scope": scope,
        "object_target_provenance": {
            "build_local_strict_target_accepted": True,
            "terrain_occlusion_clipped": True,
            "per_pixel_object_top_elevation": True,
            "target_version": REQUIRED_STRICT_OBJECT_TARGET_VERSION,
            "liquid_evidence_dry_only": True,
        },
    }
    report_path = tmp_path / "audit.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    assert validate_m0_training_audit(
        report_path, store=store, split_manifest=split, expected_scope=scope, scoped_rows=[0],
    )["safe_for_m0_build_local_training"]
    assert set(M0_AUDITED_SIGNAL_KEYS) == set(arrays)
    group["minimap_rgb"][0, 0, 0, 0] = 1
    with pytest.raises(RuntimeError, match="signal content changed"):
        validate_m0_training_audit(
            report_path, store=store, split_manifest=split, expected_scope=scope, scoped_rows=[0],
        )
    group["minimap_rgb"][0, 0, 0, 0] = 0
    (store / "index.parquet").write_bytes(b"index-v2")
    with pytest.raises(RuntimeError, match="stale"):
        validate_m0_training_audit(
            report_path, store=store, split_manifest=split, expected_scope=scope, scoped_rows=[0],
        )
