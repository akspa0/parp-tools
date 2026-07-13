from __future__ import annotations

import numpy as np

from harvester.spec102.curation import classify_tile
from harvester.spec102.strict_target_contract import REQUIRED_STRICT_OBJECT_TARGET_VERSION


def tile(**overrides):
    values = {
        "minimap_rgb": np.arange(256 * 256 * 3, dtype=np.uint8).reshape(256, 256, 3),
        "strict_mask_257": np.zeros((257, 257), dtype=np.float32),
        "strict_target_version": REQUIRED_STRICT_OBJECT_TARGET_VERSION,
        "strict_target_materialized": True,
        "strict_target_status": "CompleteEmpty",
        "strict_target_arrays_present": True,
        "strict_target_liquid_evidence_status": "Dry",
        "strict_target_liquid_covered_pixel_count": 0,
        "strict_target_liquid_surface_unknown_pixel_count": 0,
        "strict_target_liquid_covered_fragment_count": 0,
        "strict_target_liquid_hidden_fragment_count": 0,
        "strict_target_liquid_above_surface_fragment_count": 0,
        "strict_target_liquid_unknown_fragment_count": 0,
        "strict_target_fragment_trace_schema": REQUIRED_STRICT_OBJECT_TARGET_VERSION,
        "strict_target_fragment_count": 0,
        "strict_target_fragment_sha256": "a" * 64,
        "strict_target_assets_json": "[]",
        "strict_target_unresolved_placements_json": "[]",
        "strict_target_fragment_trace_arrays_present": True,
        "strict_target_fragment_trace_validated": True,
        "strict_target_fragment_trace_sidecar_start": 0,
        "strict_target_fragment_trace_sidecar_end": 0,
        "strict_target_fragment_trace_source_sidecar_sha256": "b" * 64,
        "liquid_mask_256": np.zeros((256, 256), dtype=np.uint8),
        "liquid_signal_present": False,
        "mcnk_flags_16": np.zeros((16, 16), dtype=np.int32),
        "normal_xyz_257": np.ones((257, 257, 3), dtype=np.int8),
        "height_257": np.ones((257, 257), dtype=np.float32),
        "height_repaired": False,
        "mismatch_reason": None,
        "has_paired_wdl": True,
    }
    values.update(overrides)
    return classify_tile(**values)


def test_water_occluded_tile_is_ineligible_for_every_stage() -> None:
    result = tile(
        liquid_mask_256=np.full((256, 256), 255, dtype=np.uint8),
        mcnk_flags_16=np.full((16, 16), 0x08, dtype=np.int32),
    )
    assert not result.eligible_m0 and not result.eligible_w1 and not result.eligible_h2
    assert "terrain_occluded_by_liquid" in result.rejection_reasons


def test_unmaterialized_strict_target_is_never_silently_a_negative_label() -> None:
    result = tile(strict_target_materialized=False)
    assert not result.eligible_m0
    assert "unmaterialized_strict_geometry_target" in result.rejection_reasons


def test_incomplete_or_fallback_strict_target_is_rejected_even_when_mask_is_zero() -> None:
    result = tile(
        strict_target_status="IncompleteGeometry",
        strict_target_fallback_required_placement_count=1,
    )
    assert not result.eligible_m0
    assert "incomplete_strict_geometry_target" in result.rejection_reasons
    assert "strict_target_requires_forbidden_fallback" in result.rejection_reasons


def test_non_dry_liquid_evidence_is_preserved_as_a_rejection_not_a_negative_label() -> None:
    result = tile(
        strict_target_liquid_evidence_status="LiquidCovered",
        strict_target_liquid_covered_fragment_count=1,
    )
    assert not result.eligible_m0
    assert "strict_target_liquid_evidence_not_dry" in result.rejection_reasons
    assert "strict_target_liquid_covered_fragment_count_nonzero" in result.rejection_reasons


def test_non_required_target_version_is_rejected() -> None:
    result = tile(strict_target_version="v1")
    assert not result.eligible_m0
    assert "strict_target_version_mismatch" in result.rejection_reasons


def test_repaired_or_mismatched_height_never_enters_terrain_models() -> None:
    result = tile(height_repaired=True, mismatch_reason="height_flat_vs_normal_varied")
    assert not result.eligible_m0
    assert not result.eligible_w1 and not result.eligible_h2
    assert "synthetic_height_repair" in result.rejection_reasons
    assert "known_signal_mismatch" in result.rejection_reasons


def test_w1_requires_real_paired_wdl_arrays() -> None:
    result = tile(has_paired_wdl=False)
    assert result.eligible_m0 and result.eligible_h2
    assert not result.eligible_w1
    assert "missing_real_paired_wdl" in result.rejection_reasons


def test_visually_uniform_blue_minimap_rejects_missing_liquid_facts() -> None:
    rgb = np.full((256, 256, 3), (25, 55, 95), dtype=np.uint8)
    result = tile(minimap_rgb=rgb)
    assert not result.eligible_m0 and not result.eligible_h2
    assert "minimap_near_uniform_or_water" in result.rejection_reasons
    assert "visual_ocean_without_adt_liquid" in result.rejection_reasons
    assert "visual_ocean_dominant" in result.rejection_reasons


def test_partial_visual_ocean_without_adt_liquid_rejects_m0() -> None:
    rgb = np.arange(256 * 256 * 3, dtype=np.uint8).reshape(256, 256, 3)
    rgb[:, :96] = (25, 55, 95)
    result = tile(minimap_rgb=rgb)
    assert not result.eligible_m0
    assert not result.eligible_h2
    assert "visual_ocean_without_adt_liquid" in result.rejection_reasons


def test_any_mh2o_mask_rejects_m0_without_a_valid_loss_mask() -> None:
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[:, :1] = 255
    result = tile(liquid_mask_256=mask, liquid_signal_present=True)
    assert not result.eligible_m0
    assert "terrain_occluded_by_liquid" in result.rejection_reasons


def test_decoded_empty_liquid_mask_is_a_valid_dry_tile_fact() -> None:
    result = tile(liquid_signal_present=True)
    assert result.eligible_m0
    assert "declared_liquid_signal_empty" not in result.rejection_reasons


def test_any_detected_liquid_is_not_initial_m0_terrain_shadow_training() -> None:
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[0, 0] = 255
    result = tile(liquid_mask_256=mask, liquid_signal_present=True)
    assert not result.eligible_m0
    assert result.visible_terrain_coverage < 1.0
    assert "terrain_occluded_by_liquid" in result.rejection_reasons
