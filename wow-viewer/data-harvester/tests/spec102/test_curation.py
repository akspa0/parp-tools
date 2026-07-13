from __future__ import annotations

import numpy as np

from harvester.spec102.curation import classify_tile


def tile(**overrides):
    values = {
        "minimap_rgb": np.arange(256 * 256 * 3, dtype=np.uint8).reshape(256, 256, 3),
        "precise_mask_257": np.zeros((257, 257), dtype=np.float32),
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


def test_partial_visible_coast_can_train_m0_but_not_height_without_liquid_facts() -> None:
    rgb = np.arange(256 * 256 * 3, dtype=np.uint8).reshape(256, 256, 3)
    rgb[:, :96] = (25, 55, 95)
    result = tile(minimap_rgb=rgb)
    assert result.eligible_m0
    assert not result.eligible_h2
    assert "visual_ocean_without_adt_liquid" in result.rejection_reasons


def test_mh2o_mask_does_not_require_mcnk_liquid_flags() -> None:
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[:, :64] = 255
    result = tile(liquid_mask_256=mask, liquid_signal_present=True)
    assert result.eligible_m0 and result.eligible_h2
    assert "declared_liquid_signal_empty" not in result.rejection_reasons
