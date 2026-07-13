"""Fail-closed, stage-specific curation for Spec 102 numeric models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LIQUID_FLAG_MASK = np.int64(0x3C)


@dataclass(frozen=True)
class TileEligibility:
    liquid_coverage: float
    liquid_flag_chunk_coverage: float
    visible_terrain_coverage: float
    minimap_dominant_color_fraction: float
    minimap_blue_fraction: float
    liquid_signal_present: bool
    eligible_m0: bool
    eligible_w1: bool
    eligible_h2: bool
    rejection_reasons: tuple[str, ...]


def classify_tile(
    *,
    minimap_rgb: np.ndarray,
    precise_mask_257: np.ndarray,
    liquid_mask_256: np.ndarray,
    liquid_signal_present: bool,
    mcnk_flags_16: np.ndarray,
    normal_xyz_257: np.ndarray,
    height_257: np.ndarray,
    height_repaired: bool,
    mismatch_reason: str | None,
    has_paired_wdl: bool,
    max_liquid_coverage: float = 0.80,
) -> TileEligibility:
    reasons: list[str] = []
    rgb = np.asarray(minimap_rgb)
    precise = np.asarray(precise_mask_257)
    liquid = np.asarray(liquid_mask_256)
    flags = np.asarray(mcnk_flags_16, dtype=np.int64)
    normals = np.asarray(normal_xyz_257)
    height = np.asarray(height_257)

    if rgb.shape != (256, 256, 3) or not np.isfinite(rgb).all() or float(rgb.std()) < 1.0:
        reasons.append("invalid_or_placeholder_minimap")
        dominant_color_fraction = 1.0
        blue_fraction = 0.0
    else:
        rgb_u8 = np.asarray(rgb, dtype=np.uint8)
        quantized = rgb_u8 // 8
        packed = (
            (quantized[..., 0].astype(np.int32) << 10)
            | (quantized[..., 1].astype(np.int32) << 5)
            | quantized[..., 2].astype(np.int32)
        )
        counts = np.bincount(packed.reshape(-1), minlength=32768)
        dominant_color_fraction = float(counts.max(initial=0) / packed.size)
        red, green, blue = (rgb_u8[..., index].astype(np.float32) for index in range(3))
        blue_fraction = float(((blue > red * 1.20) & (blue > green * 1.08)).mean())
        if dominant_color_fraction >= 0.80:
            reasons.append("minimap_near_uniform_or_water")
    if precise.shape != (257, 257) or not np.isfinite(precise).all():
        reasons.append("invalid_precise_object_mask")
    elif float(precise.min()) < 0.0 or float(precise.max()) > 1.0:
        reasons.append("precise_object_mask_out_of_range")
    if liquid.shape != (256, 256):
        reasons.append("invalid_liquid_mask")
        liquid_coverage = 1.0
    else:
        liquid_coverage = float((liquid > 127).mean())
    if flags.shape != (16, 16):
        reasons.append("invalid_mcnk_flags")
        flag_coverage = 0.0
    else:
        flag_coverage = float(((flags & LIQUID_FLAG_MASK) != 0).mean())
    visible = 1.0 - liquid_coverage
    if liquid_coverage >= max_liquid_coverage:
        reasons.append("terrain_occluded_by_liquid")
    if liquid_signal_present and liquid_coverage <= 0.0:
        reasons.append("declared_liquid_signal_empty")
    if not liquid_signal_present and blue_fraction >= 0.25:
        reasons.append("visual_ocean_without_adt_liquid")
    if not liquid_signal_present and blue_fraction >= 0.80:
        reasons.append("visual_ocean_dominant")
    if height.shape != (257, 257) or not np.isfinite(height).all():
        reasons.append("invalid_height_vertices")
    if normals.shape != (257, 257, 3) or not np.any(normals):
        reasons.append("missing_numeric_normals")
    if height_repaired:
        reasons.append("synthetic_height_repair")
    if mismatch_reason:
        reasons.append("known_signal_mismatch")

    base_reasons = {
        "invalid_or_placeholder_minimap", "invalid_precise_object_mask",
        "precise_object_mask_out_of_range", "invalid_liquid_mask",
        "invalid_mcnk_flags", "terrain_occluded_by_liquid",
        "declared_liquid_signal_empty", "known_signal_mismatch",
        "minimap_near_uniform_or_water", "visual_ocean_dominant",
    }
    m0_ok = not any(reason in base_reasons for reason in reasons)
    terrain_reasons = base_reasons | {
        "invalid_height_vertices", "missing_numeric_normals", "synthetic_height_repair",
        "visual_ocean_without_adt_liquid",
    }
    h2_ok = not any(reason in terrain_reasons for reason in reasons)
    w1_ok = h2_ok and has_paired_wdl
    if h2_ok and not has_paired_wdl:
        reasons.append("missing_real_paired_wdl")
    return TileEligibility(
        liquid_coverage=liquid_coverage,
        liquid_flag_chunk_coverage=flag_coverage,
        visible_terrain_coverage=visible,
        minimap_dominant_color_fraction=dominant_color_fraction,
        minimap_blue_fraction=blue_fraction,
        liquid_signal_present=bool(liquid_signal_present),
        eligible_m0=m0_ok,
        eligible_w1=w1_ok,
        eligible_h2=h2_ok,
        rejection_reasons=tuple(reasons),
    )
