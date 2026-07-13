"""Fail-closed, stage-specific curation for Spec 102 numeric models."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import numpy as np

from harvester.spec102.strict_target_contract import REQUIRED_STRICT_OBJECT_TARGET_VERSION

LIQUID_FLAG_MASK = np.int64(0x3C)
STRICT_OBJECT_TARGET_COMPLETE_STATUSES = frozenset({"CompleteEmpty", "CompleteVisible"})
STRICT_LIQUID_EVIDENCE_DRY = "Dry"
STRICT_TARGET_LIQUID_COUNTER_FIELDS = (
    "strict_target_liquid_covered_pixel_count",
    "strict_target_liquid_surface_unknown_pixel_count",
    "strict_target_liquid_covered_fragment_count",
    "strict_target_liquid_hidden_fragment_count",
    "strict_target_liquid_above_surface_fragment_count",
    "strict_target_liquid_unknown_fragment_count",
)
STRICT_TARGET_FRAGMENT_TRACE_FIELDS = (
    "strict_target_fragment_trace_schema",
    "strict_target_fragment_count",
    "strict_target_fragment_sha256",
    "strict_target_assets_json",
    "strict_target_unresolved_placements_json",
    "strict_target_fragment_trace_arrays_present",
    "strict_target_fragment_trace_validated",
    "strict_target_fragment_trace_sidecar_start",
    "strict_target_fragment_trace_sidecar_end",
    "strict_target_fragment_trace_source_sidecar_sha256",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# M0 receives RGB only and has no per-pixel valid-loss mask.  Therefore even
# one water-covered RGB pixel could become an implicit background negative for
# its object target.  Do not turn liquid into a forward input or a hidden loss
# heuristic: the initial M0 corpus is dry-only until masking is implemented.
DEFAULT_M0_MAX_LIQUID_COVERAGE = 0.0


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
    strict_mask_257: np.ndarray,
    strict_target_version: str,
    strict_target_materialized: bool,
    strict_target_status: str,
    strict_target_arrays_present: bool,
    strict_target_geometry_unresolved_placement_count: int = 0,
    strict_target_fallback_required_placement_count: int = 0,
    strict_target_terrain_unknown_pixel_count: int = 0,
    strict_target_liquid_evidence_status: str = "missing",
    strict_target_liquid_covered_pixel_count: int = 0,
    strict_target_liquid_surface_unknown_pixel_count: int = 0,
    strict_target_liquid_covered_fragment_count: int = 0,
    strict_target_liquid_hidden_fragment_count: int = 0,
    strict_target_liquid_above_surface_fragment_count: int = 0,
    strict_target_liquid_unknown_fragment_count: int = 0,
    strict_target_fragment_trace_schema: str = "missing",
    strict_target_fragment_count: int = 0,
    strict_target_fragment_sha256: str = "missing",
    strict_target_assets_json: str = "[]",
    strict_target_unresolved_placements_json: str = "[]",
    strict_target_fragment_trace_arrays_present: bool = False,
    strict_target_fragment_trace_validated: bool = False,
    strict_target_fragment_trace_sidecar_start: int = -1,
    strict_target_fragment_trace_sidecar_end: int = -1,
    strict_target_fragment_trace_source_sidecar_sha256: str = "missing",
    liquid_mask_256: np.ndarray,
    liquid_signal_present: bool,
    mcnk_flags_16: np.ndarray,
    normal_xyz_257: np.ndarray,
    height_257: np.ndarray,
    height_repaired: bool,
    mismatch_reason: str | None,
    has_paired_wdl: bool,
    max_liquid_coverage: float = DEFAULT_M0_MAX_LIQUID_COVERAGE,
) -> TileEligibility:
    if not 0.0 <= max_liquid_coverage < 1.0:
        raise ValueError("max_liquid_coverage must be in [0, 1)")
    reasons: list[str] = []
    rgb = np.asarray(minimap_rgb)
    strict = np.asarray(strict_mask_257)
    liquid = np.asarray(liquid_mask_256)
    flags = np.asarray(mcnk_flags_16, dtype=np.int64)
    normals = np.asarray(normal_xyz_257)
    height = np.asarray(height_257)

    def strict_count(value: int, *, field: str) -> int | None:
        try:
            count = int(value)
        except (TypeError, ValueError):
            reasons.append(f"invalid_{field}")
            return None
        if count < 0:
            reasons.append(f"invalid_{field}")
            return None
        return count

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
    # A literal zero-filled tensor is a negative label only when the raw V18
    # row says it came from the new transformed-triangle / raw-MCVT-Z path.
    # The historical object_precise_mask is deliberately never consulted here.
    if strict_target_version != REQUIRED_STRICT_OBJECT_TARGET_VERSION:
        reasons.append("strict_target_version_mismatch")
    if strict_target_status not in STRICT_OBJECT_TARGET_COMPLETE_STATUSES:
        reasons.append("incomplete_strict_geometry_target")
    if not strict_target_materialized:
        reasons.append("unmaterialized_strict_geometry_target")
    if not strict_target_arrays_present:
        reasons.append("strict_target_arrays_not_materialized")
    unresolved = strict_count(
        strict_target_geometry_unresolved_placement_count,
        field="strict_target_geometry_unresolved_placement_count",
    )
    fallback_required = strict_count(
        strict_target_fallback_required_placement_count,
        field="strict_target_fallback_required_placement_count",
    )
    terrain_unknown = strict_count(
        strict_target_terrain_unknown_pixel_count,
        field="strict_target_terrain_unknown_pixel_count",
    )
    if unresolved not in (None, 0):
        reasons.append("strict_target_unresolved_geometry")
    if fallback_required not in (None, 0):
        reasons.append("strict_target_requires_forbidden_fallback")
    if terrain_unknown not in (None, 0):
        reasons.append("strict_target_unknown_terrain")
    if strict_target_liquid_evidence_status != STRICT_LIQUID_EVIDENCE_DRY:
        reasons.append("strict_target_liquid_evidence_not_dry")
    liquid_evidence_counts = {
        "strict_target_liquid_covered_pixel_count": strict_target_liquid_covered_pixel_count,
        "strict_target_liquid_surface_unknown_pixel_count": strict_target_liquid_surface_unknown_pixel_count,
        "strict_target_liquid_covered_fragment_count": strict_target_liquid_covered_fragment_count,
        "strict_target_liquid_hidden_fragment_count": strict_target_liquid_hidden_fragment_count,
        "strict_target_liquid_above_surface_fragment_count": strict_target_liquid_above_surface_fragment_count,
        "strict_target_liquid_unknown_fragment_count": strict_target_liquid_unknown_fragment_count,
    }
    for field, value in liquid_evidence_counts.items():
        count = strict_count(value, field=field)
        if count not in (None, 0):
            reasons.append(f"{field}_nonzero")
    trace_count = strict_count(
        strict_target_fragment_count,
        field="strict_target_fragment_count",
    )
    try:
        trace_start = int(strict_target_fragment_trace_sidecar_start)
        trace_end = int(strict_target_fragment_trace_sidecar_end)
    except (TypeError, ValueError):
        trace_start = trace_end = -1
        reasons.append("invalid_strict_target_fragment_trace_sidecar_offsets")
    if strict_target_fragment_trace_schema != REQUIRED_STRICT_OBJECT_TARGET_VERSION:
        reasons.append("strict_target_fragment_trace_schema_mismatch")
    if not strict_target_fragment_trace_arrays_present:
        reasons.append("strict_target_fragment_trace_arrays_missing")
    if not strict_target_fragment_trace_validated:
        reasons.append("strict_target_fragment_trace_not_validated")
    if not _SHA256_RE.fullmatch(str(strict_target_fragment_sha256 or "")):
        reasons.append("invalid_strict_target_fragment_trace_sha256")
    if trace_count is not None and (trace_start < 0 or trace_end < trace_start or trace_end - trace_start != trace_count):
        reasons.append("invalid_strict_target_fragment_trace_sidecar_offsets")
    if not _SHA256_RE.fullmatch(str(strict_target_fragment_trace_source_sidecar_sha256 or "")):
        reasons.append("invalid_strict_target_fragment_trace_source_sidecar_sha256")
    try:
        assets = json.loads(str(strict_target_assets_json))
        unresolved = json.loads(str(strict_target_unresolved_placements_json))
    except (TypeError, ValueError, json.JSONDecodeError):
        assets = unresolved = None
    if not isinstance(assets, list):
        reasons.append("invalid_strict_target_fragment_trace_assets")
    if not isinstance(unresolved, list):
        reasons.append("invalid_strict_target_fragment_trace_unresolved_placements")
    elif unresolved:
        reasons.append("strict_target_fragment_trace_unresolved_placements_present")
    if strict.shape != (257, 257) or not np.isfinite(strict).all():
        reasons.append("invalid_strict_geometry_mask")
    elif float(strict.min()) < 0.0 or float(strict.max()) > 1.0:
        reasons.append("strict_geometry_mask_out_of_range")
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
    if liquid_coverage > max_liquid_coverage:
        reasons.append("terrain_occluded_by_liquid")
    # A zero-valued liquid mask is a valid, useful fact: it means no liquid
    # covers this tile.  `has_liquid_mask` records that the source was decoded,
    # not that this particular tile must contain water.
    if not liquid_signal_present and liquid_coverage > 0.0:
        reasons.append("liquid_mask_without_source_provenance")
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
        "invalid_or_placeholder_minimap", "incomplete_strict_geometry_target",
        "strict_target_version_mismatch",
        "unmaterialized_strict_geometry_target", "strict_target_arrays_not_materialized",
        "strict_target_unresolved_geometry", "strict_target_requires_forbidden_fallback",
        "strict_target_unknown_terrain", "invalid_strict_geometry_mask",
        "strict_geometry_mask_out_of_range", "invalid_liquid_mask",
        "invalid_mcnk_flags", "terrain_occluded_by_liquid",
        "liquid_mask_without_source_provenance", "known_signal_mismatch",
        "minimap_near_uniform_or_water", "visual_ocean_dominant",
        "visual_ocean_without_adt_liquid",
        "invalid_strict_target_geometry_unresolved_placement_count",
        "invalid_strict_target_fallback_required_placement_count",
        "invalid_strict_target_terrain_unknown_pixel_count",
        "strict_target_liquid_evidence_not_dry",
        "strict_target_fragment_trace_schema_mismatch",
        "strict_target_fragment_trace_arrays_missing",
        "strict_target_fragment_trace_not_validated",
        "invalid_strict_target_fragment_count",
        "invalid_strict_target_fragment_trace_sha256",
        "invalid_strict_target_fragment_trace_sidecar_offsets",
        "invalid_strict_target_fragment_trace_source_sidecar_sha256",
        "invalid_strict_target_fragment_trace_assets",
        "invalid_strict_target_fragment_trace_unresolved_placements",
        "strict_target_fragment_trace_unresolved_placements_present",
        *(f"{field}_nonzero" for field in STRICT_TARGET_LIQUID_COUNTER_FIELDS),
        *(f"invalid_{field}" for field in STRICT_TARGET_LIQUID_COUNTER_FIELDS),
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
