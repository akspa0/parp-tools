"""Spec 120 Minimap OBB Object Detector & Metadata Sidecar Package."""

from harvester.spec120.obb_contract import (
    COARSE_CLASS_INDEX,
    COARSE_INDEX_CLASS,
    STAGE_OBB_DETECTOR,
    STAGE_SIDECAR_EXPORTER,
    ObbContractError,
    derive_coarse_class,
    format_sidecar_item,
    placement_to_obb_target,
    tile_pixels_to_world,
    validate_sidecar_schema,
    world_to_tile_pixels,
)

__all__ = [
    "COARSE_CLASS_INDEX",
    "COARSE_INDEX_CLASS",
    "STAGE_OBB_DETECTOR",
    "STAGE_SIDECAR_EXPORTER",
    "ObbContractError",
    "derive_coarse_class",
    "format_sidecar_item",
    "placement_to_obb_target",
    "tile_pixels_to_world",
    "validate_sidecar_schema",
    "world_to_tile_pixels",
]
