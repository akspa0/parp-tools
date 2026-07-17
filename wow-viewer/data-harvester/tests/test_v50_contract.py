from __future__ import annotations

import pytest
import zarr

from harvester.v50_contract import (
    MIXED_STORE_SCHEMA,
    MODEL_FAMILY,
    require_metadata_release,
    migration_policy_for_signal,
    require_liquid_source_provenance,
    require_store_release,
    validate_release,
)


def test_v50_release_requires_current_family_schema_and_increment(tmp_path) -> None:
    group = zarr.open_group(str(tmp_path / "v50.1.zarr"), mode="w")
    group.attrs.update({"model_family": MODEL_FAMILY, "release": "v50.1", "schema": MIXED_STORE_SCHEMA})
    require_store_release(group, "v50.1", store=tmp_path / "v50.1.zarr")
    with pytest.raises(ValueError, match="requested v50 release"):
        require_store_release(group, "v50.2", store=tmp_path / "v50.1.zarr")
    with pytest.raises(ValueError, match="v50.N"):
        validate_release("v8")


def test_v50_archive_or_checkpoint_cannot_cross_release() -> None:
    require_metadata_release({"model_family": MODEL_FAMILY, "release": "v50.1"}, "v50.1", artifact="test")
    with pytest.raises(ValueError, match="not compatible"):
        require_metadata_release({"model_family": MODEL_FAMILY, "release": "v50.1"}, "v50.2", artifact="test")


def test_v50_liquid_signals_are_fresh_only_and_wl_sources_require_complete_provenance() -> None:
    assert migration_policy_for_signal("liquid_mask") == "fresh-only"
    assert migration_policy_for_signal("liquid_height") == "fresh-only"
    assert migration_policy_for_signal("height_257") == "copy-if-verified"
    require_liquid_source_provenance(
        [
            "wl_liquid_mask",
            "wl_liquid_height",
            "wl_liquid_surface_quads_v1",
            "wl_liquid_above_terrain_v1",
            "wl_liquid_basic_type_header_v1",
        ],
        artifact="corrected shard",
    )
    require_liquid_source_provenance(["unified_liquid_mask", "mclq_liquid_height"], artifact="non-WL shard")
    with pytest.raises(ValueError, match="re-extract"):
        require_liquid_source_provenance(["wl_liquid_mask", "wl_liquid_height"], artifact="historical shard")
