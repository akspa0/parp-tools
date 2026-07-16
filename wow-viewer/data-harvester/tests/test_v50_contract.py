from __future__ import annotations

import pytest
import zarr

from harvester.v50_contract import (
    MIXED_STORE_SCHEMA,
    MODEL_FAMILY,
    require_metadata_release,
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
