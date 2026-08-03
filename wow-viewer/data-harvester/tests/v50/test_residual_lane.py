"""Spec 125 residual lane: the store the builder writes must be the store the trainer accepts.

These are the gates that were silently broken — a zarr v3 write API mismatch, a model output grid
that disagreed with the stored target, a release gate that rejected the lane's own schema, and a
row floor that built stores the trainer then refused.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import zarr

from harvester.v50.contracts import MIXED_STORE_SCHEMA, release_identity, require_store_release
from harvester.v50.residual_extractor_model import (
    RESIDUAL_GRID,
    ResidualExtractorNet,
    residual_loss,
)
from harvester.v50.residual_extractor_train import (
    CURRICULUM_SCHEMA as EXTRACTOR_SCHEMA,
)
from harvester.v50.residual_height_model import (
    HEIGHT_GRID,
    TARGET_GRID,
    ResidualHeightNet,
    encode_relative_height,
    height_loss,
)
from harvester.v50.residual_height_train import CURRICULUM_SCHEMA as HEIGHT_SCHEMA


def test_height_net_output_matches_the_curriculums_cropped_target_grid():
    """The builder crops height_257 to the residual's 256 grid, so the default model must emit 256 —
    a 257 output makes the loss fail on shape at training step 1."""
    model = ResidualHeightNet()
    out = model(torch.rand(2, 1, 256, 256))
    assert out.shape == (2, TARGET_GRID, TARGET_GRID)
    target = torch.rand(2, TARGET_GRID, TARGET_GRID)
    assert torch.isfinite(height_loss(out, target))


def test_height_net_can_still_be_built_on_the_uncropped_world_grid():
    out = ResidualHeightNet(grid=HEIGHT_GRID)(torch.rand(1, 1, 256, 256))
    assert out.shape == (1, HEIGHT_GRID, HEIGHT_GRID)


def test_extractor_net_output_matches_the_residual_grid():
    out = ResidualExtractorNet()(torch.rand(2, 3, 256, 256))
    assert out.shape == (2, RESIDUAL_GRID, RESIDUAL_GRID)
    assert torch.isfinite(residual_loss(out, torch.rand(2, RESIDUAL_GRID, RESIDUAL_GRID)))


@pytest.mark.parametrize("schema", [HEIGHT_SCHEMA, EXTRACTOR_SCHEMA])
def test_release_gate_accepts_the_lanes_own_curriculum_schema(tmp_path, schema):
    """The lane writes its own schema; require_store_release must gate family/release against it
    rather than demanding the mixed-curriculum schema."""
    store = tmp_path / f"{schema}.zarr"
    group = zarr.open_group(str(store), mode="w")
    identity = release_identity("v50.1")
    group.attrs["model_family"] = identity["model_family"]
    group.attrs["release"] = identity["release"]
    group.attrs["schema"] = schema

    require_store_release(group, "v50.1", store=store, expected_schema=schema)  # must not raise

    with pytest.raises(ValueError):  # default still demands the mixed-curriculum schema
        require_store_release(group, "v50.1", store=store)
    with pytest.raises(ValueError):  # release mismatch still rejected
        require_store_release(group, "v50.2", store=store, expected_schema=schema)


def test_release_gate_default_schema_is_unchanged():
    assert MIXED_STORE_SCHEMA == "v50-mixed-curriculum-v1"


def test_zarr_v3_create_array_round_trips_a_stacked_curriculum(tmp_path):
    """create_dataset(data=...) without an explicit shape raises on zarr v3; create_array is the
    write path the builders must use."""
    store = tmp_path / "curriculum.zarr"
    group = zarr.open_group(str(store), mode="w")
    stack = np.zeros((3, 256, 256), dtype=np.float32)
    group.create_array("residual_256", data=stack, chunks=(1, 256, 256))
    assert zarr.open_group(str(store), mode="r")["residual_256"].shape == (3, 256, 256)


def test_relative_height_encoding_is_altitude_offset_invariant():
    height = np.random.default_rng(0).random((257, 257)).astype(np.float32) * 200.0
    base, _, _ = encode_relative_height(height)
    shifted, _, _ = encode_relative_height(height + 1000.0)
    assert np.allclose(base, shifted, atol=1e-5)
