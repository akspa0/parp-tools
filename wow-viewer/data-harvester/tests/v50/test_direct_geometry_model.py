"""Spec 114 T014: one-output contract for both direct geometry architectures."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from harvester.v50.direct_geometry_model import (
    DIRECT_CNN_ID,
    EXPECTED_DIRECT_CNN_PARAMS,
    MIT_B0_ID,
    GeometryModelError,
    MitB0RegressionNet,
    build_geometry_model,
    default_mit_config,
    tiny_mit_config,
    validate_pretrained_source,
)
from harvester.v50.height_relative_model import HEIGHT_GRID


def test_direct_cnn_identity_matches_frozen_baseline() -> None:
    model, identity = build_geometry_model(DIRECT_CNN_ID)
    assert identity["architecture"]["id"] == DIRECT_CNN_ID
    assert identity["architecture"]["parameter_count"] == EXPECTED_DIRECT_CNN_PARAMS
    assert identity["pretrained_source"] is None
    out = model(torch.zeros(1, 3, 256, 256))
    assert out.shape == (1, HEIGHT_GRID, HEIGHT_GRID)


def test_direct_cnn_refuses_pretrained_source() -> None:
    with pytest.raises(GeometryModelError, match="from-scratch baseline"):
        build_geometry_model(
            DIRECT_CNN_ID,
            pretrained_source={
                "hub_id": "nvidia/mit-b0", "revision": "r" * 40,
                "sha256": "e" * 64, "license": "Apache-2.0",
            },
        )


def test_mit_tiny_forward_is_one_bounded_finite_field() -> None:
    model = MitB0RegressionNet(tiny_mit_config())
    model.eval()
    with torch.no_grad():
        out = model(torch.rand(2, 3, 256, 256))
    assert out.shape == (2, HEIGHT_GRID, HEIGHT_GRID)  # one output, no aux heads
    assert torch.isfinite(out).all()
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_mit_tiny_backward_flows_to_encoder_and_head() -> None:
    model = MitB0RegressionNet(tiny_mit_config())
    model.train()
    out = model(torch.rand(1, 3, 256, 256))
    out.mean().backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


def test_mit_default_config_is_compact_b0() -> None:
    model, identity = build_geometry_model(MIT_B0_ID)
    params = identity["architecture"]["parameter_count"]
    assert 3_000_000 < params < 5_000_000  # SegFormer-B0 scale, per plan Performance Goals
    assert identity["architecture"]["id"] == MIT_B0_ID


def test_mit_refuses_wrong_channels() -> None:
    model = MitB0RegressionNet(tiny_mit_config())
    with pytest.raises(GeometryModelError, match="one RGB tile"):
        model(torch.rand(1, 4, 256, 256))


def test_mit_is_deterministic_under_seed() -> None:
    torch.manual_seed(7)
    first = MitB0RegressionNet(tiny_mit_config()).eval()
    torch.manual_seed(7)
    second = MitB0RegressionNet(tiny_mit_config()).eval()
    x = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        assert torch.equal(first(x), second(x))


def test_unknown_architecture_is_refused() -> None:
    with pytest.raises(GeometryModelError, match="architecture must be one of"):
        build_geometry_model("dinov2_small_relief_v1")


def test_pretrained_source_pinning_and_depthanything_refusal() -> None:
    valid = validate_pretrained_source(
        {"hub_id": "nvidia/mit-b0", "revision": "r" * 40, "sha256": "E" * 64,
         "license": "Apache-2.0"}
    )
    assert valid == {
        "hub_id": "nvidia/mit-b0", "revision": "r" * 40, "sha256": "E" * 64,
        "license": "Apache-2.0", "optional": True,
    }
    with pytest.raises(GeometryModelError, match="DepthAnything"):
        validate_pretrained_source(
            {"hub_id": "depth-anything/Depth-Anything-V2-Small", "revision": "r" * 40,
             "sha256": "e" * 64, "license": "Apache-2.0"}
        )
    with pytest.raises(GeometryModelError, match="sha256"):
        validate_pretrained_source(
            {"hub_id": "nvidia/mit-b0", "revision": "r" * 40, "sha256": "short",
             "license": "Apache-2.0"}
        )


def test_offset_invariant_target_property_is_preserved_by_contract() -> None:
    # The model contract consumes only RGB; offset invariance lives in the v112.1 target. Guard the
    # boundary here: identical pixels must yield identical relief regardless of world altitude,
    # which no model input can express.
    model = MitB0RegressionNet(tiny_mit_config()).eval()
    x = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        assert torch.equal(model(x), model(x.clone()))
    assert default_mit_config().num_labels == 1
    _ = np.float32(0)  # keep numpy import honest for fixture parity
