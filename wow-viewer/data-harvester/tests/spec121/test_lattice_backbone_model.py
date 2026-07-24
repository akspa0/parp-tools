"""Spec 121 T006: MitB0LatticeNet tests (CPU, tiny config — never touches the Hub)."""

from __future__ import annotations

import pytest
import torch

from harvester.spec117.lattice_contract import SAMPLE_COUNT
from harvester.spec117.lattice_model import LatticeTargetError, lattice_loss
from harvester.spec121.lattice_backbone_model import (
    LATTICE_NET_ID,
    MIT_B0_LATTICE_ID,
    PARAM_BAND_MIN,
    LatticeBackboneError,
    MitB0LatticeNet,
    backbone_config_payload,
    build_stage_a_model,
    config_from_payload,
    default_lattice_mit_config,
    parameter_band_ok,
    parameter_count,
    tiny_lattice_mit_config,
)


def test_tiny_forward_shape_and_range():
    model = MitB0LatticeNet(tiny_lattice_mit_config())
    out = model(torch.randn(2, 3, 256, 256))
    assert out.shape == (2, SAMPLE_COUNT)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_rejects_wrong_channel_count():
    model = MitB0LatticeNet(tiny_lattice_mit_config())
    with pytest.raises(LatticeTargetError):
        model(torch.randn(1, 4, 256, 256))


def test_native_direct_heads_no_interpolation_in_output_path():
    # Same contract as LatticeNet v5: the module must contain no upsample/interpolate wrapper
    # between the heads and the output; outer comes off a learned k2/s2/p1 conv.
    model = MitB0LatticeNet(tiny_lattice_mit_config())
    assert model.outer_reduce.kernel_size == (2, 2)
    assert model.outer_reduce.stride == (2, 2)
    module_types = {type(m).__name__ for m in model.modules()}
    assert "Upsample" not in module_types


def test_grads_flow_through_encoder_and_both_heads():
    model = MitB0LatticeNet(tiny_lattice_mit_config())
    out = model(torch.randn(1, 3, 256, 256))
    target = torch.rand(1, SAMPLE_COUNT)
    mask = torch.ones(1, SAMPLE_COUNT)
    loss = lattice_loss(out, target, mask)
    loss.backward()
    enc_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())
    outer_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.outer_head.parameters())
    inner_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.inner_head.parameters())
    assert enc_grad and outer_grad and inner_grad


def test_config_payload_round_trips_to_identical_config():
    payload = backbone_config_payload(tiny_lattice_mit_config())
    rebuilt = config_from_payload(payload)
    assert rebuilt.to_dict() == tiny_lattice_mit_config().to_dict()


def test_reconstructable_from_payload_alone_state_dict_shapes_match():
    original = MitB0LatticeNet(tiny_lattice_mit_config())
    payload = backbone_config_payload(original.config)
    clone = MitB0LatticeNet(config_from_payload(payload))
    assert [tuple(p.shape) for p in clone.parameters()] == [tuple(p.shape) for p in original.parameters()]
    clone.load_state_dict(original.state_dict())


def test_default_b0_config_lands_inside_param_band():
    model = MitB0LatticeNet(default_lattice_mit_config())
    assert parameter_band_ok(model)
    assert PARAM_BAND_MIN <= parameter_count(model) <= 30_000_000


def test_tiny_config_is_flagged_outside_band():
    assert not parameter_band_ok(MitB0LatticeNet(tiny_lattice_mit_config()))


def test_build_stage_a_model_mit_returns_full_config_payload():
    model, payload = build_stage_a_model(MIT_B0_LATTICE_ID, mit_config=tiny_lattice_mit_config())
    assert isinstance(model, MitB0LatticeNet)
    assert payload["arch"] == MIT_B0_LATTICE_ID
    assert payload["hidden_sizes"] == [8, 16, 32, 64]


def test_build_stage_a_model_fallback_constructable_from_base_alone():
    model, payload = build_stage_a_model(LATTICE_NET_ID, base=8)
    assert payload == {
        "class": "LatticeNet", "arch": "lattice_net_v5", "base": 8,
        "input": "3x256x256", "output": str(SAMPLE_COUNT),
    }
    out = model(torch.randn(1, 3, 256, 256))
    assert out.shape == (1, SAMPLE_COUNT)


def test_build_stage_a_model_rejects_unknown_architecture():
    with pytest.raises(LatticeBackboneError):
        build_stage_a_model("dinov2_retrieval")  # the dead lane stays dead
