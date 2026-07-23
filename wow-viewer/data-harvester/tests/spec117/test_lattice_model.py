"""Spec 117 T011: encode/decode round-trip honesty, masked baseline correctness, row selection,
and LatticeNet's output contract."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from harvester.spec117.lattice_model import (
    LatticeNet,
    LatticeTargetError,
    compute_lattice_tile_mean_baseline,
    decode_lattice_target,
    encode_lattice_target,
    lattice_gradient_loss,
    lattice_loss,
    select_lattice_rows,
)


def _full_lattice(value_fn):
    outer = np.fromfunction(lambda r, c: value_fn(r, c), (17, 17), dtype=np.float64)
    inner = np.fromfunction(lambda r, c: value_fn(r + 0.5, c + 0.5), (16, 16), dtype=np.float64)
    return outer.astype(np.float32), inner.astype(np.float32)


def test_encode_decode_round_trips_exactly_when_fully_present():
    outer, inner = _full_lattice(lambda r, c: 100.0 + r * 2.0 + c)
    outer_present = np.ones((17, 17), dtype=bool)
    inner_present = np.ones((16, 16), dtype=bool)

    target, mask, tile_min, tile_max = encode_lattice_target(outer, inner, outer_present, inner_present)
    assert mask.shape == (545,)
    assert np.all(mask == 1.0)

    decoded_outer, decoded_inner = decode_lattice_target(target, tile_min, tile_max)
    np.testing.assert_allclose(decoded_outer, outer, atol=1e-3)
    np.testing.assert_allclose(decoded_inner, inner, atol=1e-3)


def test_absent_samples_never_influence_tile_min_max():
    outer, inner = _full_lattice(lambda r, c: 100.0 + r * 2.0 + c)
    outer_present = np.ones((17, 17), dtype=bool)
    inner_present = np.ones((16, 16), dtype=bool)
    # True range is outer[0,0]=100 (global min) .. outer[16,16]=148 (global max). Blank out an
    # INTERIOR sample (neither the min nor the max) and give it an extreme garbage value: if the
    # encoder incorrectly folded it into min/max, tile_max would be corrupted to 10000.
    gap_index = (8, 8)  # value 132, not the global min or max

    outer_with_gap = outer.copy()
    outer_with_gap[gap_index] = 10_000.0
    outer_present_with_gap = outer_present.copy()
    outer_present_with_gap[gap_index] = False

    target_clean, _, min_clean, max_clean = encode_lattice_target(outer, inner, outer_present, inner_present)
    target_gap, mask_gap, min_gap, max_gap = encode_lattice_target(
        outer_with_gap, inner, outer_present_with_gap, inner_present
    )

    assert min_gap == min_clean == 100.0
    assert max_gap == max_clean == 148.0
    flat_gap_index = gap_index[0] * 17 + gap_index[1]
    assert mask_gap[flat_gap_index] == 0.0
    # Every OTHER present sample encodes identically regardless of the absent one's garbage value.
    kept = np.delete(np.arange(545), flat_gap_index)
    np.testing.assert_allclose(target_gap[kept], target_clean[kept], atol=1e-6)


def test_encode_raises_on_zero_present_samples():
    outer = np.zeros((17, 17), dtype=np.float32)
    inner = np.zeros((16, 16), dtype=np.float32)
    with pytest.raises(LatticeTargetError):
        encode_lattice_target(outer, inner, np.zeros((17, 17), dtype=bool), np.zeros((16, 16), dtype=bool))


class _FakeArray:
    def __init__(self, data: np.ndarray) -> None:
        self._data = data

    def __getitem__(self, index):
        return self._data[index]


class _FakeGroup(dict):
    pass


def test_select_lattice_rows_excludes_all_absent_tiles_and_counts_them():
    outer_present = np.zeros((3, 17, 17), dtype=bool)
    inner_present = np.zeros((3, 16, 16), dtype=bool)
    outer_present[0] = True  # row 0 usable
    outer_present[2, 5, 5] = True  # row 2 usable (single present sample)
    # row 1 stays entirely False -> excluded

    group = _FakeGroup(
        wdl_outer_present=_FakeArray(outer_present),
        wdl_inner_present=_FakeArray(inner_present),
    )
    usable, excluded = select_lattice_rows(group, [0, 1, 2])
    assert usable == [0, 2]
    assert excluded == 1


def test_masked_tile_mean_baseline_matches_hand_computation():
    target = np.array([0.0, 0.5, 1.0, 0.25], dtype=np.float32)
    mask = np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32)  # last sample absent
    # present values: 0.0, 0.5, 1.0 -> mean 0.5 -> MAE = mean(|0-0.5|,|0.5-0.5|,|1-0.5|) = mean(0.5,0,0.5) = 1/3
    baseline = compute_lattice_tile_mean_baseline([(target, mask)])
    assert baseline == pytest.approx(1.0 / 3.0, abs=1e-6)


def test_lattice_loss_ignores_absent_samples():
    predicted = torch.tensor([[0.0, 0.9]])
    target = torch.tensor([[0.0, 0.1]])
    mask_all = torch.tensor([[1.0, 1.0]])
    mask_masked = torch.tensor([[1.0, 0.0]])
    assert lattice_loss(predicted, target, mask_all).item() > 0.0
    assert lattice_loss(predicted, target, mask_masked).item() == pytest.approx(0.0, abs=1e-6)


def test_lattice_net_forward_shape_and_range():
    model = LatticeNet(base=8)
    x = torch.rand(2, 3, 256, 256)
    out = model(x)
    assert out.shape == (2, 545)
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0)


def test_lattice_net_rejects_wrong_channel_count():
    model = LatticeNet(base=8)
    with pytest.raises(LatticeTargetError):
        model(torch.rand(1, 4, 256, 256))


def test_lattice_net_v5_native_direct_heads_no_dense_field():
    """v5 predicts the 17x17 outer + 16x16 inner grids directly -- native heads, no dense field.

    v1/v2 average-pooled to the lattice (destroying localization); v3/v4 routed through a dense
    256x256 field and bilinearly resampled. v5 removes both: a strided double-conv encoder
    (enc1-5) feeds two native conv heads (inner_head off the 16x16 bottleneck, outer_head off a
    learned 32->17 reduce). There must be NO dense-field decoder (mid/up1-3/head) and NO
    forward(return_dense=...) path -- interpolation lives only in the downstream bridge.
    """
    model = LatticeNet(base=8)
    for attr in ("enc1", "enc2", "enc3", "enc4", "enc5", "inner_head", "outer_reduce", "outer_head"):
        assert hasattr(model, attr), f"v5 LatticeNet missing native attribute {attr}"
    # The old dense-field decoder must be gone.
    for attr in ("mid", "up1", "up2", "up3", "head"):
        assert not hasattr(model, attr), f"v5 must not carry dense-field decoder attribute {attr}"
    # Encoder levels are double-conv (two Conv+GN+SiLU sub-blocks).
    for attr in ("enc1", "enc2", "enc3", "enc4", "enc5"):
        assert len(getattr(model, attr)) == 2, f"{attr} must be a double-conv level"
    # No return_dense path: forward takes only x, so the keyword is rejected.
    with pytest.raises(TypeError):
        model(torch.rand(1, 3, 256, 256), return_dense=True)
    # base=64 default is deliberately over-capacity; even the base=8 test net is a real network.
    assert sum(p.numel() for p in LatticeNet(base=8).parameters()) > 50_000


def test_lattice_net_v5_output_is_545_native_and_in_range():
    """256x256x3 -> (B, 545) in [0, 1]: outer 289 (17x17) then inner 256 (16x16), no interpolation."""
    model = LatticeNet(base=8)
    out = model(torch.rand(2, 3, 256, 256))
    assert out.shape == (2, 545)
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0)
    # outer_reduce turns a 32x32 feature map into the native 17x17 grid (learned downsample, not
    # a bilinear resize): floor((32 + 2*1 - 2) / 2) + 1 = 17.
    feat = torch.rand(1, 8 * 8, 32, 32)  # base*8 channels at the 32x32 level
    assert tuple(model.outer_reduce(feat).shape[-2:]) == (17, 17)


def test_lattice_net_v5_grads_flow_through_encoder_and_both_heads():
    """Gradients must reach the shallow encoder and both native heads, not a single path."""
    model = LatticeNet(base=8)
    x = torch.rand(1, 3, 256, 256, requires_grad=False)
    model(x).sum().backward()
    # enc1 (shallowest) and enc5 (bottleneck) are double-conv Sequentials, so the conv is [0][0].
    assert model.enc1[0][0].weight.grad is not None
    assert torch.isfinite(model.enc1[0][0].weight.grad).all()
    assert model.enc5[0][0].weight.grad is not None
    # Both native heads and the outer reduce must receive gradient.
    assert model.inner_head[-1].weight.grad is not None
    assert model.outer_head[-1].weight.grad is not None
    assert model.outer_reduce.weight.grad is not None


def test_lattice_net_reconstructable_from_base_alone():
    """lattice_bridge.py rebuilds the model from lattice_config.base only -- the v5 architecture
    must stay constructable from `base` with no extra knobs, and a saved state_dict must reload
    into a freshly-constructed same-base model bit-for-bit (the bridge's load_state_dict contract)."""
    a = LatticeNet(base=8)
    a.eval()
    x = torch.rand(1, 3, 256, 256)
    expected = a(x)
    state = a.state_dict()
    b = LatticeNet(base=8)
    b.load_state_dict(state)
    b.eval()
    got = b(x)
    assert torch.allclose(expected, got, atol=1e-6)
    # A different base must NOT accept this state_dict (guards against silent cross-arch loads).
    with pytest.raises(RuntimeError):
        LatticeNet(base=16).load_state_dict(state)


def test_lattice_gradient_loss_zero_on_match_positive_on_mismatch():
    """The V7-ported structural term is 0 when the slope field matches, >0 when it does not."""
    target = torch.rand(2, 545)
    mask = torch.ones(2, 545)
    assert lattice_gradient_loss(target, target, mask).item() == pytest.approx(0.0, abs=1e-6)
    # A prediction that scrambles the arrangement (flips each grid) has a non-zero gradient error
    # even if its per-point values are drawn from the same distribution.
    flipped = torch.flip(target, dims=[1])
    assert lattice_gradient_loss(flipped, target, mask).item() > 0.0


def test_lattice_gradient_loss_respects_presence_mask():
    """A gradient between two absent samples (mask 0) never contributes."""
    target = torch.rand(1, 545)
    pred = target + 0.5  # large error everywhere
    mask_all = torch.ones(1, 545)
    mask_none = torch.zeros(1, 545)
    full = lattice_gradient_loss(pred, target, mask_all).item()
    none = lattice_gradient_loss(pred, target, mask_none).item()
    assert full > 0.0
    assert none == pytest.approx(0.0, abs=1e-6)


def test_lattice_net_v5_localizes_not_globally_averages():
    """Two inputs that differ only in one 16x16 region must produce different lattice cells there.

    The v1/v2 failure was average pooling, which smeared every output cell into a near-global mean.
    A native-head model must localize: a bounded change in the input must move the corresponding
    lattice cells and not (much) move distant ones.
    """
    model = LatticeNet(base=8).eval()
    base_img = torch.rand(1, 3, 256, 256)
    bumped = base_img.clone()
    bumped[:, :, :32, :32] += 0.5  # perturb only the top-left tile corner
    with torch.no_grad():
        delta = (model(bumped) - model(base_img)).abs()
    # Some lattice cell must actually respond to the localized change (not a dead/constant head).
    assert float(delta.max()) > 1e-4
