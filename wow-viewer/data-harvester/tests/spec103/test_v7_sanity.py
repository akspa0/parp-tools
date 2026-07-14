"""Spec 103 T006 — CPU sanity for the ported v7: forward/loss/backward, the pinned
13-channel order, the trestle residual path, and WDL-prior dropout. No GPU required."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from harvester.spec103.v7_inputs import (
    HEIGHT_GLOBAL_MAX,
    HEIGHT_GLOBAL_MIN,
    MISSING_PRIOR_FILL,
    assemble_v7_input,
    build_v7_targets,
    normalize_height,
    prediction_to_height257,
    render_wdl_prior_channel,
    wdl_lattice_from_height257,
)
from harvester.spec103.v7_losses import combined_loss, derive_recovery_mask_from_inputs
from harvester.spec103.v7_model import MODEL_INPUT_CHANNELS, MultiChannelUNetV7

SMALL = 64  # divisible by 32; keeps the 2048-channel bottleneck cheap on CPU


def _tile_arrays(height_scale: float = 200.0) -> dict:
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[0:257, 0:257].astype(np.float32) / 256.0
    height = (xx * 0.75 + yy * 0.25) * height_scale  # diagonal ramp
    return {
        "minimap_rgb": rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8),
        "height_257": height,
        "normal_xyz": np.stack([np.zeros_like(height), np.zeros_like(height), np.ones_like(height)], axis=-1),
        "liquid_mask": np.zeros((256, 256), dtype=np.float32),
        "liquid_height": np.zeros((256, 256), dtype=np.float32),
        "object_mask": np.zeros((257, 257), dtype=np.float32),
    }


def test_wdl_lattice_pairing_is_the_verified_transform() -> None:
    height = np.arange(257 * 257, dtype=np.float32).reshape(257, 257)
    outer, inner = wdl_lattice_from_height257(height)
    assert outer.shape == (17, 17)
    assert inner.shape == (16, 16)
    np.testing.assert_array_equal(outer, height[::16, ::16])
    np.testing.assert_array_equal(inner, height[8::16, 8::16])
    # the prohibited ::8 raster would be 33x33; both real grids differ from it
    assert height[::8, ::8].shape == (33, 33)


def test_assembler_channel_order_and_values() -> None:
    arrays = _tile_arrays()
    arrays["liquid_mask"][10:20, 10:20] = 1.0
    arrays["liquid_height"][10:20, 10:20] = 50.0
    arrays["object_mask"][100:120, 100:120] = 1.0

    x = assemble_v7_input(size=SMALL, **arrays)
    assert x.shape == (MODEL_INPUT_CHANNELS, SMALL, SMALL)
    assert torch.isfinite(x).all()

    # ch 6: WDL prior equals normalized outer-lattice upsample
    outer, _ = wdl_lattice_from_height257(arrays["height_257"])
    expected_prior = render_wdl_prior_channel(outer, SMALL)
    torch.testing.assert_close(x[6:7], expected_prior)

    # ch 7/8: gt hints are constant planes at the tile height bounds (normalized)
    expected_min = float(normalize_height(arrays["height_257"].min()))
    expected_max = float(normalize_height(arrays["height_257"].max()))
    assert torch.allclose(x[7], torch.full((SMALL, SMALL), expected_min))
    assert torch.allclose(x[8], torch.full((SMALL, SMALL), expected_max))

    # ch 9 liquid, ch 10 liquid-height (only inside the mask), ch 11 object, ch 12 brush
    assert x[9].max() == 1.0 and x[9].min() == 0.0
    assert float(x[10].max()) > 0.0
    assert torch.all(x[10][x[9] == 0.0] == 0.0)
    assert x[11].max() == 1.0
    assert torch.all(x[12] == 0.0)

    # loss-side recovery mask reads the same channels
    recovery = derive_recovery_mask_from_inputs(x.unsqueeze(0))
    assert float(recovery.max()) == 1.0


def test_wdl_prior_dropout_fills_missing_prior_constant() -> None:
    arrays = _tile_arrays()
    dropped = assemble_v7_input(size=SMALL, height_hints="wdl", drop_wdl_prior=True, **arrays)
    assert torch.all(dropped[6] == MISSING_PRIOR_FILL)
    # wdl-mode hints neutralize with the dropped prior
    assert torch.all(dropped[7] == 0.0)
    assert torch.all(dropped[8] == 1.0)

    kept = assemble_v7_input(size=SMALL, height_hints="wdl", drop_wdl_prior=False, **arrays)
    assert not torch.all(kept[6] == MISSING_PRIOR_FILL)


def test_targets_and_bounds() -> None:
    arrays = _tile_arrays(height_scale=300.0)
    target, bounds = build_v7_targets(arrays["height_257"], size=SMALL)
    assert target.shape == (2, SMALL, SMALL)
    assert bounds.shape == (4,)
    # local channel spans [0, 1]; global stays within the normalized band
    assert float(target[1].min()) == pytest.approx(0.0, abs=1e-5)
    assert float(target[1].max()) == pytest.approx(1.0, abs=1e-5)
    assert float(bounds[2]) == 0.0 and float(bounds[3]) == 1.0
    expected_max = (300.0 - HEIGHT_GLOBAL_MIN) / (HEIGHT_GLOBAL_MAX - HEIGHT_GLOBAL_MIN)
    assert float(bounds[1]) == pytest.approx(expected_max, abs=1e-5)


def test_forward_loss_backward_and_trestle() -> None:
    torch.manual_seed(0)
    model = MultiChannelUNetV7(use_wdl_global_trestle=True, output_size=SMALL)
    model.train()

    arrays = _tile_arrays()
    x = assemble_v7_input(size=SMALL, **arrays).unsqueeze(0)
    target, bounds = build_v7_targets(arrays["height_257"], size=SMALL)
    target = target.unsqueeze(0)
    bounds = bounds.unsqueeze(0)

    outputs, predicted_bounds = model(x)
    assert outputs.shape == (1, 2, SMALL, SMALL)
    assert predicted_bounds.shape == (1, 4)

    # trestle: global output stays within ±global_residual_scale of the WDL base (then clamped)
    wdl_base = x[:, 6:7]
    delta = (outputs[:, 0:1] - wdl_base).abs()
    unclamped = (outputs[:, 0:1] > 0.0) & (outputs[:, 0:1] < 1.0)
    assert float(delta[unclamped].max()) <= model.global_residual_scale + 1e-5

    loss, components = combined_loss(outputs, predicted_bounds, target, bounds, input_context=x)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert set(components) >= {"heightmap_global", "heightmap_local", "bounds", "recovery"}


def test_prior_dropout_still_resolves_forward() -> None:
    torch.manual_seed(0)
    model = MultiChannelUNetV7(use_wdl_global_trestle=True, output_size=SMALL)
    model.eval()
    arrays = _tile_arrays()
    x = assemble_v7_input(size=SMALL, height_hints="wdl", drop_wdl_prior=True, **arrays).unsqueeze(0)
    with torch.no_grad():
        outputs, bounds = model(x)
    assert torch.isfinite(outputs).all() and torch.isfinite(bounds).all()
    # with a 0.5 trestle base the global head can still cover ±scale around 0.5
    assert float(outputs[:, 0:1].min()) >= 0.0 and float(outputs[:, 0:1].max()) <= 1.0


def test_prediction_roundtrip_to_world_units() -> None:
    arrays = _tile_arrays(height_scale=150.0)
    normalized = normalize_height(arrays["height_257"])
    # feed the exact normalized GT as a fake prediction raster at 256
    import torch.nn.functional as F  # local import to mirror assembler convention

    raster = F.interpolate(
        torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0), size=(256, 256), mode="bilinear", align_corners=True
    ).squeeze().numpy()
    recovered = prediction_to_height257(raster)
    assert recovered.shape == (257, 257)
    # corners survive the vertex-grid round trip exactly; interior within interpolation error
    assert abs(float(recovered[0, 0]) - float(arrays["height_257"][0, 0])) < 1e-2
    assert abs(float(recovered[-1, -1]) - float(arrays["height_257"][-1, -1])) < 1e-2
    assert float(np.abs(recovered - arrays["height_257"]).mean()) < 1.0
