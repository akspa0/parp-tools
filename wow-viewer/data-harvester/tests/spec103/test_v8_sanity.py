"""Spec 103 — CPU sanity for V8LeanUNet: the v7 I/O contract (forward/loss/backward,
trestle residual, prior dropout, bounds) plus the lean-budget guard. No GPU required."""

from __future__ import annotations

import numpy as np
import torch

from harvester.spec103.v7_inputs import assemble_v7_input, build_v7_targets
from harvester.spec103.v7_losses import combined_loss
from harvester.spec103.v7_model import MODEL_INPUT_CHANNELS
from harvester.spec103.v8_model import V8LeanUNet

SMALL = 64  # divisible by 16 (4 downsamples); cheap on CPU


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


def test_v8_is_actually_lean() -> None:
    model = V8LeanUNet(use_wdl_global_trestle=True)
    n_params = sum(p.numel() for p in model.parameters())
    # the whole point: an order of magnitude+ under v7's 117M
    assert n_params < 10_000_000, f"v8 grew to {n_params:,} params; the lean budget is <10M"


def test_forward_loss_backward_and_trestle() -> None:
    torch.manual_seed(0)
    model = V8LeanUNet(use_wdl_global_trestle=True, output_size=SMALL)
    model.train()

    arrays = _tile_arrays()
    x = assemble_v7_input(size=SMALL, **arrays).unsqueeze(0)
    assert x.shape == (1, MODEL_INPUT_CHANNELS, SMALL, SMALL)
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
    model = V8LeanUNet(use_wdl_global_trestle=True, output_size=SMALL)
    model.eval()
    arrays = _tile_arrays()
    x = assemble_v7_input(size=SMALL, height_hints="wdl", drop_wdl_prior=True, **arrays).unsqueeze(0)
    with torch.no_grad():
        outputs, bounds = model(x)
    assert torch.isfinite(outputs).all() and torch.isfinite(bounds).all()
    assert float(outputs[:, 0:1].min()) >= 0.0 and float(outputs[:, 0:1].max()) <= 1.0


def test_detail_head_third_channel() -> None:
    torch.manual_seed(0)
    model = V8LeanUNet(out_channels=3, use_wdl_global_trestle=True, use_detail_head=True, output_size=SMALL)
    model.eval()
    arrays = _tile_arrays()
    x = assemble_v7_input(size=SMALL, **arrays).unsqueeze(0)
    with torch.no_grad():
        outputs, _bounds = model(x)
    assert outputs.shape == (1, 3, SMALL, SMALL)
    # detail channel is a tanh-scaled residual: bounded by ±detail_residual_scale
    assert float(outputs[:, 2:3].abs().max()) <= model.detail_residual_scale + 1e-5


def test_output_interpolates_to_requested_size() -> None:
    torch.manual_seed(0)
    model = V8LeanUNet(use_wdl_global_trestle=True, output_size=48)
    model.eval()
    arrays = _tile_arrays()
    x = assemble_v7_input(size=SMALL, **arrays).unsqueeze(0)
    with torch.no_grad():
        outputs, _ = model(x)
    assert outputs.shape == (1, 2, 48, 48)


def test_state_dict_round_trip() -> None:
    """The trainer/infer contract: a fresh instance loads a saved state dict strictly."""
    torch.manual_seed(0)
    model = V8LeanUNet(use_wdl_global_trestle=True, output_size=SMALL)
    state = model.state_dict()
    fresh = V8LeanUNet(use_wdl_global_trestle=True, output_size=SMALL)
    fresh.load_state_dict(state, strict=True)
