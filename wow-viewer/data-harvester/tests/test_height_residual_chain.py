"""Tests for the Spec 077 H0/H1 residual height chain."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
for _entry in (_SRC_DIR, _SCRIPTS_DIR):
    _entry_str = str(_entry)
    if _entry_str not in sys.path:
        sys.path.insert(0, _entry_str)

from harvester.height_residual_chain import (  # noqa: E402
    build_height_chain_input,
    compose_refined_height,
    downsample_height_target,
    height_chain_input_channels,
    residual_target,
    upsample_coarse_height,
)
from harvester.v18_models import V18HeightCoarseModel, V18HeightResidualModel  # noqa: E402
import train_height_coarse_prior  # noqa: E402
import train_height_residual_prior  # noqa: E402


def test_height_chain_input_channels() -> None:
    assert height_chain_input_channels() == 3
    assert height_chain_input_channels(use_albedo=True) == 6
    assert height_chain_input_channels(use_albedo=True, use_density=True) == 9
    assert height_chain_input_channels(use_albedo=True, use_density=True, include_base=True) == 10


def test_build_height_chain_input_appends_albedo_density_and_base() -> None:
    batch = {
        "input_prior": torch.rand(2, 5, 16, 16),
        "albedo_rgb": torch.rand(2, 3, 16, 16),
    }
    base = torch.rand(2, 1, 16, 16)

    x = build_height_chain_input(
        batch,
        device=torch.device("cpu"),
        use_albedo=True,
        use_density=True,
        base_height_257=base,
    )

    assert x.shape == (2, 10, 16, 16)
    assert torch.allclose(x[:, :3], batch["input_prior"][:, :3].float())
    assert torch.allclose(x[:, -1:], base)


def test_coarse_downsample_and_residual_composition_shapes() -> None:
    height = torch.rand(2, 1, 257, 257)
    weight = torch.ones(2, 1, 257, 257)

    coarse, coarse_weight = downsample_height_target(height, weight, coarse_size=65)
    base = upsample_coarse_height(coarse, size=257)
    delta = residual_target(height, base)
    refined = compose_refined_height(base, delta)

    assert coarse.shape == (2, 1, 65, 65)
    assert coarse_weight.shape == (2, 1, 65, 65)
    assert base.shape == (2, 1, 257, 257)
    assert delta.shape == (2, 1, 257, 257)
    assert torch.allclose(refined, height, atol=1e-6)


def test_h0_h1_model_shapes() -> None:
    h0 = V18HeightCoarseModel(in_channels=9, norm="group", decoder_upsample="nearest", coarse_size=65)
    h1 = V18HeightResidualModel(in_channels=10, norm="group", decoder_upsample="nearest")
    x = torch.rand(1, 9, 64, 64)

    coarse = h0(x)
    base = upsample_coarse_height(coarse, size=257)
    delta = h1(torch.cat([torch.rand(1, 9, 257, 257), base], dim=1))

    assert coarse.shape == (1, 1, 65, 65)
    assert base.shape == (1, 1, 257, 257)
    assert delta.shape == (1, 1, 257, 257)


def test_h1_residual_model_starts_as_zero_delta() -> None:
    model = V18HeightResidualModel(in_channels=10, norm="group", decoder_upsample="nearest")
    x = torch.rand(2, 10, 257, 257)

    delta = model(x)

    assert torch.count_nonzero(delta) == 0


def test_training_scripts_parse_smoke_args() -> None:
    h0_args = train_height_coarse_prior._parse_args([
        "--prior", "prior.zarr",
        "--v18", "v18.zarr",
        "--albedo",
        "--density",
        "--steps", "1",
    ])
    h1_args = train_height_residual_prior._parse_args([
        "--coarse-checkpoint", "h0.pt",
        "--prior", "prior.zarr",
        "--v18", "v18.zarr",
        "--albedo",
        "--density",
        "--steps", "1",
    ])

    assert h0_args.albedo is True
    assert h0_args.density is True
    assert h1_args.coarse_checkpoint == Path("h0.pt")
    assert h1_args.delta_weight == 0.25
