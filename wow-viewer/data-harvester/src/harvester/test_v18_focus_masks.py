from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import torch


_ROOT = Path(__file__).resolve().parents[2]


def _load_script_module(name: str, relative_path: str):
    path = _ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_train_common = _load_script_module("train_v16_1_common_test", "scripts/train_v16_1_common.py")
_curation = _load_script_module("build_v16_curation_manifest_test", "scripts/build_v16_curation_manifest.py")


class _FixedModel(torch.nn.Module):
    def __init__(self, output: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("_output", output.clone())

    def forward(self, _inp: torch.Tensor) -> torch.Tensor:
        return self._output.clone()


def test_height_loss_ignores_non_trainable_regions() -> None:
    pred = torch.tensor([[[[1.0, 10.0], [10.0, 1.0]]]], dtype=torch.float32)
    model = _FixedModel(pred)
    batch = {
        "input": torch.zeros((1, 3, 2, 2), dtype=torch.float32),
        "height_norm": torch.zeros((1, 1, 2, 2), dtype=torch.float32),
        "terrain_valid_mask_257": torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=torch.float32),
    }

    loss, metrics, outputs = _train_common._height_loss(model, batch, torch.device("cpu"), argparse.Namespace())

    assert torch.isclose(loss, torch.tensor(1.0))
    assert metrics["height_mask_cov"] == 0.5
    assert torch.equal(outputs["weight"], batch["terrain_valid_mask_257"])


def test_normal_loss_respects_terrain_valid_mask() -> None:
    pred = torch.tensor(
        [[
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]],
            [[1.0, -1.0], [-1.0, 1.0]],
        ]],
        dtype=torch.float32,
    )
    target = torch.tensor(
        [[
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]],
            [[1.0, 1.0], [1.0, 1.0]],
        ]],
        dtype=torch.float32,
    )
    model = _FixedModel(pred)
    batch = {
        "input": torch.zeros((1, 3, 2, 2), dtype=torch.float32),
        "normals": target,
        "normal_mask": torch.ones((1, 1, 2, 2), dtype=torch.float32),
        "terrain_valid_mask_257": torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=torch.float32),
    }

    loss, metrics, outputs = _train_common._normal_loss(model, batch, torch.device("cpu"), argparse.Namespace())

    assert torch.isclose(loss, torch.tensor(0.0))
    assert metrics["normal_mask_cov"] == 0.5
    assert torch.equal(outputs["train_mask"], batch["terrain_valid_mask_257"])


def test_curation_rejects_low_trainable_terrain_tiles() -> None:
    row = {
        "what_plate": False,
        "minimap_gray_std": 8.0,
        "height_std": 5.0,
        "alpha_cov": 0.0,
        "liquid_cov": 0.85,
        "normal_cov": 0.9,
        "has_normals": True,
        "normal_relief_mean": 0.1,
        "normal_edge_frac": 0.0,
        "minimap_edge_frac": 0.0,
        "modf_cov": 0.0,
        "loss_gate_cov": 0.0,
        "wmo_loss_share": 0.0,
        "trainable_cov": 0.05,
    }
    args = argparse.Namespace(
        profile="v18_focus_terrain_v1",
        min_minimap_gray_std=4.0,
        min_height_std=3.0,
        min_normal_coverage=0.25,
        min_edge_frac=0.01,
        min_normal_edge_f1=0.10,
        min_wmo_wipeout_modf_cov=0.25,
        min_wmo_wipeout_loss_gate_cov=0.35,
        min_wmo_wipeout_share=0.75,
        max_wmo_wipeout_trainable_cov=0.30,
        min_trainable_cov=0.20,
    )

    keep, payload = _curation._evaluate_profile(row, args)

    assert not keep
    assert payload["reject_reason"] == "insufficient_trainable_terrain"
