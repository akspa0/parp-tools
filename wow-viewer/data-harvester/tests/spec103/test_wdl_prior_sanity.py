from __future__ import annotations

import numpy as np
import pytest
import torch

from harvester.spec103.v7_inputs import assemble_v7_input, wdl_lattice_from_height257
from harvester.spec103.wdl_prior_io import read_prediction_archive, write_prediction_archive
from harvester.spec103.wdl_prior_model import (
    WDL_VALUE_COUNT,
    WdlPriorNet,
    build_wdl_target,
    decode_wdl_target,
    normalize_minimap_rgb,
)


def test_target_is_exact_paired_wdl_mapping():
    height = np.linspace(-100.0, 100.0, 257 * 257, dtype=np.float32).reshape(257, 257)
    target = build_wdl_target(height)
    outer, inner = decode_wdl_target(target)
    expected_outer, expected_inner = wdl_lattice_from_height257(height)
    assert target.shape == (WDL_VALUE_COUNT,)
    # Normalized float32 targets introduce sub-millimetre world-unit quantization.
    np.testing.assert_allclose(outer, expected_outer, atol=1e-3)
    np.testing.assert_allclose(inner, expected_inner, atol=1e-3)


def test_model_consumes_rgb_tensor_only_and_returns_545_values():
    model = WdlPriorNet().eval()
    rgb = np.full((256, 256, 3), 127, dtype=np.uint8)
    with torch.no_grad():
        result = model(normalize_minimap_rgb(rgb).unsqueeze(0))
    assert tuple(result.shape) == (1, WDL_VALUE_COUNT)
    assert torch.isfinite(result).all()


def test_generated_outer_changes_v8_prior_channel():
    height = np.zeros((257, 257), dtype=np.float32)
    generated = np.full((17, 17), 200.0, dtype=np.float32)
    rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    ground_truth_input = assemble_v7_input(rgb, height_257=height, height_hints="wdl")
    generated_input = assemble_v7_input(rgb, height_257=None, wdl_outer_17=generated, height_hints="wdl")
    assert tuple(generated_input.shape) == (13, 256, 256)
    assert not torch.equal(ground_truth_input[6], generated_input[6])


def test_prediction_archive_rejects_duplicate_rows(tmp_path):
    path = tmp_path / "prior.npz"
    outer = np.zeros((2, 17, 17), dtype=np.float32)
    inner = np.zeros((2, 16, 16), dtype=np.float32)
    with pytest.raises(ValueError, match="duplicate"):
        write_prediction_archive(path, np.array([4, 4]), outer, inner, {})
    write_prediction_archive(path, np.array([4, 7]), outer, inner, {"schema": "test"})
    values, metadata = read_prediction_archive(path)
    assert sorted(values) == [4, 7]
    assert metadata["schema"] == "test"
