from __future__ import annotations

import numpy as np
import pytest
import torch
import subprocess
import sys
from pathlib import Path
from PIL import Image

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from harvester.spec103.v7_inputs import assemble_v7_input, wdl_lattice_from_height257
from harvester.spec103.wdl_prior_io import read_prediction_archive, write_prediction_archive
from harvester.spec103.wdl_prior_model import (
    INPUT_CONTRACT,
    MODEL_VARIANT_WDL_PRIOR,
    TARGET_CONTRACT,
    WDL_VALUE_COUNT,
    WdlPriorNet,
    build_wdl_target,
    decode_wdl_target,
    normalize_minimap_rgb,
)
from train_spec103_wdl_prior import filter_deployable_rows
from spec103_make_synthetic_adts import DEFAULT_PATTERNS, _pattern_height


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


def test_standalone_png_cli_needs_no_store_or_wdl(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({
        "model_variant": MODEL_VARIANT_WDL_PRIOR,
        "input_contract": INPUT_CONTRACT,
        "target_contract": TARGET_CONTRACT,
        "model": WdlPriorNet().state_dict(),
    }, checkpoint)
    image = tmp_path / "minimap.png"
    Image.fromarray(np.full((32, 32, 3), 127, dtype=np.uint8), mode="RGB").save(image)
    output = tmp_path / "lattice.npz"
    script = Path(__file__).resolve().parents[2] / "scripts" / "infer_spec103_wdl_prior.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--image", str(image), "--checkpoint", str(checkpoint), "--output", str(output), "--device", "cpu"],
        check=True, capture_output=True, text=True,
    )
    assert "RGB image only" in completed.stdout
    with np.load(output, allow_pickle=False) as archive:
        assert archive["outer_17"].shape == (17, 17)
        assert archive["inner_16"].shape == (16, 16)


def test_training_filter_rejects_dark_and_occluded_minimap_rows():
    group = {
        "minimap_rgb": np.stack([
            np.zeros((8, 8, 3), dtype=np.uint8),
            np.full((8, 8, 3), 100, dtype=np.uint8),
            np.full((8, 8, 3), 100, dtype=np.uint8),
        ]),
        "object_precise_mask": np.stack([
            np.zeros((9, 9), dtype=np.float32),
            np.ones((9, 9), dtype=np.float32),
            np.zeros((9, 9), dtype=np.float32),
        ]),
    }
    rows, report = filter_deployable_rows(group, [0, 1, 2], min_rgb_mean=25.0, max_object_coverage=0.0)
    assert rows == [2]
    assert report == {"dropped_dark": 1, "dropped_object": 1}


def test_synthetic_generator_has_deterministic_family_variation():
    first, first_params = _pattern_height("ridge", 180.0, np.random.default_rng(103))
    second, second_params = _pattern_height("ridge", 180.0, np.random.default_rng(104))
    assert first.shape == (257, 257)
    assert float(first.max() - first.min()) > 170.0
    assert first_params != second_params
    assert not np.array_equal(first, second)
    assert {"hills", "valley", "terraces", "saddle", "dunes", "basin"}.issubset(DEFAULT_PATTERNS)
