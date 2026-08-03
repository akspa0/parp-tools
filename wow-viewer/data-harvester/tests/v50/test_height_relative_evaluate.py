from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from harvester.v50.height_relative_evaluate import (
    build_model_input_channels,
    compute_row_metrics,
    render_validation_sheet,
    select_error_quantile_rows,
    select_fixed_preview_rows,
)


def test_row_metrics_report_model_baseline_gradient_and_border_errors() -> None:
    target = np.linspace(0.0, 1.0, 81, dtype=np.float32).reshape(9, 9)
    predicted = np.zeros_like(target)

    metrics = compute_row_metrics(predicted, target, border_width=2)

    assert metrics["mae"] == np.mean(target)
    assert metrics["gradient_mae"] > 0.0
    assert metrics["border_mae"] > 0.0
    assert metrics["tile_mean_baseline_mae"] < metrics["mae"]
    assert metrics["mae_delta_vs_baseline"] > 0.0


def test_error_quantile_and_fixed_preview_selection_include_endpoints() -> None:
    records = [{"row_id": row, "mae": float(row)} for row in range(10)]

    assert select_error_quantile_rows(records, 4) == [0, 3, 6, 9]
    assert select_fixed_preview_rows(list(range(10)), 4) == [0, 3, 6, 9]


def test_validation_sheet_uses_all_required_panels(tmp_path: Path) -> None:
    output = tmp_path / "sheet.png"
    target = np.linspace(0.0, 1.0, 25, dtype=np.float32).reshape(5, 5)
    predicted = np.flipud(target).copy()
    render_validation_sheet(
        [
            {
                "label": "row 7 Kalimdor 1,2",
                "rgb": np.full((5, 5, 3), 100, dtype=np.uint8),
                "target": target,
                "predicted": predicted,
                "metrics": compute_row_metrics(predicted, target),
            }
        ],
        output,
        title="test",
        panel_size=32,
    )

    with Image.open(output) as image:
        assert image.width == 210 + (6 * 32) + (5 * 4)
        assert image.height == 42 + 36


class _StubExtractor(torch.nn.Module):
    """Stands in for the frozen Spec 125 US7 extractor: (B, 3, H, W) -> (B, H, W) in [0, 1]."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1).sigmoid()


def test_input_channels_order_is_rgb_then_residual_then_features() -> None:
    rgb = np.random.default_rng(0).random((8, 8, 3)).astype(np.float32)
    channels = build_model_input_channels(
        rgb, 0, bindings=[], residual_extractor=_StubExtractor()
    )

    assert channels.shape == (4, 8, 8)
    assert torch.allclose(channels[:3], torch.from_numpy(rgb).permute(2, 0, 1))
    # The residual sits at index 3 — after RGB, before any feature channels.
    assert torch.allclose(channels[3], torch.from_numpy(rgb).mean(axis=2).sigmoid())

    assert build_model_input_channels(rgb, 0, bindings=[]).shape == (3, 8, 8)
    with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
        build_model_input_channels(rgb[..., :2], 0, bindings=[])


def test_prediction_paths_feed_a_four_channel_model_its_residual_channel() -> None:
    """Regression: the trainer added the residual channel in its OWN dataset only, so a
    4-channel model got 3 channels from the preview/eval paths and the run died at its first
    best-epoch checkpoint ("expected input[1, 3, 256, 256] to have 4 channels")."""
    from harvester.v50.height_relative_evaluate import _predict_samples
    from harvester.v50.height_relative_model import HEIGHT_GRID, HeightRelativeNet

    rng = np.random.default_rng(1)
    group = {
        "minimap_rgb": rng.integers(0, 256, size=(2, 32, 32, 3), dtype=np.uint8),
        "height_257": rng.random((2, HEIGHT_GRID, HEIGHT_GRID)).astype(np.float32) * 40.0,
    }
    index_rows = [{"map": "Kalimdor", "tile_x": 1, "tile_y": 2} for _ in range(2)]
    model = HeightRelativeNet(base=4, in_channels=4).eval()

    samples = _predict_samples(
        model, group, index_rows, [0, 1], torch.device("cpu"),
        use_amp=False, residual_extractor=_StubExtractor(),
    )

    assert len(samples) == 2
    assert all(np.isfinite(s["predicted"]).all() for s in samples)


def test_preview_and_final_evaluation_survive_a_stacked_model(tmp_path: Path) -> None:
    """Both call sites in the traceback — the best-epoch preview and the final evaluation."""
    from harvester.v50.height_relative_evaluate import evaluate_height_model, render_fixed_model_preview
    from harvester.v50.height_relative_model import HEIGHT_GRID, HeightRelativeNet

    rng = np.random.default_rng(3)
    group = {
        "minimap_rgb": rng.integers(0, 256, size=(4, 32, 32, 3), dtype=np.uint8),
        "height_257": rng.random((4, HEIGHT_GRID, HEIGHT_GRID)).astype(np.float32) * 40.0,
    }
    index_rows = [{"map": "Kalimdor", "tile_x": i, "tile_y": 0} for i in range(4)]
    model = HeightRelativeNet(base=4, in_channels=4).eval()
    device = torch.device("cpu")
    extractor = _StubExtractor()

    render_fixed_model_preview(
        model, group, index_rows, [0, 2], device, tmp_path / "preview.png",
        epoch=1, val_mae=0.5, use_amp=False, residual_extractor=extractor,
    )
    summary = evaluate_height_model(
        model, group, index_rows, [0, 1, 2, 3], device, tmp_path / "final",
        batch_size=2, workers=0, checkpoint_epoch=1, use_amp=False, review_count=2,
        residual_extractor=extractor,
    )

    assert (tmp_path / "preview.png").is_file()
    assert (tmp_path / "final" / "worst_cases.png").is_file()
    assert summary["val_rows"] == 4


def test_four_channel_model_still_refuses_a_missing_residual_extractor() -> None:
    """The fix must not paper over a genuine mismatch: no extractor still means 3 channels."""
    from harvester.v50.height_relative_evaluate import _predict_samples
    from harvester.v50.height_relative_model import HEIGHT_GRID, HeightRelativeNet

    rng = np.random.default_rng(2)
    group = {
        "minimap_rgb": rng.integers(0, 256, size=(1, 32, 32, 3), dtype=np.uint8),
        "height_257": rng.random((1, HEIGHT_GRID, HEIGHT_GRID)).astype(np.float32) * 40.0,
    }
    with pytest.raises(RuntimeError, match="4 channels"):
        _predict_samples(
            HeightRelativeNet(base=4, in_channels=4).eval(), group,
            [{"map": "Kalimdor", "tile_x": 0, "tile_y": 0}], [0],
            torch.device("cpu"), use_amp=False,
        )
