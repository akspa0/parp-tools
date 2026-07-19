from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from harvester.v50.height_relative_evaluate import (
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
