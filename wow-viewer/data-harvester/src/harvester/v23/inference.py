"""V23 deterministic inference helpers for Spec 089 Phase 6."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch


def _shift_view(x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    shifted = torch.zeros_like(x)
    src_y0 = max(dy, 0)
    src_y1 = x.shape[-2] + min(dy, 0)
    dst_y0 = max(-dy, 0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    src_x0 = max(dx, 0)
    src_x1 = x.shape[-1] + min(dx, 0)
    dst_x0 = max(-dx, 0)
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    shifted[..., dst_y0:dst_y1, dst_x0:dst_x1] = x[..., src_y0:src_y1, src_x0:src_x1]
    return shifted


def _predict_metric_height(model: Any, tile_input: torch.Tensor, device: torch.device | None = None) -> torch.Tensor:
    parameter = next(model.parameters(), None)
    resolved_device = device or (parameter.device if parameter is not None else torch.device("cpu"))
    batch = tile_input.unsqueeze(0).to(resolved_device, dtype=torch.float32)
    output = model(batch)
    return output.metric_height.detach().cpu()[0, 0]


def run_cai_inference(
    model: Any,
    tile_inputs: Sequence[Sequence[torch.Tensor]] | Sequence[torch.Tensor],
    *,
    cai_r: int = 16,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Run deterministic CAI-style running-mean stitching over a tile grid."""
    if not tile_inputs:
        raise ValueError("tile_inputs cannot be empty")
    if isinstance(tile_inputs[0], torch.Tensor):  # type: ignore[index]
        grid = [list(tile_inputs)]  # type: ignore[list-item]
    else:
        grid = [list(row) for row in tile_inputs]  # type: ignore[arg-type]

    rows = len(grid)
    cols = len(grid[0])
    base_stride = 256
    full_height = base_stride * (rows - 1) + 257
    full_width = base_stride * (cols - 1) + 257
    accum = torch.zeros((full_height, full_width), dtype=torch.float32)
    weight = torch.zeros((full_height, full_width), dtype=torch.float32)

    shift_count = max(1, int(cai_r))
    max_shift = 0 if shift_count == 1 else min(16, grid[0][0].shape[-1] // 8)
    shifts = [(0, 0)]
    if shift_count > 1:
        for idx in range(1, shift_count):
            frac = idx / max(1, shift_count - 1)
            shift = int(round(frac * max_shift))
            shifts.append((shift, shift))

    for row_idx, row in enumerate(grid):
        if len(row) != cols:
            raise ValueError("All tile-input rows must have the same length")
        for col_idx, tile_input in enumerate(row):
            for dy, dx in shifts:
                shifted = _shift_view(tile_input, dy, dx) if (dy or dx) else tile_input
                prediction = _predict_metric_height(model, shifted, device=device)
                canvas_y = row_idx * base_stride
                canvas_x = col_idx * base_stride
                accum[canvas_y:canvas_y + 257, canvas_x:canvas_x + 257] += prediction
                weight[canvas_y:canvas_y + 257, canvas_x:canvas_x + 257] += 1.0

    return accum / weight.clamp_min(1.0)


__all__ = ["run_cai_inference"]
