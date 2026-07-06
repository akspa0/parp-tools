"""FR-010: minimap cleaner (User Story 2). Pure NumPy, no model.

Operates at the minimap's native 256x256 resolution (spec amendment A4).
Object pixels — per V18's tile-level ``object_precise_mask`` (257x257 float32,
> 0.5 = object, max-pooled to 256) — are replaced by the median of their
non-object 8-connected neighbourhood, iterating inward until the masked region
is filled; pixels that never gain a non-object neighbour fall back to the
global mean colour of the unmasked area.

Where the V18 store carries a viewer-rendered ``no_object_minimap`` (present
on 0_5_3_3368), that render is preferred outright — the viewer is the working
renderer, so its object-free composite beats any mask-based fill.
"""

from __future__ import annotations

import warnings

import numpy as np

_MAX_FILL_PASSES = 512


def object_mask_256(object_precise_mask_257: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Reduce the 257x257 corner-grid mask to the 256x256 minimap cell grid.

    A minimap cell is "object" when any of its four corner samples is masked.
    """
    mask = np.asarray(object_precise_mask_257, dtype=np.float32)
    if mask.shape != (257, 257):
        raise ValueError(f"object_precise_mask must be (257, 257); got {mask.shape}")
    corners = np.stack([mask[:-1, :-1], mask[1:, :-1], mask[:-1, 1:], mask[1:, 1:]])
    return corners.max(axis=0) > threshold


def clean_minimap(
    minimap_rgb: np.ndarray,
    object_precise_mask: np.ndarray,
    no_object_minimap: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Clean a V18 minimap tile. Returns ``(cleaned (256,256,3) float32 in [0,1], meta)``.

    ``meta`` carries ``cleaned_minimap_unavailable`` (True when every pixel was
    masked and the output is the global mean colour) and ``source`` (one of
    ``no_object_minimap`` / ``median_fill`` / ``identity``).
    """
    rgb = np.asarray(minimap_rgb)
    if rgb.shape != (256, 256, 3):
        raise ValueError(f"minimap_rgb must be (256, 256, 3); got {rgb.shape}")
    rgb = rgb.astype(np.float32)
    if rgb.max() > 1.5:
        rgb = rgb / 255.0

    if no_object_minimap is not None:
        rendered = np.asarray(no_object_minimap)
        if rendered.shape == (256, 256, 3) and rendered.any():
            cleaned = rendered.astype(np.float32)
            if cleaned.max() > 1.5:
                cleaned = cleaned / 255.0
            return cleaned, {"cleaned_minimap_unavailable": False, "source": "no_object_minimap"}

    mask = object_mask_256(object_precise_mask)
    if not mask.any():
        return rgb.copy(), {"cleaned_minimap_unavailable": False, "source": "identity"}

    if mask.all():
        mean_colour = rgb.reshape(-1, 3).mean(axis=0)
        cleaned = np.broadcast_to(mean_colour, (256, 256, 3)).astype(np.float32).copy()
        return cleaned, {"cleaned_minimap_unavailable": True, "source": "median_fill"}

    cleaned = rgb.copy()
    unresolved = mask.copy()
    for _ in range(_MAX_FILL_PASSES):
        if not unresolved.any():
            break

        valid = ~unresolved
        padded = np.pad(cleaned, ((1, 1), (1, 1), (0, 0)), mode="edge")
        padded_valid = np.pad(valid, 1, mode="constant", constant_values=False)

        neighbours = []
        neighbour_valid = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                neighbours.append(padded[1 + dy : 257 + dy, 1 + dx : 257 + dx])
                neighbour_valid.append(padded_valid[1 + dy : 257 + dy, 1 + dx : 257 + dx])

        stack = np.stack(neighbours)  # (8, 256, 256, 3)
        stack_valid = np.stack(neighbour_valid)  # (8, 256, 256)
        fillable = unresolved & (stack_valid.sum(axis=0) > 0)
        if not fillable.any():
            break

        masked_stack = np.where(stack_valid[..., None], stack, np.nan)
        with warnings.catch_warnings():
            # Pixels with zero valid neighbours produce all-NaN slices; they are
            # excluded by `fillable`, so the warning is noise.
            warnings.simplefilter("ignore", category=RuntimeWarning)
            median = np.nanmedian(masked_stack, axis=0)
        cleaned[fillable] = median[fillable]
        unresolved[fillable] = False

    if unresolved.any():
        mean_colour = rgb[~mask].reshape(-1, 3).mean(axis=0)
        cleaned[unresolved] = mean_colour

    return cleaned, {"cleaned_minimap_unavailable": False, "source": "median_fill"}
