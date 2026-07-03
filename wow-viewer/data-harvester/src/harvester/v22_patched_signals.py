"""V22 patched signals derived from V18 store arrays in pure Python.

Each function takes a tile dict (per-tile numpy arrays) and optional metadata,
returns a single numpy array matching the V22 spec. No I/O, no game client
reparse. Reference implementations match the C# algorithms in
``RawArraySerializer.cs``.

All functions are deterministic and allocation-light.
"""

from __future__ import annotations

import numpy as np


def derive_mcnr_mask_257(tile: dict) -> np.ndarray:
    """MCNR checkerboard validity mask.

    If the V18 store carries ``mcnr_mask_257``, copy it directly.
    Otherwise derive as ``(x % 2 == y % 2)`` checkerboard — MCNR normals
    are stored on a checkerboard grid, so cells with real data are those
    where x and y have the same parity.
    """
    mask = tile.get("mcnr_mask_257")
    if mask is not None:
        return np.asarray(mask, dtype=bool)

    # Fallback: positive checkerboard
    y, x = np.ogrid[:257, :257]
    return (x % 2 == y % 2)


def derive_liquid_type_256(tile: dict) -> np.ndarray:
    """Liquid type mask at 256×256.

    Match ``RawArraySerializer.BuildLiquidType256``:
    1. If ``liquid_basic_type_257`` is present, crop 257→256.
    2. Values of 0xFF become 0; other values become ``value + 1``.
    3. If the source is missing, return zeros.
    """
    source = tile.get("liquid_basic_type_257")
    if source is None:
        return np.zeros((256, 256), dtype=np.uint8)

    source = np.asarray(source, dtype=np.uint8)
    if source.shape != (257, 257):
        return np.zeros((256, 256), dtype=np.uint8)

    cropped = source[:256, :256].copy()
    result = np.where(cropped == 0xFF, 0, cropped.astype(np.uint16) + 1).astype(np.uint8)
    return result


def derive_ground_intent_height_257(tile: dict) -> np.ndarray:
    """Inpaint ``height_257`` over ``object_precise_mask``.

    Match ``RawArraySerializer.BuildGroundIntentHeight257``:
    1. Start with a copy of ``height_257``.
    2. Pixels where ``object_precise_mask >= 0.05`` are flagged as unresolved.
    3. Iteratively replace each unresolved pixel with the mean of its
       resolved 4-neighbours. Repeat until all resolved or no progress.
    4. Max iterations = H + W = 514.
    """
    height = tile.get("height_257")
    precise = tile.get("object_precise_mask")

    if height is None:
        raise ValueError("V18 tile is missing height_257 — cannot derive ground_intent_height_257")

    height = np.asarray(height, dtype=np.float32)
    if height.shape != (257, 257):
        raise ValueError(f"height_257 has unexpected shape {height.shape}")

    result = height.copy()
    if precise is None:
        return result

    precise = np.asarray(precise, dtype=np.float32)
    if precise.shape != (257, 257):
        return result

    unresolved = precise >= 0.05
    if not np.any(unresolved):
        return result

    h, w = 257, 257
    max_iter = h + w
    for iteration in range(max_iter):
        next_unresolved = np.zeros_like(unresolved)
        made_progress = False

        # Find unresolved pixels
        unresolved_ys, unresolved_xs = np.nonzero(unresolved)
        if len(unresolved_ys) == 0:
            break

        for y, x in zip(unresolved_ys, unresolved_xs):
            neighbors = []
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and not unresolved[ny, nx]:
                    neighbors.append(result[ny, nx])
            if neighbors:
                result[y, x] = np.mean(neighbors)
                made_progress = True
            else:
                next_unresolved[y, x] = True

        unresolved = next_unresolved
        if not made_progress:
            break

    return result.astype(np.float32)


def derive_model_focus_mask(tile: dict) -> np.ndarray:
    """Alias of ``object_filtered_mask``.

    ``model_focus_mask`` is the renamed V22 successor to
    ``object_filtered_mask``. Both names refer to the same data.
    """
    mask = tile.get("object_filtered_mask")
    if mask is not None:
        return np.asarray(mask, dtype=np.float32)
    return np.zeros((257, 257), dtype=np.float32)


def derive_model_above_terrain_mask(
    tile: dict,
    mddf_placements: list[dict],
    modf_placements: list[dict],
    tile_x: int,
    tile_y: int,
) -> np.ndarray:
    """Mask of placements with Z above terrain height.

    Match ``RawArraySerializer.BuildModelAboveTerrainMask``:
    For each MDDF/MODF placement, project ``(posX, posY, posZ)`` to a tile
    pixel using four candidate projection modes. If the pixel's height value
    ``is >= posZ - 1.0``, set the mask pixel to 1.0.

    Placements whose projection falls outside the tile or whose Z is below
    the terrain surface are excluded — they are underground and invisible
    on the minimap.
    """
    tile_size = 257
    map_origin = 17066.666
    tile_world = 533.33333
    epsilon = 1.0

    mask = np.zeros((tile_size, tile_size), dtype=np.float32)
    height = np.asarray(tile.get("height_257", np.zeros((257, 257), dtype=np.float32)), dtype=np.float32)

    if height.shape != (257, 257):
        return mask

    def _project(pos_x: float, pos_y: float, pos_z: float) -> tuple[int, int] | None:
        candidates = [
            ((pos_x / tile_world) - tile_x, (pos_z / tile_world) - tile_y),
            (((map_origin - pos_z) / tile_world) - tile_x, ((map_origin - pos_x) / tile_world) - tile_y),
            ((pos_x / tile_world) - tile_x, (pos_y / tile_world) - tile_y),
            (((map_origin - pos_y) / tile_world) - tile_x, ((map_origin - pos_x) / tile_world) - tile_y),
        ]

        best_overflow = float("inf")
        best_u = best_v = 0.0
        found = False

        for cand_u, cand_v in candidates:
            overflow = max(0.0, -cand_u) + max(0.0, cand_u - 1.0) + max(0.0, -cand_v) + max(0.0, cand_v - 1.0)
            if overflow < best_overflow:
                best_overflow = overflow
                best_u, best_v = cand_u, cand_v
                found = True
                if overflow <= 1e-6:
                    break

        if not found:
            return None

        px = int(best_u * (tile_size - 1))
        py = int(best_v * (tile_size - 1))
        if 0 <= px < tile_size and 0 <= py < tile_size:
            return px, py
        return None

    for placement in mddf_placements + modf_placements:
        pos_x = placement.get("posX", 0.0)
        pos_y = placement.get("posY", 0.0)
        pos_z = placement.get("posZ", 0.0)

        proj = _project(pos_x, pos_y, pos_z)
        if proj is None:
            continue

        px, py = proj
        terrain_z = height[py, px]
        if pos_z >= terrain_z - epsilon:
            mask[py, px] = 1.0

    return mask
