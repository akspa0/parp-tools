"""Spec 116 US4: spatially-isolated held-out split.

Builds a held-out set in which **no held-out tile shares an edge or corner (8-neighbour) with any
training tile** (FR-010 / SC-005). The construction is a deterministic graph problem over the tile
grid:

- Held-out tiles are grown as contiguous **blocks** (efficient: a block of side ``s`` needs only a
  ~``4s`` perimeter buffer, vs single-tile held-out which wastes 8 buffer tiles each).
- A **buffer ring** of width ``buffer_rings`` around every held-out block is marked ``excluded``
  -- neither train nor held_out -- so no held-out tile is ever 8-adjacent to a training tile.
- Everything remaining is training.

The builder computes and reports ``verified_violation_count`` (the number of held-out/train
8-adjacent pairs), which MUST be zero. Rebuilding the split **invalidates absolute comparison**
with all prior results (FR-017); the report states this and names the baseline requiring re-run.
"""

from __future__ import annotations

from collections import deque
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from harvester.spec116.structure_contract import sha256_file, validate_held_out_split

HELD_OUT_SPLIT_SCHEMA = "v50-held-out-split-v1"
TAXONOMY_REVISION = "v115.1"
DEFAULT_HELD_OUT_FRACTION = 0.15
DEFAULT_BUFFER_RINGS = 1
DEFAULT_BLOCK_SIZE = 8
SPLIT_TRAIN = "train"
SPLIT_HELD_OUT = "held_out"
SPLIT_EXCLUDED = "excluded"


class HeldOutSplitError(ValueError):
    """Raised when a spatially-isolated held-out split cannot be built as declared."""


def _neighbours(coord: tuple[int, int], tile_set: set[tuple[int, int]]) -> list[tuple[int, int]]:
    x, y = coord
    out = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            n = (x + dx, y + dy)
            if n in tile_set:
                out.append(n)
    return out


def _ring(coords: set[tuple[int, int]], tile_set: set[tuple[int, int]], width: int) -> set[tuple[int, int]]:
    """``width`` rings of 8-neighbours around ``coords`` (excluding ``coords`` themselves)."""
    ring: set[tuple[int, int]] = set()
    frontier = set(coords)
    for _ in range(width):
        next_frontier: set[tuple[int, int]] = set()
        for c in frontier:
            for n in _neighbours(c, tile_set):
                if n not in coords and n not in ring:
                    ring.add(n)
                    next_frontier.add(n)
        frontier = next_frontier
    return ring


def _verify(held_out: set[tuple[int, int]], train: set[tuple[int, int]],
             tile_set: set[tuple[int, int]]) -> int:
    """Count held-out/train 8-adjacent pairs (the violation count; MUST be 0)."""
    violations = 0
    for h in held_out:
        for n in _neighbours(h, tile_set):
            if n in train:
                violations += 1
    return violations


def build_held_out_split(
    *,
    tiles: list[dict],
    held_out_fraction: float = DEFAULT_HELD_OUT_FRACTION,
    buffer_rings: int = DEFAULT_BUFFER_RINGS,
    block_size: int = DEFAULT_BLOCK_SIZE,
    seed: int = 0,
) -> dict:
    """Assign each tile to train / held_out / excluded so no held-out tile touches a train tile.

    ``tiles`` is the curriculum index as a list of dicts with ``tile_row, map, tile_x, tile_y``.
    Returns a dict with ``assignments`` (list of {tile_row, map, tile_x, tile_y, split}) and the
    split counts + verified violation count. Raises if the violation count is nonzero (a bug) or
    if the corpus is too small to isolate any held-out tile.
    """
    if not 0.0 < held_out_fraction < 1.0:
        raise HeldOutSplitError(f"held_out_fraction must be in (0, 1), got {held_out_fraction!r}")
    if buffer_rings < 1:
        raise HeldOutSplitError(f"buffer_rings must be >= 1, got {buffer_rings!r}")
    if block_size < 1:
        raise HeldOutSplitError(f"block_size must be >= 1, got {block_size!r}")

    # Group tiles by map; adjacency is within-map only.
    by_map: dict[str, list[dict]] = {}
    for t in tiles:
        by_map.setdefault(str(t["map"]), []).append(t)

    assignments: list[dict] = []
    total = len(tiles)
    target_held = max(1, int(round(held_out_fraction * total)))

    rng = np.random.default_rng(seed)
    held_count = 0
    excluded_count = 0

    # Deterministic per-map block selection. Process maps in sorted order; within a map, visit
    # tiles in a seeded shuffle and grow contiguous held-out blocks.
    for map_name in sorted(by_map):
        map_tiles = by_map[map_name]
        tile_set: set[tuple[int, int]] = {(int(t["tile_x"]), int(t["tile_y"])) for t in map_tiles}
        coord_to_tile: dict[tuple[int, int], dict] = {(int(t["tile_x"]), int(t["tile_y"])): t for t in map_tiles}

        assigned: set[tuple[int, int]] = set()  # held_out or excluded
        held_out: set[tuple[int, int]] = set()

        order = list(tile_set)
        rng.shuffle(order)  # deterministic given seed

        for start in order:
            if held_count >= target_held:
                break
            if start in assigned:
                continue
            # Grow a contiguous block from `start` via BFS over unassigned tiles.
            block: set[tuple[int, int]] = set()
            q: deque[tuple[int, int]] = deque([start])
            while q and len(block) < block_size:
                c = q.popleft()
                if c in assigned or c in block:
                    continue
                block.add(c)
                for n in _neighbours(c, tile_set):
                    if n not in assigned and n not in block:
                        q.append(n)
            if not block:
                continue
            # The buffer ring around the block becomes excluded.
            buf = _ring(block, tile_set, buffer_rings)
            # Refuse a block whose buffer would eat already-held tiles (shouldn't happen by
            # construction, but guard against adjacency overlap).
            if buf & held_out:
                continue
            held_out |= block
            assigned |= block | buf
            held_count += len(block)
            excluded_count += len(buf)

        train_set = tile_set - assigned

        # Verify isolation within this map.
        violations = _verify(held_out, train_set, tile_set)
        if violations != 0:
            raise HeldOutSplitError(
                f"{map_name}: {violations} held-out/train 8-adjacent pairs after construction (bug)"
            )

        for coord in tile_set:
            t = coord_to_tile[coord]
            if coord in held_out:
                split = SPLIT_HELD_OUT
            elif coord in train_set:
                split = SPLIT_TRAIN
            else:
                split = SPLIT_EXCLUDED
            assignments.append({
                "tile_row": int(t["tile_row"]), "map": str(t["map"]),
                "tile_x": int(t["tile_x"]), "tile_y": int(t["tile_y"]), "split": split,
            })

    train_count = sum(1 for a in assignments if a["split"] == SPLIT_TRAIN)
    held_out_count = sum(1 for a in assignments if a["split"] == SPLIT_HELD_OUT)
    if held_out_count == 0:
        raise HeldOutSplitError("corpus too small to isolate any held-out tile with this configuration")
    if train_count == 0:
        raise HeldOutSplitError(
            "corpus too small: the isolation buffer consumed every non-held-out tile; no training tiles remain"
        )

    # Global re-verify (cross-map adjacency is impossible, but keep the gate explicit).
    global_violations = 0
    held_coords = {(a["map"], a["tile_x"], a["tile_y"]) for a in assignments if a["split"] == SPLIT_HELD_OUT}
    train_coords = {(a["map"], a["tile_x"], a["tile_y"]) for a in assignments if a["split"] == SPLIT_TRAIN}
    for m, x, y in held_coords:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                if (m, x + dx, y + dy) in train_coords:
                    global_violations += 1

    return {
        "assignments": assignments,
        "split_counts": {"train": train_count, "held_out": held_out_count},
        "excluded_count": excluded_count,
        "verified_violation_count": global_violations,
        "buffer_rings": buffer_rings,
        "block_size": block_size,
        "held_out_fraction": held_out_fraction,
        "seed": seed,
    }


def write_split(
    *,
    store: Path,
    output: Path,
    split: dict,
    build_id: str = "",
) -> dict:
    """Persist ``split`` as ``split.parquet`` + ``split.json`` (``v50-held-out-split-v1``)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    output.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(split["assignments"]), output / "split.parquet")

    manifest = {
        "schema": HELD_OUT_SPLIT_SCHEMA,
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "build_id": build_id,
        "store": {"path": str(store.resolve()), "sha256": sha256_file(store / "index.parquet")},
        "taxonomy_revision": TAXONOMY_REVISION,
        "adjacency_rule": "8-neighbour",
        "buffer_rings": split["buffer_rings"],
        "seed": split["seed"],
        "split_counts": split["split_counts"],
        "excluded_count": split["excluded_count"],
        "verified_violation_count": split["verified_violation_count"],
        "absolute_comparison_to_prior_invalid": True,
        "baseline_requiring_rerun": "tile-mean (Spec 114 geometry) and majority-class (Spec 116 structure)",
    }
    validate_held_out_split(manifest)
    (output / "split.json").write_text(__import__("json").dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_split(split_dir: Path) -> tuple[dict, list[dict]]:
    """Read a persisted split: (manifest, assignment rows)."""
    import json

    import pyarrow.parquet as pq

    manifest = json.loads((split_dir / "split.json").read_text(encoding="utf-8"))
    validate_held_out_split(manifest)
    rows = pq.read_table(split_dir / "split.parquet").to_pylist()
    return manifest, rows


__all__ = [
    "HeldOutSplitError",
    "build_held_out_split",
    "write_split",
    "load_split",
    "DEFAULT_HELD_OUT_FRACTION",
    "DEFAULT_BUFFER_RINGS",
    "DEFAULT_BLOCK_SIZE",
    "HELD_OUT_SPLIT_SCHEMA",
]
