"""Fast, bounded selection for Spec 108's mixed real/synthetic curriculum.

This is deliberately tile-local: it reads each candidate's 16x16 alpha/height
cell lattice once, scores irregular (non-rectangular) brush/paste activity, and
uses that compact descriptor for diversity selection.  It never builds a world
canvas, connected zone, or rectangular crop.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict, deque
from typing import Any

import numpy as np


def real_brush_descriptor(alpha_256: np.ndarray, height_257: np.ndarray) -> dict[str, float | int]:
    alpha = np.asarray(alpha_256, dtype=np.float32).reshape(16, 16, 16, 16, -1)
    height = np.asarray(height_257, dtype=np.float32)
    alpha_var = alpha.std(axis=(1, 3, 4))
    relief = np.empty((16, 16), dtype=np.float32)
    for y in range(16):
        for x in range(16):
            patch = height[y * 16 : y * 16 + 17, x * 16 : x * 16 + 17]
            relief[y, x] = float(patch.max() - patch.min())
    active = (alpha_var >= 0.025) | (relief >= 2.0)
    count = int(active.sum())
    if count:
        ys, xs = np.where(active)
        bbox_area = int((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1))
        irregularity = 1.0 - count / bbox_area
    else:
        irregularity = 0.0
    return {
        "active_cells": count,
        "irregularity": round(float(irregularity), 6),
        "alpha_variation": round(float(alpha_var.mean()), 6),
        "height_relief": round(float(relief.mean()), 6),
        "brush_score": round(float(irregularity * count + alpha_var.mean() + relief.mean() / 8.0), 6),
    }


def select_real_rows(rows: list[dict[str, Any]], *, total: int) -> list[dict[str, Any]]:
    """Quota maps evenly, then round-robin irregular brush/paste descriptor buckets."""
    by_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_map[str(row["map"])].append(row)
    maps = sorted(by_map)
    if not maps:
        return []
    base, extra = divmod(total, len(maps))
    selected: list[dict[str, Any]] = []
    remaining_by_map: dict[str, list[dict[str, Any]]] = {}
    for map_index, map_name in enumerate(maps):
        quota = base + (1 if map_index < extra else 0)
        buckets: dict[tuple[int, int, int], deque[dict[str, Any]]] = defaultdict(deque)
        for row in by_map[map_name]:
            d = row["descriptor"]
            bucket = (min(5, int(float(d["irregularity"]) * 6)), min(7, int(float(d["height_relief"]) / 8)), min(7, int(int(d["active_cells"]) / 32)))
            buckets[bucket].append(row)
        for key, values in list(buckets.items()):
            buckets[key] = deque(sorted(values, key=lambda item: (-float(item["descriptor"]["brush_score"]), _stable_key(item))))
        ordered = sorted(buckets, key=lambda key: (-key[0], -key[1], -key[2]))
        while len([row for row in selected if row["map"] == map_name]) < quota and ordered:
            for key in list(ordered):
                if len([row for row in selected if row["map"] == map_name]) >= quota:
                    break
                if buckets[key]:
                    selected.append(buckets[key].popleft())
                if not buckets[key]:
                    ordered.remove(key)
        remaining_by_map[map_name] = [row for values in buckets.values() for row in values]
    # Small maps may legitimately have fewer alpha-bearing pages than their even
    # quota. Preserve their available examples, then redistribute only the
    # unfilled slots to the other maps rather than lowering the declared cap.
    for map_name in maps:
        if len(selected) >= total:
            break
        extras = sorted(remaining_by_map[map_name], key=lambda item: (-float(item["descriptor"]["brush_score"]), _stable_key(item)))
        take = min(total - len(selected), len(extras))
        selected.extend(extras[:take])
    return selected


def select_synthetic_rows(rows: list[dict[str, Any]], *, total: int) -> list[dict[str, Any]]:
    """Keep lighting siblings together and distribute source groups across terrain families."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("source_group_id") or row["tile_id"])].append(row)
    by_pattern: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    for group in groups.values():
        by_pattern[str(group[0].get("pattern") or "unknown")].append(sorted(group, key=lambda item: int(item["tile_id"])))
    selected: list[dict[str, Any]] = []
    queues = {pattern: deque(sorted(values, key=lambda group: _stable_key(group[0]))) for pattern, values in by_pattern.items()}
    while len(selected) < total and any(queues.values()):
        for pattern in sorted(queues):
            if not queues[pattern] or len(selected) >= total:
                continue
            group = queues[pattern].popleft()
            take = min(len(group), total - len(selected))
            selected.extend(group[:take])
    return selected


def assign_group_splits(rows: list[dict[str, Any]], *, val_fraction: float = 0.2) -> None:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["source_group_id"])].append(row)
    ordered = sorted(groups, key=lambda value: _stable_key({"source_group_id": value}))
    val_groups = set(ordered[:max(1, round(len(ordered) * val_fraction))])
    for group_id, members in groups.items():
        for row in members:
            row["split"] = "val" if group_id in val_groups else "train"


def _stable_key(row: dict[str, Any]) -> str:
    return hashlib.sha256(str(row.get("source_group_id") or f"{row.get('build')}|{row.get('map')}|{row.get('tile_id')}").encode()).hexdigest()
