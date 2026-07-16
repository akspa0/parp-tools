from __future__ import annotations

import numpy as np

from harvester.spec108_mixed_curriculum import (
    assign_group_splits,
    real_brush_descriptor,
    select_real_rows,
    select_synthetic_rows,
)


def test_descriptor_rewards_irregular_cell_motif() -> None:
    alpha = np.zeros((256, 256, 4), dtype=np.float32)
    height = np.zeros((257, 257), dtype=np.float32)
    for x, y in ((0, 0), (1, 0), (1, 1), (2, 1)):
        alpha[y * 16 : (y + 1) * 16, x * 16 : (x + 1) * 16, 0] = np.indices((16, 16)).sum(axis=0) % 2
    descriptor = real_brush_descriptor(alpha, height)
    assert descriptor["active_cells"] == 4
    assert float(descriptor["irregularity"]) > 0.0


def test_selection_is_capped_diverse_and_group_safe() -> None:
    real = [
        {"build": "0_5_3_3368", "map": map_name, "tile_id": index, "source_group_id": f"r:{map_name}:{index}", "descriptor": {"irregularity": 0.5, "height_relief": 8.0, "active_cells": 4, "brush_score": 5.0}}
        for index, map_name in enumerate(("Azeroth", "Kalimdor", "PVPZone02", "DeadminesInstance") * 3)
    ]
    selected_real = select_real_rows(real, total=8)
    assert len(selected_real) == 8
    assert len({row["map"] for row in selected_real}) == 4
    synthetic = [
        {"tile_id": index, "pattern": f"p{index % 4}", "source_group_id": f"s:{index // 3}"}
        for index in range(24)
    ]
    selected = selected_real + select_synthetic_rows(synthetic, total=12)
    assign_group_splits(selected)
    groups = {}
    for row in selected:
        groups.setdefault(row["source_group_id"], set()).add(row["split"])
    assert len(selected) == 20
    assert all(len(splits) == 1 for splits in groups.values())
