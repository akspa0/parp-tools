from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import zarr

_SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from harvester.fractal_canvas import CanvasTileRecord  # noqa: E402, I001
from harvester.prefab_fragments import build_fragment_families, extract_prefab_fragments  # noqa: E402, I001


def test_repeated_local_motif_is_grouped_without_emitting_a_zone(tmp_path: Path) -> None:
    root = zarr.open_group(str(tmp_path / "tiny.zarr"), mode="w")
    alpha = np.zeros((2, 256, 256, 1), dtype=np.float32)
    # Two equal motifs live inside much larger, continuous painted zones.
    alpha[:, 16:240, 16:240, 0] = 0.8
    alpha[:, 64:80, 64:96, 0] = 0.0
    alpha[:, 80:96, 80:112, 0] = 0.0
    height = np.zeros((2, 257, 257), dtype=np.float32)
    root.create_array("alpha_256", data=alpha)
    root.create_array("height_257", data=height)
    records = [
        CanvasTileRecord("test", "Map", 0, 10, 10, has_alpha_256=True, has_height_257=True),
        CanvasTileRecord("test", "Map", 1, 11, 10, has_alpha_256=True, has_height_257=True),
    ]

    fragments = extract_prefab_fragments(
        root,
        records,
        supports=(32,),
        stride=16,
        min_alpha_coverage=0.05,
        min_height_range=999.0,
        max_candidates_per_tile=64,
    )

    assert fragments
    assert all(fragment.support_px == 32 for fragment in fragments)
    assert all(fragment.local_x + fragment.support_px <= 256 for fragment in fragments)
    assert all(fragment.local_y + fragment.support_px <= 256 for fragment in fragments)
    assert any(int(family["member_count"]) >= 2 for family in build_fragment_families(fragments))
