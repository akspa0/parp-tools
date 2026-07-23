"""Shared Spec 119 fixtures: a tiny synthetic object-library store (zarr + parquet)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import zarr

ASSET_SPECS = [
    # (family dir, filename stem, asset_type, coverage)
    ("world/wmo/azeroth/buildings/castle", "castle01", "wmo", 0.40),
    ("world/wmo/azeroth/buildings/castle", "castle02", "wmo", 0.38),
    ("world/wmo/azeroth/buildings/chapel", "chapel", "wmo", 0.55),
    ("world/m2/kalimdor/trees", "tree01", "m2", 0.20),
    ("world/mdx/kalimdor/effects", "glow", "mdx", 0.12),
    ("world/m2/kalimdor/rocks", "rock", "m2", 0.005),  # blank capture -> 'empty' class
]


def make_library_store(path: Path, target_size: int = 128) -> Path:
    """Write a minimal object-library store matching the Spec 118 capture layout."""
    from harvester.object_library import library_id_from_asset_path

    n = len(ASSET_SPECS)
    rng = np.random.default_rng(7)
    group = zarr.open_group(str(path), mode="w")
    rgb = (rng.random((n, target_size, target_size, 3)) * 255).astype(np.uint8)
    mask = np.zeros((n, target_size, target_size), dtype=np.uint8)
    for i, (_family, _stem, _atype, coverage) in enumerate(ASSET_SPECS):
        count = int(round(coverage * target_size * target_size))
        if count:
            flat = mask[i].reshape(-1)
            flat[:count] = 255
    group.create_array("capture_rgb", data=rgb)
    group.create_array("capture_mask", data=mask)
    group.attrs.update({"schema": "spec-077-object-library", "schema_version": "1",
                        "target_size": target_size, "entry_count": n})
    rows = []
    for family, stem, atype, _coverage in ASSET_SPECS:
        ext = {"wmo": ".wmo", "m2": ".m2", "mdx": ".mdx"}[atype]
        full = f"{family}/{stem}{ext}"
        rows.append(
            {
                "library_id": library_id_from_asset_path(full),
                "original_asset_path": full.replace("/", "\\"),
                "normalized_asset_path": full,
                "asset_type": atype,
                "capture_status": "captured",
                "visibility_class": "roof_visible",
                "review_state": "unreviewed",
                "source_builds": ["0.5.3.3368"],
                "placement_observation_count": 0,
                "preferred_variant_id": "",
            }
        )
    pq.write_table(pa.Table.from_pylist(rows), path / "assets.parquet")
    pq.write_table(pa.Table.from_pylist(
        [{"row": i, "library_id": r["library_id"]} for i, r in enumerate(rows)]
    ), path / "index.parquet")
    return path


@pytest.fixture
def library_store(tmp_path: Path) -> Path:
    return make_library_store(tmp_path / "library.zarr")


@pytest.fixture
def library_split(tmp_path: Path, library_store: Path) -> Path:
    """A real family-isolated split over the fixture store (castle+chapel held out)."""
    from harvester.spec119.split import build_family_split, read_asset_rows

    rows = read_asset_rows(library_store)
    split = build_family_split(rows, held_out_fraction=0.34, seed=0)
    assert split["verified_violation_count"] == 0
    import json

    path = tmp_path / "split.json"
    path.write_text(json.dumps(split, indent=2), encoding="utf-8")
    return path
