"""Spec 119 contract tests (T005): label/family/target derivations are pure and total."""

from __future__ import annotations

import numpy as np
import pytest

from harvester.spec119.object_library_contract import (
    BLANK_THRESHOLD_DEFAULT,
    COARSE_CLASS_INDEX,
    COARSE_INDEX_CLASS,
    ObjectLibraryContractError,
    coarse_index_for_label,
    coarse_label_for_row,
    derive_asset_family,
    derive_fine_family_label,
    is_blank_capture,
    mask_coverage,
    segmentation_target,
    variant_stem,
)


def test_coarse_class_index_round_trip() -> None:
    assert COARSE_CLASS_INDEX == {"empty": 0, "m2": 1, "mdx": 2, "wmo": 3}
    for name, index in COARSE_CLASS_INDEX.items():
        assert COARSE_INDEX_CLASS[index] == name
        assert coarse_index_for_label(name) == index


def test_derive_asset_family_uses_parent_directory() -> None:
    assert (
        derive_asset_family("world/wmo/azeroth/buildings/castle/castle01.wmo")
        == "world/wmo/azeroth/buildings/castle"
    )
    # Numbered variants in one directory share a family (FR-004).
    assert derive_asset_family("world/wmo/azeroth/buildings/castle/castle02.wmo") == (
        derive_asset_family("world/wmo/azeroth/buildings/castle/castle01.wmo")
    )
    # Compound WMO wrapper suffixes still derive the same family.
    assert derive_asset_family("world/wmo/keep.wmo.mpq") == "world/wmo"
    # Backslash input is normalized.
    assert derive_asset_family("World\\wmo\\Keep.wmo".lower()) == "world/wmo"
    # A top-level file is its own family.
    assert derive_asset_family("lonely.m2") == "lonely.m2"


def test_derive_fine_family_label() -> None:
    assert (
        derive_fine_family_label("world/wmo/azeroth/buildings/castle/castle01.wmo") == "castle"
    )
    assert derive_fine_family_label("tree.m2") == "unknown"


def test_blank_threshold_relabels_low_coverage_to_empty() -> None:
    # 0.005 coverage < default 0.01 threshold -> empty (D-04).
    assert coarse_label_for_row("wmo", 0.005) == "empty"
    # 0.02 coverage >= threshold -> authoritative asset_type label kept.
    assert coarse_label_for_row("wmo", 0.02) == "wmo"
    assert coarse_label_for_row("m2", 0.02) == "m2"
    assert is_blank_capture(0.005, BLANK_THRESHOLD_DEFAULT)
    assert not is_blank_capture(0.02, BLANK_THRESHOLD_DEFAULT)


def test_coarse_label_refuses_unknown_asset_type() -> None:
    with pytest.raises(ObjectLibraryContractError, match="unknown asset_type"):
        coarse_label_for_row("blp", 0.5)


def test_segmentation_target_shape_and_dtype() -> None:
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:4, 3:6] = 255
    target = segmentation_target(mask)
    assert target.shape == (8, 8)
    assert target.dtype == np.int64
    assert int(target.sum()) == 6
    assert set(np.unique(target).tolist()) == {0, 1}


def test_mask_coverage() -> None:
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[0, 0] = 255
    assert mask_coverage(mask) == pytest.approx(1 / 16)
    assert mask_coverage(np.zeros((4, 4), dtype=np.uint8)) == 0.0


def test_variant_stem_strips_numeric_suffixes() -> None:
    assert variant_stem("world/wmo/azeroth/buildings/castle/castle01.wmo") == "castle"
    assert variant_stem("world/wmo/azeroth/buildings/castle/castle02.wmo") == "castle"
    assert variant_stem("world/m2/tree_000.m2") == "tree"
    assert variant_stem("world/m2/tree_001.m2") == "tree"
    assert variant_stem("world/m2/tree.m2") == "tree"
    assert variant_stem("world/wmo/keep.wmo.mpq") == "keep"
