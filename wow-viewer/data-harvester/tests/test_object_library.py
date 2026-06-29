"""Tests for the spec 077 per-object capture library contract (object_library.py).

These tests guard the deterministic ID rules, enum validation, and
default-state invariants defined in data-model.md §1.1 and §1.2. They
mirror ``ObjectLibraryContractsTests`` in C#; if you change one side,
change the other.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from harvester.object_library import (  # noqa: E402
    ObjectCaptureVariant,
    ObjectLibraryEntry,
    detect_asset_type,
    is_clutter_asset,
    library_id_from_asset_path,
    make_entry_from_path,
    make_variant_id,
    normalize_asset_path,
)


class TestNormalizeAssetPath:
    def test_lowercases_and_flips_separators(self) -> None:
        assert normalize_asset_path("World\\WMO\\Azeroth\\Stormwind.wmo") == "world/wmo/azeroth/stormwind.wmo"

    def test_dedupes_double_slashes(self) -> None:
        assert normalize_asset_path("world//wmo//azeroth") == "world/wmo/azeroth"

    def test_blank_input_returns_empty(self) -> None:
        assert normalize_asset_path("") == ""
        assert normalize_asset_path("   ") == ""


class TestDetectAssetType:
    @pytest.mark.parametrize(
        ("path", "expected"),
        [
            ("world/wmo/azeroth/stormwind.wmo", "wmo"),
            ("world/wmo/azeroth/stormwind.wmo.MPQ", "wmo"),
            ("World/Generic/Fruit_Apple.m2", "m2"),
            ("World/Model/Tree.mdx", "mdx"),
            ("World/Generic/Rock.unk", "unknown"),
            ("", "unknown"),
        ],
    )
    def test_returns_expected_kind(self, path: str, expected: str) -> None:
        assert detect_asset_type(path) == expected


class TestLibraryId:
    def test_stable_across_calls(self) -> None:
        first = library_id_from_asset_path("world/wmo/azeroth/stormwind.wmo")
        second = library_id_from_asset_path("world/wmo/azeroth/stormwind.wmo")
        assert first == second
        assert first.startswith("objlib_")
        assert len(first) == 21

    def test_differs_for_distinct_paths(self) -> None:
        a = library_id_from_asset_path("world/wmo/azeroth/stormwind.wmo")
        b = library_id_from_asset_path("world/wmo/azeroth/ironforge.wmo")
        assert a != b

    def test_blank_returns_empty(self) -> None:
        assert library_id_from_asset_path("") == ""


class TestMakeEntry:
    def test_defaults_match_data_model(self) -> None:
        entry = make_entry_from_path("World\\Generic\\Apple.m2")
        assert entry.original_asset_path == "World\\Generic\\Apple.m2"
        assert entry.normalized_asset_path == "world/generic/apple.m2"
        assert entry.asset_type == "m2"
        assert entry.capture_status == "not_attempted"
        assert entry.visibility_class == "unknown"
        assert entry.review_state == "unreviewed"
        assert entry.placement_observation_count == 0
        assert entry.preferred_variant_id is None
        assert entry.source_builds == ()
        assert entry.source_maps == ()

    def test_library_id_is_deterministic(self) -> None:
        entry = make_entry_from_path("World\\WMO\\Azeroth\\Stormwind.wmo")
        assert entry.library_id == library_id_from_asset_path(entry.normalized_asset_path)


class TestMakeVariantId:
    def test_stable_for_same_pose(self) -> None:
        a = make_variant_id(
            library_id="objlib_abc",
            capture_build="3_3_5_12340",
            capture_mode="orthographic_topdown",
            rot_x=0.0,
            rot_y=0.0,
            rot_z=0.0,
            scale=1.0,
        )
        b = make_variant_id(
            library_id="objlib_abc",
            capture_build="3_3_5_12340",
            capture_mode="orthographic_topdown",
            rot_x=0.0,
            rot_y=0.0,
            rot_z=0.0,
            scale=1.0,
        )
        assert a == b
        assert a.startswith("objvar_")
        assert len(a) == 23

    def test_differs_for_distinct_pose(self) -> None:
        a = make_variant_id(
            library_id="objlib_abc",
            capture_build="3_3_5_12340",
            capture_mode="orthographic_topdown",
            rot_x=0.0,
            rot_y=0.0,
            rot_z=0.0,
            scale=1.0,
        )
        b = make_variant_id(
            library_id="objlib_abc",
            capture_build="3_3_5_12340",
            capture_mode="orthographic_topdown",
            rot_x=0.0,
            rot_y=0.0,
            rot_z=1.5707963,
            scale=1.0,
        )
        assert a != b

    def test_blank_library_id_returns_empty(self) -> None:
        assert (
            make_variant_id(
                library_id="",
                capture_build="3_3_5_12340",
                capture_mode="orthographic_topdown",
                rot_x=0.0,
                rot_y=0.0,
                rot_z=0.0,
                scale=1.0,
            )
            == ""
        )


class TestObjectLibraryEntryValidation:
    def test_invalid_capture_status_raises(self) -> None:
        with pytest.raises(ValueError):
            ObjectLibraryEntry(
                library_id="objlib_abc",
                original_asset_path="x.m2",
                normalized_asset_path="x.m2",
                capture_status="done",
            )

    def test_invalid_review_state_raises(self) -> None:
        with pytest.raises(ValueError):
            ObjectLibraryEntry(
                library_id="objlib_abc",
                original_asset_path="x.m2",
                normalized_asset_path="x.m2",
                review_state="pending",
            )

    def test_invalid_visibility_raises(self) -> None:
        with pytest.raises(ValueError):
            ObjectLibraryEntry(
                library_id="objlib_abc",
                original_asset_path="x.m2",
                normalized_asset_path="x.m2",
                visibility_class="hidden",
            )


class TestObjectCaptureVariantValidation:
    def test_invalid_capture_mode_raises(self) -> None:
        with pytest.raises(ValueError):
            ObjectCaptureVariant(
                variant_id="objvar_abc",
                library_id="objlib_abc",
                capture_build="3_3_5_12340",
                capture_mode="render",
            )

    def test_confidence_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError):
            ObjectCaptureVariant(
                variant_id="objvar_abc",
                library_id="objlib_abc",
                capture_build="3_3_5_12340",
                capture_confidence=1.5,
            )

    def test_bbox_wh_reports_width_height(self) -> None:
        variant = ObjectCaptureVariant(
            variant_id="objvar_abc",
            library_id="objlib_abc",
            capture_build="3_3_5_12340",
            bbox_x0=2,
            bbox_y0=3,
            bbox_x1=10,
            bbox_y1=7,
        )
        assert variant.bbox_wh == (8, 4)


class TestIsClutterAsset:
    @pytest.mark.parametrize(
        ("path", "expected"),
        [
            ("world/generic/oak01.m2", False),
            ("world/trees/oak01.m2", True),
            ("world/grass/grass01.m2", True),
            ("world/wmo/azeroth/stormwind.wmo", False),
            ("", False),
        ],
    )
    def test_detects_clutter_tokens(self, path: str, expected: bool) -> None:
        assert is_clutter_asset(path) is expected
