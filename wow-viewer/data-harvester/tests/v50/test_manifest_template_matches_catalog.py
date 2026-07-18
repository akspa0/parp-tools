"""Spec 112 T009: the manifest template must be mechanically derived from the frozen catalog —
catalog-dropped signals cannot appear, era-impossible signals become explicit unavailability
records, and the output must load through the real `_load_manifest` path, not merely look right."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from harvester.v50.contracts import REASON_ERA_UNAVAILABLE, classify_reason
from harvester.v50.manifest_template import build_manifest_template, build_signals_config
from harvester.v50.signal_catalog import parse_catalog_table

_REAL_DOC = Path(__file__).parents[3] / "docs" / "architecture" / "v50-clean-room-dataset-repo-audit-2026-07-15.md"


def _catalog():
    return parse_catalog_table(_REAL_DOC)


def test_template_for_053_omits_dropped_and_era_impossible_signals():
    manifest = build_manifest_template(_catalog(), build_id="0_5_3_3368", release="v50.1")
    declared = {s.name for s in manifest.signals}

    for dropped in ("mddf_mask", "modf_mask", "object_filtered_mask", "model_focus_mask",
                    "object_roof_mask", "object_roof_confidence", "model_above_terrain_mask",
                    "object_precise_mask", "object_instance_mask"):
        assert dropped not in declared, f"catalog-dropped signal {dropped!r} leaked into the template"

    assert "mccv_rgb" not in declared
    era_records = {u.name: u.reason for u in manifest.unavailable_signals}
    assert "mccv_rgb" in era_records
    assert classify_reason(era_records["mccv_rgb"]) == REASON_ERA_UNAVAILABLE

    # blacklisted signals are excluded entirely, matching the prior template's holes_16 handling
    assert "holes_16" not in declared
    assert "holes_16" not in era_records

    # core signals present with catalog-declared properties
    by_name = {s.name: s for s in manifest.signals}
    assert by_name["height_257"].required and by_name["height_257"].row_shape == (257, 257)
    assert by_name["minimap_rgb"].authoritative_source == "synthetic-minimap"
    assert by_name["mcnk_flags_16"].authoritative_source == "harvest-stream"


def test_template_for_wotlk_era_build_declares_mccv():
    manifest = build_manifest_template(_catalog(), build_id="3_3_5_12340", release="v50.1")
    assert "mccv_rgb" in {s.name for s in manifest.signals}
    assert "mccv_rgb" not in {u.name for u in manifest.unavailable_signals}


def test_template_round_trips_through_the_real_loader(tmp_path: Path):
    manifest = build_manifest_template(_catalog(), build_id="0_5_3_3368", release="v50.1")
    path = tmp_path / "template.json"
    path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")

    scripts_dir = Path(__file__).parents[2] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        from v50_build_dataset import _load_manifest  # the real consumer, not a reimplementation
        loaded = _load_manifest(path)
    finally:
        sys.path.remove(str(scripts_dir))

    assert loaded.build_id == "0_5_3_3368"
    assert {s.name for s in loaded.signals} == {s.name for s in manifest.signals}
    assert loaded.row_count == 0


def test_generation_is_deterministic():
    first = build_manifest_template(_catalog(), build_id="0_5_3_3368", release="v50.1")
    second = build_manifest_template(_catalog(), build_id="0_5_3_3368", release="v50.1")
    assert first.to_dict() == second.to_dict()


def test_signals_config_carries_blacklist_through():
    config = build_signals_config(_catalog())
    by_name = {entry["name"]: entry for entry in config["signals"]}
    assert by_name["holes_16"]["blacklisted"] is True
    assert by_name["holes_16"]["blacklist_reason"]
    assert by_name["height_257"]["blacklisted"] is False
