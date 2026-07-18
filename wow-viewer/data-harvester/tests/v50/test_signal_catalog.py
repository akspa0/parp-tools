"""Spec 112 T004: the catalog parser must read the real frozen table exactly and fail closed on
any structural drift — a silently partial parse is the defect class this module exists to end."""

from __future__ import annotations

from pathlib import Path

import pytest

from harvester.v50.contracts import (
    REASON_ERA_UNAVAILABLE,
    REASON_NO_SOURCE_DATA,
    classify_reason,
)
from harvester.v50.signal_catalog import CatalogParseError, parse_catalog_table

_REAL_DOC = Path(__file__).parents[3] / "docs" / "architecture" / "v50-clean-room-dataset-repo-audit-2026-07-15.md"

_FIXTURE_TABLE = """# Some doc

## Frozen Signal Catalog (T002)

intro prose

| Signal | dtype | Shape | V50 Policy | Required | Notes |
|---|---|---|---|---|---|
| `height_257` | float32 | (257,257) | copy-if-verified | **yes** | Core terrain heightmap. |
| `holes_16` | bool | (16,16) | **blacklisted** | no | Uncorrected hole masks. |
| `mccv_rgb` | float32 | (257,257,3) | copy-if-verified | no | MCCV Vertex Colors. |

### After the table
"""


def test_parses_the_fixture_table_with_policies_required_and_shapes(tmp_path: Path):
    doc = tmp_path / "doc.md"
    doc.write_text(_FIXTURE_TABLE, encoding="utf-8")

    signals = {s.name: s for s in parse_catalog_table(doc)}

    assert set(signals) == {"height_257", "holes_16", "mccv_rgb"}
    assert signals["height_257"].required is True
    assert signals["height_257"].shape == (257, 257)
    assert signals["holes_16"].policy == "blacklisted"
    assert signals["holes_16"].required is False
    assert signals["mccv_rgb"].shape == (257, 257, 3)


def test_era_restriction_applies_only_to_the_allow_listed_signal(tmp_path: Path):
    doc = tmp_path / "doc.md"
    doc.write_text(_FIXTURE_TABLE, encoding="utf-8")

    signals = {s.name: s for s in parse_catalog_table(doc)}

    assert signals["mccv_rgb"].available_for_build("0_5_3_3368") is False
    assert signals["mccv_rgb"].available_for_build("3_3_5_12340") is True
    assert signals["mccv_rgb"].era_reason  # human reason recorded, never blank
    assert signals["height_257"].available_for_build("0_5_3_3368") is True


def test_parses_the_real_frozen_catalog_doc():
    signals = {s.name: s for s in parse_catalog_table(_REAL_DOC)}

    # Pinned facts from the frozen table itself — if the doc changes these, the test must too.
    assert "height_257" in signals and signals["height_257"].required
    assert "minimap_rgb" in signals and signals["minimap_rgb"].required
    assert signals["mcnk_flags_16"].shape == (16, 16)
    assert signals["holes_16"].policy == "blacklisted"
    assert signals["liquid_mask"].policy == "fresh-only"
    # The four signals the catalog's Dropped section removed must NOT be in the frozen table.
    for dropped in ("mddf_mask", "modf_mask", "object_filtered_mask", "model_focus_mask"):
        assert dropped not in signals


def test_fails_closed_on_missing_heading_and_changed_header(tmp_path: Path):
    no_heading = tmp_path / "a.md"
    no_heading.write_text("# nothing here\n", encoding="utf-8")
    with pytest.raises(CatalogParseError, match="no '## Frozen Signal Catalog'"):
        parse_catalog_table(no_heading)

    changed_header = tmp_path / "b.md"
    changed_header.write_text(
        "## Frozen Signal Catalog\n\n| Name | Type |\n|---|---|\n| x | y |\n", encoding="utf-8"
    )
    with pytest.raises(CatalogParseError, match="header changed"):
        parse_catalog_table(changed_header)


def test_reason_vocabulary_classification():
    assert classify_reason("era_unavailable: MCCV introduced WotLK+") == REASON_ERA_UNAVAILABLE
    assert classify_reason("no_source_data: tile has no MCLY layers") == REASON_NO_SOURCE_DATA
    assert classify_reason("no rows passed audit as copy-if-verified") is None  # legacy free text
