"""Spec 112: synthesis-sourced signals must never be filled from the harvest stream — the v22
stream carries the AUTHORED client minimap under the same `minimap_rgb` key, and accepting it
wherever synthesis skipped is exactly how authored imagery leaked into the clean-room store
labeled `freshly_extracted` (the real mechanism behind the 256/1024 parity gap)."""

from __future__ import annotations

import sys
from pathlib import Path

from harvester.v50.manifest_template import build_manifest_template
from harvester.v50.signal_catalog import parse_catalog_table

_REAL_DOC = Path(__file__).parents[3] / "docs" / "architecture" / "v50-clean-room-dataset-repo-audit-2026-07-15.md"

_SCRIPTS = Path(__file__).parents[2] / "scripts"
sys.path.insert(0, str(_SCRIPTS))
from v50_build_dataset import signal_takes_stream_data  # noqa: E402


def test_synthesis_sourced_signals_refuse_stream_data():
    manifest = build_manifest_template(
        parse_catalog_table(_REAL_DOC), build_id="0_5_3_3368", release="v50.1"
    )
    by_name = {s.name: s for s in manifest.signals}

    assert signal_takes_stream_data(by_name["minimap_rgb"]) is False
    assert signal_takes_stream_data(by_name["minimap_rgb_1024"]) is False
    # Stream-sourced signals are unaffected.
    assert signal_takes_stream_data(by_name["height_257"]) is True
    assert signal_takes_stream_data(by_name["mcnk_flags_16"]) is True
