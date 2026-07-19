"""Spec 112: synthesis-sourced signals must never be filled from the harvest stream — the v22
stream carries the AUTHORED client minimap under the same `minimap_rgb` key, and accepting it
wherever synthesis skipped is exactly how authored imagery leaked into the clean-room store
labeled `freshly_extracted` (the real mechanism behind the 256/1024 parity gap)."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

from harvester.v50.contracts import RowLineage
from harvester.v50.manifest_template import build_manifest_template
from harvester.v50.signal_catalog import parse_catalog_table

_REAL_DOC = Path(__file__).parents[3] / "docs" / "architecture" / "v50-clean-room-dataset-repo-audit-2026-07-15.md"

_SCRIPTS = Path(__file__).parents[2] / "scripts"
sys.path.insert(0, str(_SCRIPTS))
from v50_build_dataset import (  # noqa: E402
    build_synthetic_minimap_command,
    reconcile_manifest_policy,
    signal_takes_stream_data,
)


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


def test_v50_build_uses_material_average_at_256_and_detail_texels_at_1024():
    common = {
        "dll_path": Path("harvest.dll"),
        "client_root": Path("configured-client"),
        "map_name": "Kalimdor",
    }

    command_256 = build_synthetic_minimap_command(
        **common, resolution=256, detail_texels=False
    )
    command_1024 = build_synthetic_minimap_command(
        **common, resolution=1024, detail_texels=True
    )

    assert "--detail" not in command_256
    assert "--detail" in command_1024
    assert command_1024[command_1024.index("--resolution") + 1] == "1024"


def test_reconcile_manifest_policy_keeps_partial_synthesis_honest_without_reharvest():
    policy = build_manifest_template(
        parse_catalog_table(_REAL_DOC), build_id="0_5_3_3368", release="v50.1"
    )
    stale_manifest = replace(
        policy,
        row_count=3,
        signals=tuple(
            replace(signal, required=True, coverage_count=3)
            if signal.name == "minimap_rgb"
            else signal
            for signal in policy.signals
        ),
    )
    lineages = [
        RowLineage(
            store_row=row,
            build_id="0_5_3_3368",
            map_name="Kalimdor",
            tile_x=row,
            tile_y=0,
            source_group="Kalimdor",
            signal_actions={
                "height_257": "freshly_extracted",
                "minimap_rgb": "freshly_extracted" if row < 2 else "unavailable",
            },
        )
        for row in range(3)
    ]

    reconciled = reconcile_manifest_policy(stale_manifest, policy, lineages)
    signals = {signal.name: signal for signal in reconciled.signals}

    assert signals["minimap_rgb"].required is False
    assert signals["minimap_rgb"].coverage_count == 2
    assert signals["height_257"].coverage_count == 3
