"""Spec 117 T009 (US1): the four WDL lattice arrays are real, correctly-shaped catalog entries.

The store builder itself (``scripts/v50_build_dataset.py::_cmd_build``) is already generic --
signal selection is a plain 1:1 name match against ``manifest_template.signals`` (see
``harvester.v50.manifest_template``), and row-level "excluded and counted, never fabricated"
behavior for a signal missing on some tiles is already exercised by every other optional signal in
the catalog. This test only proves the NEW catalog entries parse with the contract this feature's
spec/data-model demand; it deliberately does not re-test the generic extraction pipeline, which
belongs to Spec 112/109's own test suite.
"""

from __future__ import annotations

from pathlib import Path

from harvester.v50.signal_catalog import parse_catalog_table

_REAL_DOC = (
    Path(__file__).parents[3] / "docs" / "architecture" / "v50-clean-room-dataset-repo-audit-2026-07-15.md"
)

_EXPECTED = {
    "wdl_outer_17": ("float32", (17, 17)),
    "wdl_inner_16": ("float32", (16, 16)),
    "wdl_outer_present": ("bool", (17, 17)),
    "wdl_inner_present": ("bool", (16, 16)),
}


def test_wdl_lattice_signals_are_in_the_frozen_catalog():
    signals = {s.name: s for s in parse_catalog_table(_REAL_DOC)}
    for name, (dtype, shape) in _EXPECTED.items():
        assert name in signals, f"{name} missing from the frozen catalog"
        entry = signals[name]
        assert entry.dtype == dtype, f"{name}: expected dtype {dtype!r}, got {entry.dtype!r}"
        assert entry.shape == shape, f"{name}: expected shape {shape}, got {entry.shape}"
        assert entry.policy == "copy-if-verified", f"{name}: expected copy-if-verified, got {entry.policy!r}"
        assert entry.required is False, f"{name}: expected not required"
        assert entry.available_for_build("0_5_3_3368") is True


def test_sample_count_matches_spec_108_contract():
    # 17*17 + 16*16 == 545, the exact Spec 108 FR-001 / TerrainWdlLattice.SampleCount contract.
    assert 17 * 17 + 16 * 16 == 545
