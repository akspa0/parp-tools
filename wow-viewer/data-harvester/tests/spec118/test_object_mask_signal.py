"""Spec 118 T012 (US1): the three strict visible-object arrays are real, correctly-shaped
catalog entries.

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
    "object_geometry_visible_mask_257": ("float32", (257, 257)),
    "object_geometry_visible_source_257": ("uint8", (257, 257)),
    "object_geometry_visible_instance_257": ("int32", (257, 257)),
}


def test_visible_object_signals_are_in_the_frozen_catalog():
    signals = {s.name: s for s in parse_catalog_table(_REAL_DOC)}
    for name, (dtype, shape) in _EXPECTED.items():
        assert name in signals, f"{name} missing from the frozen catalog"
        entry = signals[name]
        assert entry.dtype == dtype, f"{name}: expected dtype {dtype!r}, got {entry.dtype!r}"
        assert entry.shape == shape, f"{name}: expected shape {shape}, got {entry.shape}"
        assert entry.policy == "copy-if-verified", f"{name}: expected copy-if-verified, got {entry.policy!r}"
        assert entry.required is False, f"{name}: expected not required"
        assert entry.available_for_build("0_5_3_3368") is True


def test_v18_footprint_masks_are_cataloged_for_the_alpha_path():
    # Re-cataloged 2026-07-22: the strict object_geometry_visible_* signals only populate via
    # AdtTensorPackBuilder; the 0.5.3 alpha harvest (AlphaTensorPackBuilder) paints these footprint
    # masks from placements instead, so they are the ones that carry real data on this corpus.
    signals = {s.name: s for s in parse_catalog_table(_REAL_DOC)}
    for name, dtype in (("object_mask", "float32"), ("object_precise_mask", "float32"),
                        ("object_instance_mask", "int32")):
        assert name in signals, f"{name} missing from the catalog"
        assert signals[name].dtype == dtype
        assert signals[name].shape == (257, 257)
