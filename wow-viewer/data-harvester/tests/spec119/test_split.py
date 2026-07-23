"""Spec 119 split tests (T008): family isolation, leakage refusal, determinism."""

from __future__ import annotations

import json

import pytest

from harvester.spec119.object_library_contract import derive_asset_family
from harvester.spec119.split import (
    SPLIT_SCHEMA,
    SplitError,
    apply_family_split,
    build_family_split,
    leakage_check,
    load_split,
)


def _row(path: str) -> dict:
    return {"normalized_asset_path": path, "capture_status": "captured"}


def _fixture_rows() -> list[dict]:
    rows = []
    # 6 families x 4 assets; castle family has numbered near-duplicate variants.
    for family in ("castle", "chapel", "tower", "barn", "mill", "keep"):
        for n in range(1, 5):
            rows.append(_row(f"world/wmo/azeroth/buildings/{family}/{family}0{n}.wmo"))
    return rows


def test_split_isolates_by_family() -> None:
    rows = _fixture_rows()
    split = build_family_split(rows, held_out_fraction=0.25, seed=0)
    train_families = set(split["train_families"])
    held_out_families = set(split["held_out_families"])
    assert not train_families & held_out_families  # no family in both halves
    train_idx, held_out_idx = apply_family_split(rows, split)
    assert len(train_idx) == split["train_row_count"]
    assert len(held_out_idx) == split["held_out_row_count"]
    assert len(train_idx) + len(held_out_idx) == len(rows)
    # Every held-out row's family is in the held-out family list.
    for index in held_out_idx:
        assert derive_asset_family(rows[index]["normalized_asset_path"]) in held_out_families
    assert split["verified_violation_count"] == 0
    assert split["schema"] == SPLIT_SCHEMA


def test_leakage_check_flags_synthetic_leaky_fixture() -> None:
    rows = [
        _row("world/wmo/azeroth/buildings/castle/castle01.wmo"),
        _row("world/wmo/azeroth/buildings/castle/castle02.wmo"),
        _row("world/wmo/azeroth/buildings/chapel/chapel.wmo"),
    ]
    # castle01 in train, castle02 in held-out -> one straddling variant stem.
    assert leakage_check(rows, [0, 2], [1]) == 1
    # Both variants together on one side -> no violation.
    assert leakage_check(rows, [0, 1], [2]) == 0


def test_load_split_refuses_leaky_document(tmp_path) -> None:
    doc = {
        "schema": SPLIT_SCHEMA,
        "seed": 0,
        "held_out_fraction": 0.2,
        "train_families": ["a"],
        "held_out_families": ["b"],
        "train_row_count": 1,
        "held_out_row_count": 1,
        "verified_violation_count": 2,
    }
    path = tmp_path / "split.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(SplitError, match="verified_violation_count"):
        load_split(path)


def test_split_deterministic_from_seed() -> None:
    rows = _fixture_rows()
    first = build_family_split(rows, held_out_fraction=0.25, seed=42)
    second = build_family_split(rows, held_out_fraction=0.25, seed=42)
    assert first == second
    third = build_family_split(rows, held_out_fraction=0.25, seed=43)
    # A different seed may (and here does) change which families are held out.
    assert first["held_out_families"] != third["held_out_families"]


def test_split_row_count_accounting_and_refusals() -> None:
    rows = _fixture_rows()
    split = build_family_split(rows, held_out_fraction=0.2, seed=1)
    assert split["train_row_count"] + split["held_out_row_count"] == len(rows)
    assert split["held_out_row_count"] >= 1
    with pytest.raises(SplitError, match="at least 2"):
        build_family_split([_row("world/wmo/a/one.wmo"), _row("world/wmo/a/two.wmo")])
    with pytest.raises(SplitError, match="no rows"):
        build_family_split([])
    with pytest.raises(SplitError, match="held_out_fraction"):
        build_family_split(rows, held_out_fraction=0.0)
