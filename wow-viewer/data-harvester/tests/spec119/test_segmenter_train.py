"""Spec 119 segmenter trainer tests (T020): dry-run plan, blank exclusion, refusals."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harvester.spec119 import segmenter_train


def _run(monkeypatch, argv: list[str]) -> int:
    monkeypatch.setattr("sys.argv", ["spec119_train_segmenter.py", *argv])
    return segmenter_train.main()


def _base_argv(store: Path, split: Path, tmp_path: Path) -> list[str]:
    return [
        "--store", str(store),
        "--split", str(split),
        "--output-root", str(tmp_path / "runs"),
        "--run-name", "t020",
    ]


def test_dry_run_plan_shape_and_blank_exclusion(
    monkeypatch, capsys, library_store, library_split, tmp_path
) -> None:
    code = _run(monkeypatch, _base_argv(library_store, library_split, tmp_path))
    assert code == 0
    out = capsys.readouterr().out
    plan = json.loads(out[: out.index("\nDRY RUN ONLY")])
    assert plan["schema"] == "v119-segmenter-plan-v1"
    assert plan["architecture"]["parameter_count"] > 0
    assert plan["split_counts"]["train"] > 0
    assert plan["split_counts"]["held_out"] > 0
    # The fixture's 0.005-coverage rock row is excluded (D-04).
    assert plan["blank_excluded_count"] >= 1
    assert plan["trivial_baselines"]["all_foreground"] > 0.0
    assert plan["trivial_baselines"]["all_background"] == 0.0
    assert "DRY RUN ONLY" in out
    assert not (tmp_path / "runs").exists()


def test_missing_split_refusal(monkeypatch, library_store, tmp_path) -> None:
    with pytest.raises(SystemExit):
        _run(monkeypatch, _base_argv(library_store, tmp_path / "nope.json", tmp_path))


def test_help_argparse(capsys) -> None:
    import sys

    saved = sys.argv
    try:
        sys.argv = ["spec119_train_segmenter.py", "--help"]
        with pytest.raises(SystemExit) as exc:
            segmenter_train.main()
    finally:
        sys.argv = saved
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    for flag in ("--store", "--split", "--output-root", "--run-name", "--base",
                 "--epochs", "--lr", "--pct-start", "--blank-threshold", "--confirm-run"):
        assert flag in help_text
