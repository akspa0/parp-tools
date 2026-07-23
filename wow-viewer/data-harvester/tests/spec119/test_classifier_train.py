"""Spec 119 classifier trainer tests (T014): dry-run plan, refusals, --help."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harvester.spec119 import classifier_train


def _run(monkeypatch, capsys, argv: list[str]) -> int:
    monkeypatch.setattr("sys.argv", ["spec119_train_classifier.py", *argv])
    code = classifier_train.main()
    return code


def _base_argv(store: Path, split: Path, tmp_path: Path) -> list[str]:
    return [
        "--store", str(store),
        "--split", str(split),
        "--output-root", str(tmp_path / "runs"),
        "--run-name", "t014",
    ]


def test_dry_run_plan_shape(monkeypatch, capsys, library_store, library_split, tmp_path) -> None:
    code = _run(monkeypatch, capsys, _base_argv(library_store, library_split, tmp_path))
    assert code == 0
    out = capsys.readouterr().out
    plan = json.loads(out[: out.index("\nDRY RUN ONLY")])
    assert plan["schema"] == "v119-classifier-plan-v1"
    assert plan["architecture"]["parameter_count"] > 0
    assert plan["split_counts"]["train"] > 0
    assert plan["split_counts"]["held_out"] > 0
    assert 0.0 < plan["majority_class_baseline"] < 1.0
    assert len(plan["class_weights"]) == len(plan["class_index"])
    assert plan["blank_relabeled_to_empty"] == 1  # the 0.005-coverage rock row
    assert plan["fine_labels_heuristic"] is False
    assert "DRY RUN ONLY" in out
    # Nothing written without --confirm-run.
    assert not (tmp_path / "runs").exists()


def test_fine_labels_marks_run_heuristic(monkeypatch, capsys, library_store, library_split, tmp_path) -> None:
    code = _run(
        monkeypatch, capsys, [*_base_argv(library_store, library_split, tmp_path), "--fine-labels"]
    )
    assert code == 0
    plan = json.loads(capsys.readouterr().out.split("\nDRY RUN ONLY")[0])
    assert plan["fine_labels_heuristic"] is True
    assert "castle" in plan["class_index"]  # heuristic fine family present


def test_missing_split_refusal(monkeypatch, capsys, library_store, tmp_path) -> None:
    with pytest.raises(SystemExit):
        _run(monkeypatch, capsys, _base_argv(library_store, tmp_path / "nope.json", tmp_path))


def test_missing_store_refusal(monkeypatch, capsys, library_split, tmp_path) -> None:
    with pytest.raises(SystemExit):
        _run(monkeypatch, capsys, _base_argv(tmp_path / "nope.zarr", library_split, tmp_path))


def test_help_argparse(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        import sys

        sys_argv = sys.argv
        try:
            sys.argv = ["spec119_train_classifier.py", "--help"]
            classifier_train.main()
        finally:
            sys.argv = sys_argv
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    for flag in ("--store", "--split", "--output-root", "--run-name", "--base",
                 "--epochs", "--lr", "--pct-start", "--blank-threshold",
                 "--fine-labels", "--confirm-run"):
        assert flag in help_text
