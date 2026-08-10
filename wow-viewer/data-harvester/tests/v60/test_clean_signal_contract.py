from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from harvester.v60.clean_signal_corpus import (
    COARSE_SIGNAL,
    CORPUS_SCHEMA,
    DETAIL_SIGNAL,
    HEIGHT_SIGNAL,
    LUMA_SIGNAL,
    RELATIVE_HEIGHT_SIGNAL,
    validate_clean_signal_corpus,
)
from harvester.v60.clean_signal_inputs import build_clean_observation
from harvester.v60.clean_signal_targets import decompose_relative_height


def _hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array, dtype="<f4").tobytes()).hexdigest()


def _write_valid_corpus(root: Path, *, row_overrides: dict | None = None) -> None:
    y, x = np.mgrid[0:256, 0:256]
    luma = ((x + y) / 510.0).astype(np.float32)
    package = build_clean_observation(luma, np.ones((256, 256), dtype=np.float32))
    target = decompose_relative_height(np.sin(np.mgrid[0:257, 0:257][0] / 19.0).astype(np.float32))
    arrays = {
        **package.arrays(),
        HEIGHT_SIGNAL: target.height_257,
        RELATIVE_HEIGHT_SIGNAL: target.relative_height_257,
        COARSE_SIGNAL: target.coarse_relief_257,
        DETAIL_SIGNAL: target.detail_residual_257,
    }
    np.savez(root / "row-00.npz", **arrays)
    row = {
        "row_id": "row-00",
        "source_kind": "synthetic_control",
        "source_group_id": "group-00",
        "family": "smooth_relief",
        "split": "train",
        "npz": "row-00.npz",
        "confidence_status": "measured",
        "observation_status": "accepted",
        "observation_provenance": {"operation": "synthetic-albedo-v1", "artifact_status": "fresh"},
        "forbidden_signals": [],
        "array_hashes": {name: _hash(array) for name, array in arrays.items()},
    }
    row.update(row_overrides or {})
    manifest = {
        "schema": CORPUS_SCHEMA,
        "row_count": 1,
        "split_mode": "complete_family",
        "required_families": ["smooth_relief"],
        "forbidden_signals_seen": [],
        "rows": [row],
    }
    (root / "clean_signal_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_clean_signal_corpus_accepts_valid_hashes_and_recomposition(tmp_path: Path) -> None:
    _write_valid_corpus(tmp_path)
    report = validate_clean_signal_corpus(tmp_path)
    assert report["valid"] is True
    assert report["split_counts"] == {"train": 1}
    assert report["family_counts"] == {"smooth_relief": 1}


def test_clean_signal_corpus_rejects_stale_hash_and_forbidden_array(tmp_path: Path) -> None:
    _write_valid_corpus(
        tmp_path,
        row_overrides={"array_hashes": {LUMA_SIGNAL: "0" * 64}, "forbidden_signals": ["height_257"]},
    )
    report = validate_clean_signal_corpus(tmp_path)
    assert report["valid"] is False
    assert any("hash mismatch" in failure for failure in report["failures"])
    assert any("forbidden" in failure for failure in report["failures"])


def test_clean_signal_corpus_rejects_group_leakage(tmp_path: Path) -> None:
    _write_valid_corpus(tmp_path)
    manifest_path = tmp_path / "clean_signal_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["rows"].append({**manifest["rows"][0], "row_id": "row-01", "split": "validation"})
    manifest["row_count"] = 2
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    report = validate_clean_signal_corpus(tmp_path)
    assert report["valid"] is False
    assert any("source_group_id" in failure and "crosses" in failure for failure in report["failures"])
