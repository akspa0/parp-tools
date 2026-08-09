from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from harvester.v60.control_corpus import CONTROL_FAMILY_BUCKETS, validate_control_corpus


def _hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array, dtype="<f4").tobytes()).hexdigest()


def _write_row(root: Path, row_id: str, family: str, split: str, value: float) -> dict:
    shadow = np.full((256, 256), value, dtype=np.float32)
    height = np.full((257, 257), value * 100.0, dtype=np.float32)
    normals = np.zeros((257, 257, 3), dtype=np.float32)
    normals[:, :, 2] = 1.0
    name = f"{row_id}.npz"
    np.savez(root / name, terrain_shadow_256=shadow, height_257=height, mcnr_normal_xyz=normals)
    return {
        "row_id": row_id,
        "control_family": family,
        "complexity_bucket": CONTROL_FAMILY_BUCKETS.get(family, "hard"),
        "split": split,
        "npz": name,
        "input_sha256": _hash(shadow),
        "target_sha256": _hash(height),
    }


def _write_manifest(root: Path, rows: list[dict]) -> None:
    (root / "control_manifest.json").write_text(
        json.dumps({"schema": "v60-control-corpus-v1", "row_count": len(rows), "rows": rows}),
        encoding="utf-8",
    )


def test_control_corpus_validates_exact_pairs_and_family_split(tmp_path: Path) -> None:
    rows = [
        _write_row(tmp_path, "ridge-v00", "ridge", "train", 0.4),
        _write_row(tmp_path, "ridge-v01", "ridge", "train", 0.5),
        _write_row(tmp_path, "noise-v00", "noise", "validation", 0.6),
    ]
    _write_manifest(tmp_path, rows)

    report = validate_control_corpus(tmp_path)

    assert report["valid"] is True
    assert report["split_counts"] == {"train": 2, "validation": 1}
    assert report["family_splits"] == {"noise": "validation", "ridge": "train"}
    assert report["complexity_bucket_counts"] == {"hard": 2, "pathological": 1}


def test_control_corpus_rejects_hash_and_family_leak(tmp_path: Path) -> None:
    row = _write_row(tmp_path, "ridge-v00", "ridge", "train", 0.4)
    row["input_sha256"] = "0" * 64
    rows = [row, {**row, "row_id": "ridge-v01", "split": "validation", "npz": "ridge-v00.npz"}]
    _write_manifest(tmp_path, rows)

    report = validate_control_corpus(tmp_path)

    assert report["valid"] is False
    assert any("hash mismatch" in failure for failure in report["failures"])
    assert any("crosses" in failure for failure in report["failures"])


def test_control_corpus_rejects_wrong_complexity_bucket(tmp_path: Path) -> None:
    row = _write_row(tmp_path, "island-v00", "island_sea", "train", 0.4)
    row["complexity_bucket"] = "easy"
    _write_manifest(tmp_path, [row])

    report = validate_control_corpus(tmp_path)

    assert report["valid"] is False
    assert any("expected 'hard'" in failure for failure in report["failures"])


def test_control_corpus_requires_complete_cross_tile_pattern(tmp_path: Path) -> None:
    rows = []
    for variant, (tile_x, tile_y) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
        row = _write_row(tmp_path, f"cross-v{variant:02d}", "cross_tile_lightning", "train", 0.4)
        row.update(
            {
                "pattern_id": "cross_tile_lightning-pattern-00",
                "pattern_tile_x": tile_x,
                "pattern_tile_y": tile_y,
                "pattern_tile_span": 2,
                "pattern_continuity": "continuous_global_2x2",
            }
        )
        rows.append(row)
    _write_manifest(tmp_path, rows)

    report = validate_control_corpus(tmp_path)

    assert report["valid"] is True
    assert report["cross_tile_positions"]["cross_tile_lightning"] == [[0, 0], [0, 1], [1, 0], [1, 1]]

    _write_manifest(tmp_path, rows[:-1])
    incomplete = validate_control_corpus(tmp_path)
    assert incomplete["valid"] is False
    assert any("missing positions" in failure for failure in incomplete["failures"])
