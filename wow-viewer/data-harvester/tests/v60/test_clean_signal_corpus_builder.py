from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from harvester.v60.clean_signal_corpus import (
    CORPUS_SCHEMA,
    build_clean_signal_corpus,
    clean_signal_build_plan,
    validate_clean_signal_corpus,
)
from harvester.v60.control_corpus import CONTROL_FAMILY_BUCKETS


def _hash(array: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(np.ascontiguousarray(array, dtype="<f4").tobytes()).hexdigest()


def _write_control_corpus(root: Path) -> None:
    rows = []
    for index, (family, split) in enumerate((("flat", "train"), ("ridge", "validation"))):
        shadow = np.full((256, 256), 0.2 + index * 0.3, dtype=np.float32)
        y, x = np.mgrid[0:257, 0:257]
        height = (x + y + index).astype(np.float32)
        name = f"{family}-{index:02d}.npz"
        np.savez(root / name, terrain_shadow_256=shadow, height_257=height)
        rows.append(
            {
                "row_id": f"{family}-v{index:02d}",
                "control_family": family,
                "complexity_bucket": CONTROL_FAMILY_BUCKETS[family],
                "source_group_id": f"group-{index}",
                "variant": index,
                "split": split,
                "npz": name,
                "input_sha256": _hash(shadow),
                "target_sha256": _hash(height),
            }
        )
    (root / "control_manifest.json").write_text(
        json.dumps(
            {
                "schema": "v60-control-corpus-v1",
                "row_count": len(rows),
                "rows": rows,
            }
        ),
        encoding="utf-8",
    )


def test_clean_signal_build_plan_is_no_write_and_complete(tmp_path: Path) -> None:
    source = tmp_path / "control"
    source.mkdir()
    _write_control_corpus(source)
    output = tmp_path / "clean"
    plan = clean_signal_build_plan(source)
    assert plan["dry_run"] is True
    assert plan["row_count"] == 2
    assert plan["families"] == ["flat", "ridge"]
    assert not output.exists()


def test_clean_signal_builder_publishes_valid_hashed_corpus(tmp_path: Path) -> None:
    source = tmp_path / "control"
    source.mkdir()
    _write_control_corpus(source)
    output = tmp_path / "clean"
    result = build_clean_signal_corpus(source, output)

    assert result["dry_run"] is False
    assert result["row_count"] == 2
    assert (output / "clean_signal_manifest.json").is_file()
    assert not (tmp_path / "clean.partial").exists()
    report = validate_clean_signal_corpus(output)
    assert report["valid"] is True
    assert report["manifest_schema"] == CORPUS_SCHEMA
    assert report["split_counts"] == {"train": 1, "validation": 1}


def test_clean_signal_builder_refuses_overwrite_and_bad_confidence(tmp_path: Path) -> None:
    source = tmp_path / "control"
    source.mkdir()
    _write_control_corpus(source)
    with pytest.raises(ValueError, match="confidence_value"):
        clean_signal_build_plan(source, confidence_value=2.0)
    output = tmp_path / "clean"
    output.mkdir()
    with pytest.raises(FileExistsError, match="overwrite"):
        build_clean_signal_corpus(source, output)
