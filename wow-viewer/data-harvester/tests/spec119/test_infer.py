"""Spec 119 inference tests (T024): loose-PNG inference, no store present."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from harvester.spec119 import infer
from harvester.spec119.classifier_model import ObjectClassifier
from harvester.spec119.object_library_contract import ObjectLibraryContractError
from harvester.spec119.segmenter_model import ObjectSegmenter


def _save_classifier_checkpoint(path: Path, base: int = 8) -> None:
    torch.manual_seed(0)
    model = ObjectClassifier(base=base, num_classes=4)
    torch.save(
        {
            "kind": "classifier",
            "state_dict": model.state_dict(),
            "architecture": {"base": base, "num_classes": 4,
                             "class_index": {"empty": 0, "m2": 1, "mdx": 2, "wmo": 3}},
            "config": {},
            "epoch": 1,
        },
        path,
    )


def _save_segmenter_checkpoint(path: Path, base: int = 8) -> None:
    torch.manual_seed(0)
    model = ObjectSegmenter(base=base)
    torch.save(
        {"kind": "segmenter", "state_dict": model.state_dict(),
         "architecture": {"base": base}, "config": {}, "epoch": 1},
        path,
    )


def _write_png(path: Path, size: int = 128) -> None:
    from PIL import Image

    rng = np.random.default_rng(0)
    Image.fromarray((rng.random((size, size, 3)) * 255).astype(np.uint8)).save(path)


def test_classifier_json_shape_and_no_store_needed(tmp_path, monkeypatch, capsys) -> None:
    checkpoint = tmp_path / "classifier.pt"
    _save_classifier_checkpoint(checkpoint)
    png = tmp_path / "loose.png"
    _write_png(png)
    output = tmp_path / "predictions.json"
    monkeypatch.setattr(
        "sys.argv",
        ["spec119_infer.py", "--checkpoint", str(checkpoint),
         "--inputs", str(png), "--output", str(output)],
    )
    assert infer.main() == 0
    results = json.loads(output.read_text(encoding="utf-8"))
    assert len(results) == 1
    entry = results[0]
    assert entry["input"] == str(png)
    assert entry["predicted_class"] in {"empty", "m2", "mdx", "wmo"}
    assert 0.0 <= entry["confidence"] <= 1.0
    assert set(entry["per_class_probs"]) == {"empty", "m2", "mdx", "wmo"}
    assert sum(entry["per_class_probs"].values()) == pytest.approx(1.0)


def test_segmenter_mask_png_write(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "segmenter.pt"
    _save_segmenter_checkpoint(checkpoint)
    png = tmp_path / "loose.png"
    _write_png(png)
    out_dir = tmp_path / "masks"
    monkeypatch.setattr(
        "sys.argv",
        ["spec119_infer.py", "--checkpoint", str(checkpoint),
         "--inputs", str(png), "--output", str(out_dir)],
    )
    assert infer.main() == 0
    from PIL import Image

    mask = np.asarray(Image.open(out_dir / "loose_mask.png"))
    assert mask.shape == (128, 128)
    assert set(np.unique(mask).tolist()) <= {0, 255}


def test_refuses_checkpoint_missing_class_index(tmp_path) -> None:
    bad = tmp_path / "bad.pt"
    torch.save({"kind": "classifier", "state_dict": {}, "architecture": {}}, bad)
    with pytest.raises(ObjectLibraryContractError, match="missing class_index"):
        infer.load_classifier_checkpoint(bad)


def test_refuses_wrong_kind(tmp_path) -> None:
    seg = tmp_path / "segmenter.pt"
    _save_segmenter_checkpoint(seg)
    with pytest.raises(ObjectLibraryContractError, match="not a classifier"):
        infer.load_classifier_checkpoint(seg)
