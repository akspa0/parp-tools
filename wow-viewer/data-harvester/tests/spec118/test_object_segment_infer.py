"""Spec 118 T024 (US3): two-mode inference -- loose OOD images with no store, store batch with
ground-truth scoring, mutual exclusivity, and checkpoint-mismatch refusal."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import zarr
from PIL import Image

from harvester.spec118.object_contract import architecture_identity
from harvester.spec118.object_segment_infer import (
    ObjectInferError,
    infer_loose_inputs,
    infer_store,
    main,
)
from harvester.spec118.object_segment_model import ObjectSegmentNet


def _checkpoint(tmp_path: Path, *, base: int = 8) -> Path:
    model = ObjectSegmentNet(base=base)
    checkpoint = {
        "architecture": architecture_identity(model, architecture_id="object_segment_net", config={"base": base}),
        "object_config": {"base": base},
        "model": model.state_dict(),
        "epoch": 1,
    }
    path = tmp_path / "checkpoint_best.pt"
    torch.save(checkpoint, path)
    return path


def _loose_tile(tmp_path: Path, name: str = "tile.png") -> Path:
    rng = np.random.default_rng(0)
    tile = tmp_path / name
    Image.fromarray((rng.random((256, 256, 3)) * 255).astype(np.uint8)).save(tile)
    return tile


def test_loose_mode_runs_without_any_store(tmp_path: Path):
    checkpoint = _checkpoint(tmp_path)
    tile = _loose_tile(tmp_path)
    output = tmp_path / "ood"

    audit = infer_loose_inputs(checkpoint=checkpoint, inputs=[tile], output=output, write=True)

    assert audit["schema"] == "v118-object-infer-v1"
    assert audit["mode"] == "loose_inputs"
    assert audit["ground_truth"] == "unavailable"
    assert len(audit["tiles"]) == 1
    record = audit["tiles"][0]
    assert record["ground_truth"] == "unavailable"
    assert 0.0 <= record["marked_fraction"] <= 1.0
    assert (output / record["class_png"]).exists()
    persisted = json.loads((output / "object_infer_audit.json").read_text(encoding="utf-8"))
    assert persisted["checkpoint"]["sha256"] == audit["checkpoint"]["sha256"]


def test_loose_mode_dry_run_writes_nothing(tmp_path: Path):
    checkpoint = _checkpoint(tmp_path)
    tile = _loose_tile(tmp_path)
    output = tmp_path / "ood"

    plan = infer_loose_inputs(checkpoint=checkpoint, inputs=[tile], output=output, write=False)
    assert plan["tile_count"] == 1
    assert not output.exists()


def test_store_mode_scores_against_ground_truth(tmp_path: Path):
    checkpoint = _checkpoint(tmp_path)
    store = tmp_path / "store.zarr"
    group = zarr.open_group(str(store), mode="w")
    rng = np.random.default_rng(1)
    group.create_array("minimap_rgb", data=(rng.random((2, 256, 256, 3)) * 255).astype(np.uint8))
    source = np.zeros((2, 257, 257), dtype=np.uint8)
    source[0, :8, :8] = 2  # one building region on tile 0
    group.create_array("object_geometry_visible_source_257", data=source)
    pq.write_table(
        pa.Table.from_pylist([{"map": "Kalimdor", "tile_x": i, "tile_y": 0} for i in range(2)]),
        store / "index.parquet",
    )
    dumps = tmp_path / "dumps"

    audit = infer_store(checkpoint=checkpoint, store=store, dumps=dumps, write=True)

    assert audit["mode"] == "store"
    assert audit["ground_truth"] == "strict_visible_object_source"
    assert len(audit["tiles"]) == 2
    for record in audit["tiles"]:
        assert record["ground_truth"] == "strict_visible_object_source"
        assert "per_class" in record
        assert "visible_object_iou" in record
        assert (dumps / record["class_png"]).exists()


def test_checkpoint_without_object_config_is_refused(tmp_path: Path):
    model = ObjectSegmentNet(base=8)
    path = tmp_path / "old_checkpoint.pt"
    torch.save({"model": model.state_dict()}, path)  # no object_config
    tile = _loose_tile(tmp_path)

    with pytest.raises(ObjectInferError, match="object_config"):
        infer_loose_inputs(checkpoint=path, inputs=[tile], output=tmp_path / "out", write=True)


def test_base_mismatch_load_is_refused(tmp_path: Path):
    wrong = ObjectSegmentNet(base=16)
    path = tmp_path / "mismatch.pt"
    torch.save({"object_config": {"base": 8}, "model": wrong.state_dict()}, path)
    tile = _loose_tile(tmp_path)

    with pytest.raises(ObjectInferError, match="does not fit"):
        infer_loose_inputs(checkpoint=path, inputs=[tile], output=tmp_path / "out", write=True)


def test_modes_are_mutually_exclusive(tmp_path: Path, capsys):
    checkpoint = _checkpoint(tmp_path)
    code = main([
        "--checkpoint", str(checkpoint),
        "--inputs", str(tmp_path),
        "--store", str(tmp_path / "store.zarr"),
    ])
    assert code == 2
    assert "exactly one" in capsys.readouterr().out
