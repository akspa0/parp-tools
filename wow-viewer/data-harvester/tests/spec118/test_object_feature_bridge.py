"""Spec 118 T026 (US3): the bridge writes exactly the shape/schema/attrs the existing (unmodified)
geometry trainers already validate for ``--feature-store``, never mutates the source store, keeps
channels on the probability simplex, and refuses to clobber an existing output directory."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import zarr

from harvester.spec118.object_contract import architecture_identity
from harvester.spec118.object_feature_bridge import (
    CLASS_COUNT,
    FEATURE_ARRAY,
    FEATURE_STORE_SCHEMA,
    ObjectBridgeError,
    objects_to_feature_map,
)
from harvester.spec118.object_segment_model import ObjectSegmentNet


def _build_store(tmp_path: Path, *, rows: int = 3) -> Path:
    store = tmp_path / "store.zarr"
    group = zarr.open_group(str(store), mode="w")
    rng = np.random.default_rng(0)
    group.create_array("minimap_rgb", data=(rng.random((rows, 256, 256, 3)) * 255).astype(np.uint8))
    index_rows = [{"map": "Kalimdor", "tile_x": i, "tile_y": 0} for i in range(rows)]
    pq.write_table(pa.Table.from_pylist(index_rows), store / "index.parquet")
    return store


def _build_checkpoint(tmp_path: Path, *, base: int = 8) -> Path:
    model = ObjectSegmentNet(base=base)
    identity = architecture_identity(model, architecture_id="object_segment_net", config={"base": base})
    checkpoint = {
        "architecture": identity,
        "object_config": {"base": base},
        "model": model.state_dict(),
        "epoch": 1,
    }
    path = tmp_path / "checkpoint_best.pt"
    torch.save(checkpoint, path)
    return path


def test_dry_run_returns_plan_and_writes_nothing(tmp_path: Path):
    store = _build_store(tmp_path)
    checkpoint = _build_checkpoint(tmp_path)
    output = tmp_path / "feature_store"

    plan = objects_to_feature_map(store=store, checkpoint=checkpoint, output=output, write=False)
    assert plan["schema"] == "v118-object-bridge-plan-v1"
    assert plan["output_array"]["shape"] == [3, CLASS_COUNT, 256, 256]
    assert plan["channels"] == ["object_softmax"]
    assert not output.exists()


def test_write_produces_the_exact_contract_the_trainers_already_validate(tmp_path: Path):
    store = _build_store(tmp_path)
    checkpoint = _build_checkpoint(tmp_path)
    output = tmp_path / "feature_store"
    original_index_bytes = (store / "index.parquet").read_bytes()

    result = objects_to_feature_map(store=store, checkpoint=checkpoint, output=output, write=True)
    assert result["schema"] == FEATURE_STORE_SCHEMA
    assert output.exists()

    group = zarr.open_group(str(output), mode="r")
    assert dict(group.attrs)["schema"] == FEATURE_STORE_SCHEMA
    assert dict(group.attrs)["class_count"] == 1
    assert dict(group.attrs)["source_signal"] == "object_mask"
    assert dict(group.attrs)["checkpoint_sha256"]
    feature = np.asarray(group[FEATURE_ARRAY])
    assert feature.shape == (3, 1, 256, 256)
    assert np.all(feature >= 0.0) and np.all(feature <= 1.0)
    # The single object-probability channel is 1 - none, a valid per-pixel probability.
    assert np.all(feature.sum(axis=1) <= 1.0 + 1e-2)

    derived_index = pq.read_table(output / "index.parquet").to_pylist()
    assert len(derived_index) == 3
    assert derived_index[0]["source_row_index"] == 0

    # Source store must be byte-for-byte untouched.
    assert (store / "index.parquet").read_bytes() == original_index_bytes


def test_refuses_to_overwrite_a_nonempty_output(tmp_path: Path):
    store = _build_store(tmp_path)
    checkpoint = _build_checkpoint(tmp_path)
    output = tmp_path / "feature_store"
    objects_to_feature_map(store=store, checkpoint=checkpoint, output=output, write=True)

    with pytest.raises(ObjectBridgeError, match="non-empty"):
        objects_to_feature_map(store=store, checkpoint=checkpoint, output=output, write=True)


def test_refuses_a_store_without_minimap_rgb(tmp_path: Path):
    store = tmp_path / "empty_store.zarr"
    zarr.open_group(str(store), mode="w")
    pq.write_table(pa.Table.from_pylist([]), store / "index.parquet")
    checkpoint = _build_checkpoint(tmp_path)

    with pytest.raises(ObjectBridgeError, match="minimap_rgb"):
        objects_to_feature_map(store=store, checkpoint=checkpoint, output=tmp_path / "out", write=False)
