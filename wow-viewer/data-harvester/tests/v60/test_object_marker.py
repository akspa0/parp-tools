from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import zarr

from harvester.v60.object_marker import (
    FOOTPRINT_SIGNAL,
    IMAGE_SIGNAL,
    KNOWN_SIGNAL,
    MARKER_SIGNAL,
    ObjectMarkerNet,
    build_marker_map,
    build_object_marker_corpus,
    marker_input_tensor,
    marker_loss,
    retrieve_library_identity,
    validate_object_marker_corpus,
)


def _write_sieve(root: Path) -> Path:
    root.mkdir()
    image = np.full((256, 256), 0.55, dtype=np.float32)
    instance = np.zeros((256, 256), dtype=np.uint16)
    instance[80:112, 96:128] = 1
    np.savez_compressed(
        root / "terrain-lib.npz",
        objectified_terrain_shadow_256=image,
        object_instance_id_256=instance,
    )
    manifest = {
        "schema": "v60-object-library-sieve-v1",
        "rows": [
            {
                "row_id": "terrain-lib",
                "npz": "terrain-lib.npz",
                "split": "train",
                "object_instances": [
                    {
                        "instance_id": 1,
                        "library_id": "obj-1",
                        "asset_path": "world/family/object.mdx",
                        "library_family": "world/family",
                    }
                ],
            }
        ],
    }
    (root / "object_library_sieve_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _write_library(root: Path) -> Path:
    root.mkdir()
    group = zarr.open_group(str(root), mode="w")
    group.create_array("capture_rgb", data=np.full((1, 128, 128, 3), 128, dtype=np.uint8))
    mask = np.zeros((1, 128, 128), dtype=np.uint8)
    mask[:, 32:96, 32:96] = 255
    group.create_array("capture_mask", data=mask)
    row = {
        "library_id": "obj-1",
        "normalized_asset_path": "world/family/object.mdx",
        "asset_type": "mdx",
        "capture_status": "captured",
    }
    pq.write_table(pa.Table.from_pylist([row]), root / "assets.parquet")
    pq.write_table(pa.Table.from_pylist([{"library_id": "obj-1"}]), root / "index.parquet")
    return root


def test_marker_model_emits_knownness_and_normalized_embedding() -> None:
    model = ObjectMarkerNet(base=2, embedding_dim=8)
    outputs = model(torch.zeros((2, 4, 256, 256)))
    assert outputs["known_logit"].shape == (2,)
    assert outputs["embedding"].shape == (2, 8)
    assert torch.allclose(outputs["embedding"].norm(dim=1), torch.ones(2), atol=1e-5)


def test_marker_loss_has_metric_term_for_repeated_library_ids() -> None:
    model = ObjectMarkerNet(base=2, embedding_dim=8)
    outputs = model(torch.rand((4, 4, 256, 256)))
    values = marker_loss(
        outputs,
        torch.tensor([1, 1, 1, 0]),
        torch.tensor([0, 0, 1, -1]),
    )
    assert values["total_loss"].item() >= values["known_loss"].item()
    assert values["metric_loss"].item() >= 0.0


def test_retrieval_rejects_low_knownness_and_returns_ranked_ids() -> None:
    result = retrieve_library_identity(
        np.asarray([1.0, 0.0], dtype=np.float32),
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        ["known", "other"],
        known_confidence=0.9,
        known_threshold=0.55,
    )
    assert result["known"] is True
    assert result["best_library_id"] == "known"
    rejected = retrieve_library_identity(
        np.asarray([1.0, 0.0], dtype=np.float32),
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        ["known"],
        known_confidence=0.1,
        known_threshold=0.55,
    )
    assert rejected["known"] is False


def test_marker_map_only_writes_accepted_candidates() -> None:
    mask = np.zeros((256, 256), dtype=np.float32)
    mask[10:20, 10:20] = 1.0
    marker, rows = build_marker_map(
        (256, 256),
        [{"candidate_id": "known", "mask": mask}, {"candidate_id": "reject", "mask": mask}],
        [
            {"known": True, "known_confidence": 0.9, "best_library_id": "obj", "best_similarity": 0.8},
            {"known": False, "known_confidence": 0.1, "best_library_id": "obj", "best_similarity": 0.8},
        ],
    )
    assert marker.dtype == np.uint16
    assert set(np.unique(marker)) == {0, 1}
    assert len(rows) == 1
    assert rows[0]["library_id"] == "obj"


def test_marker_corpus_uses_silhouette_instances_and_negative_candidates(tmp_path: Path) -> None:
    sieve = _write_sieve(tmp_path / "sieve")
    library = _write_library(tmp_path / "library.zarr")
    output = tmp_path / "marker"
    result = build_object_marker_corpus(
        sieve_corpus=sieve,
        object_library=library,
        output=output,
    )
    assert result["candidate_count"] == 2
    assert result["skipped_instance_count"] == 0
    report = validate_object_marker_corpus(output)
    assert report["valid"] is True
    manifest = json.loads((output / "object_marker_manifest.json").read_text(encoding="utf-8"))
    assert {row["candidate_kind"] for row in manifest["rows"]} == {
        "known_library_object",
        "shifted_or_unknown",
    }
    positive = next(row for row in manifest["rows"] if row["known_object"])
    with np.load(output / positive["npz"], allow_pickle=False) as payload:
        assert payload[IMAGE_SIGNAL].shape == (256, 256, 3)
        assert payload[FOOTPRINT_SIGNAL].shape == (256, 256)
        assert int(payload[KNOWN_SIGNAL]) == 1


def test_marker_corpus_skips_fully_occluded_sieve_instances(tmp_path: Path) -> None:
    sieve = _write_sieve(tmp_path / "sieve")
    manifest_path = sieve / "object_library_sieve_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["rows"][0]["object_instances"].append(
        {
            "instance_id": 3,
            "library_id": "obj-1",
            "asset_path": "world/family/object.mdx",
            "library_family": "world/family",
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = build_object_marker_corpus(
        sieve_corpus=sieve,
        object_library=_write_library(tmp_path / "library.zarr"),
        output=tmp_path / "marker",
    )
    assert result["candidate_count"] == 2
    assert result["skipped_instance_count"] == 1
    marker_manifest = json.loads((tmp_path / "marker" / "object_marker_manifest.json").read_text())
    assert marker_manifest["skipped_instances"][0]["reason"] == "occluded_or_overwritten_in_instance_id_map"


def test_marker_builder_refuses_existing_partial_output(tmp_path: Path) -> None:
    sieve = _write_sieve(tmp_path / "sieve")
    partial = tmp_path / "marker.partial"
    partial.mkdir()
    try:
        build_object_marker_corpus(
            sieve_corpus=sieve,
            object_library=_write_library(tmp_path / "library.zarr"),
            output=tmp_path / "marker",
        )
    except ValueError as exc:
        assert "incomplete output" in str(exc)
    else:
        raise AssertionError("expected partial output refusal")


def test_marker_input_tensor_has_four_channels() -> None:
    tensor = marker_input_tensor(np.zeros((256, 256, 3), dtype=np.float32), np.zeros((256, 256), dtype=np.float32))
    assert tensor.shape == (1, 4, 256, 256)
    assert MARKER_SIGNAL == "known_object_marker_256"
