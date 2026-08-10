from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr

from harvester.v60.object_library_sieve import (
    CLEAN_SIGNAL,
    INPUT_SIGNAL,
    INSTANCE_SIGNAL,
    MASK_SIGNAL,
    _family_split,
    build_object_library_sieve_corpus,
    validate_object_library_sieve_corpus,
)


def _hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array, dtype="<f4").tobytes()).hexdigest()


def _write_control(root: Path) -> Path:
    root.mkdir()
    clean = np.full((256, 256), 0.62, dtype=np.float32)
    height = np.full((257, 257), 25.0, dtype=np.float32)
    np.savez(root / "ridge-v00.npz", terrain_shadow_256=clean, height_257=height)
    row = {
        "row_id": "ridge-v00",
        "control_family": "ridge",
        "complexity_bucket": "hard",
        "split": "train",
        "npz": "ridge-v00.npz",
        "input_sha256": _hash(clean),
        "target_sha256": _hash(height),
    }
    (root / "control_manifest.json").write_text(
        json.dumps({"schema": "v60-control-corpus-v1", "row_count": 1, "rows": [row]}),
        encoding="utf-8",
    )
    return root


def _write_library(root: Path) -> Path:
    root.mkdir()
    count = 20
    rgb = np.zeros((count, 128, 128, 3), dtype=np.uint8)
    masks = np.zeros((count, 128, 128), dtype=np.uint8)
    rows = []
    for index in range(count):
        rgb[index, 32:96, 40:88] = [40 + index, 120, 200 - index]
        masks[index, 32:96, 40:88] = 255
        path = f"family{index}/objects/object{index}.mdx"
        rows.append(
            {
                "library_id": f"objlib_{index:02d}",
                "original_asset_path": path,
                "normalized_asset_path": path,
                "asset_type": "mdx",
                "capture_status": "captured",
                "visibility_class": "likely_visible",
                "review_state": "unreviewed",
                "source_builds": ["0.5.3.3368"],
                "placement_observation_count": 0,
                "preferred_variant_id": "",
            }
        )
    group = zarr.open_group(str(root), mode="w")
    group.create_array("capture_rgb", data=rgb)
    group.create_array("capture_mask", data=masks)
    group.attrs.update({"schema": "spec-077-object-library", "run_name": "fixture", "entry_count": count})
    pq.write_table(pa.Table.from_pylist(rows), root / "assets.parquet")
    pq.write_table(pa.Table.from_pylist([{"library_id": row["library_id"]} for row in rows]), root / "index.parquet")
    assert { _family_split(f"family{index}/objects", 6001) for index in range(count) } == {"train", "validation"}
    return root


def test_library_sieve_uses_real_capture_masks_and_is_repeatable(tmp_path: Path) -> None:
    control = _write_control(tmp_path / "control")
    library = _write_library(tmp_path / "library.zarr")
    assets_before = (library / "assets.parquet").read_bytes()
    first = tmp_path / "first"
    second = tmp_path / "second"

    result = build_object_library_sieve_corpus(
        control_corpus=control,
        object_library=library,
        output=first,
        seed=6001,
    )
    build_object_library_sieve_corpus(
        control_corpus=control,
        object_library=library,
        output=second,
        seed=6001,
    )

    assert result["row_count"] == 5
    report = validate_object_library_sieve_corpus(first)
    assert report["valid"] is True
    assert report["library_object_count"] >= 1
    manifest = json.loads((first / "object_library_sieve_manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_policy"].startswith("real_v50_object_library")
    assert any(row["object_instance_count"] > 1 for row in manifest["rows"])
    assert any(row["object_instances"] for row in manifest["rows"])
    for row in manifest["rows"]:
        with np.load(first / row["npz"], allow_pickle=False) as payload:
            assert payload[INPUT_SIGNAL].shape == (256, 256)
            assert payload[CLEAN_SIGNAL].shape == (256, 256)
            assert payload[MASK_SIGNAL].shape == (256, 256)
            assert payload[INSTANCE_SIGNAL].shape == (256, 256)
            assert np.array_equal(payload[INSTANCE_SIGNAL] > 0, payload[MASK_SIGNAL] >= 0.5)
    first_row = manifest["rows"][1]
    second_manifest = json.loads((second / "object_library_sieve_manifest.json").read_text(encoding="utf-8"))
    second_row = second_manifest["rows"][1]
    assert first_row["input_sha256"] == second_row["input_sha256"]
    assert (first / first_row["npz"]).read_bytes() == (second / second_row["npz"]).read_bytes()
    assert (library / "assets.parquet").read_bytes() == assets_before
