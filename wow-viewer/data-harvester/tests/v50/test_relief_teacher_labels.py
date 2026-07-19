from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import zarr
from PIL import Image

from harvester.v50.relief_teacher_labels import (
    build_teacher_label_plan,
    default_teacher_identity,
    main,
    normalize_teacher_relief,
    validate_teacher_identity,
    write_teacher_label_store,
)


def _write_rgb(path, width: int, height: int, value: int) -> None:
    pixels = np.full((height, width, 3), value, dtype=np.uint8)
    Image.fromarray(pixels, mode="RGB").save(path)


def test_default_teacher_is_full_revision_safe_tensor_and_not_depthanything() -> None:
    identity = default_teacher_identity()
    validate_teacher_identity(identity)

    assert len(identity.revision) == 40
    assert identity.weight_file == "model.safetensors"
    assert len(identity.weights_sha256) == 64
    assert identity.license == "apache-2.0"


def test_depthanything_teacher_is_refused() -> None:
    identity = replace(default_teacher_identity(), hub_id="vendor/depth-anything-v2")
    with pytest.raises(ValueError, match="forbidden"):
        validate_teacher_identity(identity)


def test_teacher_normalization_preserves_larger_is_higher_orientation() -> None:
    predicted = np.arange(100, dtype=np.float32).reshape(10, 10)
    relief = normalize_teacher_relief(predicted, low_percentile=0.0, high_percentile=100.0)

    assert relief[0, 0] == 0.0
    assert relief[-1, -1] == 1.0
    assert np.all(np.diff(relief.ravel()) >= 0.0)


def test_constant_teacher_prediction_becomes_stable_zero() -> None:
    relief = normalize_teacher_relief(np.full((8, 5), 7.0, dtype=np.float32))
    assert np.array_equal(relief, np.zeros((8, 5), dtype=np.float32))


def test_plan_requires_data_authority_and_discovers_only_decodable_images(tmp_path) -> None:
    input_dir = tmp_path / "images"
    input_dir.mkdir()
    _write_rgb(input_dir / "one.png", 9, 7, 50)
    (input_dir / "bad.png").write_text("not an image", encoding="utf-8")

    with pytest.raises(ValueError, match="exactly one"):
        build_teacher_label_plan(
            input_root=input_dir,
            output_store=tmp_path / "labels.zarr",
            visual_family="art",
        )

    plan = build_teacher_label_plan(
        input_root=input_dir,
        output_store=tmp_path / "labels.zarr",
        visual_family="art",
        byod=True,
    )
    assert len(plan.sources) == 1
    assert plan.sources[0].relative_path == "one.png"
    assert plan.data_authority == "private-byod"


def test_fake_teacher_writes_variable_shape_zarr_and_separates_pseudo_authority(tmp_path) -> None:
    input_dir = tmp_path / "images"
    input_dir.mkdir()
    _write_rgb(input_dir / "wide.png", 13, 5, 80)
    _write_rgb(input_dir / "tall.png", 4, 11, 120)
    output = tmp_path / "labels.zarr"
    plan = build_teacher_label_plan(
        input_root=input_dir,
        output_store=output,
        visual_family="photos",
        license_id="cc-by-4.0",
    )

    def fake_predictor(image: Image.Image) -> np.ndarray:
        return np.arange(image.height * image.width, dtype=np.float32).reshape(
            image.height, image.width
        )

    summary = write_teacher_label_store(plan, fake_predictor)

    assert summary["target_authority"] == "teacher_pseudo"
    assert summary["source_count"] == 2
    assert all(len(row["relief_sha256"]) == 64 for row in summary["rows"])
    assert output.with_suffix(".summary.json").is_file()
    store = zarr.open_group(str(output), mode="r")
    shapes = sorted(tuple(store["rows"][row_id]["relative_relief"].shape) for row_id in store["rows"])
    assert shapes == [(5, 13), (11, 4)]
    assert store.attrs["teacher"]["revision"] == default_teacher_identity().revision
    assert all(len(store["rows"][row_id].attrs["relief_sha256"]) == 64 for row_id in store["rows"])


def test_cli_is_dry_run_by_default_and_creates_no_output(tmp_path, capsys) -> None:
    input_dir = tmp_path / "images"
    input_dir.mkdir()
    _write_rgb(input_dir / "one.png", 6, 6, 10)
    output = tmp_path / "labels.zarr"

    assert (
        main(
            [
                "--input-dir",
                str(input_dir),
                "--output",
                str(output),
                "--family",
                "drawings",
                "--byod",
            ]
        )
        == 0
    )
    assert not output.exists()
    assert "DRY RUN" in capsys.readouterr().out
