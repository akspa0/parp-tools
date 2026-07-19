from __future__ import annotations

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import zarr
from PIL import Image

from harvester.v50.relief_teacher_labels import (
    build_teacher_label_plan,
    write_teacher_label_store,
)
from harvester.v50.universal_relief_curriculum import (
    TeacherStoreBinding,
    UniversalCurriculumError,
    build_universal_curriculum_plan,
    main,
    write_universal_curriculum,
)


def _make_v50_store(tmp_path, *, leak: bool = False):
    store_path = tmp_path / "v50.zarr"
    group = zarr.open_group(str(store_path), mode="w")
    group.create_array(
        "minimap_rgb", data=np.zeros((3, 256, 256, 3), dtype=np.uint8), chunks=(1, 256, 256, 3)
    )
    group.create_array(
        "height_257", data=np.zeros((3, 257, 257), dtype=np.float32), chunks=(1, 257, 257)
    )
    groups = ["terrain:a", "terrain:a" if leak else "terrain:b", "terrain:c"]
    splits = ["train", "val", "train"]
    rows = [
        {
            "source_group_id": groups[index],
            "minimap_source": "authored",
            "split": splits[index],
        }
        for index in range(3)
    ]
    pq.write_table(pa.Table.from_pylist(rows), store_path / "index.parquet")
    (store_path / "summary.json").write_text(json.dumps({"rows": 3}), encoding="utf-8")
    return store_path


def _make_teacher_store(tmp_path, family: str, value: int):
    image_root = tmp_path / f"{family}-images"
    image_root.mkdir()
    pixels = np.full((7 + value, 9, 3), value, dtype=np.uint8)
    Image.fromarray(pixels, mode="RGB").save(image_root / "source.png")
    output = tmp_path / f"{family}.zarr"
    plan = build_teacher_label_plan(
        input_root=image_root,
        output_store=output,
        visual_family=family,
        byod=True,
    )

    def predictor(image: Image.Image) -> np.ndarray:
        return np.arange(image.height * image.width, dtype=np.float32).reshape(
            image.height, image.width
        )

    write_teacher_label_store(plan, predictor)
    return TeacherStoreBinding(family, output)


def _bindings(tmp_path):
    return tuple(
        _make_teacher_store(tmp_path, family, value)
        for family, value in (("aerial", 10), ("photos", 20), ("paintings", 30), ("drawings", 40))
    )


def test_plan_requires_five_real_families_and_a_whole_family_holdout(tmp_path) -> None:
    v50 = _make_v50_store(tmp_path)
    bindings = _bindings(tmp_path)

    with pytest.raises(UniversalCurriculumError, match="at least 5"):
        build_universal_curriculum_plan(
            v50_store=v50,
            teacher_stores=bindings[:3],
            holdout_families=["aerial"],
            output=tmp_path / "short.zarr",
        )
    with pytest.raises(UniversalCurriculumError, match="at least one whole"):
        build_universal_curriculum_plan(
            v50_store=v50,
            teacher_stores=bindings,
            holdout_families=[],
            output=tmp_path / "no-holdout.zarr",
        )


def test_plan_keeps_exact_and_pseudo_authorities_and_holds_out_family(tmp_path) -> None:
    v50 = _make_v50_store(tmp_path)
    bindings = _bindings(tmp_path)
    plan = build_universal_curriculum_plan(
        v50_store=v50,
        teacher_stores=bindings,
        holdout_families=["paintings"],
        output=tmp_path / "universal.zarr",
    )

    assert len(plan.summary["visual_families"]) == 5
    assert plan.summary["held_out_families"] == ["paintings"]
    assert plan.summary["target_authorities"] == {"exact_numeric": 3, "teacher_pseudo": 4}
    assert plan.summary["split_counts"]["compatibility"] == 1
    assert plan.summary["group_leak_count"] == 0
    assert plan.summary["family_leak_count"] == 0
    assert all(
        row["split"] == "compatibility"
        for row in plan.rows
        if row["visual_family"] == "paintings"
    )


def test_v50_source_group_cross_split_leak_is_refused(tmp_path) -> None:
    v50 = _make_v50_store(tmp_path, leak=True)
    with pytest.raises(UniversalCurriculumError, match="source-group leakage"):
        build_universal_curriculum_plan(
            v50_store=v50,
            teacher_stores=_bindings(tmp_path),
            holdout_families=["drawings"],
            output=tmp_path / "leak.zarr",
        )


def test_same_teacher_image_cannot_be_renamed_as_two_families(tmp_path) -> None:
    v50 = _make_v50_store(tmp_path)
    bindings = list(_bindings(tmp_path))
    copied = tmp_path / "copied.zarr"
    original = zarr.open_group(str(bindings[0].path), mode="r")
    copied_group = zarr.open_group(str(copied), mode="w")
    copied_group.attrs.update(dict(original.attrs))
    copied_group.attrs["visual_family"] = "fake-extra"
    copied_rows = copied_group.create_group("rows")
    for key in original["rows"].group_keys():
        row = copied_rows.create_group(key)
        row.attrs.update(dict(original["rows"][key].attrs))
        row.create_array("relative_relief", data=original["rows"][key]["relative_relief"][:])
    bindings.append(TeacherStoreBinding("fake-extra", copied))

    with pytest.raises(UniversalCurriculumError, match="same source content"):
        build_universal_curriculum_plan(
            v50_store=v50,
            teacher_stores=bindings,
            holdout_families=["fake-extra"],
            output=tmp_path / "renamed.zarr",
        )


def test_write_produces_immutable_zarr_index_and_summary(tmp_path) -> None:
    plan = build_universal_curriculum_plan(
        v50_store=_make_v50_store(tmp_path),
        teacher_stores=_bindings(tmp_path),
        holdout_families=["aerial"],
        output=tmp_path / "universal.zarr",
    )
    summary = write_universal_curriculum(plan)

    assert (plan.output / "index.parquet").is_file()
    assert (plan.output / "summary.json").is_file()
    assert summary["curriculum_id"].startswith("sha256:")
    assert pq.read_table(plan.output / "index.parquet").num_rows == len(plan.rows)
    with pytest.raises(FileExistsError):
        write_universal_curriculum(plan)


def test_cli_is_dry_run_and_creates_no_output(tmp_path, capsys) -> None:
    v50 = _make_v50_store(tmp_path)
    bindings = _bindings(tmp_path)
    output = tmp_path / "dry.zarr"
    args = ["--v50-store", str(v50), "--output", str(output), "--holdout-family", "photos"]
    for binding in bindings:
        args.extend(["--teacher-store", f"{binding.visual_family}={binding.path}"])

    assert main(args) == 0
    assert not output.exists()
    assert "DRY RUN" in capsys.readouterr().out
