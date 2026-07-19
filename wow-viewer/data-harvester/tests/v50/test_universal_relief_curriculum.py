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
from harvester.v50.universal_relief_train import (
    UniversalReliefDataset,
    UniversalTrainingError,
    build_training_plan,
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


def _make_exact_v50_store(tmp_path):
    store_path = tmp_path / "v50-exact.zarr"
    group = zarr.open_group(str(store_path), mode="w")
    group.create_array(
        "minimap_rgb", data=np.zeros((6, 256, 256, 3), dtype=np.uint8), chunks=(1, 256, 256, 3)
    )
    group.create_array(
        "height_257", data=np.zeros((6, 257, 257), dtype=np.float32), chunks=(1, 257, 257)
    )
    rows = [
        {
            "source_group_id": f"terrain:{map_name}:{index}",
            "minimap_source": "authored",
            "split": split,
            "map": map_name,
        }
        for index, (map_name, split) in enumerate(
            (
                ("Kalimdor", "train"),
                ("Kalimdor", "train"),
                ("Kalimdor", "val"),
                ("Azeroth", "train"),
                ("Azeroth", "train"),
                ("Azeroth", "val"),
            )
        )
    ]
    pq.write_table(pa.Table.from_pylist(rows), store_path / "index.parquet")
    (store_path / "summary.json").write_text(json.dumps({"rows": 6}), encoding="utf-8")
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


def test_exact_v50_only_plan_uses_own_rgb_height_and_whole_map_holdout(tmp_path) -> None:
    plan = build_universal_curriculum_plan(
        v50_store=_make_exact_v50_store(tmp_path),
        v50_source="authored",
        teacher_stores=(),
        holdout_families=(),
        holdout_maps=["Azeroth"],
        output=tmp_path / "exact-only.zarr",
    )

    assert plan.summary["visual_families"] == {
        "wow_authored:Azeroth": 3,
        "wow_authored:Kalimdor": 3,
    }
    assert plan.summary["held_out_families"] == ["wow_authored:Azeroth"]
    assert plan.summary["target_authorities"] == {"exact_numeric": 6, "teacher_pseudo": 0}
    assert plan.summary["split_counts"] == {
        "train": 2,
        "validation": 1,
        "test": 0,
        "compatibility": 3,
    }
    assert all(
        row["split"] == "compatibility" for row in plan.rows if row["map"] == "Azeroth"
    )
    assert all(row["target_authority"] == "exact_numeric" for row in plan.rows)


def test_exact_v50_only_curriculum_is_accepted_by_training_plan(tmp_path) -> None:
    curriculum = tmp_path / "exact-only.zarr"
    plan = build_universal_curriculum_plan(
        v50_store=_make_exact_v50_store(tmp_path),
        teacher_stores=(),
        holdout_families=(),
        holdout_maps=["Azeroth"],
        output=curriculum,
    )
    write_universal_curriculum(plan)

    training = build_training_plan(
        curriculum=curriculum,
        output=tmp_path / "run",
        batch_size=2,
        epochs=2,
        workers=0,
        seed=114,
        overlap=28,
        learning_rate=2e-4,
        weight_decay=1e-4,
        pseudo_weight=0.5,
        freeze_backbone=True,
    )

    assert training["source_rows"] == 6
    assert training["held_out_families"] == ["wow_authored:Azeroth"]
    assert training["tile_counts"]["compatibility"] > 0


def test_exact_v50_only_cli_is_dry_run_and_writes_nothing(tmp_path, capsys) -> None:
    output = tmp_path / "exact-dry.zarr"
    assert (
        main(
            [
                "--v50-store",
                str(_make_exact_v50_store(tmp_path)),
                "--v50-source",
                "authored",
                "--holdout-map",
                "Azeroth",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert not output.exists()
    assert "DRY RUN" in capsys.readouterr().out


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


def test_teacher_source_image_drift_is_refused_before_curriculum_build(tmp_path) -> None:
    v50 = _make_v50_store(tmp_path)
    bindings = _bindings(tmp_path)
    aerial = zarr.open_group(str(bindings[0].path), mode="r")
    input_root = tmp_path / "aerial-images"
    Image.fromarray(np.full((17, 9, 3), 255, dtype=np.uint8), mode="RGB").save(
        input_root / "source.png"
    )
    assert aerial.attrs["input_root"] == str(input_root.resolve())

    with pytest.raises(UniversalCurriculumError, match="source image drift"):
        build_universal_curriculum_plan(
            v50_store=v50,
            teacher_stores=bindings,
            holdout_families=["aerial"],
            output=tmp_path / "drift.zarr",
        )


def test_teacher_relief_target_drift_is_refused_before_curriculum_build(tmp_path) -> None:
    v50 = _make_v50_store(tmp_path)
    bindings = _bindings(tmp_path)
    aerial = zarr.open_group(str(bindings[0].path), mode="a")
    row_id = next(iter(aerial["rows"].group_keys()))
    aerial["rows"][row_id]["relative_relief"][:] = 0.0

    with pytest.raises(UniversalCurriculumError, match="relief target drift"):
        build_universal_curriculum_plan(
            v50_store=v50,
            teacher_stores=bindings,
            holdout_families=["aerial"],
            output=tmp_path / "target-drift.zarr",
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
    table = pq.read_table(plan.output / "index.parquet")
    assert table.num_rows == len(plan.rows)
    teacher_rows = [row for row in table.to_pylist() if row["target_authority"] == "teacher_pseudo"]
    assert all(row["input_path"] and len(row["target_sha256"]) == 64 for row in teacher_rows)
    with pytest.raises(FileExistsError):
        write_universal_curriculum(plan)


def test_training_refuses_teacher_target_mutated_after_curriculum_build(tmp_path) -> None:
    bindings = _bindings(tmp_path)
    plan = build_universal_curriculum_plan(
        v50_store=_make_v50_store(tmp_path),
        teacher_stores=bindings,
        holdout_families=["aerial"],
        output=tmp_path / "universal.zarr",
    )
    write_universal_curriculum(plan)
    photos = zarr.open_group(str(bindings[1].path), mode="a")
    row_id = next(iter(photos["rows"].group_keys()))
    photos["rows"][row_id]["relative_relief"][:] = 0.0
    dataset = UniversalReliefDataset(plan.output, {"train"}, augment=False)
    sample_index = next(
        index
        for index, sample in enumerate(dataset.samples)
        if sample.row["visual_family"] == "photos"
    )

    with pytest.raises(UniversalTrainingError, match="relief target drift"):
        dataset[sample_index]


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
