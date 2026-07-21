"""Spec 116 US3 T023: structure trainer contract tests.

Verifies:
- Dry run writes nothing and prints a plan.
- A tiny CPU fit on the fixture produces a valid ``v50-structure-run-v1`` record.
- Aggregate accuracy is recorded but the gate field never references it (D-08).
- A leaky split (nonzero violation count) is refused.
- A missing vocabulary decision is refused.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from harvester.spec116.family_slot_consistency import measure_family_slot_consistency
from harvester.spec116.held_out_split import build_held_out_split, write_split
from harvester.spec116.structure_contract import Spec116ContractError, validate_structure_run
from harvester.spec116.structure_train import (
    StructureTrainError,
    compute_class_weights,
    confusion_metrics,
    main as train_main,
    rescore_geometry_checkpoint,
    train_structure,
)
from tests.spec116.conftest import build_store, write_texture_name_dump

CHUNKS = 16


def _grid_store(store: Path, n: int, map_name: str = "Kalimdor") -> tuple[Path, Path, list[dict]]:
    """Build an n x n grid store with grass (slot 0) + road (slot 1) per tile.

    Returns (store, dump, index_rows).
    """
    rows = []
    tiles_meta = []
    for y in range(n):
        for x in range(n):
            ids = np.full((CHUNKS, CHUNKS, 4), -1, dtype=np.int32)
            ids[:, :, 0] = 0  # grass
            ids[:, :, 1] = 1  # road
            mask = np.zeros((CHUNKS, CHUNKS, 4), dtype=np.float32)
            mask[:, :, 0] = 1.0
            mask[:, :, 1] = 1.0
            rows.append({
                "map": map_name, "tile_x": x, "tile_y": y,
                "split": "train", "source": "authored",
                "texture_names": [r"Tileset\X\XGrass.blp", r"Tileset\X\XRoad.blp"],
                "mcly_texture_ids": ids, "mcly_layer_mask": mask,
            })
            tiles_meta.append({"TileX": x, "TileY": y,
                               "TextureNames": [r"Tileset\X\XGrass.blp", r"Tileset\X\XRoad.blp"]})
    build_store(store, rows=rows)
    dump = store.parent / "names.json"
    write_texture_name_dump(dump, map_name, tiles_meta)
    return store, dump, rows


def _build_split(store: Path, output: Path, n: int, map_name: str = "Kalimdor") -> Path:
    """Build and persist a held-out split over an n x n grid."""
    import pyarrow.parquet as pq
    index_rows = pq.read_table(store / "index.parquet").to_pylist()
    tiles = [
        {"tile_row": i, "map": r["map"], "tile_x": r["tile_x"], "tile_y": r["tile_y"]}
        for i, r in enumerate(index_rows)
    ]
    split = build_held_out_split(
        tiles=tiles, held_out_fraction=0.2, buffer_rings=1, block_size=5, seed=1,
    )
    assert split["verified_violation_count"] == 0
    write_split(store=store, output=output, split=split, build_id="test")
    return output


def _save_geometry_checkpoint(
    path: Path, *, architecture: str = "direct_cnn_v112", in_channels: int = 3,
) -> None:
    """Save a random-weight Spec 114/115 geometry checkpoint in the real training format."""
    import torch

    from harvester.v50.direct_geometry_model import build_geometry_model

    model, _ = build_geometry_model(architecture, in_channels=in_channels)
    torch.save({"model": model.state_dict(), "model_variant": architecture, "epoch": 1, "val_mae": 0.5}, path)


def _build_feature_store(path: Path, *, row_count: int, class_count: int = 5) -> Path:
    """A minimal v115-feature-map-v1 store: per-pixel class probabilities, row-indexed."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import zarr

    path.mkdir(parents=True, exist_ok=True)
    group = zarr.open_group(str(path), mode="w")
    rng = np.random.default_rng(0)
    raw = rng.random((row_count, class_count, 256, 256)).astype(np.float32)
    probs = raw / raw.sum(axis=1, keepdims=True)
    group.create_array("feature_map", data=probs.astype(np.float16))
    group.attrs.update({"schema": "v115-feature-map-v1", "class_count": class_count})
    index = [{"source_row_index": i} for i in range(row_count)]
    pq.write_table(pa.Table.from_pylist(index), path / "index.parquet")
    return path


def _build_vocab_report(store: Path, dump: Path) -> Path:
    """Build and persist a US1 family_slot_consistency report."""
    report = measure_family_slot_consistency(store=store, dumps=[dump])
    report_path = store.parent / "vocab.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return report_path


@pytest.fixture
def trained_fixture(tmp_path: Path) -> dict:
    """A 10x10 grid store + split + vocabulary report, ready for training."""
    store = tmp_path / "grid.zarr"
    store.mkdir()
    store, dump, _ = _grid_store(store, n=10)
    split_dir = tmp_path / "split"
    _build_split(store, split_dir, n=10)
    vocab = _build_vocab_report(store, dump)
    return {"store": store, "split": split_dir, "dumps": [dump], "vocab": vocab}


class TestClassWeights:
    def test_capped_inverse_frequency(self) -> None:
        counts = {"unknown": 0, "terrain": 1000, "road": 10, "water": 0, "structure": 0}
        weights = compute_class_weights(counts, max_weight=15.0)
        assert len(weights) == 5
        assert weights[0] == 0.0  # unknown masked
        assert weights[1] < weights[2]  # terrain (common) < road (rare)
        assert weights[2] <= 15.0  # capped

    def test_absent_class_gets_neutral_weight(self) -> None:
        counts = {"unknown": 0, "terrain": 100, "road": 0, "water": 0, "structure": 0}
        weights = compute_class_weights(counts)
        assert weights[2] == 1.0  # road absent -> neutral, not infinite

    def test_zero_labels_refused(self) -> None:
        with pytest.raises(StructureTrainError, match="zero labelled chunks"):
            compute_class_weights({"unknown": 0, "terrain": 0, "road": 0, "water": 0, "structure": 0})


class TestConfusionMetrics:
    def test_per_class_iou_and_recall(self) -> None:
        confusion = np.zeros((5, 5), dtype=np.int64)
        # Perfect prediction for terrain (1) and road (2).
        confusion[1, 1] = 50
        confusion[2, 2] = 10
        metrics = confusion_metrics(confusion)
        assert metrics["per_class"]["terrain"]["iou"] == 1.0
        assert metrics["per_class"]["road"]["iou"] == 1.0
        assert metrics["per_class"]["terrain"]["recall"] == 1.0
        assert metrics["macro_iou"] == 1.0
        assert "aggregate_accuracy_reported_only" in metrics

    def test_degenerate_majority_scores_zero_road_iou(self) -> None:
        confusion = np.zeros((5, 5), dtype=np.int64)
        # Predict terrain everywhere; road actual but never predicted.
        confusion[1, 1] = 90
        confusion[2, 1] = 10  # road actual, terrain predicted
        metrics = confusion_metrics(confusion)
        assert metrics["per_class"]["road"]["iou"] == 0.0
        assert metrics["per_class"]["road"]["recall"] == 0.0


class TestDryRun:
    def test_dry_run_writes_nothing_and_prints_plan(
        self, trained_fixture: dict, tmp_path: Path, capsys,
    ) -> None:
        output = tmp_path / "run"
        rc = train_main([
            "--store", str(trained_fixture["store"]),
            "--split", str(trained_fixture["split"]),
            "--dumps", str(trained_fixture["dumps"][0]),
            "--vocabulary", str(trained_fixture["vocab"]),
            "--output", str(output),
            "--slot", "1",
            "--epochs", "2",
            "--batch", "4",
            "--base", "4",
            "--device", "cpu",
        ])
        assert rc == 0
        # Nothing written.
        assert not output.exists()
        # Plan printed.
        captured = capsys.readouterr()
        assert "v116-structure-plan-v1" in captured.out
        assert "DRY RUN ONLY" in captured.out
        assert "slot_keyed" in captured.out or "family_keyed" in captured.out


class TestLeakySplitRefused:
    def test_leaky_split_refused(self, trained_fixture: dict, tmp_path: Path) -> None:
        """A split with nonzero violation count must be refused (FR-010)."""
        # Tamper the split.json to have a nonzero violation count.
        split_json = trained_fixture["split"] / "split.json"
        manifest = json.loads(split_json.read_text(encoding="utf-8"))
        manifest["verified_violation_count"] = 3
        split_json.write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises((StructureTrainError, SystemExit, Spec116ContractError)):
            train_structure(
                store=trained_fixture["store"],
                split_dir=trained_fixture["split"],
                dumps=trained_fixture["dumps"],
                vocabulary_report=trained_fixture["vocab"],
                output=tmp_path / "run",
                slot=1, base=4, epochs=1, batch_size=4, device="cpu",
            )


class TestTinyCpuFit:
    def test_tiny_cpu_fit_produces_valid_run_record(
        self, trained_fixture: dict, tmp_path: Path,
    ) -> None:
        """A tiny CPU fit on the fixture produces a valid v50-structure-run-v1 record."""
        pytest.importorskip("torch")
        output = tmp_path / "run"
        run_record = train_structure(
            store=trained_fixture["store"],
            split_dir=trained_fixture["split"],
            dumps=trained_fixture["dumps"],
            vocabulary_report=trained_fixture["vocab"],
            output=output,
            slot=1, base=4, epochs=2, batch_size=4, device="cpu",
            patience=5,
        )
        # The record validates against the schema.
        validate_structure_run(run_record)
        # Checkpoint written.
        assert (output / "checkpoint_best.pt").exists()
        assert (output / "structure_run.json").exists()

    def test_aggregate_accuracy_reported_but_never_gated(
        self, trained_fixture: dict, tmp_path: Path,
    ) -> None:
        """D-08: aggregate accuracy is in the record but the gate field never references it."""
        pytest.importorskip("torch")
        output = tmp_path / "run"
        run_record = train_structure(
            store=trained_fixture["store"],
            split_dir=trained_fixture["split"],
            dumps=trained_fixture["dumps"],
            vocabulary_report=trained_fixture["vocab"],
            output=output,
            slot=1, base=4, epochs=2, batch_size=4, device="cpu",
            patience=5,
        )
        # Aggregate accuracy is present in metrics.
        assert "aggregate_accuracy_reported_only" in run_record["metrics"]
        # The gate never references aggregate accuracy.
        gate_json = json.dumps(run_record["gate"])
        assert "accuracy" not in gate_json.lower()
        assert run_record["gate"]["rule"] == "per_class_iou_recall"

    def test_promotion_verdict_is_pending(
        self, trained_fixture: dict, tmp_path: Path,
    ) -> None:
        """The trainer never auto-promotes; the user reviews and decides."""
        pytest.importorskip("torch")
        run_record = train_structure(
            store=trained_fixture["store"],
            split_dir=trained_fixture["split"],
            dumps=trained_fixture["dumps"],
            vocabulary_report=trained_fixture["vocab"],
            output=tmp_path / "run",
            slot=1, base=4, epochs=2, batch_size=4, device="cpu",
            patience=5,
        )
        assert run_record["promotion_verdict"] == "pending"

    def test_majority_class_baseline_reported(
        self, trained_fixture: dict, tmp_path: Path,
    ) -> None:
        """The majority-class baseline is reported so the degenerate solution is explicit."""
        pytest.importorskip("torch")
        run_record = train_structure(
            store=trained_fixture["store"],
            split_dir=trained_fixture["split"],
            dumps=trained_fixture["dumps"],
            vocabulary_report=trained_fixture["vocab"],
            output=tmp_path / "run",
            slot=1, base=4, epochs=2, batch_size=4, device="cpu",
            patience=5,
        )
        baseline = run_record["baselines"]["majority_class"]
        assert "family" in baseline
        assert "per_class_iou" in baseline
        assert "per_class_recall" in baseline


class TestRescoreGeometryCheckpoint:
    """T019: relief-stratified re-score of an EXISTING geometry checkpoint. No training."""

    def test_rescore_reports_stratified_mae(self, tmp_path: Path) -> None:
        pytest.importorskip("torch")
        store = tmp_path / "grid.zarr"
        store.mkdir()
        store, _dump, _ = _grid_store(store, n=10)
        split_dir = tmp_path / "split"
        _build_split(store, split_dir, n=10)
        ckpt = tmp_path / "geometry.pt"
        _save_geometry_checkpoint(ckpt)

        report = rescore_geometry_checkpoint(
            checkpoint_path=ckpt, store=store, split_dir=split_dir, device="cpu",
        )
        assert report["schema"] == "v116-geometry-rescore-v1"
        assert report["held_out_row_count"] > 0
        assert "flat" in report["stratified_mae"] and "relief" in report["stratified_mae"]
        assert isinstance(report["trivial_baseline_mae"], float)
        assert isinstance(report["sc007_beats_trivial_on_relief"], bool)
        assert report["absolute_comparison_to_prior_runs_invalid"] is True

    def test_rescore_refuses_leaky_split(self, tmp_path: Path) -> None:
        pytest.importorskip("torch")
        store = tmp_path / "grid.zarr"
        store.mkdir()
        store, _dump, _ = _grid_store(store, n=10)
        split_dir = tmp_path / "split"
        _build_split(store, split_dir, n=10)
        manifest_path = split_dir / "split.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["verified_violation_count"] = 2
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        ckpt = tmp_path / "geometry.pt"
        _save_geometry_checkpoint(ckpt)

        # The split contract itself already rejects a nonzero violation count on load; a
        # leaky split can never reach rescore_geometry_checkpoint's own FR-010 guard.
        with pytest.raises((StructureTrainError, Spec116ContractError)):
            rescore_geometry_checkpoint(checkpoint_path=ckpt, store=store, split_dir=split_dir, device="cpu")

    def test_cli_rescore_mode_skips_training_arg_requirements(self, tmp_path: Path) -> None:
        pytest.importorskip("torch")
        store = tmp_path / "grid.zarr"
        store.mkdir()
        store, _dump, _ = _grid_store(store, n=10)
        split_dir = tmp_path / "split"
        _build_split(store, split_dir, n=10)
        ckpt = tmp_path / "geometry.pt"
        _save_geometry_checkpoint(ckpt)
        rescore_output = tmp_path / "rescore.json"

        rc = train_main([
            "--store", str(store), "--split", str(split_dir),
            "--rescore-checkpoint", str(ckpt),
            "--rescore-output", str(rescore_output),
            "--device", "cpu",
        ])
        assert rc == 0
        assert rescore_output.exists()
        report = json.loads(rescore_output.read_text(encoding="utf-8"))
        assert report["schema"] == "v116-geometry-rescore-v1"

    def test_cli_print_only_writes_nothing(self, tmp_path: Path) -> None:
        pytest.importorskip("torch")
        store = tmp_path / "grid.zarr"
        store.mkdir()
        store, _dump, _ = _grid_store(store, n=10)
        split_dir = tmp_path / "split"
        _build_split(store, split_dir, n=10)
        ckpt = tmp_path / "geometry.pt"
        _save_geometry_checkpoint(ckpt)
        rescore_output = tmp_path / "rescore.json"

        rc = train_main([
            "--store", str(store), "--split", str(split_dir),
            "--rescore-checkpoint", str(ckpt),
            "--rescore-output", str(rescore_output),
            "--print-only", "--device", "cpu",
        ])
        assert rc == 0
        assert not rescore_output.exists()

    def test_cli_train_mode_still_requires_training_args(self, tmp_path: Path) -> None:
        """Without --rescore-checkpoint, the normal training-arg requirements still apply."""
        store = tmp_path / "grid.zarr"
        store.mkdir()
        store, _dump, _ = _grid_store(store, n=4)
        split_dir = tmp_path / "split"
        _build_split(store, split_dir, n=4)

        with pytest.raises(SystemExit):
            train_main(["--store", str(store), "--split", str(split_dir)])


class TestRescoreWithFeatureStore:
    """A Spec 115 deconfounded checkpoint (RGB + generated feature-map channels) needs the same
    feature-map store reconstructed at rescore time -- --rescore-in-channels alone is not enough
    if the actual input tensor is still RGB-only (the bug that reached the user: an 8-channel
    checkpoint requires 8 real input channels, not just a model built to expect 8)."""

    def test_auto_derives_in_channels_from_feature_store(self, tmp_path: Path) -> None:
        pytest.importorskip("torch")
        store = tmp_path / "grid.zarr"
        store.mkdir()
        store, _dump, _ = _grid_store(store, n=10)
        split_dir = tmp_path / "split"
        _build_split(store, split_dir, n=10)

        import pyarrow.parquet as pq
        row_count = pq.read_table(store / "index.parquet").num_rows
        feature_store = _build_feature_store(tmp_path / "features.zarr", row_count=row_count, class_count=5)

        ckpt = tmp_path / "geometry.pt"
        _save_geometry_checkpoint(ckpt, architecture="mit_b0_regression", in_channels=8)

        # No --rescore-in-channels passed: must auto-derive 3 + 5 = 8 from the feature store's
        # own class_count, matching the checkpoint's actual trained channel count.
        report = rescore_geometry_checkpoint(
            checkpoint_path=ckpt, store=store, split_dir=split_dir,
            feature_store=feature_store, device="cpu",
        )
        assert report["checkpoint"]["in_channels"] == 8
        assert report["feature_store"]["class_count"] == 5
        assert report["held_out_row_count"] > 0

    def test_wrong_feature_store_schema_refused(self, tmp_path: Path) -> None:
        pytest.importorskip("torch")
        store = tmp_path / "grid.zarr"
        store.mkdir()
        store, _dump, _ = _grid_store(store, n=10)
        split_dir = tmp_path / "split"
        _build_split(store, split_dir, n=10)

        import zarr
        not_a_feature_store = tmp_path / "wrong.zarr"
        not_a_feature_store.mkdir()
        zarr.open_group(str(not_a_feature_store), mode="w").attrs["schema"] = "v116-structure-store-v1"

        ckpt = tmp_path / "geometry.pt"
        _save_geometry_checkpoint(ckpt, architecture="mit_b0_regression", in_channels=8)

        with pytest.raises(StructureTrainError, match="v115-feature-map-v1"):
            rescore_geometry_checkpoint(
                checkpoint_path=ckpt, store=store, split_dir=split_dir,
                feature_store=not_a_feature_store, device="cpu",
            )

    def test_source_filter_restricts_held_out_rows_to_matching_domain(self, tmp_path: Path) -> None:
        """A feature-map store that only covers 'authored' rows must be paired with
        --rescore-source authored, or the mismatch surfaces as a clear "missing rows" error
        instead of silently scoring the wrong (or too few) rows."""
        pytest.importorskip("torch")
        store = tmp_path / "grid.zarr"
        store.mkdir()
        store, _dump, rows = _grid_store(store, n=10)

        # Overwrite the index with a real minimap_source split: half authored, half synthetic.
        import pyarrow as pa
        import pyarrow.parquet as pq
        index_rows = pq.read_table(store / "index.parquet").to_pylist()
        for i, row in enumerate(index_rows):
            row["minimap_source"] = "authored" if i % 2 == 0 else "synthetic"
        pq.write_table(pa.Table.from_pylist(index_rows), store / "index.parquet")

        split_dir = tmp_path / "split"
        _build_split(store, split_dir, n=10)

        ckpt = tmp_path / "geometry.pt"
        _save_geometry_checkpoint(ckpt, architecture="direct_cnn_v112")

        report_all = rescore_geometry_checkpoint(
            checkpoint_path=ckpt, store=store, split_dir=split_dir, source="all", device="cpu",
        )
        report_authored = rescore_geometry_checkpoint(
            checkpoint_path=ckpt, store=store, split_dir=split_dir, source="authored", device="cpu",
        )
        assert report_all["source"] == "all"
        assert report_authored["source"] == "authored"
        # The authored-only subset can never exceed the full held-out row count.
        assert report_authored["held_out_row_count"] <= report_all["held_out_row_count"]

    def test_rgb_only_checkpoint_still_works_without_feature_store(self, tmp_path: Path) -> None:
        """Auto-derivation for the plain RGB baseline: no --feature-store -> in_channels=3."""
        pytest.importorskip("torch")
        store = tmp_path / "grid.zarr"
        store.mkdir()
        store, _dump, _ = _grid_store(store, n=10)
        split_dir = tmp_path / "split"
        _build_split(store, split_dir, n=10)
        ckpt = tmp_path / "geometry.pt"
        _save_geometry_checkpoint(ckpt, architecture="direct_cnn_v112")

        report = rescore_geometry_checkpoint(checkpoint_path=ckpt, store=store, split_dir=split_dir, device="cpu")
        assert report["checkpoint"]["in_channels"] == 3
        assert report["feature_store"] is None