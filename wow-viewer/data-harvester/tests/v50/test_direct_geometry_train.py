"""Spec 114 T015: baselines, SC-001/SC-002 gates, and the stage-run record contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from harvester.v50.direct_geometry_train import (
    SC001_RELATIVE_MARGIN,
    SPEC112_FROZEN_BEST_VAL_MAE,
    TrainerContractError,
    apply_held_out_split,
    build_direct_plan,
    build_stage_run_summary,
    check_sc002,
    compute_flat_baseline,
    evaluate_sc001,
)
from harvester.v50.height_relative_train import compute_tile_mean_baseline
from harvester.v50.model_stage_contract import validate_model_stage_run


def test_flat_baseline_is_constant_half_prediction_error() -> None:
    flat = np.full((9, 9), 0.5, dtype=np.float32)
    ramp = np.linspace(0, 1, 81, dtype=np.float32).reshape(9, 9)
    assert compute_flat_baseline([flat]) == pytest.approx(0.0)
    assert compute_flat_baseline([ramp]) == pytest.approx(0.25, abs=0.01)
    with pytest.raises(TrainerContractError, match="zero validation"):
        compute_flat_baseline([])


def _record(border: float, interior: float) -> dict:
    return {"border_mae": border, "interior_mae": interior}


def test_sc002_border_within_interior_p95_passes() -> None:
    records = [_record(border=0.10, interior=0.10) for _ in range(20)]
    result = check_sc002(records)
    assert result["passes"] is True
    assert result["held_out_rows"] == 20
    assert result["interior_mae_p95"] == pytest.approx(0.10)


def test_sc002_border_exceeding_interior_p95_fails() -> None:
    records = [_record(border=0.05, interior=0.05) for _ in range(19)]
    records.append(_record(border=0.30, interior=0.05))
    result = check_sc002(records)
    # pooled border mean (0.06125) vs interior p95 (0.05): a border-specific seam failure
    assert result["passes"] is False
    with pytest.raises(TrainerContractError, match="at least one"):
        check_sc002([])


def test_sc001_requires_five_percent_over_all_references() -> None:
    tile_mean = compute_tile_mean_baseline([np.linspace(0, 1, 81, dtype=np.float32).reshape(9, 9)])
    flat = compute_flat_baseline([np.linspace(0, 1, 81, dtype=np.float32).reshape(9, 9)])
    good = evaluate_sc001(best_val_mae=0.10, tile_mean_baseline=tile_mean, flat_baseline=flat)
    assert good["passes"] is True
    assert good["beats_spec112_frozen"] is True

    # The rejected bootstrap's own number must fail its own gate.
    failed = evaluate_sc001(
        best_val_mae=SPEC112_FROZEN_BEST_VAL_MAE,
        tile_mean_baseline=0.1387469612,
        flat_baseline=0.30,
    )
    assert failed["passes"] is False
    assert failed["beats_tile_mean"] is False

    marginal = evaluate_sc001(
        best_val_mae=SPEC112_FROZEN_BEST_VAL_MAE * (1.0 - SC001_RELATIVE_MARGIN / 2),
        tile_mean_baseline=0.5,
        flat_baseline=0.5,
    )
    assert marginal["beats_spec112_frozen"] is False
    assert marginal["passes"] is False


def _stage_run_kwargs() -> dict:
    return {
        "run_id": "mit_b0-authored-v1",
        "architecture": {"id": "mit_b0_regression", "config_sha256": "c" * 64,
                         "parameter_count": 3_800_000},
        "pretrained_source": None,
        "curriculum": {"path": "store.zarr", "sha256": "a" * 64},
        "checkpoint": {"path": "out/checkpoint_best.pt", "sha256": "d" * 64, "best_epoch": 12},
        "baselines": {"tile_mean": {"val_mae": 0.13}, "flat": {"val_mae": 0.25}},
        "metrics": {"best_val_mae": 0.10},
        "visual_evidence": {"error_quantiles": "validation/final_best/error_quantiles.png"},
    }


def test_stage_run_summary_validates_and_stays_pending() -> None:
    summary = build_stage_run_summary(**_stage_run_kwargs(), created_utc="2026-07-19T00:00:00Z")
    validate_model_stage_run(summary)
    assert summary["stage"] == "direct_geometry"
    assert summary["output_signal"] == "relative_height_257"
    assert summary["promotion_verdict"] == "pending"
    assert summary["upstream_models"] == []


def test_stage_run_summary_fails_closed_on_bad_identity() -> None:
    kwargs = _stage_run_kwargs()
    kwargs["checkpoint"] = {"path": "x", "sha256": "short", "best_epoch": 1}
    with pytest.raises(TrainerContractError, match="violates its own contract"):
        build_stage_run_summary(**kwargs)


def test_direct_plan_records_recipe_and_sc001_references() -> None:
    rows = [
        {"map": "Kalimdor", "source_group_id": f"g{i}", "minimap_source": "authored",
         "split": "train" if i < 40 else "val"}
        for i in range(50)
    ]
    plan = build_direct_plan(
        architecture_identity={"id": "direct_cnn_v112", "config_sha256": "c" * 64,
                               "parameter_count": 1_561_537},
        pretrained_source=None,
        source="authored",
        index_rows=rows,
        selected_rows=list(range(50)),
        batch_size=16,
        epochs=100,
        seed=114,
        lr=2e-4,
        lr_schedule="onecycle",
        amp=True,
        amp_dtype="bf16",
        clip=1.0,
    )
    assert plan["schema"] == "v114-direct-geometry-plan-v2"
    assert plan["wdl_prior"] is False
    assert plan["deployment_inputs"] == ["minimap_rgb"]
    assert plan["lr_schedule"] == "onecycle"
    assert plan["amp"] is True
    assert plan["amp_dtype"] == "bf16"
    assert plan["sc001_references"]["spec112_frozen_best_val_mae"] == SPEC112_FROZEN_BEST_VAL_MAE
    with pytest.raises(TrainerContractError, match="lr schedule"):
        build_direct_plan(
            architecture_identity=plan["architecture"], pretrained_source=None, source="authored",
            index_rows=rows, selected_rows=[0], batch_size=16, epochs=100, seed=114,
            lr=2e-4, lr_schedule="cosmic", amp=False, amp_dtype="fp16", clip=0.0,
        )
    with pytest.raises(TrainerContractError, match="amp_dtype"):
        build_direct_plan(
            architecture_identity=plan["architecture"], pretrained_source=None, source="authored",
            index_rows=rows, selected_rows=[0], batch_size=16, epochs=100, seed=114,
            lr=2e-4, lr_schedule="constant", amp=False, amp_dtype="int8", clip=0.0,
        )


class TestApplyHeldOutSplit:
    """Spec 116 US5 5b wiring: an external held-out-split directory overrides train/val
    selection instead of the curriculum's own (possibly stale) "split" column."""

    def _build_split_dir(self, tmp_path: Path, index_rows: list[dict]) -> Path:
        from harvester.spec116.held_out_split import build_held_out_split, write_split

        tiles = [
            {"tile_row": i, "map": r["map"], "tile_x": r["tile_x"], "tile_y": r["tile_y"]}
            for i, r in enumerate(index_rows)
        ]
        split = build_held_out_split(tiles=tiles, held_out_fraction=0.2, buffer_rings=1, block_size=3, seed=7)
        assert split["verified_violation_count"] == 0
        split_dir = tmp_path / "split"
        # write_split only needs store/index.parquet's sha256; a lightweight decoy is enough.
        store = tmp_path / "decoy.zarr"
        store.mkdir()
        import pyarrow as pa
        import pyarrow.parquet as pq

        pq.write_table(pa.Table.from_pylist(index_rows), store / "index.parquet")
        write_split(store=store, output=split_dir, split=split, build_id="test")
        return split_dir

    def _grid_index_rows(self, n: int) -> list[dict]:
        return [
            {"map": "Kalimdor", "tile_x": x, "tile_y": y, "split": "train", "source": "authored"}
            for y in range(n) for x in range(n)
        ]

    def test_overrides_train_val_from_external_split(self, tmp_path: Path) -> None:
        rows = self._grid_index_rows(10)
        split_dir = self._build_split_dir(tmp_path, rows)

        train_rows, val_rows, manifest = apply_held_out_split(
            index_rows=rows, selected_rows=list(range(len(rows))), split_dir=split_dir,
        )
        assert manifest["verified_violation_count"] == 0
        assert len(train_rows) > 0
        assert len(val_rows) > 0
        assert set(train_rows).isdisjoint(val_rows)
        # Every selected row lands in train or val or neither (excluded buffer) -- never both.
        assert len(train_rows) + len(val_rows) <= len(rows)

    def test_refuses_leaky_split(self, tmp_path: Path) -> None:
        import json as _json

        from harvester.spec116.structure_contract import Spec116ContractError

        rows = self._grid_index_rows(10)
        split_dir = self._build_split_dir(tmp_path, rows)
        manifest_path = split_dir / "split.json"
        manifest = _json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["verified_violation_count"] = 3
        manifest_path.write_text(_json.dumps(manifest), encoding="utf-8")

        # The split contract itself already rejects a nonzero violation count on load; a leaky
        # split can never reach apply_held_out_split's own FR-010 guard.
        with pytest.raises((TrainerContractError, Spec116ContractError)):
            apply_held_out_split(index_rows=rows, selected_rows=list(range(len(rows))), split_dir=split_dir)

    def test_ignores_stale_split_column_on_index_rows(self, tmp_path: Path) -> None:
        """The curriculum's own "split" column (train/val) is irrelevant once an external
        held-out-split directory is supplied -- only (map, tile_x, tile_y) membership matters."""
        rows = self._grid_index_rows(10)
        for row in rows:
            row["split"] = "val"  # every row claims val on the stale column
        split_dir = self._build_split_dir(tmp_path, rows)

        train_rows, val_rows, _manifest = apply_held_out_split(
            index_rows=rows, selected_rows=list(range(len(rows))), split_dir=split_dir,
        )
        # Real assignment comes from the spatial split, not the stale "val" column.
        assert len(train_rows) > 0
