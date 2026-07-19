"""Spec 112 T020: the trainer's contract gates must be enforceable without CUDA — map restriction
(FR-011), the in-run tile-mean baseline (SC-004), and the run summary incl. the epoch-1-best
structural-failure flag (execution contract §3)."""

from __future__ import annotations

import numpy as np
import pytest

from harvester.v50.height_relative_train import (
    TrainerContractError,
    build_run_summary,
    compute_tile_mean_baseline,
    validate_curriculum_contract,
    validate_curriculum_maps,
)


def _valid_rows() -> list[dict]:
    return [
        {
            "map": "Kalimdor",
            "source_group_id": "real:0_5_3_3368:Kalimdor:1",
            "minimap_source": source,
            "split": "train",
        }
        for source in ("authored", "synthetic")
    ]


def test_curriculum_schema_gate_refuses_wrong_schema_missing_arrays_and_row_mismatch():
    rows = _valid_rows()
    with pytest.raises(TrainerContractError, match="schema"):
        validate_curriculum_contract(
            attrs={"schema": "legacy-curriculum"},
            array_lengths={"minimap_rgb": 2, "height_257": 2},
            index_rows=rows,
        )
    with pytest.raises(TrainerContractError, match="missing required arrays"):
        validate_curriculum_contract(
            attrs={"schema": "v50-mixed-curriculum-v1"},
            array_lengths={"minimap_rgb": 2},
            index_rows=rows,
        )
    with pytest.raises(TrainerContractError, match="row-aligned"):
        validate_curriculum_contract(
            attrs={"schema": "v50-mixed-curriculum-v1"},
            array_lengths={"minimap_rgb": 2, "height_257": 1},
            index_rows=rows,
        )


def test_curriculum_schema_gate_rechecks_group_leak_safety():
    rows = _valid_rows()
    rows[1]["split"] = "val"
    with pytest.raises(TrainerContractError, match="leaks source groups"):
        validate_curriculum_contract(
            attrs={"schema": "v50-mixed-curriculum-v1"},
            array_lengths={"minimap_rgb": 2, "height_257": 2},
            index_rows=rows,
        )


def test_out_of_scope_maps_are_refused():
    rows = [{"map": "Kalimdor"}, {"map": "Azeroth"}, {"map": "PVPZone02"}]
    with pytest.raises(TrainerContractError, match="PVPZone02"):
        validate_curriculum_maps(rows)

    validate_curriculum_maps([{"map": "Kalimdor"}, {"map": "Azeroth"}])  # must not raise


def test_tile_mean_baseline_is_the_per_tile_constant_predictor_error():
    flat = np.full((9, 9), 0.5, dtype=np.float32)          # baseline error 0
    ramp = np.linspace(0, 1, 81, dtype=np.float32).reshape(9, 9)  # known MAE vs its mean (0.25 for uniform ramp)

    assert compute_tile_mean_baseline([flat]) == pytest.approx(0.0)
    assert compute_tile_mean_baseline([ramp]) == pytest.approx(0.25, abs=0.01)
    with pytest.raises(TrainerContractError, match="zero validation"):
        compute_tile_mean_baseline([])


def test_run_summary_records_contract_and_flags_epoch1_best_as_structural_failure():
    # A mocked two-epoch loop is enough to prove the summary contract without touching CUDA.
    per_epoch = [{"epoch": 1, "val_mae": 0.05}, {"epoch": 2, "val_mae": 0.09}]

    summary = build_run_summary(
        identity="sha256:" + "a" * 64, split_mode="within_map_stratified:0.15",
        per_epoch=per_epoch, baseline_mae=0.08, train_rows=100, val_rows=20,
        source_counts={"authored": 60, "synthetic": 60},
    )

    assert summary["schema"] == "v112-height-run-v1"
    assert summary["target_contract_version"] == "v112.1"
    assert summary["best_epoch"] == 1
    assert summary["structural_failure_epoch1_best"] is True  # the rejected lane's signature
    assert summary["beats_baseline"] is True  # numerically better, yet still flagged structural


def test_run_summary_healthy_case():
    per_epoch = [{"epoch": e, "val_mae": 0.2 / e} for e in range(1, 6)]

    summary = build_run_summary(
        identity="sha256:" + "b" * 64, split_mode="within_map_stratified:0.15",
        per_epoch=per_epoch, baseline_mae=0.1, train_rows=100, val_rows=20,
        source_counts={"authored": 60, "synthetic": 60},
    )

    assert summary["best_epoch"] == 5
    assert summary["structural_failure_epoch1_best"] is False
    assert summary["beats_baseline"] is True
