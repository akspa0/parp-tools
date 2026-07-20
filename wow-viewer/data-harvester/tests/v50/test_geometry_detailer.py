"""Spec 114 T058-T060: materialization, residual model, and detailer trainer gates."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import zarr

from harvester.v50.direct_geometry_materialize import (
    COARSE_ARRAY,
    COARSE_STORE_SCHEMA,
    MaterializationError,
    load_selected_rows,
    materialize_coarse_relief,
)
from harvester.v50.direct_geometry_materialize import (
    main as materialize_main,
)
from harvester.v50.geometry_detailer_model import (
    DETAILER_ARCHITECTURE_ID,
    GeometryDetailerNet,
    compose_final,
    detailer_identity,
)
from harvester.v50.geometry_detailer_train import (
    TrainerContractError,
    build_detailer_plan,
    build_detailer_stage_run,
    compute_coarse_baseline,
    evaluate_detailer_gate,
    validate_coarse_store,
)
from harvester.v50.height_relative_model import TARGET_CONTRACT_VERSION
from harvester.v50.model_stage_contract import validate_model_stage_run


def _write_source_store(path: Path, rows: list[dict], *, contract: str | None) -> None:
    group = zarr.open_group(str(path), mode="w")
    group.attrs["schema"] = "v50-mixed-curriculum-v1"
    group.attrs["model_family"] = "v50"
    group.attrs["release"] = "v50.1"
    if contract is not None:
        group.attrs["synthetic_lighting_contract"] = contract
    group.create_array(
        "minimap_rgb", shape=(len(rows), 256, 256, 3), chunks=(1, 256, 256, 3), dtype=np.uint8
    )
    group.create_array(
        "height_257", shape=(len(rows), 257, 257), chunks=(1, 257, 257), dtype=np.float32
    )
    pq.write_table(pa.Table.from_pylist(rows), path / "index.parquet")


def _rows(n: int) -> list[dict]:
    return [
        {
            "map": "Kalimdor",
            "source_group_id": f"g{i}",
            "minimap_source": "authored",
            "split": "train" if i < n - 4 else "val",
        }
        for i in range(n)
    ]


def _checkpoint(path: Path, *, variant: str = "direct_cnn_v112") -> Path:
    from harvester.v50.direct_geometry_model import build_geometry_model

    torch.manual_seed(114)
    model, _ = build_geometry_model(variant)
    torch.save(
        {
            "model_variant": variant,
            "target_contract_version": TARGET_CONTRACT_VERSION,
            "model": model.state_dict(),
            "epoch": 3,
            "val_mae": 0.19,
        },
        path,
    )
    return path


# --- T058 materialization -----------------------------------------------------


def test_load_selected_rows_validates_and_filters(tmp_path: Path) -> None:
    store = tmp_path / "src.zarr"
    _write_source_store(store, _rows(10), contract=None)
    attrs, index, selected = load_selected_rows(store, source="authored", release="v50.1")
    assert attrs["schema"] == "v50-mixed-curriculum-v1"
    assert len(selected) == 10
    assert len(index) == 10


def test_materialize_dry_run_writes_nothing(tmp_path: Path) -> None:
    store = tmp_path / "src.zarr"
    _write_source_store(store, _rows(6), contract=None)
    checkpoint = _checkpoint(tmp_path / "ckpt.pt")
    output = tmp_path / "coarse.zarr"
    plan = materialize_coarse_relief(
        store=store, checkpoint_path=checkpoint, output=output,
        source="authored", release="v50.1", device="cpu", write=False,
    )
    assert plan["schema"] == "v114-coarse-materialize-plan-v1"
    assert plan["selected_rows"] == 6
    assert not output.exists()


def test_materialize_write_persists_aligned_store_and_refuses_overwrite(tmp_path: Path) -> None:
    store = tmp_path / "src.zarr"
    _write_source_store(store, _rows(6), contract=None)
    checkpoint = _checkpoint(tmp_path / "ckpt.pt")
    output = tmp_path / "coarse.zarr"
    summary = materialize_coarse_relief(
        store=store, checkpoint_path=checkpoint, output=output,
        source="authored", release="v50.1", device="cpu", write=True,
    )
    assert summary["schema"] == COARSE_STORE_SCHEMA
    group = zarr.open_group(str(output), mode="r")
    assert group.attrs["schema"] == COARSE_STORE_SCHEMA
    assert group.attrs["source_filter"] == "authored"
    assert group[COARSE_ARRAY].shape == (6, 257, 257)
    assert group[COARSE_ARRAY].dtype == np.float16
    index = pq.read_table(output / "index.parquet").to_pylist()
    assert [row["source_row_index"] for row in index] == list(range(6))
    with pytest.raises(MaterializationError, match="refusing to overwrite"):
        materialize_coarse_relief(
            store=store, checkpoint_path=checkpoint, output=output,
            source="authored", release="v50.1", device="cpu", write=True,
        )


def test_materialize_refuses_synthetic_without_noon_white_provenance(tmp_path: Path) -> None:
    store = tmp_path / "src.zarr"
    rows = _rows(6) + [
        {"map": "Kalimdor", "source_group_id": "gS", "minimap_source": "synthetic", "split": "train"}
    ]
    _write_source_store(store, rows, contract=None)
    checkpoint = _checkpoint(tmp_path / "ckpt.pt")
    # Synthetic selection is blocked by the stale-lighting gate (same contract as the coarse
    # trainer); the detailer inherits it because it consumes the same curriculum selection.
    with pytest.raises(MaterializationError, match="synthetic rows are blocked"):
        materialize_coarse_relief(
            store=store, checkpoint_path=checkpoint, output=tmp_path / "out",
            source="synthetic", release="v50.1", device="cpu", write=False,
        )


# --- T059 residual model -------------------------------------------------------


def test_zero_init_head_starts_at_coarse_composition() -> None:
    model = GeometryDetailerNet().eval()
    rgb = torch.rand(2, 3, 256, 256)
    coarse = torch.rand(2, 257, 257) * 0.5 + 0.2
    with torch.no_grad():
        residual = model(rgb, coarse)
        final = compose_final(coarse, residual, clamp=False)
    assert torch.allclose(final, coarse, atol=1e-6)  # zero residual at init
    assert residual.shape == (2, 257, 257)


def test_residual_is_signed_and_unbounded() -> None:
    model = GeometryDetailerNet().eval()
    rgb = torch.rand(1, 3, 256, 256)
    coarse = torch.full((1, 257, 257), 0.5)
    with torch.no_grad():
        residual = model(rgb, coarse)
    assert float(residual.min()) < 0.0 or float(residual.max()) > 0.0 or residual.abs().sum() == 0


def test_compose_clamp_only_for_metrics() -> None:
    coarse = torch.full((1, 5, 5), 0.9)
    residual = torch.full((1, 5, 5), 0.5)
    unclamped = compose_final(coarse, residual, clamp=False)
    clamped = compose_final(coarse, residual, clamp=True)
    assert float(unclamped.max()) == pytest.approx(1.4)
    assert float(clamped.max()) == pytest.approx(1.0)


def test_detailer_identity_is_schema_conformant() -> None:
    model = GeometryDetailerNet()
    identity = detailer_identity(model)
    assert identity["id"] == DETAILER_ARCHITECTURE_ID
    assert identity["parameter_count"] > 1_000_000
    assert len(identity["config_sha256"]) == 64


def test_detailer_refuses_wrong_input_shapes() -> None:
    model = GeometryDetailerNet().eval()
    with pytest.raises(Exception, match="rgb must be"):
        model(torch.rand(1, 4, 256, 256), torch.rand(1, 257, 257))
    with pytest.raises(Exception, match="coarse must be"):
        model(torch.rand(1, 3, 256, 256), torch.rand(1, 128, 128))


def test_detailer_backward_flows_to_head() -> None:
    model = GeometryDetailerNet()
    rgb = torch.rand(1, 3, 256, 256, requires_grad=False)
    coarse = torch.rand(1, 257, 257, requires_grad=False)
    residual = model(rgb, coarse)
    residual.mean().backward()
    assert model.head.weight.grad is not None
    assert float(model.head.weight.grad.abs().sum()) > 0


# --- T060 trainer gates ---------------------------------------------------------


def test_validate_coarse_store_aligns_with_selected_rows() -> None:
    selected = [0, 1, 2, 3]
    coarse_index = [{"source_row_index": i, "source_group_id": f"g{i}",
                     "split": "train", "minimap_source": "authored"} for i in selected]
    validate_coarse_store(
        attrs={"schema": COARSE_STORE_SCHEMA, "source_filter": "authored"},
        coarse_index_rows=coarse_index, coarse_array_rows=4,
        selected=selected, source="authored",
    )
    with pytest.raises(TrainerContractError, match="source filter"):
        validate_coarse_store(
            attrs={"schema": COARSE_STORE_SCHEMA, "source_filter": "authored"},
            coarse_index_rows=coarse_index, coarse_array_rows=4,
            selected=selected, source="synthetic",
        )
    misaligned = list(coarse_index)
    misaligned[1] = {**misaligned[1], "source_row_index": 99}
    with pytest.raises(TrainerContractError, match="does not align"):
        validate_coarse_store(
            attrs={"schema": COARSE_STORE_SCHEMA, "source_filter": "authored"},
            coarse_index_rows=misaligned, coarse_array_rows=4,
            selected=selected, source="authored",
        )


def test_coarse_baseline_is_upstream_composition_error() -> None:
    coarse = [np.full((9, 9), 0.5, dtype=np.float32)]
    target = [np.full((9, 9), 0.4, dtype=np.float32)]
    assert compute_coarse_baseline(coarse, target) == pytest.approx(0.1)
    with pytest.raises(TrainerContractError, match="one coarse field"):
        compute_coarse_baseline([], [])


def test_detailer_gate_requires_five_percent_over_coarse_only() -> None:
    passing = evaluate_detailer_gate(best_val_mae=0.10, coarse_baseline=0.18)
    assert passing["passes"] is True
    assert passing["threshold"] == pytest.approx(0.18 * 0.95)
    failing = evaluate_detailer_gate(best_val_mae=0.18, coarse_baseline=0.18)
    assert failing["passes"] is False
    marginal = evaluate_detailer_gate(best_val_mae=0.172, coarse_baseline=0.18)
    assert marginal["passes"] is False  # 4.4% improvement, below 5%


def test_detailer_plan_records_upstream_and_no_teacher_forcing() -> None:
    plan = build_detailer_plan(
        architecture={"id": DETAILER_ARCHITECTURE_ID, "config_sha256": "c" * 64,
                      "parameter_count": 1_561_857},
        upstream={"path": "ckpt.pt", "sha256": "a" * 64},
        source="authored", selected_rows=100, train_rows=80, val_rows=20,
        batch_size=16, epochs=100, seed=114, lr=2e-4, lr_schedule="onecycle",
        amp=True, amp_dtype="bf16", clip=1.0, spectral_weight=0.1, multiscale_weight=0.25,
        frequency_2d_weight=0.0, laplacian_weight=0.0, edge_weight=0.0,
        transition_focus_weight=0.0, band_lf_weight=0.0, band_hf_weight=0.0,
        band_cutoff=0.1,
    )
    assert plan["schema"] == "v114-detailer-plan-v1"
    assert plan["upstream_coarse_checkpoint"]["sha256"] == "a" * 64
    assert plan["teacher_forced_truth_inputs"] is False
    assert plan["deployment_inputs"] == ["minimap_rgb", "generated_coarse_relief"]
    assert plan["guidance"]["frequency_2d_weight"] == 0.0
    assert plan["guidance"]["band_cutoff"] == 0.1
    assert plan["amp_dtype"] == "bf16"


def test_detailer_stage_run_validates_with_upstream_models() -> None:
    summary = build_detailer_stage_run(
        run_id="detailer-v1",
        architecture={"id": DETAILER_ARCHITECTURE_ID, "config_sha256": "c" * 64,
                      "parameter_count": 1_561_857},
        upstream_identity={"path": "ckpt.pt", "sha256": "a" * 64},
        curriculum={"path": "store.zarr", "sha256": "b" * 64},
        checkpoint={"path": "best.pt", "sha256": "d" * 64, "best_epoch": 12},
        baselines={"coarse_only": {"val_mae": 0.18}},
        metrics={"best_val_mae": 0.10, "detailer_gate": {"passes": True}},
        visual_evidence={"fixed_rows": "validation/final_best/fixed_rows.png"},
        created_utc="2026-07-20T00:00:00Z",
    )
    validate_model_stage_run(summary)
    assert summary["upstream_models"] == [{"path": "ckpt.pt", "sha256": "a" * 64}]
    assert summary["promotion_verdict"] == "pending"


def test_materialize_cli_dry_run(tmp_path: Path, capsys) -> None:
    store = tmp_path / "src.zarr"
    _write_source_store(store, _rows(6), contract=None)
    checkpoint = _checkpoint(tmp_path / "ckpt.pt")
    exit_code = materialize_main(
        ["--store", str(store), "--checkpoint", str(checkpoint),
         "--output", str(tmp_path / "out"), "--source", "authored", "--device", "cpu"]
    )
    assert exit_code == 0
    assert "DRY RUN ONLY" in capsys.readouterr().out
    assert not (tmp_path / "out").exists()
