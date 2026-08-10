from __future__ import annotations

import json
from pathlib import Path

import pytest

from harvester.v60.rgb_method_benchmark import (
    RGB_METHOD_BENCHMARK_SCHEMA,
    RGBMethodBenchmarkError,
    build_rgb_method_benchmark_plan,
)
from harvester.v60.terrain_method_translation import FORBIDDEN_INPUTS


def _write_authored(root: Path, *, inference_target_reads: list[str] | None = None) -> Path:
    root.mkdir()
    rows = []
    for index, map_name in enumerate(("Kalimdor", "Azeroth")):
        rows.append(
            {
                "row_id": f"real_minimap_diagnostic-authored-alpha-{map_name}-{index:02d}-24",
                "source_kind": "real_minimap_diagnostic",
                "source_group_id": f"real:alpha:{map_name}:{index}",
                "family": f"alpha:{map_name}",
                "complexity_bucket": "real_observation",
                "split": "validation" if map_name == "Azeroth" else "train",
                "map": map_name,
                "tile_x": index,
                "tile_y": 24,
                "forbidden_signals": [],
                "observation_status": "accepted",
                "observation_provenance": {
                    "source_signal": "minimap_rgb",
                    "inference_target_reads": inference_target_reads or [],
                },
            }
        )
    manifest = {
        "schema": "v7-clean-signal-corpus-v1",
        "row_count": len(rows),
        "source_schema": "v7-clean-signal-real-minimap-rgb-v1",
        "source_filter": "authored",
        "input_contract": "minimap_rgb_to_raw_luma_diagnostic_v1",
        "rows": rows,
    }
    (root / "clean_signal_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _write_object_sieve(root: Path) -> Path:
    root.mkdir()
    rows = []
    for index, regime in enumerate(("none", "boundary_crossing")):
        rows.append(
            {
                "row_id": f"ridge-v00-libobj-{regime}-s00",
                "source_kind": "real_v50_object_library_composite",
                "source_control_row_id": "ridge-v00",
                "terrain_control_family": "ridge",
                "split": "validation" if regime == "boundary_crossing" else "train",
                "placement_regime": regime,
                "input": "objectified_terrain_shadow_256",
                "targets": ["terrain_shadow_256", "object_contamination_mask_256"],
                "object_instance_count": index,
            }
        )
    manifest = {
        "schema": "v60-object-library-sieve-v1",
        "source_policy": "real_v50_object_library_over_project_control_terrain",
        "source_control_schema": "v60-control-corpus-v1",
        "row_count": len(rows),
        "rows": rows,
    }
    (root / "object_library_sieve_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_authored_plan_only_marks_no_mask_runtime_eligible(tmp_path: Path) -> None:
    plan = build_rgb_method_benchmark_plan(
        source="authored",
        authored_corpus=_write_authored(tmp_path / "authored"),
    )

    assert plan["schema"] == RGB_METHOD_BENCHMARK_SCHEMA
    assert plan["valid"] is True
    assert plan["runtime_eligible_conditions"] == ["no_mask"]
    conditions = {condition["condition_id"]: condition for condition in plan["conditions"]}
    assert conditions["no_mask"]["eligible_row_count"] == 2
    assert conditions["predicted_mask"]["status"] == "blocked"
    assert conditions["withheld_mask"]["status"] == "blocked"
    assert plan["baselines"][0]["baseline_id"] == "tile_mean_height_v1"


def test_object_sieve_is_explicit_control_only_and_withheld_mask_is_available(tmp_path: Path) -> None:
    plan = build_rgb_method_benchmark_plan(
        source="object_library",
        object_library_sieve=_write_object_sieve(tmp_path / "sieve"),
    )

    report = plan["source_reports"][0]
    assert plan["valid"] is True
    assert report["runtime_compatible"] is False
    assert report["source_modality"] == "synthetic_luma_object_control"
    assert report["available_conditions"]["withheld_mask"] == 2
    assert plan["runtime_eligible_conditions"] == []
    assert set(report["evaluation_only_arrays"]) >= {"terrain_shadow_256", "object_contamination_mask_256"}


def test_both_sources_keep_separate_split_identities(tmp_path: Path) -> None:
    plan = build_rgb_method_benchmark_plan(
        source="both",
        authored_corpus=_write_authored(tmp_path / "authored"),
        object_library_sieve=_write_object_sieve(tmp_path / "sieve"),
    )

    reports = {report["source_key"]: report for report in plan["source_reports"]}
    assert set(reports) == {"authored", "object_library"}
    assert reports["authored"]["split"]["split_identity_sha256"] != reports["object_library"]["split"]["split_identity_sha256"]
    assert plan["runtime_eligible_conditions"] == ["no_mask"]


def test_authored_target_read_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(RGBMethodBenchmarkError, match="inference_target_reads"):
        build_rgb_method_benchmark_plan(
            source="authored",
            authored_corpus=_write_authored(
                tmp_path / "authored",
                inference_target_reads=["height_257"],
            ),
        )


def test_model_input_arrays_have_no_forbidden_names(tmp_path: Path) -> None:
    plan = build_rgb_method_benchmark_plan(
        source="both",
        authored_corpus=_write_authored(tmp_path / "authored"),
        object_library_sieve=_write_object_sieve(tmp_path / "sieve"),
    )

    for report in plan["source_reports"]:
        assert not set(report["model_input_arrays"]) & set(FORBIDDEN_INPUTS)
