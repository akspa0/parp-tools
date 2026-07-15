"""Focused CPU proof for Spec 103 prefab-aware corpus curation.

These tests deliberately treat ADTs as provenance pages inside one map canvas.
No training, GPU work, or large client corpus is involved.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import random
from pathlib import Path
from types import ModuleType

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import zarr

from harvester.spec103.prefab_curation import (
    COVERAGE_SCHEMA,
    EVIDENCE_SCHEMA_VERSION,
    LEDGER_SCHEMA,
    TRANSFORMS,
    PrefabCurationConfig,
    add_map_composition_features,
    assign_group_safe_splits,
    build_curation_manifest,
    canonical_alpha_signature,
    cluster_d4_signatures,
    discover_canvas_sources,
    discover_member_paths,
    discover_region_paths,
    multiscale_alpha_descriptor,
    normalize_placements,
    run_prefab_curation,
    select_representative_tiles,
    sha256_tree,
    validate_manifest_rows,
    write_typed_parquet,
)
from harvester.spec103.prefab_curation import _object_spatial_context


def _load_packager() -> ModuleType:
    path = Path(__file__).parents[2] / "scripts" / "package_spec103_runpod.py"
    spec = importlib.util.spec_from_file_location("package_spec103_runpod_tests", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _coverage_row(
    tile_id: int,
    *,
    map_name: str = "Map",
    families: tuple[str, ...] = (),
    tokens: tuple[str, ...] = (),
    selected: bool = False,
    completeness: float = 1.0,
) -> dict[str, object]:
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "build": "0.5.3",
        "map": map_name,
        "tile_id": tile_id,
        "tile_x": tile_id,
        "tile_y": 0,
        "store_row": tile_id,
        "clean_eligible": True,
        "clean_reason": "kept",
        "prefab_family_ids": list(families),
        "placement_ids": [f"placement-{tile_id}"],
        "transforms": ["identity"],
        "tileset_variant_ids": [],
        "tileset_anomaly_ids": [],
        "arrangement_classes": ["isolated"],
        "coverage_tokens": list(tokens),
        "coverage_weight": 0.0,
        "evidence_completeness": completeness,
        "selected": selected,
        "selection_reason": "representative_coverage" if selected else "not_selected",
        "representative_tile_key": "",
        "split": "excluded",
    }


def _composition_rows(offset_x: int) -> list[dict[str, object]]:
    # The first placement crosses the x=256 ADT page boundary at offset 244.
    return [
        {
            "build": "0.5.3",
            "map_name": "Canvas",
            "placement_id": f"p{index}",
            "prefab_family_id": "prefab-road",
            "layer_idx": 1,
            "bbox_xywh": (offset_x + index * 32, 80, 32, 24),
        }
        for index in range(4)
    ]


def test_d4_signature_and_fallback_grouping_are_shuffle_deterministic() -> None:
    alpha = np.zeros((9, 9), dtype=np.float32)
    alpha[1:8, 2] = 1.0
    alpha[6, 2:7] = 1.0
    alpha[2:4, 5:7] = 1.0

    transformed_signatures = []
    rows: list[dict[str, object]] = []
    for index, (_name, transform) in enumerate(TRANSFORMS):
        signature, canonical_transform, bits, _canonical = canonical_alpha_signature(
            transform(alpha), size=16, threshold=0.05
        )
        transformed_signatures.append(signature)
        rows.append(
            {
                "placement_id": f"placement-{index}",
                "prefab_family_id": "",
                "canonical_alpha_hash": signature,
                "canonical_alpha_bits": bits,
                "transform_to_canonical": canonical_transform,
            }
        )

    assert len(set(transformed_signatures)) == 1

    ordered = copy.deepcopy(rows)
    cluster_d4_signatures(ordered, hamming_radius=0)
    shuffled = copy.deepcopy(rows)
    random.Random(103).shuffle(shuffled)
    cluster_d4_signatures(shuffled, hamming_radius=0)

    ordered_assignment = {
        str(row["placement_id"]): str(row["prefab_family_id"]) for row in ordered
    }
    shuffled_assignment = {
        str(row["placement_id"]): str(row["prefab_family_id"]) for row in shuffled
    }
    assert ordered_assignment == shuffled_assignment
    assert len(set(ordered_assignment.values())) == 1


def test_membership_catalog_annotates_without_dropping_non_member_regions(
    tmp_path: Path,
) -> None:
    canvas_path = tmp_path / "canvas.zarr"
    canvas = zarr.open_group(str(canvas_path), mode="w")
    alpha = np.zeros((64, 64, 1), dtype=np.float32)
    alpha[4:20, 4:20, 0] = 1.0
    alpha[36:56, 40:52, 0] = 1.0
    canvas.create_array("alpha_256", data=alpha)
    canvas.attrs["layout"] = {"build": "0.5.3", "map_name": "WholeMap"}
    canvases = discover_canvas_sources(tmp_path)
    regions = [
        {
            "region_id": "member-region",
            "build": "0.5.3",
            "map_name": "WholeMap",
            "layer_slot": 0,
            "layer_idx": 0,
            "bbox_xywh": [4, 4, 16, 16],
            "curation_label": "accepted_candidate",
        },
        {
            "region_id": "one-off-region",
            "build": "0.5.3",
            "map_name": "WholeMap",
            "layer_slot": 0,
            "layer_idx": 0,
            "bbox_xywh": [40, 36, 12, 20],
            "curation_label": "one_off_detail",
        },
    ]
    members = [{"region_id": "member-region", "cluster_id": "prefab-known"}]

    placements = normalize_placements(
        regions,
        members,
        canvases,
        config=PrefabCurationConfig(family_hamming_radius=0),
    )

    assert {row["region_id"] for row in placements} == {
        "member-region",
        "one-off-region",
    }
    by_region = {row["region_id"]: row for row in placements}
    assert by_region["member-region"]["prefab_family_id"] == "prefab-known"
    assert by_region["one-off-region"]["family_source"] == "d4_hamming_fallback"


def test_discovery_applies_authority_per_map_instead_of_globally(tmp_path: Path) -> None:
    def write(path: Path, build: str, map_name: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pylist([{"build": build, "map_name": map_name, "region_id": path.stem}]),
            path,
        )

    preferred_a = tmp_path / "A" / "fractal_regions.parquet"
    fallback_a = tmp_path / "A" / "regions.parquet"
    fallback_b = tmp_path / "B" / "regions.parquet"
    write(preferred_a, "0.5.3", "MapA")
    write(fallback_a, "0.5.3", "MapA")
    write(fallback_b, "1.12.1", "MapB")
    preferred_member_a = tmp_path / "A" / "prefab_members.parquet"
    fallback_member_a = tmp_path / "A" / "fractal_region_members.parquet"
    fallback_member_b = tmp_path / "B" / "fractal_region_members.parquet"
    write(preferred_member_a, "0.5.3", "MapA")
    write(fallback_member_a, "0.5.3", "MapA")
    write(fallback_member_b, "1.12.1", "MapB")

    assert discover_region_paths(tmp_path) == [preferred_a, fallback_b]
    assert discover_member_paths(tmp_path) == [preferred_member_a, fallback_member_b]


def test_missing_alpha_is_typed_and_never_becomes_a_zero_mask_family(
    tmp_path: Path,
) -> None:
    canvas_path = tmp_path / "canvas.zarr"
    canvas = zarr.open_group(str(canvas_path), mode="w")
    canvas.attrs["layout"] = {"build": "0.5.3", "map_name": "MissingAlpha"}
    canvases = discover_canvas_sources(tmp_path)
    regions = [
        {
            "region_id": "missing-layer",
            "build": "0.5.3",
            "map_name": "MissingAlpha",
            "layer_slot": 0,
            "layer_idx": 0,
            "bbox_xywh": [0, 0, 16, 16],
            "curation_label": "accepted_candidate",
        }
    ]

    placements = normalize_placements(
        regions, [], canvases, config=PrefabCurationConfig(family_hamming_radius=0)
    )

    assert len(placements) == 1
    assert placements[0]["alpha_evidence_status"] == "missing_alpha_array"
    assert placements[0]["canonical_alpha_hash"] == ""
    assert placements[0]["prefab_family_id"] == ""
    assert placements[0]["family_source"] == ""
    assert "alpha:missing_alpha_array" in placements[0]["missing_evidence"]


def test_multiscale_and_cellular_features_ignore_adt_page_boundaries() -> None:
    alpha = np.zeros((24, 40), dtype=np.float32)
    alpha[2:22, 4:8] = 1.0
    alpha[16:20, 4:34] = 1.0
    occupancy_a, transitions_a = multiscale_alpha_descriptor(alpha, levels=(2, 4, 8))
    occupancy_b, transitions_b = multiscale_alpha_descriptor(alpha.copy(), levels=(2, 4, 8))
    assert occupancy_a == occupancy_b
    assert transitions_a == transitions_b

    crossing = _composition_rows(244)
    page_local = _composition_rows(64)
    config = PrefabCurationConfig(neighbor_radii=(48.0, 128.0, 512.0))
    add_map_composition_features(crossing, config=config)
    add_map_composition_features(page_local, config=config)

    assert [row["arrangement_class"] for row in crossing] == [
        row["arrangement_class"] for row in page_local
    ]
    assert [row["cellular_ring_sector_counts"] for row in crossing] == [
        row["cellular_ring_sector_counts"] for row in page_local
    ]
    assert [row["cellular_signature"] for row in crossing] == [
        row["cellular_signature"] for row in page_local
    ]


def test_object_context_filters_to_region_and_preserves_asset_position_pairing() -> None:
    tile_world = 533.33333

    def placement(asset: str, px: float, py: float) -> dict[str, object]:
        return {
            "instance_type": "m2",
            "asset_path": asset,
            "posX": px / 255.0 * tile_world,
            "posY": 0.0,
            "posZ": py / 255.0 * tile_world,
            "rotX": 0.0,
            "rotY": 0.0,
            "rotZ": 0.0,
            "scale": 1.0,
        }

    first = [
        placement("A.m2", 44, 50),
        placement("B.m2", 56, 50),
        placement("Far.m2", 220, 220),
    ]
    swapped = [
        placement("B.m2", 44, 50),
        placement("A.m2", 56, 50),
        placement("Far.m2", 220, 220),
    ]

    related_first, signature_first = _object_spatial_context(
        first,
        tile_x=0,
        tile_y=0,
        local_bbox=(40, 40, 20, 20),
        max_distance_px=16,
    )
    related_swapped, signature_swapped = _object_spatial_context(
        swapped,
        tile_x=0,
        tile_y=0,
        local_bbox=(40, 40, 20, 20),
        max_distance_px=16,
    )

    assert {row["asset_path"] for row in related_first} == {"A.m2", "B.m2"}
    assert {row["asset_path"] for row in related_swapped} == {"A.m2", "B.m2"}
    assert signature_first != signature_swapped


def test_weighted_set_cover_prefers_prefab_family_over_background_noise() -> None:
    coverage = [
        _coverage_row(
            0,
            tokens=("background:0.5.3:Map:flat", "background:0.5.3:Map:rolling"),
            completeness=1.0,
        ),
        _coverage_row(
            1,
            families=("family-a",),
            tokens=("family:family-a",),
            completeness=0.5,
        ),
    ]

    selected, uncovered = select_representative_tiles(
        coverage, config=PrefabCurationConfig(max_selected_tiles=1)
    )

    assert selected == {1}
    assert coverage[1]["selection_reason"] == "representative_coverage"
    assert uncovered == {
        "background:0.5.3:Map:flat",
        "background:0.5.3:Map:rolling",
    }


def test_family_connected_components_propagate_holdout_without_leakage() -> None:
    coverage = [
        _coverage_row(0, map_name="Holdout", families=("family-a",), selected=True),
        _coverage_row(
            1, map_name="TrainMap", families=("family-a", "family-b"), selected=True
        ),
        _coverage_row(2, map_name="TrainMap", families=("family-b",), selected=True),
        _coverage_row(3, map_name="TrainMap", families=("family-c",), selected=True),
        _coverage_row(4, map_name="Holdout", families=("family-z",), selected=True),
    ]

    audit = assign_group_safe_splits(coverage, val_maps={"Holdout"})

    assert [coverage[index]["split"] for index in (0, 1, 2)] == ["val", "val", "val"]
    assert coverage[3]["split"] == "train"
    assert coverage[4]["split"] == "val"
    assert audit == {
        "component_count": 3,
        "selected_count": 5,
        "family_count": 4,
        "family_leakage_count": 0,
        "holdout_miss_count": 0,
        "holdout_eligible_count": 2,
        "holdout_selected_count": 2,
        "split_counts": {"val": 4, "train": 1},
    }


def test_complete_map_holdout_constrains_set_cover_before_family_split() -> None:
    coverage = [
        _coverage_row(
            0,
            map_name="AaaTrain",
            families=("shared-family",),
            tokens=("family:shared-family",),
        ),
        _coverage_row(
            1,
            map_name="ZzzHoldout",
            families=("shared-family",),
            tokens=("family:shared-family",),
        ),
    ]

    selected, uncovered = select_representative_tiles(
        coverage,
        config=PrefabCurationConfig(max_selected_tiles=1),
        val_maps={"ZzzHoldout"},
    )
    audit = assign_group_safe_splits(coverage, val_maps={"ZzzHoldout"})

    assert selected == {1}
    assert not uncovered
    assert coverage[1]["selection_reason"] == "complete_map_holdout"
    assert coverage[1]["split"] == "val"
    assert audit["holdout_eligible_count"] == 1
    assert audit["holdout_selected_count"] == 1


def test_typed_evidence_schemas_and_manifest_contract(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.parquet"
    coverage_path = tmp_path / "coverage.parquet"
    write_typed_parquet(
        ledger_path,
        [
            {
                "schema_version": EVIDENCE_SCHEMA_VERSION,
                "evidence_state": "recovered_evidence",
                "build": "0.5.3",
                "map": "Map",
                "tile_id": 7,
                "tile_x": 2,
                "tile_y": 3,
                "store_row": 0,
                "chunk_keys": ["0,0"],
                "placement_id": "placement-7",
                "prefab_family_id": "family-a",
                "missing_evidence": [],
            }
        ],
        LEDGER_SCHEMA,
    )
    coverage = [
        _coverage_row(
            7,
            families=("family-a",),
            tokens=("family:family-a",),
            selected=True,
        )
    ]
    coverage[0]["split"] = "train"
    write_typed_parquet(coverage_path, coverage, COVERAGE_SCHEMA)

    assert pq.read_schema(ledger_path).equals(LEDGER_SCHEMA)
    assert pq.read_schema(coverage_path).equals(COVERAGE_SCHEMA)
    assert pq.read_table(ledger_path).to_pylist()[0]["chunk_keys"] == ["0,0"]

    manifest = build_curation_manifest(coverage, clean_rows=None)
    assert manifest[0]["keep"] is True
    assert manifest[0]["partition"] == "train"
    assert manifest[0]["prefab_family_ids"] == ["family-a"]
    validate_manifest_rows(manifest)
    with pytest.raises(ValueError, match="Duplicate curation manifest tile"):
        validate_manifest_rows([*manifest, dict(manifest[0])])


def test_runtime_manifest_rejects_duplicate_build_tile_identity() -> None:
    manifest = [
        {
            "build": "0.5.3",
            "map": "Azeroth",
            "tile_id": 140,
            "keep": True,
            "partition": "train",
            "prefab_family_ids": ["prefab-a"],
        },
        {
            "build": "0.5.3",
            "map": "Kalimdor",
            "tile_id": 140,
            "keep": True,
            "partition": "train",
            "prefab_family_ids": ["prefab-b"],
        },
    ]

    with pytest.raises(ValueError, match="Duplicate runtime curation-manifest key"):
        from harvester.spec103.prefab_curation import resolve_manifest_rows

        resolve_manifest_rows([], manifest, val_key="map", val_value="Kalimdor")


def test_runtime_manifest_rejects_prefab_family_partition_leakage() -> None:
    manifest = [
        {
            "build": "0.5.3",
            "map": "Azeroth",
            "tile_id": 1,
            "keep": True,
            "partition": "train",
            "prefab_family_ids": ["prefab-shared"],
        },
        {
            "build": "0.5.3",
            "map": "Kalimdor",
            "tile_id": 2,
            "keep": True,
            "partition": "val",
            "prefab_family_ids": ["prefab-shared"],
        },
    ]

    with pytest.raises(ValueError, match="Prefab-family partition leakage.*prefab-shared"):
        from harvester.spec103.prefab_curation import resolve_manifest_rows

        resolve_manifest_rows([], manifest, val_key="map", val_value="Kalimdor")


def test_runtime_manifest_resolves_non_contiguous_ids_to_index_positions() -> None:
    from harvester.spec103.prefab_curation import resolve_manifest_rows

    index_rows = [
        {"build": "0.5.3", "map": "Holdout", "tile_id": 900},
        {"build": "0.5.3", "map": "Unused", "tile_id": 10},
        {"build": "0.5.3", "map": "Train", "tile_id": 42},
    ]
    manifest = [
        {
            "build": "0.5.3",
            "tile_id": 42,
            "keep": True,
            "partition": "train",
            "prefab_family_ids": ["prefab-train"],
        },
        {
            "build": "0.5.3",
            "tile_id": 900,
            "keep": True,
            "partition": "val",
            "prefab_family_ids": ["prefab-val"],
        },
        {
            "build": "0.5.3",
            "tile_id": 10,
            "keep": False,
            "partition": "excluded",
            "prefab_family_ids": [],
        },
    ]

    train, val, mode = resolve_manifest_rows(
        index_rows, manifest, val_key="map", val_value="Holdout"
    )

    assert train == [2]
    assert val == [0]
    assert mode == "manifest_partition"


def test_packager_preserves_source_identity_and_all_evidence_columns(tmp_path: Path) -> None:
    packager = _load_packager()
    source_store = tmp_path / "source.zarr"
    root = zarr.open_group(str(source_store), mode="w")
    shapes = {
        "minimap_rgb": (3, 2, 2, 3),
        "height_257": (3, 3, 3),
        "normal_xyz": (3, 3, 3, 3),
        "liquid_mask": (3, 2, 2),
        "liquid_height": (3, 2, 2),
        "object_precise_mask": (3, 2, 2),
    }
    for field_index, (name, shape) in enumerate(shapes.items(), start=1):
        values = np.zeros(shape, dtype=np.float32)
        for row_index in range(3):
            values[row_index] = field_index * 10 + row_index
        root.create_array(name, data=values)

    source_index = pa.Table.from_pylist(
        [
            {
                "tile_id": 900,
                "build": "0.5.3",
                "map": "Azeroth",
                "tile_x": 31,
                "tile_y": 20,
                "custom_index_evidence": "first-source-row",
            },
            {
                "tile_id": 10,
                "build": "1.12.1",
                "map": "Unused",
                "tile_x": 2,
                "tile_y": 3,
                "custom_index_evidence": "excluded-source-row",
            },
            {
                "tile_id": 42,
                "build": "3.3.5",
                "map": "Kalimdor",
                "tile_x": 8,
                "tile_y": 9,
                "custom_index_evidence": "third-source-row",
            },
        ]
    )
    pq.write_table(source_index, source_store / "index.parquet")

    runtime_store = tmp_path / "runtime.zarr"
    report, remap = packager._subset_store(source_store, runtime_store, [900, 42])
    assert report["kept_rows"] == 2
    assert remap == {42: 0, 900: 1}

    runtime_index = pq.read_table(runtime_store / "index.parquet").to_pylist()
    assert runtime_index == [
        {
            "tile_id": 0,
            "build": "3.3.5",
            "map": "Kalimdor",
            "tile_x": 8,
            "tile_y": 9,
            "custom_index_evidence": "third-source-row",
            "source_tile_id": 42,
            "source_build": "3.3.5",
        },
        {
            "tile_id": 1,
            "build": "0.5.3",
            "map": "Azeroth",
            "tile_x": 31,
            "tile_y": 20,
            "custom_index_evidence": "first-source-row",
            "source_tile_id": 900,
            "source_build": "0.5.3",
        },
    ]
    runtime_root = zarr.open_group(str(runtime_store), mode="r")
    assert float(runtime_root["height_257"][0, 0, 0]) == 22.0
    assert float(runtime_root["height_257"][1, 0, 0]) == 20.0

    evidence_source = tmp_path / "curation-source"
    evidence_source.mkdir()
    manifest_rows = [
        {
            "tile_id": 900,
            "build": "0.5.3",
            "map": "Azeroth",
            "keep": True,
            "partition": "val",
            "prefab_family_ids": ["prefab-a"],
            "custom_manifest_evidence": "source-900",
        },
        {
            "tile_id": 10,
            "build": "1.12.1",
            "map": "Unused",
            "keep": False,
            "partition": "excluded",
            "prefab_family_ids": [],
            "custom_manifest_evidence": "source-10",
        },
        {
            "tile_id": 42,
            "build": "3.3.5",
            "map": "Kalimdor",
            "keep": True,
            "partition": "train",
            "prefab_family_ids": ["prefab-b"],
            "custom_manifest_evidence": "source-42",
        },
    ]
    manifest_path = evidence_source / "curation_manifest.parquet"
    pq.write_table(pa.Table.from_pylist(manifest_rows), manifest_path)

    for name in ("pattern_evidence_ledger.parquet", "tile_pattern_coverage.parquet"):
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {
                        "tile_id": row["tile_id"],
                        "build": row["build"],
                        "map": row["map"],
                        "prefab_family_ids": row["prefab_family_ids"],
                        "custom_evidence_column": f"evidence-{row['tile_id']}",
                    }
                    for row in manifest_rows
                ]
            ),
            evidence_source / name,
        )
    (evidence_source / "curation_summary.json").write_text(
        json.dumps({"schema": EVIDENCE_SCHEMA_VERSION, "selected": [900, 42]}),
        encoding="utf-8",
    )

    subset_manifest_dir = tmp_path / "bundle" / "manifests"
    count = packager._subset_curation_manifest(manifest_path, subset_manifest_dir, remap)
    assert count == 2
    subset_manifest = pq.read_table(
        subset_manifest_dir / "curation_manifest.parquet"
    ).to_pylist()
    assert [row["tile_id"] for row in subset_manifest] == [1, 0]
    assert [row["source_tile_id"] for row in subset_manifest] == [900, 42]
    assert [row["source_build"] for row in subset_manifest] == ["0.5.3", "3.3.5"]
    assert [row["custom_manifest_evidence"] for row in subset_manifest] == [
        "source-900",
        "source-42",
    ]

    evidence_dest = tmp_path / "bundle" / "curation_evidence"
    package_report = packager._package_curation_evidence(
        evidence_source, evidence_dest, remap
    )
    assert set(package_report["source"]) == {
        "curation_manifest.parquet",
        "pattern_evidence_ledger.parquet",
        "tile_pattern_coverage.parquet",
        "curation_summary.json",
    }
    assert set(package_report["runtime"]) == {
        "pattern_evidence_ledger.parquet",
        "tile_pattern_coverage.parquet",
    }
    assert (evidence_dest / "source" / "curation_manifest.parquet").read_bytes() == (
        evidence_source / "curation_manifest.parquet"
    ).read_bytes()

    for name in ("pattern_evidence_ledger.parquet", "tile_pattern_coverage.parquet"):
        source_rows = pq.read_table(evidence_dest / "source" / name).to_pylist()
        assert [row["tile_id"] for row in source_rows] == [900, 10, 42]
        runtime_rows = pq.read_table(evidence_dest / "runtime" / name).to_pylist()
        assert [row["tile_id"] for row in runtime_rows] == [1, 0]
        assert [row["source_tile_id"] for row in runtime_rows] == [900, 42]
        assert [row["source_build"] for row in runtime_rows] == ["0.5.3", "3.3.5"]
        assert [row["custom_evidence_column"] for row in runtime_rows] == [
            "evidence-900",
            "evidence-42",
        ]


def test_empty_region_catalog_uses_hash_bound_mapwide_canvas_fallback(
    tmp_path: Path,
) -> None:
    analysis_root = tmp_path / "analysis"
    canvas_path = analysis_root / "Fallback" / "canvas.zarr"
    canvas = zarr.open_group(str(canvas_path), mode="w")
    alpha = np.zeros((256, 256, 1), dtype=np.float32)
    alpha[48:112, 60:124, 0] = 1.0
    canvas.create_array("alpha_256", data=alpha)
    canvas.create_array("tile_id_256", data=np.full((256, 256), 7, dtype=np.int32))
    canvas.attrs["layout"] = {
        "build": "0.5.3",
        "map_name": "FallbackMap",
        "min_tile_x": 0,
        "min_tile_y": 0,
    }
    empty_regions = analysis_root / "Fallback" / "fractal_regions.parquet"
    pq.write_table(
        pa.table(
            {
                "build": pa.array([], type=pa.string()),
                "map_name": pa.array([], type=pa.string()),
                "region_id": pa.array([], type=pa.string()),
            }
        ),
        empty_regions,
    )

    store_path = tmp_path / "store.zarr"
    store = zarr.open_group(str(store_path), mode="w")
    store.create_array(
        "object_precise_mask", data=np.zeros((1, 256, 256), dtype=np.float32)
    )
    store.create_array("liquid_mask", data=np.zeros((1, 256, 256), dtype=np.float32))
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "build": "0.5.3",
                    "map": "FallbackMap",
                    "tile_id": 7,
                    "tile_x": 0,
                    "tile_y": 0,
                }
            ]
        ),
        store_path / "index.parquet",
    )

    output = tmp_path / "curated"
    summary = run_prefab_curation(
        store_paths=[store_path],
        analysis_root=analysis_root,
        output_dir=output,
        val_maps={"FallbackMap"},
        config=PrefabCurationConfig(family_hamming_radius=0),
    )

    assert summary["derived_region_count"] == 1
    assert summary["derived_region_scopes"] == [["0.5.3", "FallbackMap"]]
    assert len(summary["canvases"][0]["evidence_sha256"]) == 64
    assert len(summary["stores"][0]["evidence_sha256"]) == 64
    assert all(
        len(output_evidence["sha256"]) == 64
        for output_evidence in summary["outputs"].values()
    )
    ledger = pq.read_table(output / "pattern_evidence_ledger.parquet").to_pylist()
    assert ledger[0]["region_evidence_source"] == "spec103_canvas_segmentation_fallback_v1"
    assert len(ledger[0]["source_canvas_sha256"]) == 64
    assert len(ledger[0]["source_store_evidence_sha256"]) == 64


def test_evidence_tree_hash_is_deterministic_and_content_sensitive(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    first = evidence / "a.bin"
    second = evidence / "nested" / "b.bin"
    second.parent.mkdir()
    first.write_bytes(b"alpha")
    second.write_bytes(b"beta")

    initial = sha256_tree(evidence)
    assert sha256_tree(evidence) == initial
    second.write_bytes(b"changed")
    assert sha256_tree(evidence) != initial


def test_run_prefab_curation_tiny_cross_tile_canvas(tmp_path: Path) -> None:
    analysis_root = tmp_path / "analysis"
    canvas_path = analysis_root / "Canvas" / "canvas.zarr"
    canvas = zarr.open_group(str(canvas_path), mode="w")
    alpha = np.zeros((256, 512, 1), dtype=np.float32)
    # One asymmetric map-global placement intersects both ADT provenance pages.
    alpha[64:96, 240:246, 0] = 1.0
    alpha[84:90, 240:272, 0] = 1.0
    canvas.create_array("alpha_256", data=alpha)
    canvas.attrs["layout"] = {
        "build": "0.5.3",
        "map_name": "CanvasMap",
        "min_tile_x": 0,
        "min_tile_y": 0,
    }

    regions_path = analysis_root / "Canvas" / "fractal_regions.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "region_id": "region-cross-page",
                    "build": "0.5.3",
                    "map_name": "CanvasMap",
                    "layer_slot": 0,
                    "layer_idx": 0,
                    "bbox_xywh": [240, 64, 32, 32],
                    "tile_coverage_count": 2,
                    "curation_label": "accepted_candidate",
                    "height_mean": 12.0,
                    "height_std": 2.0,
                    "height_range": 6.0,
                    "normal_mean_xyz": [0.0, 0.0, 1.0],
                }
            ]
        ),
        regions_path,
    )

    store_path = tmp_path / "store.zarr"
    store = zarr.open_group(str(store_path), mode="w")
    object_mask = np.zeros((2, 256, 256), dtype=np.float32)
    object_mask[0, 80:88, 240:256] = 1.0
    store.create_array("object_precise_mask", data=object_mask)
    store.create_array("liquid_mask", data=np.zeros_like(object_mask))
    mcly_ids = np.full((2, 16, 16, 4), -1, dtype=np.int32)
    mcly_ids[:, :, :, 0] = 0
    mcly_mask = np.zeros((2, 16, 16, 4), dtype=np.uint8)
    mcly_mask[:, :, :, 0] = 1
    store.create_array("mcly_texture_ids", data=mcly_ids)
    store.create_array("mcly_layer_mask", data=mcly_mask)

    pq.write_table(
        pa.Table.from_pylist(
            [
                {"build": "0.5.3", "map": "CanvasMap", "tile_id": 10, "tile_x": 0, "tile_y": 0},
                {"build": "0.5.3", "map": "CanvasMap", "tile_id": 11, "tile_x": 1, "tile_y": 0},
            ]
        ),
        store_path / "index.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "tile_id": tile_id,
                    "source_adt_path": f"World/Maps/CanvasMap/CanvasMap_{tile_x}_0.adt",
                    "decoded_metadata_json": json.dumps(
                        {"mcly_texture_names": ["tileset/desert/road.blp"]}
                    ),
                }
                for tile_id, tile_x in ((10, 0), (11, 1))
            ]
        ),
        store_path / "decoded_metadata.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "tile_id": 10,
                    "instance_idx": 0,
                    "asset_path": "World/Generic/PassiveDoodads/RoadMarker.m2",
                    "posX": 0.0,
                    "posY": 0.0,
                    "posZ": 0.0,
                    "rotX": 0.0,
                    "rotY": 0.0,
                    "rotZ": 0.0,
                }
            ]
        ),
        store_path / "placements.parquet",
    )

    output = tmp_path / "curated"
    summary = run_prefab_curation(
        store_paths=[store_path],
        analysis_root=analysis_root,
        output_dir=output,
        val_maps={"CanvasMap"},
        config=PrefabCurationConfig(family_hamming_radius=0),
    )

    assert summary["placement_count"] == 1
    assert summary["ledger_row_count"] == 2
    assert summary["selected_tile_count"] == 2
    assert summary["split_audit"]["family_leakage_count"] == 0

    ledger = pq.read_table(output / "pattern_evidence_ledger.parquet").to_pylist()
    assert pq.read_schema(output / "pattern_evidence_ledger.parquet").equals(LEDGER_SCHEMA)
    assert {row["tile_id"] for row in ledger} == {10, 11}
    assert all(row["crosses_adt_boundary"] for row in ledger)
    assert {tuple(row["chunk_keys"]) for row in ledger} == {(("15,4", "15,5")), (("0,4", "0,5"))}
    assert {Path(row["source_adt_path"]).name for row in ledger} == {
        "CanvasMap_0_0.adt",
        "CanvasMap_1_0.adt",
    }
    assert all(len(row["source_canvas_sha256"]) == 64 for row in ledger)
    assert all(len(row["source_store_evidence_sha256"]) == 64 for row in ledger)

    manifest = pq.read_table(output / "curation_manifest.parquet").to_pylist()
    assert len(manifest) == 2
    kept = [row for row in manifest if row["keep"]]
    assert len(kept) == 2
    assert all(row["partition"] == "val" for row in kept)
    assert all(row["schema_version"] == EVIDENCE_SCHEMA_VERSION for row in manifest)
