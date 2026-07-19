"""Immutable multi-family curriculum index for Spec 114 universal relief.

The builder references source Zarr stores instead of copying their potentially large arrays. Every
source is content-identified, every row records exact versus teacher-pseudo authority, and at least
one complete visual family is held out. It refuses a random row-only validation contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import zarr

UNIVERSAL_CURRICULUM_SCHEMA = "v50-universal-relief-curriculum-v1"
MIN_VISUAL_FAMILIES = 5
V50_VISUAL_FAMILY = "wow_minimap"
V50_REQUIRED_ARRAYS = frozenset({"minimap_rgb", "height_257"})


class UniversalCurriculumError(ValueError):
    """Raised when a universal curriculum would violate identity or family gates."""


@dataclass(frozen=True)
class TeacherStoreBinding:
    visual_family: str
    path: Path


@dataclass(frozen=True)
class UniversalCurriculumPlan:
    v50_store: Path
    v50_source: str
    teacher_stores: tuple[TeacherStoreBinding, ...]
    holdout_families: tuple[str, ...]
    output: Path
    rows: tuple[dict[str, Any], ...]
    summary: dict[str, Any]


def source_store_identity(store_path: Path) -> str:
    digest = hashlib.sha256()
    found = False
    for candidate in (
        store_path / "index.parquet",
        store_path / "summary.json",
        store_path.with_suffix(".summary.json"),
    ):
        if candidate.is_file():
            digest.update(candidate.name.encode())
            digest.update(candidate.read_bytes())
            found = True
    if not found:
        group = zarr.open_group(str(store_path), mode="r")
        digest.update(json.dumps(dict(group.attrs), sort_keys=True).encode())
    return digest.hexdigest()


def _row_id(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _load_v50_rows(store_path: Path, source_filter: str) -> tuple[list[dict[str, Any]], str]:
    if not store_path.is_dir():
        raise UniversalCurriculumError(f"v50 curriculum store does not exist: {store_path}")
    index_path = store_path / "index.parquet"
    if not index_path.is_file():
        raise UniversalCurriculumError(f"v50 curriculum index is missing: {index_path}")
    group = zarr.open_group(str(store_path), mode="r")
    missing_arrays = sorted(V50_REQUIRED_ARRAYS - set(group.array_keys()))
    if missing_arrays:
        raise UniversalCurriculumError(f"v50 curriculum missing arrays: {missing_arrays}")
    table = pq.read_table(index_path)
    required_columns = {"source_group_id", "minimap_source", "split"}
    missing_columns = sorted(required_columns - set(table.column_names))
    if missing_columns:
        raise UniversalCurriculumError(f"v50 curriculum index missing columns: {missing_columns}")
    if source_filter not in {"authored", "synthetic", "all"}:
        raise UniversalCurriculumError("v50_source must be authored, synthetic, or all")
    if source_filter != "authored":
        lighting = group.attrs.get("synthetic_lighting_contract")
        if lighting != "NoonWhiteGlobal":
            raise UniversalCurriculumError(
                "synthetic v50 rows require synthetic_lighting_contract=NoonWhiteGlobal"
            )

    identity = source_store_identity(store_path)
    rows = []
    for source_index, source in enumerate(table.to_pylist()):
        minimap_source = str(source["minimap_source"])
        if source_filter != "all" and minimap_source != source_filter:
            continue
        original_split = str(source["split"])
        split = "validation" if original_split in {"val", "validation"} else original_split
        if split not in {"train", "validation", "test"}:
            raise UniversalCurriculumError(f"unsupported v50 split {original_split!r}")
        source_group_id = f"v50:{identity}:{source['source_group_id']}"
        row_payload = {
            "source_identity": identity,
            "source_index": source_index,
            "source_group_id": source_group_id,
            "input_origin": f"v50_{minimap_source}",
        }
        rows.append(
            {
                "row_id": _row_id(row_payload),
                "source_group_id": source_group_id,
                "source_content_id": "",
                "visual_family": V50_VISUAL_FAMILY,
                "split": split,
                "input_origin": f"v50_{minimap_source}",
                "target_authority": "exact_numeric",
                "target_signal": "relative_relief",
                "source_store": str(store_path),
                "source_store_sha256": identity,
                "source_row_key": "",
                "source_index": source_index,
                "input_path": "",
                "input_sha256": "",
                "width": 256,
                "height": 256,
                "mode": "RGB",
                "teacher_revision": "",
                "teacher_weights_sha256": "",
            }
        )
    if not rows:
        raise UniversalCurriculumError(f"no v50 rows selected by --v50-source {source_filter}")
    return rows, identity


def _load_teacher_rows(
    binding: TeacherStoreBinding,
    *,
    holdout_families: set[str],
) -> tuple[list[dict[str, Any]], str]:
    store_path = binding.path.resolve()
    if not store_path.is_dir():
        raise UniversalCurriculumError(f"teacher store does not exist: {store_path}")
    group = zarr.open_group(str(store_path), mode="r")
    attrs = dict(group.attrs)
    recorded_family = attrs.get("visual_family")
    if recorded_family != binding.visual_family:
        raise UniversalCurriculumError(
            f"teacher family mismatch for {store_path}: binding={binding.visual_family!r}, "
            f"store={recorded_family!r}"
        )
    if attrs.get("target_authority") != "teacher_pseudo":
        raise UniversalCurriculumError(f"teacher store lacks teacher_pseudo authority: {store_path}")
    teacher = attrs.get("teacher")
    if not isinstance(teacher, dict):
        raise UniversalCurriculumError(f"teacher identity is missing from {store_path}")
    if "depthanything" in str(teacher.get("hub_id", "")).lower().replace("-", ""):
        raise UniversalCurriculumError("DepthAnything-family teacher store is forbidden")
    if "rows" not in group:
        raise UniversalCurriculumError(f"teacher store has no rows group: {store_path}")

    identity = source_store_identity(store_path)
    split = "compatibility" if binding.visual_family in holdout_families else "train"
    rows = []
    row_group = group["rows"]
    for source_row_key in sorted(row_group.group_keys()):
        source = dict(row_group[source_row_key].attrs)
        required = {"source_id", "source_sha256", "relative_path", "width", "height", "mode"}
        missing = sorted(required - set(source))
        if missing:
            raise UniversalCurriculumError(
                f"teacher row {source_row_key} in {store_path} missing {missing}"
            )
        source_group_id = f"image:{source['source_id']}"
        input_path = Path(str(attrs["input_root"])) / str(source["relative_path"])
        if not input_path.is_file():
            raise UniversalCurriculumError(f"teacher source image is missing: {input_path}")
        observed_source_sha256 = _sha256_file(input_path)
        if observed_source_sha256 != source["source_sha256"]:
            raise UniversalCurriculumError(
                f"teacher source image drift for {input_path}: expected {source['source_sha256']}, "
                f"observed {observed_source_sha256}"
            )
        row_payload = {
            "source_identity": identity,
            "source_row_key": source_row_key,
            "source_group_id": source_group_id,
        }
        rows.append(
            {
                "row_id": _row_id(row_payload),
                "source_group_id": source_group_id,
                "source_content_id": str(source["source_sha256"]),
                "visual_family": binding.visual_family,
                "split": split,
                "input_origin": "teacher_labeled_image",
                "target_authority": "teacher_pseudo",
                "target_signal": "relative_relief",
                "source_store": str(store_path),
                "source_store_sha256": identity,
                "source_row_key": source_row_key,
                "source_index": -1,
                "input_path": str(input_path),
                "input_sha256": str(source["source_sha256"]),
                "width": int(source["width"]),
                "height": int(source["height"]),
                "mode": str(source["mode"]),
                "teacher_revision": str(teacher.get("revision", "")),
                "teacher_weights_sha256": str(teacher.get("weights_sha256", "")),
            }
        )
    if not rows:
        raise UniversalCurriculumError(f"teacher store is empty: {store_path}")
    return rows, identity


def _audit_rows(
    rows: list[dict[str, Any]],
    *,
    holdout_families: set[str],
) -> tuple[int, int]:
    groups: dict[str, set[str]] = defaultdict(set)
    family_splits: dict[str, set[str]] = defaultdict(set)
    teacher_content_families: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        groups[str(row["source_group_id"])].add(str(row["split"]))
        family_splits[str(row["visual_family"])].add(str(row["split"]))
        content_id = str(row["source_content_id"])
        if content_id:
            teacher_content_families[content_id].add(str(row["visual_family"]))
    duplicated_content = {
        content: families for content, families in teacher_content_families.items() if len(families) > 1
    }
    if duplicated_content:
        first_content, first_families = next(iter(duplicated_content.items()))
        raise UniversalCurriculumError(
            f"same source content {first_content} was relabeled into families {sorted(first_families)}"
        )
    leaking_groups = {group: splits for group, splits in groups.items() if len(splits) > 1}
    if leaking_groups:
        first_group, first_splits = next(iter(leaking_groups.items()))
        raise UniversalCurriculumError(
            f"source-group leakage: {first_group} appears in {sorted(first_splits)}"
        )
    for family in holdout_families:
        splits = family_splits.get(family, set())
        if splits != {"compatibility"}:
            raise UniversalCurriculumError(
                f"held-out family {family!r} must appear only in compatibility, got {sorted(splits)}"
            )
    return 0, 0


def build_universal_curriculum_plan(
    *,
    v50_store: str | Path,
    teacher_stores: Sequence[TeacherStoreBinding],
    holdout_families: Sequence[str],
    output: str | Path,
    v50_source: str = "authored",
) -> UniversalCurriculumPlan:
    output_path = Path(output).resolve()
    if output_path.suffix.lower() != ".zarr":
        raise UniversalCurriculumError("universal curriculum output must end with .zarr")
    if output_path.exists():
        raise FileExistsError(f"universal curriculum output already exists: {output_path}")
    bindings = tuple(
        TeacherStoreBinding(binding.visual_family, binding.path.resolve())
        for binding in teacher_stores
    )
    family_names = [V50_VISUAL_FAMILY, *(binding.visual_family for binding in bindings)]
    if len(set(family_names)) != len(family_names):
        raise UniversalCurriculumError("every teacher store must have a unique visual family")
    if len(family_names) < MIN_VISUAL_FAMILIES:
        raise UniversalCurriculumError(
            f"universal curriculum requires at least {MIN_VISUAL_FAMILIES} visual families, "
            f"got {len(family_names)}"
        )
    holdouts = set(holdout_families)
    if not holdouts:
        raise UniversalCurriculumError("at least one whole visual family must be held out")
    unknown_holdouts = sorted(holdouts - set(family_names))
    if unknown_holdouts:
        raise UniversalCurriculumError(f"unknown held-out visual families: {unknown_holdouts}")

    v50_path = Path(v50_store).resolve()
    rows, v50_identity = _load_v50_rows(v50_path, v50_source)
    source_inputs = [{"path": str(v50_path), "sha256": v50_identity}]
    for binding in bindings:
        teacher_rows, identity = _load_teacher_rows(binding, holdout_families=holdouts)
        rows.extend(teacher_rows)
        source_inputs.append({"path": str(binding.path), "sha256": identity})

    group_leak_count, family_leak_count = _audit_rows(rows, holdout_families=holdouts)
    split_counts = Counter(str(row["split"]) for row in rows)
    visual_families = Counter(str(row["visual_family"]) for row in rows)
    input_origins = Counter(str(row["input_origin"]) for row in rows)
    authorities = Counter(str(row["target_authority"]) for row in rows)
    summary = {
        "schema": UNIVERSAL_CURRICULUM_SCHEMA,
        "created_utc": datetime.now(UTC).isoformat(),
        "curriculum_id": "pending-write",
        "source_inputs": source_inputs,
        "row_count": len(rows),
        "source_group_count": len({str(row["source_group_id"]) for row in rows}),
        "split_counts": {
            "train": split_counts["train"],
            "validation": split_counts["validation"],
            "test": split_counts["test"],
            "compatibility": split_counts["compatibility"],
        },
        "input_origins": dict(sorted(input_origins.items())),
        "visual_families": dict(sorted(visual_families.items())),
        "held_out_families": sorted(holdouts),
        "target_signal": "relative_relief",
        "target_authorities": {
            "exact_numeric": authorities["exact_numeric"],
            "teacher_pseudo": authorities["teacher_pseudo"],
        },
        "synthetic_lighting_contract": (
            "not_applicable_no_synthetic" if v50_source == "authored" else "NoonWhiteGlobal"
        ),
        "group_leak_count": group_leak_count,
        "family_leak_count": family_leak_count,
        "excluded_counts": {},
    }
    identity_payload = {key: value for key, value in summary.items() if key not in {"created_utc", "curriculum_id"}}
    summary["curriculum_id"] = "sha256:" + hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True).encode()
    ).hexdigest()
    return UniversalCurriculumPlan(
        v50_store=v50_path,
        v50_source=v50_source,
        teacher_stores=bindings,
        holdout_families=tuple(sorted(holdouts)),
        output=output_path,
        rows=tuple(rows),
        summary=summary,
    )


def write_universal_curriculum(plan: UniversalCurriculumPlan) -> dict[str, Any]:
    if plan.output.exists():
        raise FileExistsError(f"universal curriculum output already exists: {plan.output}")
    group = zarr.open_group(str(plan.output), mode="w")
    group.attrs.update(plan.summary)
    table = pa.Table.from_pylist(list(plan.rows))
    pq.write_table(table, plan.output / "index.parquet", compression="zstd")
    (plan.output / "summary.json").write_text(
        json.dumps(plan.summary, indent=2) + "\n", encoding="utf-8"
    )
    return plan.summary


def _parse_teacher_binding(value: str) -> TeacherStoreBinding:
    if "=" not in value:
        raise argparse.ArgumentTypeError("teacher store must use FAMILY=PATH")
    family, path = value.split("=", 1)
    if not family.strip() or not path.strip():
        raise argparse.ArgumentTypeError("teacher store must use non-empty FAMILY=PATH")
    return TeacherStoreBinding(family.strip(), Path(path.strip()))


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build universal relief curriculum index; dry-run unless --confirm-build."
    )
    parser.add_argument("--v50-store", required=True, type=Path)
    parser.add_argument("--v50-source", choices=("authored", "synthetic", "all"), default="authored")
    parser.add_argument("--teacher-store", action="append", required=True, type=_parse_teacher_binding)
    parser.add_argument("--holdout-family", action="append", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--confirm-build", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    plan = build_universal_curriculum_plan(
        v50_store=args.v50_store,
        v50_source=args.v50_source,
        teacher_stores=args.teacher_store,
        holdout_families=args.holdout_family,
        output=args.output,
    )
    print(json.dumps(plan.summary, indent=2))
    if not args.confirm_build:
        print("DRY RUN: add --confirm-build to write the immutable curriculum index.")
        return 0
    write_universal_curriculum(plan)
    print(f"wrote {plan.output}")
    return 0
