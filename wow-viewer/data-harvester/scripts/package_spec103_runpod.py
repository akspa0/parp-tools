"""Package the Spec 103 terrain regressor (v8 lean / v7 ablation) into a RunPod-ready bundle.

Unlike the V24 bundle, this trainer has no pretrained-weight dependency (v8/v7 both train
from scratch) — no HF downloads, no LoRA/bitsandbytes. The only thing worth shipping small is
the *data*: the V18 store carries 24 fields (29.4 GB uncompressed) but spec103 training reads
exactly 6 (minimap_rgb, height_257, normal_xyz, liquid_mask, liquid_height,
object_precise_mask); and curation already drops the majority of tiles as object-contaminated
or blank (FR-013). This packager subsets both fields and rows so the shipped store is only
what a training run will actually touch — smaller upload, smaller pod disk, faster cold start.

Reads the curation manifest (`spec103_curate_dataset.py` output) and copies only kept rows,
remapped to compact indices, plus a matching remapped copy of the manifest so
`--curation-manifest` still works unchanged on the pod (auditability, FR-013 — not just an
inline filter). Fields not read by `train_spec103_v7.py` / `infer_spec103_v7.py` / v7_inputs.py
are never copied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr
import zarr.storage

_SCRIPT_DIR = Path(__file__).resolve().parent
_HARVESTER_ROOT = _SCRIPT_DIR.parent
_WOW_ROOT = _HARVESTER_ROOT.parent
_DEFAULT_STORE = _WOW_ROOT / "output" / "datasets" / "v18" / "3_3_5_12340.zarr"
_DEFAULT_CURATION = _WOW_ROOT / "output" / "spec103" / "curation_v18_v1"
_DEFAULT_OUTPUT_ROOT = _WOW_ROOT / "output" / "cloud-packages" / "spec103"

# The complete set of arrays spec103 training/inference ever reads (v7_inputs.assemble_v7_input
# + train_spec103_v7.py's required/OPTIONAL_ARRAYS). Everything else in the V18 schema (alpha_256,
# mcly layers, MODF/MDDF masks, object_instance_mask, shadow_mask, ...) is irrelevant here.
_FIELDS_NEEDED = (
    "minimap_rgb",
    "height_257",
    "normal_xyz",
    "liquid_mask",
    "liquid_height",
    "object_precise_mask",
)

_SOURCE_FILES = (
    "src/harvester/__init__.py",
    "src/harvester/height_to_normal.py",
    "scripts/train_spec103_v7.py",
    "scripts/infer_spec103_v7.py",
    "runpod/spec103/install_deps.sh",
    "runpod/spec103/verify_bundle.sh",
    "runpod/spec103/smoke.sh",
    "runpod/spec103/train.sh",
)

_SOURCE_DIRS = (
    "src/harvester/spec103",
    "tests/spec103",
)

_BUNDLE_PYPROJECT = """[build-system]
requires = ["setuptools>=81"]
build-backend = "setuptools.build_meta"

[project]
name = "wowviewer-harvester-spec103-bundle"
version = "0.1.0"
description = "Spec 103 v8/v7 terrain regressor RunPod bundle"
requires-python = ">=3.11"
dependencies = [
    "numpy>=2.0",
    "pyarrow>=24.0.0",
    "torch>=2.5",
    "zarr>=2.0",
    "numcodecs>=0.13",
    "Pillow>=10.0",
    "scipy>=1.13",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
]

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]
"""

_BUNDLE_REQUIREMENTS = """numpy>=2.0
pyarrow>=24.0.0
torch>=2.5
zarr>=2.0
numcodecs>=0.13
Pillow>=10.0
scipy>=1.13
pytest>=8.0
"""

_CHUNK = 256  # rows per copy batch; normal_xyz alone is ~800KB/tile uncompressed


def _copy_file(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)


def _copy_tree(source: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(source, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def _copy_sources(bundle_root: Path) -> list[str]:
    copied: list[str] = []
    for relative in _SOURCE_FILES:
        source = _HARVESTER_ROOT / relative
        if not source.exists():
            continue
        _copy_file(source, bundle_root / relative)
        copied.append(relative.replace("\\", "/"))
    for relative in _SOURCE_DIRS:
        source = _HARVESTER_ROOT / relative
        _copy_tree(source, bundle_root / relative)
        copied.append(relative.replace("\\", "/"))
    return copied


def _write_bundle_runtime_files(bundle_root: Path) -> None:
    (bundle_root / "pyproject.toml").write_text(_BUNDLE_PYPROJECT, encoding="utf-8")
    (bundle_root / "requirements-runpod.txt").write_text(_BUNDLE_REQUIREMENTS, encoding="utf-8")
    readme = (
        "# Spec 103 Terrain Regressor RunPod Bundle\n\n"
        "Ships the v8 (lean ConvNeXt-V2, ~6.2M params, default) / v7 (117M ablation) trainer "
        "plus a curated, field-and-row-subsetted copy of the V18 store (only the 6 arrays "
        "training reads, only the curation-kept tiles). Trains from scratch -- no pretrained "
        "weights, no HF downloads, no LoRA.\n\n"
        "## Usage\n\n"
        "bash runpod/spec103/install_deps.sh   # uv sync only\n"
        "bash runpod/spec103/verify_bundle.sh  # import + manifest sanity, bundled pytest\n"
        "bash runpod/spec103/smoke.sh          # ~1 min, 2 epochs, proves the pod works\n"
        "bash runpod/spec103/train.sh          # the real run (env-var configurable)\n"
    )
    (bundle_root / "README_RunPod.md").write_text(readme, encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(_sha256_file(path).encode("utf-8"))
    return digest.hexdigest()


_ROW_IDENTITY_COLUMNS = ("build", "map", "tile_x", "tile_y")


def _read_store_index(store_source: Path) -> tuple[dict[str, list[Any]], dict[int, int]]:
    index_path = store_source / "index.parquet"
    if not index_path.is_file():
        raise ValueError(f"source store is missing index.parquet: {index_path}")
    index = pq.read_table(index_path).to_pydict()
    required = {"tile_id", *_ROW_IDENTITY_COLUMNS}
    missing = sorted(required - set(index))
    if missing:
        raise ValueError(f"source index is missing identity columns: {missing}")
    tile_ids = [int(value) for value in index["tile_id"]]
    if len(tile_ids) != len(set(tile_ids)):
        raise ValueError("source index contains duplicate tile_id values")
    return index, {tile_id: position for position, tile_id in enumerate(tile_ids)}


def _validate_rows_against_store_index(
    rows: dict[str, list[Any]],
    *,
    source_index: dict[str, list[Any]],
    source_positions: dict[int, int],
    label: str,
    require_unique_tile_ids: bool,
) -> int:
    required = {"tile_id", *_ROW_IDENTITY_COLUMNS}
    missing = sorted(required - set(rows))
    if missing:
        raise ValueError(f"{label} is missing row-identity columns: {missing}")
    row_count = len(rows["tile_id"])
    if any(len(rows[column]) != row_count for column in required):
        raise ValueError(f"{label} has inconsistent identity-column lengths")
    tile_ids = [int(value) for value in rows["tile_id"]]
    if require_unique_tile_ids and len(tile_ids) != len(set(tile_ids)):
        raise ValueError(f"{label} contains duplicate tile_id values")

    mismatches: list[dict[str, Any]] = []
    for row_position, tile_id in enumerate(tile_ids):
        source_position = source_positions.get(tile_id)
        if source_position is None:
            mismatches.append({"tile_id": tile_id, "reason": "absent_from_source_index"})
            continue
        differences: dict[str, dict[str, Any]] = {}
        for column in _ROW_IDENTITY_COLUMNS:
            expected = source_index[column][source_position]
            actual = rows[column][row_position]
            if column in {"tile_x", "tile_y"}:
                expected, actual = int(expected), int(actual)
            else:
                expected, actual = str(expected), str(actual)
            if actual != expected:
                differences[column] = {"manifest": actual, "store": expected}
        if differences:
            mismatches.append({"tile_id": tile_id, "differences": differences})
        if len(mismatches) >= 8:
            break
    if mismatches:
        raise ValueError(
            f"{label} row identity does not match source index: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return row_count


def _declared_index_digests(summary_path: Path) -> list[str]:
    if not summary_path.is_file():
        return []
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    candidates: list[object] = [summary.get("index_sha256")]
    stores = summary.get("stores")
    if isinstance(stores, list):
        candidates.extend(
            entry.get("index_sha256")
            for entry in stores
            if isinstance(entry, dict)
        )
    declared: list[str] = []
    for candidate in candidates:
        if candidate is None or not str(candidate).strip():
            continue
        digest = str(candidate).strip().lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"curation summary contains invalid index_sha256: {candidate!r}")
        declared.append(digest)
    return sorted(set(declared))


def _validate_manifest_store_identity(
    store_source: Path,
    manifest_path: Path,
) -> tuple[dict[str, list[Any]], dict[str, Any]]:
    source_index, source_positions = _read_store_index(store_source)
    manifest = pq.read_table(manifest_path).to_pydict()
    row_count = _validate_rows_against_store_index(
        manifest,
        source_index=source_index,
        source_positions=source_positions,
        label="curation manifest",
        require_unique_tile_ids=True,
    )
    if "keep" not in manifest or len(manifest["keep"]) != row_count:
        raise ValueError("curation manifest is missing a row-aligned keep column")

    index_sha256 = _sha256_file(store_source / "index.parquet")
    summary_path = manifest_path.parent / "curation_summary.json"
    declared_digests = _declared_index_digests(summary_path)
    if declared_digests and index_sha256 not in declared_digests:
        raise ValueError(
            "curation summary index digest does not match the selected source store; "
            f"store={index_sha256}, declared={declared_digests}"
        )
    digest_status = "matched" if declared_digests else "not_declared"
    return manifest, {
        "status": "verified",
        "manifest_sha256": _sha256_file(manifest_path),
        "source_index_sha256": index_sha256,
        "index_digest_status": digest_status,
        "declared_index_sha256": declared_digests,
        "row_identity_columns": ["tile_id", *_ROW_IDENTITY_COLUMNS],
        "manifest_rows": row_count,
        "kept_rows": sum(bool(value) for value in manifest["keep"]),
    }


def _declared_evidence_artifact_hashes(summary_path: Path) -> dict[str, str]:
    if not summary_path.is_file():
        return {}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    candidates: dict[str, object] = {}
    for field in ("artifact_sha256", "output_sha256"):
        value = summary.get(field)
        if isinstance(value, dict):
            candidates.update(value)
    outputs = summary.get("outputs")
    if isinstance(outputs, dict):
        for name, value in outputs.items():
            if isinstance(value, dict) and value.get("sha256"):
                candidates[str(name)] = value["sha256"]

    result: dict[str, str] = {}
    for name, candidate in candidates.items():
        digest = str(candidate or "").strip().lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(
                f"curation summary contains invalid artifact SHA-256 for {name!r}"
            )
        filename = Path(str(name)).name
        result[filename] = digest
    return result


def _subset_store(
    source: Path, dest: Path, kept_tile_ids: list[int]
) -> tuple[dict[str, Any], dict[int, int]]:
    """Copy only the fields spec103 reads, only the curation-kept rows, compact-remapped."""
    if dest.exists():
        shutil.rmtree(dest)
    unique_ids = sorted({int(r) for r in kept_tile_ids})
    remap = {old: new for new, old in enumerate(unique_ids)}
    n_rows = len(unique_ids)

    src_store = zarr.storage.LocalStore(str(source), read_only=True)
    src = zarr.open_group(store=src_store, mode="r")
    dest_store = zarr.storage.LocalStore(str(dest), read_only=False)
    dest_group = zarr.group(store=dest_store)

    copied_fields: list[str] = []
    arrays: dict[str, zarr.Array] = {}
    for name in _FIELDS_NEEDED:
        if name not in src:
            continue
        arr = src[name]
        shape = (n_rows,) + arr.shape[1:]
        src_chunks = arr.chunks
        if src_chunks is not None:
            row_chunk = min(int(src_chunks[0]), _CHUNK)
            chunks = (row_chunk,) + tuple(int(c) for c in src_chunks[1:])
        else:
            chunks = (_CHUNK,) + shape[1:]
        arrays[name] = dest_group.create_array(
            name, shape=shape, dtype=arr.dtype, chunks=chunks, fill_value=0, overwrite=True
        )
        copied_fields.append(name)

    src_index = pq.read_table(str(source / "index.parquet")).to_pydict()
    source_ids = [int(value) for value in src_index["tile_id"]]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("index.parquet contains duplicate tile_id values; cannot compact safely")
    id_to_pos = {tile_id: pos for pos, tile_id in enumerate(source_ids)}
    missing_ids = [tile_id for tile_id in unique_ids if tile_id not in id_to_pos]
    if missing_ids:
        raise KeyError(f"curation manifest references tile IDs absent from index: {missing_ids[:8]}")
    source_positions = [id_to_pos[tile_id] for tile_id in unique_ids]

    for start in range(0, n_rows, _CHUNK):
        batch = source_positions[start:start + _CHUNK]
        batch_idx = np.asarray(batch, dtype=np.int64)
        end = min(start + _CHUNK, n_rows)
        for name in copied_fields:
            arrays[name][start:end] = src[name][batch_idx]
        print(f"  subset [{start}:{end}/{n_rows}] rows", flush=True)

    # Preserve every index column. Runtime tile_id is compact; immutable source
    # identity remains available for evidence joins and audits.
    new_index: dict[str, list] = {
        column: [values[pos] for pos in source_positions]
        for column, values in src_index.items()
    }
    new_index["source_tile_id"] = list(unique_ids)
    new_index["source_build"] = [str(src_index["build"][pos]) for pos in source_positions]
    new_index["tile_id"] = [remap[tile_id] for tile_id in unique_ids]
    pq.write_table(pa.table(new_index), str(dest / "index.parquet"))

    dest_group.attrs.update({
        "spec": "103-image-only-reconstruction-runpod-bundle",
        "subset_of": str(source),
        "subset_row_count": n_rows,
        "subset_total_rows": int(src["height_257"].shape[0]),
        "fields_copied": copied_fields,
    })
    return (
        {"fields_copied": copied_fields, "kept_rows": n_rows,
         "total_rows": int(src["height_257"].shape[0])},
        remap,
    )


def _subset_curation_manifest(manifest_path: Path, dest_dir: Path, remap: dict[int, int]) -> int:
    table = pq.read_table(str(manifest_path)).to_pydict()
    kept_mask = [bool(k) for k in table["keep"]]
    out: dict[str, list] = {}
    source_ids: list[int] = []
    for col, values in table.items():
        kept_values = [v for v, k in zip(values, kept_mask, strict=True) if k]
        if col == "tile_id":
            source_ids = [int(v) for v in kept_values]
            kept_values = [remap[value] for value in source_ids]
        out[col] = kept_values
    out["source_tile_id"] = source_ids
    out["source_build"] = [str(value) for value in out.get("build", [""] * len(source_ids))]
    dest_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(out), str(dest_dir / "curation_manifest.parquet"))
    return len(out["tile_id"])


def _package_curation_evidence(
    source_dir: Path,
    dest_dir: Path,
    remap: dict[int, int],
    *,
    store_source: Path | None = None,
    identity_report: dict[str, Any] | None = None,
    require_verified_adjacent: bool = False,
) -> dict[str, Any]:
    """Preserve immutable source evidence and write compact runtime copies."""
    source_out = dest_dir / "source"
    runtime_out = dest_dir / "runtime"
    source_out.mkdir(parents=True, exist_ok=True)
    runtime_out.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"source": {}, "runtime": {}, "skipped": {}}
    if require_verified_adjacent and (
        identity_report is None or identity_report.get("status") != "verified"
    ):
        raise ValueError(
            "clean_synthetic curation evidence requires verified manifest-to-store identity"
        )
    adjacent_is_index_bound = (
        identity_report is not None
        and identity_report.get("status") == "verified"
        and identity_report.get("index_digest_status") == "matched"
    )
    source_index: dict[str, list[Any]] | None = None
    source_positions: dict[int, int] | None = None
    declared_artifact_hashes = _declared_evidence_artifact_hashes(
        source_dir / "curation_summary.json"
    )
    if store_source is not None:
        source_index, source_positions = _read_store_index(store_source)
    for name in (
        "curation_manifest.parquet",
        "pattern_evidence_ledger.parquet",
        "tile_pattern_coverage.parquet",
        "curation_summary.json",
    ):
        path = source_dir / name
        if not path.exists():
            continue
        is_requested_manifest = name == "curation_manifest.parquet"
        if require_verified_adjacent and not is_requested_manifest:
            if not adjacent_is_index_bound:
                report["skipped"][name] = (
                    "clean_synthetic_requires_matching_curation_summary_index_digest"
                )
                continue
            if name == "curation_summary.json":
                report["skipped"][name] = (
                    "clean_synthetic_does_not_copy_self_unbound_adjacent_summary"
                )
                continue
            expected_artifact_sha256 = declared_artifact_hashes.get(name)
            if expected_artifact_sha256 is None:
                report["skipped"][name] = (
                    "clean_synthetic_requires_explicit_adjacent_artifact_digest"
                )
                continue
            if _sha256_file(path) != expected_artifact_sha256:
                raise ValueError(
                    f"clean_synthetic adjacent evidence digest mismatch for {name}"
                )
        verification = "unverified_adjacent_private_byod"
        if is_requested_manifest and identity_report is not None:
            expected_manifest_sha256 = str(identity_report.get("manifest_sha256") or "")
            if _sha256_file(path) != expected_manifest_sha256:
                raise ValueError(
                    "curation_manifest.parquet changed after manifest-to-store validation"
                )
            verification = "manifest_rows_and_source_index_identity_verified"
        elif path.suffix == ".parquet" and adjacent_is_index_bound:
            if source_index is None or source_positions is None:
                raise ValueError("verified adjacent evidence requires the selected source store")
            evidence_rows = pq.read_table(path).to_pydict()
            _validate_rows_against_store_index(
                evidence_rows,
                source_index=source_index,
                source_positions=source_positions,
                label=name,
                require_unique_tile_ids=False,
            )
            verification = (
                "artifact_digest_rows_and_source_index_verified"
                if require_verified_adjacent
                else "rows_and_summary_index_digest_verified"
            )
        elif name == "curation_summary.json" and adjacent_is_index_bound:
            verification = "source_index_digest_verified"
        copied = source_out / name
        shutil.copy2(path, copied)
        report["source"][name] = {
            "path": copied.relative_to(dest_dir.parent.parent).as_posix(),
            "sha256": _sha256_file(copied),
            "verification": verification,
        }
        if path.suffix != ".parquet" or name == "curation_manifest.parquet":
            continue
        table = pq.read_table(path).to_pylist()
        runtime_rows: list[dict[str, Any]] = []
        for row in table:
            source_tile_id = int(row.get("tile_id", -1))
            if source_tile_id not in remap:
                continue
            runtime_rows.append(
                {
                    **row,
                    "source_tile_id": source_tile_id,
                    "source_build": str(row.get("build", "")),
                    "tile_id": remap[source_tile_id],
                }
            )
        runtime_path = runtime_out / name
        pq.write_table(pa.Table.from_pylist(runtime_rows), runtime_path)
        report["runtime"][name] = {
            "path": runtime_path.relative_to(dest_dir.parent.parent).as_posix(),
            "rows": len(runtime_rows),
            "sha256": _sha256_file(runtime_path),
            "verification": verification,
        }
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Package the Spec 103 terrain-regressor RunPod bundle.")
    parser.add_argument("--bundle-name", default="spec103_bundle")
    parser.add_argument("--store", type=Path, default=_DEFAULT_STORE)
    parser.add_argument("--curation-manifest", type=Path, default=_DEFAULT_CURATION,
                         help="dir containing curation_manifest.parquet (spec103_curate_dataset.py output); "
                              "only kept tiles are shipped")
    parser.add_argument("--output-root", type=Path, default=_DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-tar", type=Path, default=None)
    parser.add_argument("--archive-format", choices=["tar", "none"], default="tar")
    parser.add_argument(
        "--rights-class",
        choices=["private_byod", "clean_synthetic"],
        default="private_byod",
        help=(
            "package provenance class; clean_synthetic fails closed against the source "
            "store contract, while private_byod makes no redistribution claim"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _resolve_bundle_rights(store_source: Path, requested_class: str) -> dict[str, Any]:
    """Resolve rights metadata without attempting to make a legal determination."""
    if requested_class == "private_byod":
        return {
            "rights_class": "private_byod",
            "contains_raw_game_client_files": False,
            "contains_client_derived_training_data": True,
            "distribution_policy": "operator_private_no_redistribution_claim",
            "legal_status": "not_a_legal_determination",
        }

    contract_path = store_source / "contract.json"
    if not contract_path.is_file():
        raise ValueError(
            "--rights-class clean_synthetic requires a source store contract.json"
        )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    required = {
        "rights_class": "clean_synthetic",
        "contains_raw_game_client_files": False,
        "contains_client_derived_training_data": False,
        "distribution_policy": "operator_declared_license_only",
    }
    mismatches = {
        field: {"expected": expected, "actual": contract.get(field)}
        for field, expected in required.items()
        if contract.get(field) != expected
    }
    license_summary = contract.get("source_license_summary")
    rights_summary = contract.get("source_rights_assertion_summary")
    if (
        not isinstance(license_summary, list)
        or not license_summary
        or any(not str(value).strip() or str(value).upper() == "UNSPECIFIED" for value in license_summary)
    ):
        mismatches["source_license_summary"] = {
            "expected": "one or more explicit operator-declared licenses",
            "actual": license_summary,
        }
    if (
        not isinstance(rights_summary, list)
        or not rights_summary
        or any(not str(value).strip() or str(value).upper() == "UNSPECIFIED" for value in rights_summary)
    ):
        mismatches["source_rights_assertion_summary"] = {
            "expected": "one or more explicit operator rights assertions",
            "actual": rights_summary,
        }
    if mismatches:
        raise ValueError(
            "clean_synthetic source contract failed closed: "
            + json.dumps(mismatches, sort_keys=True)
        )

    return {
        **required,
        "legal_status": "not_a_legal_determination",
        "source_contract_sha256": _sha256_file(contract_path),
        "source_license_summary": license_summary,
        "source_rights_assertion_summary": rights_summary,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    store_source = args.store.resolve()
    if not store_source.exists():
        raise SystemExit(f"Missing store: {store_source}")
    manifest_dir = args.curation_manifest.resolve()
    manifest_path = (
        manifest_dir / "curation_manifest.parquet"
        if manifest_dir.is_dir()
        else manifest_dir
    )
    if not manifest_path.exists():
        raise SystemExit(f"Missing curation manifest: {manifest_path}")
    try:
        curation, identity_report = _validate_manifest_store_identity(
            store_source, manifest_path
        )
        rights_contract = _resolve_bundle_rights(store_source, args.rights_class)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    kept_tile_ids = [
        int(tile_id)
        for tile_id, keep in zip(curation["tile_id"], curation["keep"], strict=True)
        if keep
    ]
    if not kept_tile_ids:
        raise SystemExit(f"curation manifest {manifest_path} has zero kept tiles")

    bundle_root = Path(args.output_root).resolve() / str(args.bundle_name)
    if bundle_root.exists():
        if not args.overwrite:
            raise SystemExit(f"Bundle already exists: {bundle_root}")
        shutil.rmtree(bundle_root)
    bundle_root.mkdir(parents=True, exist_ok=False)

    copied_sources = _copy_sources(bundle_root)
    _write_bundle_runtime_files(bundle_root)

    store_dest = bundle_root / "data" / store_source.stem / f"{store_source.stem}.zarr"
    store_report, remap = _subset_store(store_source, store_dest, kept_tile_ids)
    curation_dest = bundle_root / "data" / "curation"
    n_manifest_rows = _subset_curation_manifest(manifest_path, curation_dest, remap)
    evidence_report = _package_curation_evidence(
        manifest_path.parent,
        curation_dest,
        remap,
        store_source=store_source,
        identity_report=identity_report,
        require_verified_adjacent=args.rights_class == "clean_synthetic",
    )

    print(f"[spec103] store: {store_report['kept_rows']}/{store_report['total_rows']} tiles, "
          f"fields={store_report['fields_copied']}")
    print(f"[spec103] curation manifest: {n_manifest_rows} kept rows (all keep=True by construction)")

    manifest = {
        "schema": "spec-103-v8-runpod-bundle",
        "bundle_version": 2,
        "bundle_name": str(args.bundle_name),
        # Compatibility field: means no raw archive/client files are copied into the bundle.
        # It does not mean the numeric arrays are independent of client-derived sources.
        "contains_game_client_files": False,
        **rights_contract,
        "store": {"build": store_source.stem, "dest": store_dest.relative_to(bundle_root).as_posix(), **store_report},
        "curation_manifest": {"dest": "data/curation/curation_manifest.parquet", "kept_rows": n_manifest_rows},
        "curation_store_identity": identity_report,
        "curation_evidence": evidence_report,
        "copied_sources": copied_sources,
        "default_arch": "v8",
    }
    manifest["tree_hash"] = _tree_hash(bundle_root)
    (bundle_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    if args.archive_format == "tar":
        tar_path = args.output_tar.resolve() if args.output_tar is not None else bundle_root.with_suffix(".tar")
        if tar_path.exists():
            tar_path.unlink()
        tar_path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path, mode="w") as archive:
            archive.add(bundle_root, arcname=bundle_root.name)
        print(f"Wrote bundle tar: {tar_path}")
    print(f"Wrote bundle directory: {bundle_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
