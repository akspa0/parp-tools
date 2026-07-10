"""Package the Spec 094 V24 trainer (WDL prior + lattice detailer) into a RunPod-ready bundle.

Unlike V23 (which depends on the paths-only V22 store), V24 training reads a
*small* V24 store (priors/confidence/source, a few MB regardless of tile
count) joined against the full-payload V18 store via `v18_row`. Shipping the
whole V18 store would be multiple GB per build, so this packager copies only
the V18 fields `harvester.v24.tiles.TileSource` actually reads, and only the
rows the chosen V24 store's index references, remapped to compact indices.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr
import zarr.storage


_SCRIPT_DIR = Path(__file__).resolve().parent
_HARVESTER_ROOT = _SCRIPT_DIR.parent
_WOW_ROOT = _HARVESTER_ROOT.parent
_DEFAULT_V24_STORE = _WOW_ROOT / "output" / "datasets" / "v24" / "3_3_5_12340_v24_all_v1.zarr"
_DEFAULT_OUTPUT_ROOT = _WOW_ROOT / "output" / "cloud-packages" / "v24"

_BUNDLE_PYPROJECT = """[build-system]
requires = ["setuptools>=81"]
build-backend = "setuptools.build_meta"

[project]
name = "wowviewer-harvester-v24-bundle"
version = "0.1.0"
description = "Spec 094 V24 RunPod bundle"
requires-python = ">=3.11"
dependencies = [
    "numpy>=2.0",
    "pyarrow>=24.0.0",
    "torch>=2.5",
    "zarr>=2.0",
    "numcodecs>=0.13",
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
pytest>=8.0
"""

_SOURCE_FILES = (
    "src/harvester/__init__.py",
    "scripts/train_v24_stage_a.py",
    "scripts/train_v24_stage_b.py",
    "scripts/infer_v24_stage_a.py",
    "scripts/infer_v24_stage_b.py",
    "scripts/validate_v24.py",
    "scripts/inspect_v24_dataset.py",
    "runpod/v24/install_deps.sh",
    "runpod/v24/verify_bundle.sh",
    "runpod/v24/smoke.sh",
    "runpod/v24/train.sh",
)

_SOURCE_DIRS = (
    "src/harvester/v24",
    "tests/v24",
)

# Fields harvester.v24.tiles.TileSource.load() actually reads off the V18 store.
# Everything else in the V18 schema (MCLY layers, MODF/MDDF masks, object
# instance ids, ...) is irrelevant to V24 training and would only bloat the bundle.
_V18_FIELDS_NEEDED = (
    "minimap_rgb",
    "no_object_minimap",  # optional; only some V18 stores have it
    "alpha_256",
    "normal_xyz",
    "mcnr_mask_257",
    "object_precise_mask",
    "liquid_mask",
    "holes_16",
    "height_257",
)

# Process V18 rows in chunks to avoid OOM on large datasets (2,801 tiles of
# alpha_256 = 2.74 GiB in one shot). 256 rows per chunk = ~256 MB peak.
_V18_CHUNK = 256


def _default_bundle_name() -> str:
    return "v24_bundle"


def _copy_file(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)


def _copy_tree(source: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(source, dest)


def _copy_sources(bundle_root: Path, *, include_tests: bool) -> list[str]:
    copied: list[str] = []
    for relative in _SOURCE_FILES:
        source = _HARVESTER_ROOT / relative
        if not source.exists():
            continue
        _copy_file(source, bundle_root / relative)
        copied.append(relative.replace("\\", "/"))
    for relative in _SOURCE_DIRS:
        if not include_tests and relative == "tests/v24":
            continue
        source = _HARVESTER_ROOT / relative
        _copy_tree(source, bundle_root / relative)
        copied.append(relative.replace("\\", "/"))
    return copied


def _write_bundle_runtime_files(bundle_root: Path) -> None:
    (bundle_root / "pyproject.toml").write_text(_BUNDLE_PYPROJECT, encoding="utf-8")
    (bundle_root / "requirements-runpod.txt").write_text(_BUNDLE_REQUIREMENTS, encoding="utf-8")
    readme = (
        "# Spec 094 V24 RunPod Bundle\n\n"
        "Ships the V24 Python trainer/inference code plus a right-sized V24 store\n"
        "(priors/confidence/source) and a compacted V18 substrate subset (only the\n"
        "fields and rows V24 training reads). No game client files, no C# tooling\n"
        "(the WDL shim is only needed to *build* the V24 store, not to train on it).\n"
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


def _copy_v24_store(source: Path, dest: Path) -> dict[str, Any]:
    """Copy a V24 store as-is; these are a few MB regardless of tile count."""
    _copy_tree(source, dest)
    root = zarr.open_group(str(dest), mode="r")
    n_tiles = int(root["wdl_prior_outer"].shape[0])
    return {"n_tiles": n_tiles}


def _subset_v18_store(
    source: Path, dest: Path, v18_rows_needed: list[int]
) -> tuple[dict[str, Any], dict[int, int]]:
    """Copy only the fields/rows V24 training reads, remapped to compact indices.

    Processes rows in chunks (``_V18_CHUNK``) to avoid loading the full dataset
    into memory at once — a 2,801-tile ``alpha_256`` is 2.74 GiB alone.
    """
    if dest.exists():
        shutil.rmtree(dest)
    unique_rows = sorted({int(r) for r in v18_rows_needed})
    remap = {old: new for new, old in enumerate(unique_rows)}
    n_rows = len(unique_rows)

    src_store = zarr.storage.LocalStore(str(source), read_only=True)
    src = zarr.open_group(store=src_store, mode="r")
    dest_store = zarr.storage.LocalStore(str(dest), read_only=False)
    dest_group = zarr.group(store=dest_store)

    # First pass: create arrays with estimated chunks so we can write incrementally
    # (v2 zarr doesn't support chunked append well; we allocate the full shape
    # but write it one chunk at a time).
    copied_fields: list[str] = []
    arrays: dict[str, zarr.Array] = {}
    for name in _V18_FIELDS_NEEDED:
        if name not in src:
            continue
        arr = src[name]
        shape = (n_rows,) + arr.shape[1:]
        dtype = arr.dtype
        src_chunks = arr.chunks
        if src_chunks is not None:
            # Keep source chunk sizes but cap row-chunk to _V18_CHUNK
            row_chunk = min(int(src_chunks[0]), _V18_CHUNK)
            chunks = (row_chunk,) + tuple(int(c) for c in src_chunks[1:])
        else:
            chunks = (_V18_CHUNK,) + shape[1:]
        arrays[name] = dest_group.create_array(name, shape=shape, dtype=dtype, chunks=chunks, fill_value=0, overwrite=True)
        copied_fields.append(name)

    # Second pass: read source rows in chunks and write to destination
    chunk = _V18_CHUNK
    for start in range(0, n_rows, chunk):
        batch = unique_rows[start:start + chunk]
        batch_idx = np.asarray(batch, dtype=np.int64)
        end = min(start + chunk, n_rows)
        for name in copied_fields:
            data = src[name][batch_idx]
            arrays[name][start:end] = data
        print(f"  v18 subset [{start}:{end}/{n_rows}] rows", flush=True)

    dest_group.attrs.update(
        {
            "spec": "094-wdl-prior-v24-runpod-bundle",
            "subset_of": str(source),
            "subset_row_count": len(unique_rows),
            "subset_total_rows": int(src["height_257"].shape[0]) if "height_257" in src else None,
        }
    )
    return (
        {
            "source": str(source),
            "fields_copied": copied_fields,
            "unique_rows": len(unique_rows),
            "total_rows": int(src["height_257"].shape[0]) if "height_257" in src else None,
        },
        remap,
    )


def _rewrite_v24_index(v24_dest: Path, remap: dict[int, int], v18_dest_relative: str) -> None:
    """Point the bundled V24 store's index + attrs at the compacted V18 store."""
    index_path = v24_dest / "index.parquet"
    columns = pq.read_table(str(index_path)).to_pydict()
    columns["v18_row"] = [remap[int(r)] for r in columns["v18_row"]]
    pq.write_table(pa.table(columns), str(index_path))

    group = zarr.open_group(str(v24_dest), mode="r+")
    group.attrs["v18_store_path"] = v18_dest_relative


def _package_one_build(
    bundle_root: Path, v24_store: Path, v18_store: Path | None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Copy one (V24, V18-subset) build pair into the bundle. Returns (v24_report, v18_report)."""
    v24_source = v24_store.resolve()
    if not v24_source.exists():
        raise SystemExit(f"Missing V24 store: {v24_source}")

    build_name = v24_source.stem
    v24_dest = bundle_root / "data" / "v24" / f"{build_name}.zarr"
    v24_report = _copy_v24_store(v24_source, v24_dest)

    index = pq.read_table(str(v24_source / "index.parquet")).to_pydict()
    v18_rows_needed = [int(r) for r in index["v18_row"]]

    v18_source = v18_store.resolve() if v18_store else None
    if v18_source is None:
        attr_path = zarr.open_group(str(v24_source), mode="r").attrs.get("v18_store_path")
        if not attr_path:
            raise SystemExit(f"V24 store {v24_source} has no v18_store_path attr; pass a matching --v18-store.")
        # Stored relative to the data-harvester root: it is written verbatim by
        # build_wdl_prior.py (run with that as CWD) and opened the same way by
        # TileSource at train time, so resolve it against the same root here.
        v18_source = (_HARVESTER_ROOT / attr_path).resolve()
    if not v18_source.exists():
        raise SystemExit(f"Missing V18 store: {v18_source}")

    v18_build_name = v18_source.stem
    v18_dest = bundle_root / "data" / "v18" / f"{v18_build_name}.zarr"
    v18_report, remap = _subset_v18_store(v18_source, v18_dest, v18_rows_needed)
    _rewrite_v24_index(v24_dest, remap, f"data/v18/{v18_build_name}.zarr")

    v24_report = {"build": build_name, "dest": v24_dest.relative_to(bundle_root).as_posix(), **v24_report}
    v18_report = {"build": v18_build_name, "dest": v18_dest.relative_to(bundle_root).as_posix(), **v18_report}
    return v24_report, v18_report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Package the Spec 094 V24 RunPod bundle.")
    parser.add_argument("--bundle-name", default=_default_bundle_name())
    parser.add_argument("--v24-store", type=Path, nargs="+", default=[_DEFAULT_V24_STORE],
                         help="one or more V24 stores to ship (each is small; whole stores are copied). "
                              "Multiple stores (e.g. different builds) train as one concatenated corpus.")
    parser.add_argument("--v18-store", type=Path, nargs="*", default=None,
                         help="Override V18 stores to subset from, one per --v24-store "
                              "(default: each V24 store's own v18_store_path attr).")
    parser.add_argument("--output-root", type=Path, default=_DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-tar", type=Path, default=None)
    parser.add_argument("--archive-format", choices=["tar", "none"], default="tar")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-tests", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.v18_store and len(args.v18_store) != len(args.v24_store):
        raise SystemExit(
            f"--v18-store count ({len(args.v18_store)}) must match --v24-store count ({len(args.v24_store)}) when given"
        )
    bundle_root = Path(args.output_root).resolve() / str(args.bundle_name)
    if bundle_root.exists():
        if not args.overwrite:
            raise SystemExit(f"Bundle already exists: {bundle_root}")
        shutil.rmtree(bundle_root)
    bundle_root.mkdir(parents=True, exist_ok=False)

    copied_sources = _copy_sources(bundle_root, include_tests=not bool(args.no_tests))
    _write_bundle_runtime_files(bundle_root)

    v24_reports: list[dict[str, Any]] = []
    v18_reports: list[dict[str, Any]] = []
    for i, v24_store in enumerate(args.v24_store):
        v18_store = args.v18_store[i] if args.v18_store else None
        v24_report, v18_report = _package_one_build(bundle_root, v24_store, v18_store)
        v24_reports.append(v24_report)
        v18_reports.append(v18_report)
        print(
            f"[{v24_report['build']}] V24 store: {v24_report['n_tiles']} tiles. "
            f"V18 subset: {v18_report['unique_rows']}/{v18_report['total_rows']} rows, "
            f"fields={v18_report['fields_copied']}"
        )

    manifest = {
        "schema": "spec-094-v24-runpod-bundle",
        "v24_bundle_version": 2,
        "bundle_name": str(args.bundle_name),
        "contains_game_client_files": False,
        "v24_stores": v24_reports,
        "v18_subsets": v18_reports,
        "copied_sources": copied_sources,
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
