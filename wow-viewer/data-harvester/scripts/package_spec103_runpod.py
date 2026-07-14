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


def _subset_store(
    source: Path, dest: Path, kept_tile_ids: list[int]
) -> tuple[dict[str, Any], dict[int, int]]:
    """Copy only the fields spec103 reads, only the curation-kept rows, compact-remapped."""
    if dest.exists():
        shutil.rmtree(dest)
    unique_rows = sorted({int(r) for r in kept_tile_ids})
    remap = {old: new for new, old in enumerate(unique_rows)}
    n_rows = len(unique_rows)

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

    for start in range(0, n_rows, _CHUNK):
        batch = unique_rows[start:start + _CHUNK]
        batch_idx = np.asarray(batch, dtype=np.int64)
        end = min(start + _CHUNK, n_rows)
        for name in copied_fields:
            arrays[name][start:end] = src[name][batch_idx]
        print(f"  subset [{start}:{end}/{n_rows}] rows", flush=True)

    # rewritten index.parquet: map/tile_x/tile_y (holdout key) + compact tile_id
    src_index = pq.read_table(str(source / "index.parquet")).to_pydict()
    id_to_pos = {int(t): i for i, t in enumerate(src_index["tile_id"])}
    new_index: dict[str, list] = {"tile_id": [], "build": [], "map": [], "tile_x": [], "tile_y": []}
    for old_id in unique_rows:
        pos = id_to_pos[old_id]
        new_index["tile_id"].append(remap[old_id])
        new_index["build"].append(src_index["build"][pos])
        new_index["map"].append(src_index["map"][pos])
        new_index["tile_x"].append(src_index["tile_x"][pos])
        new_index["tile_y"].append(src_index["tile_y"][pos])
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
    for col, values in table.items():
        kept_values = [v for v, k in zip(values, kept_mask) if k]
        if col == "tile_id":
            kept_values = [remap[int(v)] for v in kept_values]
        out[col] = kept_values
    dest_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(out), str(dest_dir / "curation_manifest.parquet"))
    return len(out["tile_id"])


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
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    bundle_root = Path(args.output_root).resolve() / str(args.bundle_name)
    if bundle_root.exists():
        if not args.overwrite:
            raise SystemExit(f"Bundle already exists: {bundle_root}")
        shutil.rmtree(bundle_root)
    bundle_root.mkdir(parents=True, exist_ok=False)

    copied_sources = _copy_sources(bundle_root)
    _write_bundle_runtime_files(bundle_root)

    store_source = args.store.resolve()
    if not store_source.exists():
        raise SystemExit(f"Missing store: {store_source}")
    manifest_dir = args.curation_manifest
    manifest_path = manifest_dir / "curation_manifest.parquet" if manifest_dir.is_dir() else manifest_dir
    if not manifest_path.exists():
        raise SystemExit(f"Missing curation manifest: {manifest_path}")

    curation = pq.read_table(str(manifest_path)).to_pydict()
    kept_tile_ids = [int(t) for t, k in zip(curation["tile_id"], curation["keep"]) if k]
    if not kept_tile_ids:
        raise SystemExit(f"curation manifest {manifest_path} has zero kept tiles")

    store_dest = bundle_root / "data" / store_source.stem / f"{store_source.stem}.zarr"
    store_report, remap = _subset_store(store_source, store_dest, kept_tile_ids)
    n_manifest_rows = _subset_curation_manifest(manifest_path, bundle_root / "data" / "curation", remap)

    print(f"[spec103] store: {store_report['kept_rows']}/{store_report['total_rows']} tiles, "
          f"fields={store_report['fields_copied']}")
    print(f"[spec103] curation manifest: {n_manifest_rows} kept rows (all keep=True by construction)")

    manifest = {
        "schema": "spec-103-v8-runpod-bundle",
        "bundle_version": 1,
        "bundle_name": str(args.bundle_name),
        "contains_game_client_files": False,
        "store": {"build": store_source.stem, "dest": store_dest.relative_to(bundle_root).as_posix(), **store_report},
        "curation_manifest": {"dest": "data/curation/curation_manifest.parquet", "kept_rows": n_manifest_rows},
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
