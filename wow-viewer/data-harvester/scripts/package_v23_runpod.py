"""Package the Spec 089 V23 trainer into a RunPod-ready bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
from typing import Any

import numpy as np
import zarr
import zarr.storage


_SCRIPT_DIR = Path(__file__).resolve().parent
_HARVESTER_ROOT = _SCRIPT_DIR.parent
_WOW_ROOT = _HARVESTER_ROOT.parent
_DEFAULT_DATASET_DIR = _WOW_ROOT / "output" / "datasets" / "v22"
_DEFAULT_OUTPUT_ROOT = _WOW_ROOT / "output" / "cloud-packages" / "v23"

_BUNDLE_PYPROJECT = """[build-system]
requires = ["setuptools>=81"]
build-backend = "setuptools.build_meta"

[project]
name = "wowviewer-harvester-v23-bundle"
version = "0.1.0"
description = "Spec 089 V23 RunPod bundle"
requires-python = ">=3.11"
dependencies = [
    "bitsandbytes>=0.43",
    "numpy>=2.0",
    "peft>=0.10",
    "pillow>=10.0",
    "pyarrow>=24.0.0",
    "scipy>=1.12",
    "torch>=2.5",
    "transformers>=4.52",
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

_BUNDLE_REQUIREMENTS = """bitsandbytes>=0.43
numpy>=2.0
peft>=0.10
pillow>=10.0
pyarrow>=24.0.0
scipy>=1.12
torch>=2.5
transformers>=4.52
zarr>=2.0
numcodecs>=0.13
pytest>=8.0
"""

_SOURCE_FILES = (
    "src/harvester/__init__.py",
    "src/harvester/v22_zarr_io.py",
    "src/harvester/v22_patched_signals.py",
    "scripts/train_v23_height.py",
    "scripts/infer_v23_height.py",
    "runpod/v23/install_deps.sh",
    "runpod/v23/verify_bundle.sh",
    "runpod/v23/smoke.sh",
    "runpod/v23/train.sh",
)

_SOURCE_DIRS = (
    "src/harvester/v23",
    "tests/v23",
)

_SIDECAR_FILES = (
    "index.parquet",
    "placements.parquet",
    "asset_inventory.parquet",
    "finalization.json",
)


def _default_bundle_name() -> str:
    return "v23_bundle"


def _copy_file(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)


def _copy_tree(source: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(source, dest)


def _tile_count(root: zarr.Group) -> int:
    tile_index = list(root.attrs.get("tile_index", []))
    if tile_index:
        return len(tile_index)
    if "height_257" in root:
        return int(root["height_257"].shape[0])
    raise ValueError("Unable to infer tile count from V22 store")


def _trim_tile_index(attrs: dict[str, Any], tile_limit: int, tile_count: int) -> dict[str, Any]:
    trimmed = dict(attrs)
    tile_index = list(trimmed.get("tile_index", []))
    if tile_index and len(tile_index) == tile_count:
        trimmed["tile_index"] = tile_index[:tile_limit]
    return trimmed


def _copy_subset_array(dest_root: zarr.Group, name: str, array: zarr.Array, tile_limit: int, tile_count: int) -> None:
    data = array[:]
    if data.ndim > 0 and int(data.shape[0]) == tile_count:
        data = data[:tile_limit]
    chunks = None
    if array.chunks is not None:
        chunks = tuple(min(int(dim), int(chunk)) for dim, chunk in zip(data.shape, array.chunks))
    dest_root.create_array(name, data=data, chunks=chunks, overwrite=True)


def _copy_store(source: Path, dest: Path, tile_limit: int | None) -> dict[str, Any]:
    if dest.exists():
        shutil.rmtree(dest)
    if tile_limit is None or tile_limit <= 0:
        shutil.copytree(source, dest)
        return {"mode": "full", "tile_count": None, "copied_tile_count": None}

    store = zarr.storage.LocalStore(str(source), read_only=True)
    root = zarr.open_group(store=store, mode="r")
    total_tiles = _tile_count(root)
    if tile_limit >= total_tiles:
        shutil.copytree(source, dest)
        return {"mode": "full", "tile_count": total_tiles, "copied_tile_count": total_tiles}

    dest_store = zarr.storage.LocalStore(str(dest), read_only=False)
    dest_root = zarr.group(store=dest_store)
    attrs = dict(root.attrs)
    dest_root.attrs.update(_trim_tile_index(attrs, tile_limit, total_tiles))

    for name in root.array_keys():
        _copy_subset_array(dest_root, name, root[name], tile_limit, total_tiles)

    for name in root.group_keys():
        _copy_tree(source / name, dest / name)

    for sidecar in _SIDECAR_FILES:
        sidecar_path = source / sidecar
        if sidecar_path.exists():
            _copy_file(sidecar_path, dest / sidecar)

    return {"mode": "subset", "tile_count": total_tiles, "copied_tile_count": tile_limit}


def _resolve_store_map(args: argparse.Namespace) -> dict[str, Path]:
    if args.v22_store_subset_path:
        result: dict[str, Path] = {}
        for raw_path in args.v22_store_subset_path:
            store_path = Path(raw_path).resolve()
            build = store_path.stem
            result[build] = store_path
        return result

    dataset_dir = Path(args.dataset_dir).resolve()
    result = {}
    for build in args.builds:
        result[str(build)] = dataset_dir / f"{build}.zarr"
    return result


def _copy_sources(bundle_root: Path, *, include_tests: bool) -> list[str]:
    copied: list[str] = []
    for relative in _SOURCE_FILES:
        source = _HARVESTER_ROOT / relative
        _copy_file(source, bundle_root / relative)
        copied.append(relative.replace("\\", "/"))
    for relative in _SOURCE_DIRS:
        if not include_tests and relative == "tests/v23":
            continue
        source = _HARVESTER_ROOT / relative
        _copy_tree(source, bundle_root / relative)
        copied.append(relative.replace("\\", "/"))
    return copied


def _write_bundle_runtime_files(bundle_root: Path) -> None:
    (bundle_root / "pyproject.toml").write_text(_BUNDLE_PYPROJECT, encoding="utf-8")
    (bundle_root / "requirements-runpod.txt").write_text(_BUNDLE_REQUIREMENTS, encoding="utf-8")
    readme = (
        "# Spec 089 V23 RunPod Bundle\n\n"
        "This bundle ships the V23 Python code, a bounded V22 subset, and helper scripts.\n"
        "It does not include staged clients, MPQs, or raw archive payloads.\n"
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


def _write_manifest(
    bundle_root: Path,
    *,
    bundle_name: str,
    store_reports: list[dict[str, Any]],
    copied_sources: list[str],
    tileset_prune_table: str | None,
) -> None:
    manifest = {
        "schema": "spec-089-v23-runpod-bundle",
        "v23_bundle_version": 1,
        "bundle_name": bundle_name,
        "contains_game_client_files": False,
        "store_reports": store_reports,
        "copied_sources": copied_sources,
        "tileset_prune_table": tileset_prune_table,
        "tree_hash": _tree_hash(bundle_root),
    }
    (bundle_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _make_tar(bundle_root: Path, output_tar: Path) -> Path:
    if output_tar.exists():
        output_tar.unlink()
    output_tar.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output_tar, mode="w") as archive:
        archive.add(bundle_root, arcname=bundle_root.name)
    return output_tar


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Package the Spec 089 V23 RunPod bundle.")
    parser.add_argument("--bundle-name", default=_default_bundle_name())
    parser.add_argument("--dataset-dir", type=Path, default=_DEFAULT_DATASET_DIR)
    parser.add_argument("--builds", nargs="+", default=["0_5_3_3368", "3_3_5_12340"])
    parser.add_argument("--v22-store-subset-path", action="append", default=None)
    parser.add_argument("--tileset-prune-table", default=None)
    parser.add_argument("--include-v22-subset-tiles", type=int, default=8)
    parser.add_argument("--output-root", type=Path, default=_DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-tar", type=Path, default=None)
    parser.add_argument("--archive-format", choices=["tar", "none"], default="tar")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-tests", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    bundle_root = Path(args.output_root).resolve() / str(args.bundle_name)
    if bundle_root.exists():
        if not args.overwrite:
            raise SystemExit(f"Bundle already exists: {bundle_root}")
        shutil.rmtree(bundle_root)
    bundle_root.mkdir(parents=True, exist_ok=False)

    store_map = _resolve_store_map(args)
    copied_sources = _copy_sources(bundle_root, include_tests=not bool(args.no_tests))
    _write_bundle_runtime_files(bundle_root)

    config_dir = bundle_root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    copied_prune = None
    if args.tileset_prune_table:
        prune_source = Path(args.tileset_prune_table).resolve()
        copied_prune = "config/tileset_prune_table.json"
        _copy_file(prune_source, bundle_root / copied_prune)

    data_root = bundle_root / "data" / "v22"
    data_root.mkdir(parents=True, exist_ok=True)
    store_reports: list[dict[str, Any]] = []
    for build, store_path in store_map.items():
        if not store_path.exists():
            raise SystemExit(f"Missing V22 store: {store_path}")
        dest = data_root / f"{build}.zarr"
        report = _copy_store(store_path, dest, int(args.include_v22_subset_tiles))
        report.update(
            {
                "build": build,
                "source": str(store_path),
                "dest": dest.relative_to(bundle_root).as_posix(),
            }
        )
        store_reports.append(report)

    _write_manifest(
        bundle_root,
        bundle_name=str(args.bundle_name),
        store_reports=store_reports,
        copied_sources=copied_sources,
        tileset_prune_table=copied_prune,
    )

    if args.archive_format == "tar":
        tar_path = args.output_tar.resolve() if args.output_tar is not None else bundle_root.with_suffix(".tar")
        _make_tar(bundle_root, tar_path)
        print(f"Wrote bundle tar: {tar_path}")
    print(f"Wrote bundle directory: {bundle_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
