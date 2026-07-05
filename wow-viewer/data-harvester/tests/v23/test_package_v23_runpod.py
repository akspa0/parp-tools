from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import zarr
import zarr.storage

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import package_v23_runpod  # noqa: E402
from tests.v23.support import make_synthetic_v22_store  # noqa: E402

pytestmark = pytest.mark.v23


def test_package_v23_runpod_writes_subset_bundle(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    build = "3_3_5_12340"
    store_path = make_synthetic_v22_store(dataset_dir / f"{build}.zarr", build=build, tile_count=6)
    prune_path = tmp_path / "tileset_prune.json"
    prune_path.write_text(json.dumps({"7": 0, "8": 1}), encoding="utf-8")
    curation_path = tmp_path / "kept_tiles.parquet"
    curation_path.write_bytes(b"placeholder parquet bytes for packaging contract")
    output_root = tmp_path / "dist"

    exit_code = package_v23_runpod.main(
        [
            "--bundle-name",
            "pkg",
            "--dataset-dir",
            str(dataset_dir),
            "--builds",
            build,
            "--tileset-prune-table",
            str(prune_path),
            "--curation-manifest",
            str(curation_path),
            "--include-v22-subset-tiles",
            "2",
            "--output-root",
            str(output_root),
            "--archive-format",
            "none",
        ]
    )

    assert exit_code == 0
    bundle = output_root / "pkg"
    assert (bundle / "runpod" / "v23" / "install_deps.sh").exists()
    assert (bundle / "scripts" / "train_v23_height.py").exists()
    assert (bundle / "src" / "harvester" / "v22_zarr_io.py").exists()
    assert (bundle / "src" / "harvester" / "v23" / "__init__.py").exists()
    assert (bundle / "config" / "tileset_prune_table.json").exists()
    assert (bundle / "config" / "curation_manifest.parquet").exists()

    subset_store = bundle / "data" / "v22" / f"{build}.zarr"
    root = zarr.open_group(store=zarr.storage.LocalStore(str(subset_store), read_only=True), mode="r")
    assert int(root["height_257"].shape[0]) == 2
    assert len(list(root.attrs.get("tile_index", []))) == 2

    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["contains_game_client_files"] is False
    assert manifest["tileset_prune_table"] == "config/tileset_prune_table.json"
    assert manifest["curation_manifest"] == "config/curation_manifest.parquet"
    assert manifest["store_reports"][0]["mode"] == "subset"
    assert manifest["store_reports"][0]["copied_tile_count"] == 2
    assert manifest["tree_hash"]

    assert store_path.exists()
