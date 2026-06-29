"""End-to-end proof for the spec 077 per-object capture library builder.

This test exercises the full path: capture-job enumeration JSONL → builder →
Zarr + Parquet outputs → review HTML. It uses synthetic placement rows and
PNG captures on disk so it runs without any game client or capture tool.

It is the validation gate for spec 077 T012/T013 in CI; the same code path
runs on real data once a V18 Zarr store is available.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
import zarr
import zarr.storage

# Make the scripts and harvester package importable. The scripts are not a
# real Python package; we adjust sys.path so ``import build_object_library``
# and ``from harvester import object_library`` both work.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
for _entry in (_REPO_ROOT, _SRC_DIR, _SCRIPTS_DIR):
    _entry_str = str(_entry)
    if _entry_str not in sys.path:
        sys.path.insert(0, _entry_str)

import build_object_library  # noqa: E402
import review_object_library  # noqa: E402


def _make_capture_artifact(captures_dir: Path, variant_id: str, color: tuple[int, int, int]) -> None:
    image = Image.new("RGB", (256, 256), color)
    mask = Image.new("L", (256, 256), 200)
    pose = {
        "capture_confidence": 0.9,
        "capture_notes": f"synthetic {variant_id}",
    }
    image.save(captures_dir / f"{variant_id}_image.png")
    mask.save(captures_dir / f"{variant_id}_mask.png")
    (captures_dir / f"{variant_id}_pose.json").write_text(json.dumps(pose), encoding="utf-8")


def _jobs_jsonl(jobs: list[dict]) -> str:
    return "\n".join(json.dumps(job, sort_keys=True) for job in jobs) + "\n"


def test_builder_writes_zarr_and_parquet_end_to_end() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        jobs_path = root / "jobs.jsonl"
        captures_dir = root / "captures"
        captures_dir.mkdir()
        output_root = root / "out"
        run_name = "proof_run"

        jobs = [
            {
                "build": "3_3_5_12340",
                "instance_type": "modf",
                "asset_path": "World\\wmo\\Azeroth\\Stormwind.wmo",
                "normalized_asset_path": "world/wmo/azeroth/stormwind.wmo",
                "library_id": "objlib_abc",
                "asset_type": "wmo",
                "observation_count": 5,
                "source_builds": ["3_3_5_12340"],
                "source_maps": ["Azeroth"],
                "first_tile_id": 600,
                "first_unique_id": 1234,
                "first_pos_x": 0.0,
                "first_pos_y": 0.0,
                "first_pos_z": 0.0,
                "first_rot_x": 0.0,
                "first_rot_y": 0.0,
                "first_rot_z": 0.0,
                "first_scale": 1.0,
            },
            {
                "build": "3_3_5_12340",
                "instance_type": "mddf",
                "asset_path": "World\\Generic\\Apple.m2",
                "normalized_asset_path": "world/generic/apple.m2",
                "library_id": "objlib_def",
                "asset_type": "m2",
                "observation_count": 1,
                "source_builds": ["3_3_5_12340"],
                "source_maps": ["Azeroth"],
                "first_tile_id": 600,
                "first_unique_id": 9999,
                "first_pos_x": 1.0,
                "first_pos_y": 2.0,
                "first_pos_z": 0.5,
                "first_rot_x": 0.0,
                "first_rot_y": 0.0,
                "first_rot_z": 0.0,
                "first_scale": 1.0,
            },
            {
                "build": "3_3_5_12340",
                "instance_type": "modf",
                "asset_path": "World\\wmo\\Azeroth\\Uncaptured.wmo",
                "normalized_asset_path": "world/wmo/azeroth/uncaptured.wmo",
                "library_id": "objlib_xyz",
                "asset_type": "wmo",
                "observation_count": 1,
                "source_builds": ["3_3_5_12340"],
                "source_maps": ["Azeroth"],
                "first_tile_id": 600,
                "first_unique_id": 8888,
                "first_pos_x": 0.0,
                "first_pos_y": 0.0,
                "first_pos_z": 0.0,
                "first_rot_x": 0.0,
                "first_rot_y": 0.0,
                "first_rot_z": 0.0,
                "first_scale": 1.0,
            },
        ]
        jobs_path.write_text(_jobs_jsonl(jobs), encoding="utf-8")

        # Pre-compute the variant ids the builder will use so we can stage
        # the matching capture artifacts on disk.
        from harvester.object_library import (  # noqa: PLC0415
            library_id_from_asset_path,
            make_variant_id,
        )

        for asset_path, color in [
            ("world/wmo/azeroth/stormwind.wmo", (255, 0, 0)),
            ("world/generic/apple.m2", (0, 255, 0)),
        ]:
            library_id = library_id_from_asset_path(asset_path)
            variant_id = make_variant_id(
                library_id=library_id,
                capture_build="3_3_5_12340",
                capture_mode="orthographic_topdown",
                rot_x=0.0,
                rot_y=0.0,
                rot_z=0.0,
                scale=1.0,
            )
            _make_capture_artifact(captures_dir, variant_id, color)

        # Run the builder as a real CLI invocation so we exercise arg parsing
        # and exit code, not just the inner helpers.
        exit_code = build_object_library.main_with_args(
            [
                "--jobs", str(jobs_path),
                "--captures-dir", str(captures_dir),
                "--output-root", str(output_root),
                "--run-name", run_name,
                "--target-size", "64",
            ]
        )
        assert exit_code == 0, f"builder exit code was {exit_code}"

        store_path = output_root / f"{run_name}.zarr"
        assert store_path.exists(), f"store not written: {store_path}"
        assert (store_path / "assets.parquet").exists()
        assert (store_path / "index.parquet").exists()

        # Read back the store and verify shape/content.
        store = zarr.storage.LocalStore(str(store_path), read_only=True)
        zarr_root = zarr.open_group(store, mode="r")
        rgb = np.asarray(zarr_root["capture_rgb"][:])
        mask = np.asarray(zarr_root["capture_mask"][:])
        assert rgb.shape == (3, 64, 64, 3), rgb.shape
        assert mask.shape == (3, 64, 64), mask.shape
        # Captured entries should have non-zero masks; uncaptured should be all zero.
        captured_mask_sums = mask.sum(axis=(1, 2))
        assert captured_mask_sums[0] > 0
        assert captured_mask_sums[1] > 0
        assert captured_mask_sums[2] == 0

        assets = pq.read_table(str(store_path / "assets.parquet")).to_pylist()
        statuses = sorted(a["capture_status"] for a in assets)
        assert statuses == ["captured", "captured", "not_attempted"], statuses

        variants = pq.read_table(str(store_path / "index.parquet")).to_pylist()
        assert len(variants) == 3

        # Run the reviewer and ensure HTML + families/ are written.
        review_dir = root / "review"
        exit_code = review_object_library.main_with_args(
            [
                "--library", str(store_path),
                "--output-dir", str(review_dir),
            ]
        )
        assert exit_code == 0
        assert (review_dir / "index.html").exists()
        family_sheets = list((review_dir / "families").glob("*.png"))
        assert len(family_sheets) == 3, [p.name for p in family_sheets]
