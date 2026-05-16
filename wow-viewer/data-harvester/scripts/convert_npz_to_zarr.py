"""Convert NPZ shard corpus to Zarr DirectoryStore format.

Usage:
    cd wow-viewer/data-harvester
    uv run python scripts/convert_npz_to_zarr.py                     # defaults
    uv run python scripts/convert_npz_to_zarr.py --build 3_0_1_8303  # single build
    uv run python scripts/convert_npz_to_zarr.py --dry-run            # count only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure src/ is on the path
_src_dir = Path(__file__).resolve().parent.parent / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from harvester.zarr_io import convert_npz_to_zarr  # noqa: E402

DEFAULT_NPZ_ROOT = (
    Path(__file__).resolve().parent.parent.parent
    / "output"
    / "datasets"
    / "d1_reharvest"
    / "shards"
)
DEFAULT_ZARR_ROOT = (
    Path(__file__).resolve().parent.parent.parent
    / "output"
    / "datasets"
    / "d1_reharvest"
    / "shards_zarr"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert NPZ shards to Zarr stores")
    p.add_argument(
        "--npz-root",
        type=Path,
        default=DEFAULT_NPZ_ROOT,
        help="Root of NPZ shard tree (shards/<build>/<map>/*.npz)",
    )
    p.add_argument(
        "--zarr-root", type=Path, default=DEFAULT_ZARR_ROOT, help="Output root for Zarr stores"
    )
    p.add_argument(
        "--build", type=str, default=None, help="Convert only this build (e.g. 3_0_1_8303)"
    )
    p.add_argument("--dry-run", action="store_true", help="Count shards without converting")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing Zarr stores")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    npz_root = Path(args.npz_root)
    zarr_root = Path(args.zarr_root)

    if not npz_root.exists():
        print(f"ERROR: NPZ root not found: {npz_root}")
        sys.exit(1)

    builds = (
        [args.build] if args.build else sorted(d.name for d in npz_root.iterdir() if d.is_dir())
    )

    total_npz = 0
    total_zarr = 0
    converted = 0
    skipped = 0
    errors = 0
    sw = time.perf_counter()

    for build in builds:
        build_dir = npz_root / build
        if not build_dir.is_dir():
            print(f"SKIP: not a directory: {build_dir}")
            continue

        npz_files = sorted(build_dir.glob("*/*.npz"))
        total_npz += len(npz_files)

        for npz_path in npz_files:
            rel = npz_path.relative_to(npz_root)
            zarr_path = zarr_root / rel.with_suffix(".zarr")

            if zarr_path.exists() and not args.overwrite:
                total_zarr += 1
                skipped += 1
                continue

            if args.dry_run:
                continue

            try:
                zarr_path.parent.mkdir(parents=True, exist_ok=True)
                convert_npz_to_zarr(npz_path, zarr_path, overwrite=args.overwrite)
                converted += 1
                total_zarr += 1
            except Exception as exc:
                errors += 1
                print(f"  ERROR [{npz_path.name}]: {exc}")

            if (converted + skipped + errors) % 100 == 0:
                elapsed = time.perf_counter() - sw
                rate = (converted + skipped) / max(elapsed, 0.001)
                print(f"  ... {converted + skipped}/{len(npz_files)} ({rate:.1f}/s) [{build}]")

    elapsed = time.perf_counter() - sw

    if args.dry_run:
        print(f"DRY RUN: {total_npz} NPZ shards found, {total_zarr} Zarr already exist")
    else:
        print(
            f"Done. Converted={converted} Skipped={skipped} Errors={errors} "
            f"in {elapsed:.1f}s ({total_npz / max(elapsed, 0.001):.1f}/s)"
        )
        print(f"Zarr root: {zarr_root}")


if __name__ == "__main__":
    main()
