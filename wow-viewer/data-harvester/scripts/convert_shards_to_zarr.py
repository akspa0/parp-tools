"""Convert harvested NPZ shards to Zarr ZipStore format.

Usage:
    cd wow-viewer/data-harvester
    uv run python scripts/convert_shards_to_zarr.py [--shard-root <dir>] [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure src/ is on the path so 'harvester' package is importable
_src_dir = Path(__file__).resolve().parent.parent / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from harvester.zarr_store import npz_to_zarr_zipstore  # noqa: E402

DEFAULT_SHARD_ROOT = (
    Path(__file__).resolve().parent.parent.parent
    / "output" / "datasets" / "full_shard_batch_staged_native" / "shards"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert NPZ shards to Zarr ZipStore")
    p.add_argument("--shard-root", type=Path, default=DEFAULT_SHARD_ROOT)
    p.add_argument("--dry-run", action="store_true", help="List shards without converting")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing .zarr.zip files")
    p.add_argument("--limit", type=int, default=0, help="Max shards to convert (0=all)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.shard_root)
    if not root.is_dir():
        print(f"Error: shard root not found: {root}")
        sys.exit(1)

    npz_paths = sorted(root.glob("*/*/*.npz"))
    if args.limit > 0:
        npz_paths = npz_paths[: args.limit]

    print(f"Found {len(npz_paths)} NPZ shards")
    if args.dry_run:
        for p in npz_paths[:10]:
            zarr_path = p.with_suffix("").with_suffix(".zarr.zip")
            status = "exists" if zarr_path.exists() else "new"
            print(f"  {p.relative_to(root)} -> {zarr_path.name}  [{status}]")
        if len(npz_paths) > 10:
            print(f"  ... and {len(npz_paths) - 10} more")
        return

    converted = 0
    skipped = 0
    errors = 0
    t0 = time.perf_counter()

    for i, npz_path in enumerate(npz_paths):
        zarr_path = npz_path.with_suffix("").with_suffix(".zarr.zip")
        if zarr_path.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            npz_to_zarr_zipstore(npz_path, zarr_path, overwrite=args.overwrite)
            converted += 1
        except Exception as exc:
            errors += 1
            print(f"  ERROR {npz_path.name}: {exc}")

        if (i + 1) % 500 == 0:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  {i + 1}/{len(npz_paths)}  ({rate:.1f} shards/s)")

    elapsed = time.perf_counter() - t0
    print(
        f"Done: {converted} converted, {skipped} skipped, {errors} errors "
        f"in {elapsed:.1f}s ({converted / elapsed:.1f} shards/s)"
    )


if __name__ == "__main__":
    main()
