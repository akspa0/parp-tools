"""Build a trainable spec 076 terrain-art primitive library."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from harvester.fractal_library import (  # noqa: E402
    DEFAULT_ACCEPTED_LABELS,
    build_trainable_library,
    smoke_load_library,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build trainable full-map fractal brush library.")
    parser.add_argument("--canvas-dir", type=Path, required=True)
    parser.add_argument("--regions", type=Path, required=True, help="fractal_regions.parquet or containing segments dir")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--accepted-labels", default=",".join(DEFAULT_ACCEPTED_LABELS))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--smoke-count", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    accepted_labels = tuple(label.strip() for label in str(args.accepted_labels).split(",") if label.strip())
    summary = build_trainable_library(
        canvas_dir=args.canvas_dir,
        regions_path=args.regions,
        output_dir=args.output_dir,
        crop_size=int(args.crop_size),
        accepted_labels=accepted_labels,
        max_samples=args.max_samples,
    )
    smoke = smoke_load_library(args.output_dir, count=int(args.smoke_count))
    if smoke["loaded"] < int(args.smoke_count):
        raise SystemExit(f"Smoke loader loaded {smoke['loaded']} samples, expected at least {args.smoke_count}")

    print("Full-map fractal brush library built", flush=True)
    print(f"  output_dir: {args.output_dir}", flush=True)
    print(f"  sample_count: {summary['sample_count']}", flush=True)
    print(f"  rejected_count: {summary['rejected_count']}", flush=True)
    print(f"  split_counts: {summary['split_counts']}", flush=True)
    print(f"  smoke: {json.dumps(smoke, sort_keys=True)}", flush=True)
    print(f"  samples_zarr: {Path(args.output_dir) / 'samples.zarr'}", flush=True)
    print(f"  samples_parquet: {Path(args.output_dir) / 'samples.parquet'}", flush=True)


if __name__ == "__main__":
    main()
