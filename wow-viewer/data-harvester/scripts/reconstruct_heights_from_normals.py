"""Reconstruct height fields from normals for mismatched tiles.

Reads the mismatch report, applies Frankot-Chellappa normal integration,
and writes corrected heights to a sidecar Zarr store.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import zarr
import zarr.codecs
import zarr.storage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from harvester.normal_height_reconstructor import (
    reconstruct_height_from_normals,
    anchor_heights,
)
from harvester.mismatch_detector import MismatchReport


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct heights from normals for mismatched tiles"
    )
    parser.add_argument(
        "--dataset-dir", type=str, required=True,
        help="Root directory containing <build>.zarr stores",
    )
    parser.add_argument(
        "--mismatch-report", type=str, required=True,
        help="Path to mismatch report parquet",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output sidecar Zarr store path (default: <dataset-dir>/v18_mismatch_repair.zarr)",
    )
    parser.add_argument(
        "--nz-clip", type=float, default=0.05,
        help="Minimum |nz| before gradient clamping (default: 0.05)",
    )
    parser.add_argument(
        "--preview-dir", type=str, default=None,
        help="Optional directory for before/after PNG previews",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)

    output_path = args.output
    if not output_path:
        output_path = str(dataset_dir / "v18_mismatch_repair.zarr")
    output_path = Path(output_path)

    report_path = Path(args.mismatch_report)
    if not report_path.exists():
        print(f"ERROR: mismatch report not found: {report_path}", file=sys.stderr)
        sys.exit(1)

    import pyarrow.parquet as pq
    table = pq.read_table(str(report_path))
    report_rows = table.to_pylist()

    if not report_rows:
        print("No mismatch tiles to reconstruct. Empty report.")
        return

    print(f"Loaded {len(report_rows)} mismatched tiles from report")

    preview_dir = None
    if args.preview_dir:
        preview_dir = Path(args.preview_dir)
        preview_dir.mkdir(parents=True, exist_ok=True)

    builds = sorted(set(r["build"] for r in report_rows))

    codec = zarr.codecs.BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle")

    for build in builds:
        store_path = dataset_dir / f"{build}.zarr"
        if not store_path.exists():
            print(f"WARNING: store not found: {store_path}", file=sys.stderr)
            continue

        store = zarr.storage.LocalStore(str(store_path), read_only=True)
        root = zarr.open_group(store, mode="r")

        n_tiles = root["height_257"].shape[0]

        build_rows = [r for r in report_rows if r["build"] == build]
        print(f"\n  {build}: reconstructing {len(build_rows)} tiles (store has {n_tiles} total)")

        sidecar_store_path = output_path / build
        sidecar_store_path.mkdir(parents=True, exist_ok=True)

        sstore = zarr.storage.LocalStore(str(sidecar_store_path))

        has_height_corrected = False
        try:
            existing = zarr.open_array(sstore, path="height_corrected_257")
            if existing.shape == (n_tiles, 257, 257):
                print("    sidecar height_corrected_257 already exists, will update specific tiles")
                has_height_corrected = True
        except Exception:
            pass

        if not has_height_corrected:
            print(f"    creating height_corrected_257 shape=({n_tiles}, 257, 257)")
            corrected = zarr.create_array(
                sstore,
                name="height_corrected_257",
                shape=(n_tiles, 257, 257),
                dtype=np.float32,
                chunks=(32, 129, 129),
                fill_value=np.nan,
                compressors=[codec],
            )
        else:
            corrected = zarr.open_array(sstore, path="height_corrected_257")

        for row in build_rows:
            tid = int(row["tile_id"])
            original = root["height_257"][tid].astype(np.float32, copy=False)
            normals = root["normal_xyz"][tid].astype(np.float32, copy=False)

            if "normal_mask" in root:
                nmask = root["normal_mask"][tid].astype(np.float32, copy=False)
            else:
                nmask = np.ones((257, 257), dtype=np.float32)

            reconstructed = reconstruct_height_from_normals(
                normals,
                normal_mask=nmask,
                nz_clip=args.nz_clip,
            )
            anchored = anchor_heights(reconstructed, original, normal_mask=nmask)

            corrected[tid, :, :] = anchored

            if preview_dir and tid == build_rows[0]["tile_id"]:
                _write_preview(preview_dir, build, tid, original, anchored, nmask)

    print(f"\nWrote sidecar repair store to {output_path}")


def _write_preview(
    preview_dir: Path,
    build: str,
    tile_id: int,
    original: np.ndarray,
    corrected: np.ndarray,
    normal_mask: np.ndarray,
) -> None:
    try:
        from PIL import Image

        def _to_img(data: np.ndarray) -> Image.Image:
            vmin = float(data.min())
            vmax = float(data.max())
            if vmax <= vmin:
                vmax = vmin + 1.0
            normed = ((data - vmin) / (vmax - vmin) * 255).clip(0, 255).astype(np.uint8)
            return Image.fromarray(normed, mode="L")

        orig_img = _to_img(original)
        corr_img = _to_img(corrected)
        combined = Image.new("L", (257 * 2 + 10, 257))
        combined.paste(orig_img, (0, 0))
        combined.paste(corr_img, (257 + 10, 0))

        fname = preview_dir / f"{build}_tile_{tile_id:04d}_height_before_after.png"
        combined.save(str(fname))
        print(f"    wrote preview: {fname}")
    except ImportError:
        print("    PIL not available, skipping preview PNG")


if __name__ == "__main__":
    main()
