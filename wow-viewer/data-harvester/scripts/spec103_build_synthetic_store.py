"""Spec 103 T009 — assemble the synthetic 13-channel training store.

Reads the synthetic manifest from spec103_make_synthetic_adts.py plus captured minimap PNGs
(or a clearly-labeled procedural fallback) and writes a zarr store with the same array names
the trainer reads from the real V18 store: minimap_rgb, height_257, normal_xyz, liquid_mask,
liquid_height, object_precise_mask (all zeros here — synthetic tiles have no objects/liquid).
Normals are derived analytically from the known height field. The WDL prior is NOT stored —
the assembler derives it from height_257 (the verified ::16 outer transform) at batch time.

Run from wow-viewer/data-harvester/ (fast, CPU-only):

    uv run python scripts/spec103_build_synthetic_store.py \
        --manifest ../output/spec103/synthetic/synthetic_manifest.json \
        --minimap-dir ../output/spec103/synthetic/captures \
        --output ../output/datasets/spec103/synthetic_v1.zarr

    # before a capture run exists, the loop stays testable with:
    #   --synthesize-minimaps   (hillshade render from the known height; labeled in attrs)
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr
from PIL import Image

CHUNK_METERS = 533.33333 / 16.0  # world meters per chunk; height grid step = tile/256


def normals_from_height(height_257: np.ndarray) -> np.ndarray:
    """Analytic unit normals from the known height field (finite differences)."""
    step = 533.33333 / 256.0
    gy, gx = np.gradient(height_257.astype(np.float64), step)
    nx, ny, nz = -gx, -gy, np.ones_like(gx)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    return np.stack([nx / norm, ny / norm, nz / norm], axis=-1).astype(np.float32)


def hillshade_minimap(height_257: np.ndarray) -> np.ndarray:
    """Procedural fallback minimap: fixed-light hillshade of the known height, 256×256 u8 RGB."""
    normals = normals_from_height(height_257)
    light = np.array([-0.5, -0.5, 0.72])
    light = light / np.linalg.norm(light)
    shade = np.clip(normals @ light, 0.0, 1.0)
    shade = (0.25 + 0.75 * shade)[:256, :256]
    rgb = np.stack([shade * 0.55, shade * 0.62, shade * 0.42], axis=-1)  # muted terrain green
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser(description="Spec 103 synthetic 13-channel store builder")
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--minimap-dir", type=Path, default=None, help="captured minimap PNGs named <tile_name>.png")
    ap.add_argument("--synthesize-minimaps", action="store_true",
                    help="use the procedural hillshade fallback for tiles without a captured PNG")
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    tiles = manifest["tiles"]
    if not tiles:
        raise SystemExit("manifest has no tiles")

    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing store: {output}")

    n = len(tiles)
    out = zarr.open_group(str(output), mode="w")
    arrays = {
        "minimap_rgb": out.create_array("minimap_rgb", shape=(n, 256, 256, 3), chunks=(1, 256, 256, 3), dtype=np.uint8),
        "height_257": out.create_array("height_257", shape=(n, 257, 257), chunks=(1, 257, 257), dtype=np.float32),
        "normal_xyz": out.create_array("normal_xyz", shape=(n, 257, 257, 3), chunks=(1, 257, 257, 3), dtype=np.float32),
        "liquid_mask": out.create_array("liquid_mask", shape=(n, 256, 256), chunks=(1, 256, 256), dtype=np.float32),
        "liquid_height": out.create_array("liquid_height", shape=(n, 256, 256), chunks=(1, 256, 256), dtype=np.float32),
        "object_precise_mask": out.create_array("object_precise_mask", shape=(n, 257, 257), chunks=(1, 257, 257), dtype=np.float32),
    }

    rows = []
    minimap_sources = {"captured": 0, "synthesized": 0}
    for row, tile in enumerate(tiles):
        tile_name = tile["tile_name"]
        height = np.load(tile["height_npy"]).astype(np.float32)
        if height.shape != (257, 257):
            raise SystemExit(f"{tile_name}: height must be (257, 257), got {height.shape}")

        png = (args.minimap_dir / f"{tile_name}.png") if args.minimap_dir else None
        if png is not None and png.exists():
            minimap = np.asarray(Image.open(png).convert("RGB").resize((256, 256), Image.BILINEAR), dtype=np.uint8)
            source = "captured"
        elif args.synthesize_minimaps:
            minimap = hillshade_minimap(height)
            source = "synthesized"
        else:
            raise SystemExit(
                f"{tile_name}: no captured minimap at {png} — run the capture commands or pass --synthesize-minimaps"
            )
        minimap_sources[source] += 1

        arrays["minimap_rgb"][row] = minimap
        arrays["height_257"][row] = height
        arrays["normal_xyz"][row] = normals_from_height(height)
        # zeros: synthetic tiles have no liquid, objects, or brush imprints by construction
        rows.append({
            "row": row, "map": tile["map"], "tile_x": int(tile["tile_x"]), "tile_y": int(tile["tile_y"]),
            "tile_id": row, "build": "synthetic", "pattern": tile["pattern"],
            "amplitude": float(tile["amplitude"]), "minimap_source": source,
        })

    pq.write_table(pa.Table.from_pylist(rows), output / "index.parquet")
    out.attrs.update({
        "schema": "spec103-synthetic-store-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "tile_count": n,
        "source_manifest": str(args.manifest.resolve()),
        "minimap_sources": minimap_sources,
        "signals": sorted(arrays),
        "wdl_prior_policy": "derived at batch time: outer = height_257[::16, ::16]; wdl_height_33 prohibited",
    })
    (output / "contract.json").write_text(json.dumps(dict(out.attrs), indent=2), encoding="utf-8")
    print(f"[spec103] wrote {n} tiles -> {output}")
    print(f"[spec103] minimap sources: {minimap_sources}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
