"""Spec 097 Slice 1: per-map V18 Zarr -> single stitched OBJ + baked atlas.

Reads the per-tile V18 records for one map out of a multi-map V18 Zarr store,
runs the V24 minimap-only Stage A prior on each tile, upsamples the
(17,17) + (16,16) WDL prior to a 257x257 heightmap, applies edge alignment
across tile boundaries, and writes a single OBJ mesh + atlas PNG covering
the whole map.

Usage:
    uv run python scripts/v24_export_map.py \\
        --v18-store path/to/3_3_5_12340.zarr \\
        --map Northrend \\
        [--build 3_3_5_12340] \\
        [--curation-manifest path/to/kept_tiles.parquet] \\
        [--checkpoint path/to/stage_a.pt] \\
        [--output path/to/out_dir] \\
        [--device cuda] [--seed 94]

The output directory will contain:
  <map>.obj           - the stitched mesh (one per map)
  atlas.png           - the baked source-minimap texture
  <map>_manifest.json - per-tile provenance: which tiles were used, the
                       per-tile height range, the build, etc.
  tiles/<tx>_<ty>.prior.npz - per-tile WDL prior NPZ (small, for downstream
                              use by the WDL writer in Spec 097 Slice 2)

Edge alignment: the 16-pixel border on each tile's east edge is averaged
with the 16-pixel border on the west edge of its east neighbour, and the
same for north/south. Corner cells (4-way) are averaged from all four
contributing tiles. This produces a continuous heightmap with no visible
hard step at the 256-pixel-tile borders.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pyarrow.parquet as pq
import zarr

from harvester.v24 import lattice, stage_a, train_common  # noqa: E402
from harvester.v24.tiles import TileRecord, _normalize_holes  # noqa: E402

# Tile / lattice geometry. The WDL prior is 17 outer + 16 inner over a
# 256x256 minimap, so the upsampled 257x257 heightmap has a 1px overlap on
# the corner grid (per Spec 094 amendment A6). 533.333 world units per
# tile = one standard WoW tile.
TILE_SIZE = 533.333
PRIOR_FULL = 257
HEIGHT_SCALE = 100.0  # prior normalized in [0, 1] -> world units (Spec 094)


def _read_v18_index(v18_path: Path) -> dict[str, list]:
    return pq.read_table(str(v18_path / "index.parquet")).to_pydict()


def _load_v18_record(
    v18_group: zarr.Group,
    index: dict[str, list],
    row: int,
) -> TileRecord:
    """Build a minimal TileRecord from a V18 row, suitable for Stage A prior.

    The v24 TileSource expects a V24 store, so we re-implement the minimum
    we need here. Fields the Stage A prior requires: cleaned_minimap (256,256,3)
    and the (17,17) / (16,16) priors. For the minimap-only regime only
    cleaned_minimap matters; the rest are filled with safe defaults.
    """
    raw = np.asarray(v18_group["minimap_rgb"][row], dtype=np.float32)
    if raw.max() > 1.5:
        raw = raw / 255.0
    cleaned = raw.astype(np.float32)  # no cleaner pass; raw is the input

    return TileRecord(
        row=row,
        v18_row=row,
        map_name=index["map"][row],
        tile_x=int(index["tile_x"][row]),
        tile_y=int(index["tile_y"][row]),
        audit_empty=False,
        real_available=False,
        cleaned_minimap=cleaned,
        alpha=np.zeros((256, 256, 4), dtype=np.float32),
        normal=np.zeros((257, 257, 3), dtype=np.float32),
        mcnr_mask=np.ones((257, 257), dtype=np.float32),
        object_mask=np.zeros((257, 257), dtype=np.float32),
        liquid_mask=np.zeros((256, 256), dtype=np.float32),
        holes=np.zeros((16, 16), dtype=bool),
        height=np.zeros((257, 257), dtype=np.float32),
        prior_outer=np.zeros((17, 17), dtype=np.float32),
        prior_inner=np.zeros((16, 16), dtype=np.float32),
        source_outer=np.zeros((17, 17), dtype=np.uint8),
        source_inner=np.zeros((16, 16), dtype=np.uint8),
        confidence_outer=np.zeros((17, 17), dtype=np.float32),
        confidence_inner=np.zeros((16, 16), dtype=np.float32),
        synth_outer=np.zeros((17, 17), dtype=np.float32),
        synth_inner=np.zeros((16, 16), dtype=np.float32),
    )


def _discover_checkpoint(output_root: Path) -> Path:
    """Find the most recent minimap-only Stage A checkpoint under output_root/v24_validation/."""
    v24_root = output_root / "v24_validation"
    if not v24_root.exists():
        raise FileNotFoundError(
            f"v24 validation root not found: {v24_root}. "
            f"Train a minimap-only checkpoint first (see Spec 096)."
        )
    candidates: list[tuple[float, Path]] = []
    for run_dir in v24_root.iterdir():
        if not run_dir.is_dir() or not run_dir.name.startswith("v24_minimap_only"):
            continue
        ckpt = run_dir / "stage_a.pt"
        if ckpt.exists():
            candidates.append((ckpt.stat().st_mtime, ckpt))
    if not candidates:
        raise FileNotFoundError(
            f"no minimap-only checkpoint found under {v24_root}. "
            f"Train one with: "
            f"uv run python scripts/train_v24_stage_a.py --minimap-only ..."
        )
    candidates.sort(reverse=True)
    return candidates[0][1]


def _per_tile_priors(
    source: TileSource,
    rows: list[int],
    model: stage_a.StageAMinimapOnly,
    device: torch.device,
    log: callable | None = None,
) -> dict[int, np.ndarray]:
    """Compute the per-tile (17,17)+(16,16) WDL prior for each row.

    Returns a dict {row -> (257,257) float32 heightmap in world units}.
    """
    out: dict[int, np.ndarray] = {}
    n = len(rows)
    step = max(1, n // 20)
    started = time.time()
    for i, r in enumerate(rows, 1):
        record = source.load(r)
        x = stage_a.build_minimap_only_input(record.cleaned_minimap)  # (3,64,64)
        with torch.no_grad():
            outer, inner = model(torch.from_numpy(x)[None].to(device))
        outer = (outer[0].cpu().numpy() * HEIGHT_SCALE).astype(np.float32)
        inner = (inner[0].cpu().numpy() * HEIGHT_SCALE).astype(np.float32)
        out[r] = lattice.upsample_prior_257(outer, inner)
        if log and (i % step == 0 or i == n):
            elapsed = time.time() - started
            eta = elapsed / i * (n - i)
            log(f"  [prior] {i}/{n} tiles ({100.0 * i / n:.0f}%) "
                f"elapsed={elapsed:.1f}s eta={eta:.1f}s")
    return out


def _align_seams(
    per_tile: dict[tuple[int, int], np.ndarray],
    tile_rows: list[int],
    tile_cols: list[int],
) -> np.ndarray:
    """Apply edge alignment: average each shared 16-pixel border.

    Returns a (len(tile_rows) * 257, len(tile_cols) * 257) float32 array of
    world-space heights, continuous across tile boundaries.
    """
    H = len(tile_rows) * PRIOR_FULL
    W = len(tile_cols) * PRIOR_FULL
    full = np.zeros((H, W), dtype=np.float32)
    counts = np.zeros((H, W), dtype=np.float32)

    # Lay every tile's 257x257 onto the full grid (no alignment yet).
    for ri, ty in enumerate(tile_rows):
        for ci, tx in enumerate(tile_cols):
            tile = per_tile[(ty, tx)]
            y0, y1 = ri * PRIOR_FULL, (ri + 1) * PRIOR_FULL
            x0, x1 = ci * PRIOR_FULL, (ci + 1) * PRIOR_FULL
            # No X-flip: heightmap kept in original (un-flipped) image
            # orientation so the synthetic colour atlas (derived from
            # the same un-flipped heightmap) lines up with the mesh.
            full[y0:y1, x0:x1] += tile
            counts[y0:y1, x0:x1] += 1.0

    # 1-pixel shared-border alignment. Tile (ri, ci) and tile (ri, ci+1)
    # share exactly the 1-pixel column at world x = (ci+1) * TILE. The
    # earlier 16-pixel-band average produced a visible "weird border"
    # on the tiles; the user reported this and the fix is to align only
    # the 1-pixel shared column (no wide soft band anywhere).
    for ri in range(len(tile_rows)):
        for ci in range(len(tile_cols) - 1):
            seam_x = (ci + 1) * PRIOR_FULL
            shared = (full[:, seam_x - 1] + full[:, seam_x]) * 0.5
            full[:, seam_x - 1] = shared
            full[:, seam_x] = shared
    for ri in range(len(tile_rows) - 1):
        for ci in range(len(tile_cols)):
            seam_y = (ri + 1) * PRIOR_FULL
            shared = (full[seam_y - 1, :] + full[seam_y, :]) * 0.5
            full[seam_y - 1, :] = shared
            full[seam_y, :] = shared

    # Normalise remaining cells that have only one contributor (so they
    # are the simple per-tile value).
    mask = counts > 0
    full[mask] = full[mask] / counts[mask]
    return full


def _export_atlas(
    per_tile: dict[tuple[int, int], np.ndarray],
    tile_rows: list[int],
    tile_cols: list[int],
    out_path: Path,
) -> tuple[np.ndarray, dict[tuple[int, int], tuple[int, int]]]:
    """Bake the per-tile textures into a single atlas PNG.

    The atlas layout mirrors the OBJ layout: row 0 of the OBJ is the
    south end of the map (smallest Y) and the atlas row 0 holds the
    colour for that OBJ row. ``tile_rows`` is sorted descending (north
    first, south last), so the colour for ``tile_rows[ri]`` lands at
    atlas row ``len(tile_rows) - 1 - ri`` (south end first).

    Returns the atlas array and a dict mapping (tile_y, tile_x) -> (u0, v0)
    in atlas pixel coords (top-left corner of the per-tile sub-rect in
    *PNG* image space). The OBJ writer applies the standard V-flip
    (1.0 - y/h) so the atlas opens right-side-up in any 3D viewer.
    """
    H = len(tile_rows)
    W = len(tile_cols)
    tile_w, tile_h = 256, 256
    atlas = np.zeros((H * tile_h, W * tile_w, 3), dtype=np.uint8)
    uv_origin: dict[tuple[int, int], tuple[int, int]] = {}
    for ri, ty in enumerate(tile_rows):
        for ci, tx in enumerate(tile_cols):
            prior = per_tile[(ty, tx)]
            mean_h = float(prior.mean())
            t = np.clip((mean_h + 500.0) / 4500.0, 0.0, 1.0)
            r = int(255 * (0.267 + t * (0.105 - 0.267)))
            g = int(255 * (0.005 + t * (0.491 - 0.005)))
            b = int(255 * (0.329 + t * (0.741 - 0.329)))
            colour = np.full((tile_h, tile_w, 3), (r, g, b), dtype=np.uint8)
            # OBJ row 0 = south end = largest tile_y = tile_rows[-1].
            # So the colour for tile_rows[ri] lands at atlas row
            # (H - 1 - ri) so atlas row 0 is the south end.
            atlas_row = (H - 1 - ri) * tile_h
            y0, y1 = atlas_row, atlas_row + tile_h
            x0, x1 = ci * tile_w, (ci + 1) * tile_w
            atlas[y0:y1, x0:x1] = colour
            uv_origin[(ty, tx)] = (x0, y0)
    Image.fromarray(atlas, mode="RGB").save(str(out_path))
    return atlas, uv_origin


def _write_obj(
    height: np.ndarray,
    atlas: np.ndarray,
    uv_origin: dict[tuple[int, int], tuple[int, int]],
    tile_rows: list[int],
    tile_cols: list[int],
    out_path: Path,
) -> None:
    """Write the stitched OBJ + MTL pair.

    Atlas V is flipped (image-Y -> OBJ-Y), per the convention used in
    `v24_prior_to_obj.py` so meshes open right-side up in any 3D viewer.
    """
    H, W = height.shape
    atlas_h, atlas_w = atlas.shape[:2]
    lines: list[str] = [
        "mtllib terrain.mtl",
        "usemtl terrain",
        "",
    ]
    # Vertices: row-major.
    for y in range(H):
        for x in range(W):
            wx = (x / (PRIOR_FULL - 1)) * TILE_SIZE
            wy = (y / (PRIOR_FULL - 1)) * TILE_SIZE
            wz = float(height[y, x])
            lines.append(f"v {wx:.6f} {wy:.6f} {wz:.6f}")
    lines.append("")
    # Texture coords: per-tile sub-rect of the atlas.
    for ri, _ty in enumerate(tile_rows):
        for ci, _tx in enumerate(tile_cols):
            ax0, ay0 = uv_origin[(_ty, _tx)]
            for y in range(PRIOR_FULL):
                for x in range(PRIOR_FULL):
                    u = (ax0 + x) / atlas_w
                    v = 1.0 - ((ay0 + y) / atlas_h)
                    lines.append(f"vt {u:.6f} {v:.6f}")
    lines.append("")
    # Faces: two triangles per grid cell.
    v_offset = 0
    for ri in range(len(tile_rows)):
        for ci in range(len(tile_cols)):
            for y in range(PRIOR_FULL - 1):
                for x in range(PRIOR_FULL - 1):
                    v00 = v_offset + y * PRIOR_FULL + x + 1
                    v10 = v_offset + y * PRIOR_FULL + (x + 1) + 1
                    v01 = v_offset + (y + 1) * PRIOR_FULL + x + 1
                    v11 = v_offset + (y + 1) * PRIOR_FULL + (x + 1) + 1
                    lines.append(f"f {v00}/{v00} {v10}/{v10} {v01}/{v01}")
                    lines.append(f"f {v10}/{v10} {v11}/{v11} {v01}/{v01}")
            v_offset += PRIOR_FULL * PRIOR_FULL
    out_path.write_text("\n".join(lines), encoding="utf-8")
    (out_path.parent / "terrain.mtl").write_text(
        "\n".join(
            [
                "newmtl terrain",
                f"map_Kd {out_path.parent.name}.atlas.png",
                "Ka 1.0 1.0 1.0",
                "Kd 1.0 1.0 1.0",
                "Ns 0.0",
                "d 1.0",
            ]
        ),
        encoding="utf-8",
    )


def _load_curation_keepset(manifest_path: Path, build: str) -> set[tuple[str, int]] | None:
    """Load curated kept tile_ids for one build from a V18 curation manifest."""
    import pyarrow.parquet as pq

    table = pq.read_table(str(manifest_path))
    builds = table.column("build").to_pylist()
    tile_ids = table.column("tile_id").to_pylist()
    keep = table.column("keep").to_pylist()
    out: set[tuple[str, int]] = set()
    for b, tid, k in zip(builds, tile_ids, keep, strict=True):
        if b == build and k:
            out.add((b, int(tid)))
    return out or None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v18-store", required=True, type=Path)
    parser.add_argument("--map", required=True, help="map name (e.g. Northrend)")
    parser.add_argument("--build", default="3_3_5_12340")
    parser.add_argument("--curation-manifest", type=Path, default=None,
                        help="optional V18 curation kept_tiles.parquet")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="override the auto-discovered minimap-only Stage A checkpoint")
    parser.add_argument("--output", type=Path, default=None,
                        help="output directory (default: ./output/v24_maps/<map>)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=94)
    args = parser.parse_args()

    train_common.set_determinism(args.seed, strict=True)
    device = train_common.pick_device(args.device)

    if not args.v18_store.exists():
        raise FileNotFoundError(f"v18 store not found: {args.v18_store}")

    # Read the V18 store directly (not via the V24 TileSource — that wants a
    # V24 store with prior arrays, which we don't need for a minimap-only
    # prior that is computed on the fly here).
    v18 = zarr.open_group(str(args.v18_store), mode="r")
    index = _read_v18_index(args.v18_store)
    # Per-map filter.
    rows = [r for r in range(len(index["tile_id"]))
            if index["map"][r].lower() == args.map.lower()]
    if args.curation_manifest:
        keepset = _load_curation_keepset(args.curation_manifest, args.build)
        if keepset:
            rows = [r for r in rows
                    if (index["build"][r], int(index["tile_id"][r])) in keepset]
    if not rows:
        raise RuntimeError(f"no tiles for map={args.map} in {args.v18_store}")

    # Group by (tile_y, tile_x); we'll need the (rows, cols) layout.
    by_tile: dict[tuple[int, int], int] = {}
    for r in rows:
        key = (int(index["tile_y"][r]), int(index["tile_x"][r]))
        if key in by_tile:
            continue
        by_tile[key] = r
    tile_y_set = sorted({k[0] for k in by_tile})
    tile_x_set = sorted({k[1] for k in by_tile})
    tile_rows = sorted(tile_y_set, reverse=True)
    tile_cols = sorted(tile_x_set)
    print(f"map: {args.map} ({len(tile_y_set)} rows x {len(tile_x_set)} cols = "
          f"{len(by_tile)} tiles, {len(rows)} V18 rows after curation)")

    # Checkpoint.
    if args.checkpoint is None:
        # Walk up from the V18 store to the project root: <root>/output/datasets/v18/<store>.zarr
        # -> <root> is args.v18_store.parent.parent.parent
        project_root = args.v18_store.parent.parent.parent
        args.checkpoint = _discover_checkpoint(project_root)
    print(f"checkpoint: {args.checkpoint}")
    ckpt = torch.load(str(args.checkpoint), map_location=device, weights_only=True)
    model = stage_a.StageAMinimapOnly(base=ckpt["config"]["base"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Per-tile priors.
    per_tile: dict[tuple[int, int], np.ndarray] = {}
    started = time.time()
    for ri, ty in enumerate(tile_rows, 1):
        for ci, tx in enumerate(tile_cols, 1):
            row = by_tile.get((ty, tx))
            if row is None:
                # Missing tile (boundary of a partial map). Use per-map mean.
                # We'll fill in after we know the per-map mean.
                per_tile[(ty, tx)] = None  # type: ignore[assignment]
                continue
            record = _load_v18_record(v18, index, row)
            x = stage_a.build_minimap_only_input(record.cleaned_minimap)
            with torch.no_grad():
                outer, inner = model(torch.from_numpy(x)[None].to(device))
            outer = (outer[0].cpu().numpy() * HEIGHT_SCALE).astype(np.float32)
            inner = (inner[0].cpu().numpy() * HEIGHT_SCALE).astype(np.float32)
            per_tile[(ty, tx)] = lattice.upsample_prior_257(outer, inner)
        elapsed = time.time() - started
        eta = elapsed / max(1, ri) * (len(tile_rows) - ri)
        print(f"  [tile rows] {ri}/{len(tile_rows)} done, "
              f"elapsed={elapsed:.1f}s eta={eta:.1f}s")

    # Per-map mean for missing tiles.
    real_heights = [v for v in per_tile.values() if v is not None]
    if not real_heights:
        raise RuntimeError("no real tile heights computed; nothing to export")
    map_mean = float(np.mean([v.mean() for v in real_heights]))
    map_mean_field = np.full((PRIOR_FULL, PRIOR_FULL), map_mean, dtype=np.float32)
    per_tile = {k: (v if v is not None else map_mean_field)
                for k, v in per_tile.items()}

    # Edge alignment + stitch.
    print("aligning seams ...")
    full_height = _align_seams(per_tile, tile_rows, tile_cols)

    # Atlas + OBJ.
    out_dir = (args.output or (args.v18_store.parent.parent / "output" / "v24_maps" / args.map)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    atlas, uv_origin = _export_atlas(per_tile, tile_rows, tile_cols, out_dir / f"{args.map}.atlas.png")
    obj_path = out_dir / f"{args.map}.obj"
    _write_obj(full_height, atlas, uv_origin, tile_rows, tile_cols, obj_path)

    # Per-tile prior NPZs (small, for Slice 2 WDL writer).
    tiles_dir = out_dir / "tiles"
    tiles_dir.mkdir(exist_ok=True)
    for (ty, tx), prior in per_tile.items():
        # The prior NPZ carries the original (17,17)+(16,16) grids, not the
        # 257x257 upsample. Re-derive them from the model run would be
        # wasteful; we have the per-tile 257x257, but the WDL writer needs
        # the raw outer/inner. For now, save the 257x257 and let the
        # downstream code extract the 17x17 / 16x16 if needed.
        np.savez(
            tiles_dir / f"{tx}_{ty}.prior.npz",
            height_257=prior.astype(np.float32),
            tile_x=tx, tile_y=ty,
        )

    # Manifest.
    manifest = {
        "map": args.map,
        "build": args.build,
        "checkpoint": str(args.checkpoint),
        "rows": len(tile_rows),
        "cols": len(tile_cols),
        "n_tiles": len(by_tile),
        "n_v18_rows": len(rows),
        "world_height_min": float(full_height.min()),
        "world_height_max": float(full_height.max()),
        "world_height_mean": float(full_height.mean()),
        "map_mean_for_missing_tiles": map_mean,
        "seed": args.seed,
        "device": str(device),
        "wall_s": time.time() - started,
        "obj": str(obj_path),
        "atlas": str(out_dir / f"{args.map}.atlas.png"),
        "tiles_dir": str(tiles_dir),
    }
    (out_dir / f"{args.map}_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print()
    print(f"exported: {out_dir}")
    print(f"  {args.map}.obj    ({full_height.shape[0]} x {full_height.shape[1]} heightmap, "
          f"{full_height.size} vertices, world {full_height.min():.1f}..{full_height.max():.1f})")
    print(f"  {args.map}.atlas.png  ({atlas.shape[1]} x {atlas.shape[0]})")
    print(f"  {args.map}_manifest.json")
    print(f"  tiles/  ({len(per_tile)} per-tile prior NPZs)")
    print(f"  wall time: {manifest['wall_s']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
