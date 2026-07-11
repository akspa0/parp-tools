"""Spec 097 (follow-on) — axis orientation probe using a 2x2 tile cluster.

A single tile is untextured enough that mirroring is hard to see. So
this script picks a 2x2 cluster of adjacent tiles (4 tiles), runs the
model on each, and writes 8 axis-permutation variants of the cluster's
OBJ+MTL+PNG. The cluster has enough detail (textures, edges, gradient
across the cluster) to make the orientation unambiguous.

Each variant's texture is scored against the source PNG: the variant
with the smallest mean per-pixel distance is the one where the
texture is used unchanged. The script auto-picks the right
permutation and prints the matching flags for the full-folder quilt
command.

Usage:
    uv run python scripts/v24_axis_probe.py \\
        --input-dir path/to/folder/of/tiles/
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
V24_VALIDATION_ROOT = SCRIPT_DIR.parent.parent / "output" / "v24_validation"
TILE_SIZE = 533.333
HEIGHT_SCALE = 100.0

sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))
from harvester.v24 import lattice, stage_a, train_common  # noqa: E402

PERMUTATIONS: list[tuple[bool, bool, bool, str]] = [
    (False, False, False, "identity"),
    (True,  False, False, "flip_x"),
    (False, True,  False, "flip_y"),
    (True,  True,  False, "flip_xy"),
    (False, False, True,  "flip_z"),
    (True,  False, True,  "flip_xz"),
    (False, True,  True,  "flip_yz"),
    (True,  True,  True,  "flip_xyz"),
]


def _discover_checkpoint() -> Path:
    candidates: list[tuple[float, Path]] = []
    for run_dir in V24_VALIDATION_ROOT.iterdir():
        if not run_dir.is_dir() or not run_dir.name.startswith("v24_minimap_only"):
            continue
        ckpt = run_dir / "stage_a.pt"
        if ckpt.exists():
            candidates.append((ckpt.stat().st_mtime, ckpt))
    if not candidates:
        raise FileNotFoundError(
            f"no minimap-only checkpoint found under {V24_VALIDATION_ROOT}. "
            f"Train one first."
        )
    candidates.sort(reverse=True)
    return candidates[0][1]


def _parse_xy(stem: str) -> tuple[int, int] | None:
    m = re.match(r"^tile[_-](?P<x>\d+)[_-](?P<y>\d+)$", stem, re.IGNORECASE)
    if m:
        return int(m.group("x")), int(m.group("y"))
    m = re.match(r"^(?P<x>\d+)[_-](?P<y>\d+)$", stem)
    if m:
        return int(m.group("x")), int(m.group("y"))
    return None


def _load_minimap(path: Path) -> np.ndarray:
    with Image.open(path) as src:
        rgb = src.convert("RGB")
        if rgb.size != (256, 256):
            rgb = rgb.resize((256, 256), Image.Resampling.BILINEAR)
        return np.asarray(rgb, dtype=np.float32) / 255.0


def _write_cluster(
    tiles: list[tuple[np.ndarray, np.ndarray, str, int, int]],
    obj_path: Path,
    flip_x: bool,
    flip_y: bool,
    flip_z: bool,
) -> None:
    """Write a 2x2 cluster OBJ + per-tile PNGs + per-tile MTLs.

    Each tile is a tuple of (height_257, tex_256, name, world_x, world_y).
    Adjacent tiles share edges; we apply the same flip to all of them.
    Each tile gets its OWN material so the textures are distinct.
    """
    obj_lines: list[str] = [f"mtllib {obj_path.stem}.mtl", ""]
    v_offset = 0
    for height, tex, name, world_x, world_y in tiles:
        h = height.copy()
        if flip_z:
            h = -h
        if flip_y:
            h = np.flipud(h)
        if flip_x:
            h = np.fliplr(h)
        tex = tex.copy()
        if flip_y:
            tex = np.flipud(tex)
        if flip_x:
            tex = np.fliplr(tex)
        # Save the per-tile texture PNG.
        Image.fromarray(tex, mode="RGB").save(str(obj_path.parent / f"{name}.png"))
        # Vertices (world space).
        h_grid, w_grid = h.shape
        rows, cols = h_grid - 1, w_grid - 1
        for y in range(h_grid):
            for x in range(w_grid):
                wx = world_x + (x / cols) * TILE_SIZE
                wy = world_y + (y / rows) * TILE_SIZE
                wz = float(h[y, x])
                obj_lines.append(f"v {wx:.4f} {wy:.4f} {wz:.4f}")
        # Texture coords.
        for y in range(h_grid):
            for x in range(w_grid):
                u = x / cols
                v = 1.0 - (y / rows)
                obj_lines.append(f"vt {u:.6f} {v:.6f}")
        # Faces — use a per-tile material so the textures are distinct.
        obj_lines.append(f"usemtl mtl_{name}")
        for y in range(rows):
            for x in range(cols):
                v00 = v_offset + y * w_grid + x + 1
                v10 = v_offset + y * w_grid + (x + 1) + 1
                v01 = v_offset + (y + 1) * w_grid + x + 1
                v11 = v_offset + (y + 1) * w_grid + (x + 1) + 1
                obj_lines.append(f"f {v00}/{v00} {v10}/{v10} {v01}/{v01}")
                obj_lines.append(f"f {v10}/{v10} {v11}/{v11} {v01}/{v01}")
        v_offset += h_grid * w_grid
        obj_lines.append("")
    obj_path.write_text("\n".join(obj_lines), encoding="utf-8")
    # MTL: one newmtl per tile, each pointing at its own PNG.
    mtl_lines: list[str] = []
    for _height, _tex, name, _wx, _wy in tiles:
        mtl_lines.append(f"newmtl mtl_{name}")
        mtl_lines.append(f"map_Kd {name}.png")
        mtl_lines.append("Ka 1.0 1.0 1.0")
        mtl_lines.append("Kd 1.0 1.0 1.0")
        mtl_lines.append("Ns 0.0")
        mtl_lines.append("d 1.0")
    (obj_path.parent / f"{obj_path.stem}.mtl").write_text(
        "\n".join(mtl_lines), encoding="utf-8",
    )


def _auto_pick(tiles: list[tuple[np.ndarray, np.ndarray, str, int, int]]) -> tuple[bool, bool, bool, str, float]:
    """Pick the axis permutation where each tile's texture is closest to the source PNG.

    Score = mean over all tiles of (mean per-pixel abs-diff between the
    variant's texture and the source PNG). The variant with the lowest
    score is the one where the texture is used as-is (or with the
    minimum flips needed to undo whatever mirroring the user's pipeline
    introduced).
    """
    scores: list[tuple[float, bool, bool, bool, str]] = []
    for flip_x, flip_y, flip_z, name in PERMUTATIONS:
        per_tile_diffs: list[float] = []
        for _h, tex_src, _name, _wx, _wy in tiles:
            tex = tex_src.copy()
            if flip_y:
                tex = np.flipud(tex)
            if flip_x:
                tex = np.fliplr(tex)
            diff = float(np.mean(np.abs(tex.astype(np.int32) - tex_src.astype(np.int32))))
            per_tile_diffs.append(diff)
        avg = sum(per_tile_diffs) / len(per_tile_diffs) if per_tile_diffs else float("inf")
        scores.append((avg, flip_x, flip_y, flip_z, name))
    scores.sort(key=lambda t: t[0])
    return scores[0][1], scores[0][2], scores[0][3], scores[0][4], scores[0][0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--anchor-x", type=int, default=None,
                        help="X coord of the cluster's anchor tile (default: "
                             "the smallest X among tiles with parseable names).")
    parser.add_argument("--anchor-y", type=int, default=None,
                        help="Y coord of the cluster's anchor tile (default: "
                             "the smallest Y for the chosen X).")
    parser.add_argument("--output-dir", default=None, type=Path,
                        help="default: wow-viewer/output/v24_axis_probe/<input-basename>/")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=94)
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        raise NotADirectoryError(f"input dir not found: {args.input_dir}")

    pngs = sorted(p for p in args.input_dir.iterdir()
                  if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if not pngs:
        raise FileNotFoundError(f"no PNG/JPG files in {args.input_dir}")

    # Pick a 2x2 cluster of adjacent tiles.
    parsed: list[tuple[Path, tuple[int, int]]] = []
    for p in pngs:
        xy = _parse_xy(p.stem)
        if xy is not None:
            parsed.append((p, xy))
    if len(parsed) < 4:
        print(f"only {len(parsed)} tiles have parseable names; using the first tile only")
        first = parsed[0][0] if parsed else pngs[0]
        cluster = [(first, 0, 0)]
    else:
        by_xy = {xy: p for p, xy in parsed}
        if args.anchor_x is not None and args.anchor_y is not None:
            ax, ay = args.anchor_x, args.anchor_y
        else:
            # Anchor on the smallest-X tile at the smallest-Y for that X.
            x_min = min(xy[0] for _, xy in parsed)
            candidates = sorted([(p, xy) for p, xy in parsed if xy[0] == x_min],
                                key=lambda t: t[1][1])
            ax, ay = candidates[0][1]
        cluster = []
        for tx, ty in [(ax, ay), (ax + 1, ay), (ax, ay + 1), (ax + 1, ay + 1)]:
            if (tx, ty) in by_xy:
                cluster.append((by_xy[(tx, ty)], tx, ty))
        if len(cluster) < 4:
            print("could not find a 2x2 cluster; falling back to a single-tile probe")
            cluster = [(parsed[0][0], 0, 0)]

    print(f"probing a {len(cluster)}-tile cluster:")
    for p, tx, ty in cluster:
        print(f"  {p.name}  (parsed as X={tx}, Y={ty})")
    print()

    if args.output_dir is None:
        repo_root = SCRIPT_DIR.parent.parent
        safe_stem = re.sub(r"[^A-Za-z0-9_.-]", "_", args.input_dir.name)
        args.output_dir = repo_root / "output" / "v24_axis_probe" / safe_stem
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"output: {args.output_dir}")
    print(f"  8 variants per permutation under: {args.output_dir}/<variant>/cluster.obj")
    print()

    # Run inference on each tile in the cluster.
    train_common.set_determinism(args.seed, strict=True)
    device = train_common.pick_device(args.device)
    ckpt_path = _discover_checkpoint()
    print(f"checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    model = stage_a.StageAMinimapOnly(base=ckpt["config"]["base"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    tiles: list[tuple[np.ndarray, np.ndarray, str, int, int]] = []
    for p, tx, ty in cluster:
        minimap = _load_minimap(p)
        x = stage_a.build_minimap_only_input(minimap)
        with torch.no_grad():
            outer, inner = model(torch.from_numpy(x)[None].to(device))
        outer = (outer[0].cpu().numpy() * HEIGHT_SCALE).astype(np.float32)
        inner = (inner[0].cpu().numpy() * HEIGHT_SCALE).astype(np.float32)
        height = lattice.upsample_prior_257(outer, inner)
        tex_arr = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
        name = f"tile_{tx}_{ty}"
        tiles.append((height, tex_arr, name, tx * 533, ty * 533))
        print(f"  predicted {p.name}: world {height.min():.1f} .. {height.max():.1f}")
    print()

    # Render all 8 variants. Each variant has a single combined OBJ
    # (the 2x2 cluster) plus the per-tile PNGs/MTLs. The cluster is
    # a 2-tile-by-2-tile grid in world space (X*533, Y*533) so the
    # user can see the cluster as one piece in a 3D viewer, with
    # enough detail to make mirroring obvious.
    for flip_x, flip_y, flip_z, name in PERMUTATIONS:
        variant_dir = args.output_dir / name
        variant_dir.mkdir(parents=True, exist_ok=True)
        cluster_tiles: list[tuple[np.ndarray, np.ndarray, str, int, int]] = []
        for height, tex_src, tile_name, world_x, world_y in tiles:
            h = height.copy()
            if flip_z:
                h = -h
            if flip_y:
                h = np.flipud(h)
            if flip_x:
                h = np.fliplr(h)
            tex = tex_src.copy()
            if flip_y:
                tex = np.flipud(tex)
            if flip_x:
                tex = np.fliplr(tex)
            cluster_tiles.append((h, tex, tile_name, world_x, world_y))
        cluster_obj = variant_dir / "cluster.obj"
        _write_cluster(cluster_tiles, cluster_obj, flip_x, flip_y, flip_z)
        flags = []
        if flip_x: flags.append("--flip-x")
        if flip_y: flags.append("--flip-y")
        if flip_z: flags.append("--flip-z")
        flag_str = " ".join(flags) if flags else "(no flags)"
        print(f"  {name:8s}  -> {cluster_obj}  {flag_str}")

    # Auto-pick: lowest texture-source distance.
    fx, fy, fz, name, score = _auto_pick(tiles)
    print()
    print(f"auto-pick: {name} (texture-source distance = {score:.3f})")
    flags = []
    if fx: flags.append("--flip-x")
    if fy: flags.append("--flip-y")
    if fz: flags.append("--flip-z")
    flag_str = " ".join(flags) if flags else "(no flags)"
    print()
    print("To re-run the full quilt with this orientation:")
    print()
    print(f"    uv run python scripts/v24_quilt_objs.py \\")
    print(f'        --input-dir "{args.input_dir}" \\')
    print(f"        {flag_str}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
