"""Spec 096 helper: one-shot PNG -> WDL prior runner with no path wrangling.

Picks the most recent trained minimap-only Stage A checkpoint under
``wow-viewer/output/v24_validation/`` and runs ``infer_v24_stage_a_png.py``
against the user-supplied PNG. Output paths default to sibling files in the
same directory as the input PNG, so the typical call is just:

    uv run python scripts/v24_run_on_png.py path/to/minimap.png

For a real-data sanity check, also exports a PNG of a V18 minimap from the
open-world V24 store on demand. Use --export-v18-minimap to write a random
tile's minimap to disk first if you don't have a PNG handy.

Outputs (defaults):
  <png-stem>.prior.npz      — the (17,17) outer + (16,16) inner WDL prior NPZ
  <png-stem>.prior.png      — 1024x256 4-up preview PNG
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
V24_VALIDATION_ROOT = SCRIPT_DIR.parent.parent / "output" / "v24_validation"
DEFAULT_INFER_SCRIPT = SCRIPT_DIR / "infer_v24_stage_a_png.py"
DEFAULT_OBJ_SCRIPT = SCRIPT_DIR / "v24_prior_to_obj.py"


def _discover_checkpoint(v24_root: Path) -> Path:
    """Return the most recent minimap-only stage_a.pt under v24_validation/.

    Looks for a ``stage_a.pt`` in any subdir whose companion
    ``stage_a_metrics.json`` declares ``minimap_only: true`` (only Spec 096
    checkpoints carry that flag). Falls back to the most recent
    ``v24_minimap_only_*`` subdir's stage_a.pt if no JSON has the flag yet.
    """
    if not v24_root.exists():
        raise FileNotFoundError(
            f"v24 validation root not found: {v24_root}. "
            f"Train a minimap-only checkpoint first."
        )
    candidates: list[tuple[float, Path]] = []
    for run_dir in v24_root.iterdir():
        if not run_dir.is_dir():
            continue
        if not run_dir.name.startswith("v24_minimap_only"):
            continue
        ckpt = run_dir / "stage_a.pt"
        if not ckpt.exists():
            continue
        candidates.append((ckpt.stat().st_mtime, ckpt))
    if not candidates:
        raise FileNotFoundError(
            f"no minimap-only checkpoint found under {v24_root}. "
            f"Train one with: "
            f"uv run python scripts/train_v24_stage_a.py --minimap-only ..."
        )
    candidates.sort(reverse=True)
    return candidates[0][1]


def _run_batch(args: argparse.Namespace) -> int:
    """Process every PNG in args.batch_dir end-to-end, then stitch a grid OBJ.

    Each PNG gets:
      - its own prior NPZ
      - its own per-tile mesh (terrain.obj + texture.png + height.png)
    All outputs land under ``<batch_dir>_v24_objs/`` (a sibling folder),
    so the input directory is never modified. The grid stitch lands
    under the same output dir as ``stitched_mesh/``.

    Use ``--batch-output-dir`` to override the default output location.
    """
    batch_dir = args.batch_dir.resolve()
    if not batch_dir.is_dir():
        raise NotADirectoryError(f"--batch-dir is not a directory: {batch_dir}")

    if args.batch_output_dir is None:
        # Default: repo root's output/ folder under v24_objs/<basename>/.
        repo_root = SCRIPT_DIR.parent.parent
        safe_stem = re.sub(r"[^A-Za-z0-9_.-]", "_", batch_dir.name)
        out_dir = (repo_root / "output" / "v24_objs" / safe_stem).resolve()
    else:
        out_dir = args.batch_output_dir.resolve()
    # Hard safety: never write inside the batch input dir.
    try:
        out_dir.relative_to(batch_dir)
    except ValueError:
        pass
    else:
        raise ValueError(
            f"refusing to run: --batch-output-dir {out_dir} is inside "
            f"--batch-dir {batch_dir}. Pick an --batch-output-dir outside "
            f"the input."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"batch input : {batch_dir} ({len(list(batch_dir.iterdir()))} entries)")
    print(f"batch output: {out_dir}")

    ckpt = args.checkpoint or _discover_checkpoint(V24_VALIDATION_ROOT)
    pngs = sorted(p for p in batch_dir.iterdir()
                  if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if not pngs:
        raise FileNotFoundError(f"no PNG/JPG files found in {batch_dir}")
    print(f"checkpoint: {ckpt}")
    print(f"will process: {len(pngs)} PNGs")
    print()

    failed: list[tuple[Path, int]] = []
    for i, png in enumerate(pngs, 1):
        out_npz = out_dir / f"{png.stem}.prior.npz"
        out_preview = out_dir / f"{png.stem}.prior.png"
        mesh_dir = out_dir / f"{png.stem}.mesh"
        mesh_dir.mkdir(exist_ok=True)

        infer_cmd = [
            sys.executable, str(DEFAULT_INFER_SCRIPT),
            "--checkpoint", str(ckpt),
            "--image", str(png),
            "--output", str(out_npz),
            "--preview", str(out_preview),
        ]
        if args.device:
            infer_cmd.extend(["--device", args.device])
        infer_proc = subprocess.run(infer_cmd, check=False)

        if infer_proc.returncode != 0:
            failed.append((png, infer_proc.returncode))
            print(f"[{i}/{len(pngs)}] {png.name}: infer FAILED (rc={infer_proc.returncode})")
            continue

        if not args.no_obj:
            obj_cmd = [
                sys.executable, str(DEFAULT_OBJ_SCRIPT),
                "--prior", str(out_npz),
                "--image", str(png),
                "--output-dir", str(mesh_dir),
                "--tile-size", str(args.obj_tile_size),
                "--height-scale", str(args.obj_height_scale),
            ]
            obj_proc = subprocess.run(obj_cmd, check=False)
            if obj_proc.returncode != 0:
                failed.append((png, obj_proc.returncode))
                print(f"[{i}/{len(pngs)}] {png.name}: obj FAILED (rc={obj_proc.returncode})")
                continue

        print(f"[{i}/{len(pngs)}] {png.name}: ok")

    # Stitch.
    if not args.no_obj:
        stitched_dir = out_dir / "stitched_mesh"
        stitch_cmd = [
            sys.executable, str(DEFAULT_OBJ_SCRIPT),
            "--grid-from-priors", str(out_dir),
            "--output-dir", str(stitched_dir),
            "--tile-size", str(args.obj_tile_size),
            "--height-scale", str(args.obj_height_scale),
        ]
        if args.obj_grid_cols is not None:
            stitch_cmd.extend(["--grid-cols", str(args.obj_grid_cols)])
        print()
        print(f"stitching {len(pngs) - len(failed)} tiles into {stitched_dir}...")
        stitch_proc = subprocess.run(stitch_cmd, check=False)
        if stitch_proc.returncode != 0:
            print(f"stitch FAILED (rc={stitch_proc.returncode})")
            return stitch_proc.returncode

    if failed:
        print(f"\n{len(failed)} tile(s) failed:")
        for png, rc in failed:
            print(f"  {png.name} (rc={rc})")
        return 1
    return 0


def _export_v18_minimap(rng_seed: int) -> Path:
    """Sample a random minimap from the canonical 3_3_5_12340 V18 store.

    Returns the path to the exported PNG. Used as a built-in sanity-check
    minimap so the user can run the wrapper against real V18 data without
    having a PNG of their own.
    """
    import numpy as np
    import zarr

    repo_root = SCRIPT_DIR.parent.parent
    v18 = zarr.open_group(str(repo_root / "output" / "datasets" / "v18" / "3_3_5_12340.zarr"), mode="r")
    rng = np.random.default_rng(rng_seed)
    row = int(rng.integers(0, v18["minimap_rgb"].shape[0]))
    arr = np.asarray(v18["minimap_rgb"][row])
    if arr.ndim == 3 and arr.shape[-1] == 3:
        from PIL import Image
        out = Path.cwd() / f"v18_minimap_row{row:05d}.png"
        Image.fromarray(arr, mode="RGB").save(str(out))
        return out
    raise RuntimeError(f"unexpected minimap_rgb shape: {arr.shape}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "image", nargs="?",
        help="path to a PNG minimap. If omitted and --export-v18-minimap or "
             "--batch-dir is set, that mode takes over.",
    )
    parser.add_argument("--batch-dir", type=Path, default=None,
                        help="process every PNG in this directory end-to-end; "
                             "outputs are written to <batch-dir>_v24_objs/ by "
                             "default (or to --batch-output-dir if given). The "
                             "input folder is never modified.")
    parser.add_argument("--batch-output-dir", type=Path, default=None,
                        help="override the batch output directory")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="override the auto-discovered minimap-only checkpoint")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="override the output directory (default: same as image)")
    parser.add_argument("--output-npz", type=Path, default=None,
                        help="override the output NPZ path (default: <image-stem>.prior.npz)")
    parser.add_argument("--no-preview", action="store_true",
                        help="skip the 4-up preview PNG")
    parser.add_argument("--export-v18-minimap", type=int, default=0, nargs="?",
                        const=0, metavar="SEED",
                        help="when set, export a random V18 minimap to the current dir "
                             "and run the wrapper against it (optional integer seed)")
    parser.add_argument("--no-obj", action="store_true",
                        help="skip the OBJ+MTL+texture mesh export (mesh is on by default)")
    parser.add_argument("--obj-output-dir", type=Path, default=None,
                        help="override the OBJ output directory (default: <output-dir>/mesh)")
    parser.add_argument("--obj-tile-size", type=float, default=533.333,
                        help="OBJ world-space tile size (default 533.333 = one WoW tile)")
    parser.add_argument("--obj-height-scale", type=float, default=1.0,
                        help="OBJ height multiplier (default 1.0)")
    parser.add_argument("--obj-grid-cols", type=int, default=None,
                        help="column count for the stitched grid mesh (default: auto sqrt)")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.export_v18_minimap is not None and not args.image and not args.batch_dir:
        exported = _export_v18_minimap(args.export_v18_minimap)
        print(f"exported: {exported}")
        args.image = str(exported)

    if args.batch_dir:
        return _run_batch(args)

    if not args.image:
        parser.error(
            "an image path is required (or use --export-v18-minimap / --batch-dir)"
        )

    image = Path(args.image).resolve()
    if not image.exists():
        raise FileNotFoundError(f"image not found: {image}")

    ckpt = args.checkpoint or _discover_checkpoint(V24_VALIDATION_ROOT)
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")

    out_dir = (args.output_dir or image.parent).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = args.output_npz or (out_dir / f"{image.stem}.prior.npz")
    out_preview = out_dir / f"{image.stem}.prior.png" if not args.no_preview else None

    cmd = [
        sys.executable, str(DEFAULT_INFER_SCRIPT),
        "--checkpoint", str(ckpt),
        "--image", str(image),
        "--output", str(out_npz),
    ]
    if out_preview is not None:
        cmd.extend(["--preview", str(out_preview)])
    if args.device:
        cmd.extend(["--device", args.device])

    print("checkpoint: ", ckpt)
    print("image:      ", image)
    print("output npz: ", out_npz)
    if out_preview is not None:
        print("preview png:", out_preview)
    print()
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        return proc.returncode

    if not args.no_obj:
        obj_dir = (args.obj_output_dir or (out_dir / f"{image.stem}.mesh")).resolve()
        obj_dir.mkdir(parents=True, exist_ok=True)
        obj_cmd = [
            sys.executable, str(DEFAULT_OBJ_SCRIPT),
            "--prior", str(out_npz),
            "--image", str(image),
            "--output-dir", str(obj_dir),
            "--tile-size", str(args.obj_tile_size),
            "--height-scale", str(args.obj_height_scale),
        ]
        print()
        obj_proc = subprocess.run(obj_cmd, check=False)
        if obj_proc.returncode != 0:
            return obj_proc.returncode
        print()
        print(f"OBJ mesh:    {obj_dir / 'terrain.obj'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
