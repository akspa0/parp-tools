"""One command: run every archaeology tool over a set of stores into ONE flat output folder.

Each underlying tool has its own CLI and its own output directory. Running them by hand means five
invocations and five nested paths. This drives all of them and lands everything in:

    <output>/
      images/       every PNG, flat, named <map>-<mode>.png or <map>_<x>_<y>-<what>.png
      data/         every CSV and JSON
      README.txt    what each file is

Usage:

    uv run python scripts/v50_archaeology.py \
        --store ../output/datasets/v50/v50.1/0_5_3_3368-Azeroth.zarr \
        --store ../output/datasets/v50/v50.1/0_5_3_3368-Kalimdor.zarr \
        --holes ../output/datasets/v50/v50.1/holes-0_5_3_3368.json \
        --output ../output/archaeology/0_5_3_3368
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
_SRC = _SCRIPTS.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

README = """\
ARCHAEOLOGY OUTPUT
==================

images/
  <map>-absolute.png      every tile on the map's global height scale (what the data looks like)
  <map>-autostretch.png   every tile normalized to ITSELF (the blanks fill in)
  <map>-restored.png      compressed tiles scaled toward their neighbours' real range
  <map>-liquid.png        the same terrain flooded to its liquid surface (blue = wet)
  <map>-textured.png      minimap colour over restored relief
  <map>-degenerate.png    mosaic of just the weak/blank tiles at their true grid positions

  tile-<map>_<x>_<y>.png  4-panel sheet for one degenerate tile
                          (autostretched height | hillshade | MCNR normals | minimap)
  hole-<map>_<x>_<y>.png  4-panel sheet for one HOLED tile
                          (terrain | hole mask | hidden geometry only | minimap)

data/
  tiles.csv               one row per tile: classification, height range,
                          surviving_height_levels, hole counts, neighbour refs
  tiles.json              same, plus neighbour tile keys
  inventory-summary.json  corpus counts + the tile-key lists (white plates, weak signal, ...)
  classify.csv            three-tier signal classification (strong/normal/weak) + evidence
  classify.json           full per-tile classification records
  classify-summary.json   per-tier tile counts and tile-key lists
  hidden_chunks.csv       one row per HOLED chunk: position, hole mask, and the
                          relief that is hidden underneath it
  holed_tiles.csv         per-tile hole totals
  holes-summary.json      hole totals + the 16-bit hole-pattern census
  signal-mismatch.json    near-universal A->B signal rules and the tiles that break them
  synthesis-manifest.json per-tile measurements for every degenerate tile
"""


def _run(script: str, args: list[str]) -> None:
    command = [sys.executable, str(_SCRIPTS / script), *args]
    print(f"\n>>> {script} {' '.join(args[:4])} ...", flush=True)
    result = subprocess.run(command, capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if "ZarrUserWarning" in line or "warnings.warn" in line:
            continue
        print("    " + line, flush=True)
    if result.returncode != 0:
        print(result.stderr[-2000:], flush=True)
        raise SystemExit(f"{script} failed with exit {result.returncode}")


def _collect(src: Path, dst: Path, pattern: str, rename) -> int:
    moved = 0
    for path in sorted(src.glob(pattern)):
        target = dst / rename(path)
        shutil.copyfile(path, target)
        moved += 1
    return moved


def main() -> int:
    parser = argparse.ArgumentParser(description="Run every archaeology tool into one flat folder")
    parser.add_argument("--store", required=True, type=Path, action="append", dest="stores")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--holes", type=Path, default=None,
                        help="extract-holes JSON; omit to skip the hole stage")
    parser.add_argument("--near-zero-band", default=None,
                        help="pass 'inf' for any non-alpha client (see README)")
    parser.add_argument("--cell", type=int, default=96, help="pixels per tile in whole-map images")
    parser.add_argument("--render-tiles", type=int, default=60,
                        help="how many per-tile sheets to keep for each of the degenerate/hole sets")
    args = parser.parse_args()

    images = args.output / "images"
    data = args.output / "data"
    images.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    store_args: list[str] = []
    for store in args.stores:
        store_args += ["--store", str(store)]

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)

        inv = work / "inv"
        band = ["--near-zero-band", args.near_zero_band] if args.near_zero_band else []
        _run("v50_tile_inventory.py", [*store_args, "--output", str(inv), *band])

        synth = work / "synth"
        _run("v50_synthesize_weak_tiles.py",
             ["--inventory", str(inv), *store_args, "--output", str(synth)])

        comp = work / "comp"
        _run("v50_tile_composite.py",
             ["--inventory", str(inv), *store_args, "--output", str(comp),
              "--cell", str(args.cell)])

        # Three-tier brush-signature classification (Spec 132 US1)
        classify_out = work / "classify"
        _run("v50_tile_classify.py", [*store_args, "--output", str(classify_out)])

        _run("v50_tile_mismatch.py",
             [*store_args, "--output", str(data / "signal-mismatch.json")])

        holes = None
        if args.holes is not None:
            holes = work / "holes"
            _run("v50_tile_holes.py",
                 ["--holes", str(args.holes), *store_args, "--output", str(holes),
                  "--render", str(args.render_tiles)])

        # --- flatten -------------------------------------------------------------------------
        n = 0
        n += _collect(comp, images, "composite-*.png",
                      lambda p: p.name.replace("composite-", ""))
        n += _collect(synth, images, "mosaic-*.png",
                      lambda p: p.name.replace("mosaic-", "").replace(".png", "-degenerate.png"))
        kept = 0
        for path in sorted((synth / "tiles").glob("*.png")):
            if kept >= args.render_tiles:
                break
            shutil.copyfile(path, images / f"tile-{path.name}")
            kept += 1; n += 1
        if holes is not None:
            n += _collect(holes / "tiles", images, "*.png", lambda p: f"hole-{p.name}")

        for name, target in (("tiles.csv", "tiles.csv"), ("tiles.json", "tiles.json"),
                             ("summary.json", "inventory-summary.json")):
            shutil.copyfile(inv / name, data / target)
        # Three-tier classification output (Spec 132 US1)
        for name in ("classify.csv", "classify.json", "summary.json"):
            src = classify_out / name
            if src.exists():
                target_name = f"classify-{name}" if name == "summary.json" else name
                shutil.copyfile(src, data / target_name)
        shutil.copyfile(synth / "manifest.json", data / "synthesis-manifest.json")
        if holes is not None:
            for name, target in (("hidden_chunks.csv", "hidden_chunks.csv"),
                                 ("holed_tiles.csv", "holed_tiles.csv"),
                                 ("summary.json", "holes-summary.json"),
                                 ("hidden_chunks.json", "hidden_chunks.json")):
                shutil.copyfile(holes / name, data / target)

    (args.output / "README.txt").write_text(README, encoding="utf-8")
    print(f"\n[DONE] {n} images -> {images}")
    print(f"       {len(list(data.glob('*')))} data files -> {data}")
    print(f"       what each file is -> {args.output / 'README.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
