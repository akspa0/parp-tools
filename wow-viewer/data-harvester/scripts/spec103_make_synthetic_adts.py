"""Spec 103 T007 — author synthetic known-height tiles for the v7 PoC.

Writes, per pattern, a 257×257 world-unit heightmap (.npy) plus the inference_summary.json
layout that the frozen C# `terrain-patch-adt` command consumes, and prints the exact
generate-blank / patch / capture commands. The C# tooling is used as-is (AlphaWdtWriter and
the ADT writers are frozen — this script only prepares their inputs).

Patterns (known analytic height fields): flat, ramp, ridge, crater, plateau — each at one or
more amplitudes. Tiles are placed on EVERY OTHER coordinate so no two synthetic tiles are
adjacent: `terrain-patch-adt` seam-stitches adjacent patched tiles, which would mutate the
known patterns.

Run from wow-viewer/data-harvester/ (fast, CPU-only — safe to run directly):

    uv run python scripts/spec103_make_synthetic_adts.py --output ../output/spec103/synthetic

The printed dotnet/capture commands are for the USER to run (AGENTS RULE 0).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

MAP_NAME = "synth103"
GRID = np.mgrid[0:257, 0:257].astype(np.float32) / 256.0  # (v, u) in [0, 1]


def _pattern_height(name: str, amplitude: float) -> np.ndarray:
    v, u = GRID[0], GRID[1]
    if name == "flat":
        return np.full((257, 257), amplitude * 0.5, dtype=np.float32)
    if name == "ramp":
        return (u * 0.75 + v * 0.25) * amplitude
    if name == "ridge":
        return (1.0 - np.abs(u - 0.5) * 2.0) ** 2 * amplitude
    if name == "crater":
        radius = np.sqrt((u - 0.5) ** 2 + (v - 0.5) ** 2)
        rim = np.exp(-((radius - 0.28) ** 2) / (2 * 0.06**2))
        bowl = np.clip(1.0 - radius / 0.28, 0.0, 1.0) ** 1.5
        return (rim - 0.65 * bowl) * amplitude + amplitude * 0.4
    if name == "plateau":
        soft = 1.0 / (1.0 + np.exp(-(0.32 - np.maximum(np.abs(u - 0.5), np.abs(v - 0.5))) * 40.0))
        return soft * amplitude
    raise ValueError(f"unknown pattern {name!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Spec 103 synthetic known-height tile author")
    ap.add_argument("--output", type=Path, default=Path("../output/spec103/synthetic"))
    ap.add_argument("--patterns", nargs="*", default=["flat", "ramp", "ridge", "crater", "plateau"])
    ap.add_argument("--amplitudes", nargs="*", type=float, default=[60.0, 180.0],
                    help="world-unit height amplitudes; each (pattern, amplitude) pair becomes one tile")
    ap.add_argument("--base-tile", type=int, nargs=2, default=(30, 30), metavar=("X", "Y"))
    args = ap.parse_args()

    out_root = args.output.resolve()
    heights_dir = out_root / "known_heights"
    inference_dir = out_root / "patch_inputs"
    adt_blank_dir = out_root / "adt_blank"
    adt_patched_dir = out_root / "adt_patched"
    captures_dir = out_root / "captures"
    for d in (heights_dir, inference_dir):
        d.mkdir(parents=True, exist_ok=True)

    specs = [(p, a) for p in args.patterns for a in args.amplitudes]
    manifest = []
    generate_cmds: list[str] = []
    capture_cmds: list[str] = []

    x0, y0 = args.base_tile
    per_row = 8
    for index, (pattern, amplitude) in enumerate(specs):
        # every other coordinate: no adjacency, so seam stitching never fires
        tile_x = x0 + (index % per_row) * 2
        tile_y = y0 + (index // per_row) * 2
        tile_name = f"{MAP_NAME}_{tile_x}_{tile_y}"

        height = _pattern_height(pattern, amplitude)
        npy_path = heights_dir / f"{tile_name}.npy"
        np.save(npy_path, height)

        summary_dir = inference_dir / tile_name
        summary_dir.mkdir(parents=True, exist_ok=True)
        (summary_dir / "inference_summary.json").write_text(json.dumps({
            "tile_name": tile_name,
            "predicted_height_257_path": str(npy_path.resolve()).replace("\\", "/"),
        }, indent=2), encoding="utf-8")

        manifest.append({
            "tile_name": tile_name, "map": MAP_NAME, "tile_x": tile_x, "tile_y": tile_y,
            "pattern": pattern, "amplitude": amplitude,
            "height_npy": str(npy_path.resolve()).replace("\\", "/"),
            "height_min": float(height.min()), "height_max": float(height.max()),
        })
        generate_cmds.append(
            f"dotnet run --project tools/inspect/WowViewer.Tool.Inspect -c Release -- "
            f"map generate-blank --tile-x {tile_x} --tile-y {tile_y} --map-name {MAP_NAME} "
            f"--format lk --output-dir \"{adt_blank_dir}\""
        )
        capture_cmds.append(
            f"dotnet run --project tools/capture/WowViewer.Tool.Capture -c Release -- "
            f"render --client-root \"{adt_patched_dir}\" --tile-name {tile_name} "
            f"--output \"{captures_dir / (tile_name + '.png')}\" --resolution 256"
        )

    (out_root / "synthetic_manifest.json").write_text(json.dumps({
        "schema": "spec103-synthetic-manifest-v1",
        "map": MAP_NAME,
        "tiles": manifest,
    }, indent=2), encoding="utf-8")

    patch_cmd = (
        f"dotnet run --project tools/converter/WowViewer.Tool.Converter -c Release -- "
        f"terrain-patch-adt --input-adt-dir \"{adt_blank_dir}\" "
        f"--inference-dir \"{inference_dir}\" --output-dir \"{adt_patched_dir}\" --no-export-guide-textures"
    )

    print(f"[spec103] wrote {len(manifest)} known-height tiles under {heights_dir}")
    print(f"[spec103] manifest: {out_root / 'synthetic_manifest.json'}")
    print()
    print("=== USER-RUN COMMANDS (from wow-viewer/, in order) ===")
    print()
    print("# 1. blank ADT + WDT + WDL per tile (frozen writers, used as-is):")
    for cmd in generate_cmds:
        print(cmd)
    print()
    print("# 2. patch the known heights into the blank ADTs:")
    print(patch_cmd)
    print()
    print("# 3. capture per-tile renders (GPU; perspective-camera caveat is recorded in")
    print("#    research-v7-contract.md §7 — or skip and use --synthesize-minimaps in step 4):")
    for cmd in capture_cmds:
        print(cmd)
    print()
    print("# 4. assemble the 13-channel store (fast, CPU):")
    print(f"uv run python scripts/spec103_build_synthetic_store.py --manifest \"{out_root / 'synthetic_manifest.json'}\" "
          f"--minimap-dir \"{captures_dir}\" --output ../output/datasets/spec103/synthetic_v1.zarr")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
