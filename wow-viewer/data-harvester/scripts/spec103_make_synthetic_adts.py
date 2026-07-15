"""Spec 103 T007 — author synthetic known-height tiles for the v7 PoC.

Writes, per pattern, a 257×257 world-unit heightmap (.npy) plus the inference_summary.json
layout that the frozen C# `terrain-patch-adt` command consumes, and prints the exact
generate-blank / patch / capture commands. The C# tooling is used as-is (AlphaWdtWriter and
the ADT writers are frozen — this script only prepares their inputs).

The original ten tiles were a smoke fixture, not a corpus. The default creates deterministic,
varied analytic terrain families: each has independently sampled geometry/orientation/offset and
amplitude. Tiles are placed on EVERY OTHER coordinate so no two synthetic tiles are adjacent:
`terrain-patch-adt` seam-stitches adjacent patched tiles, which would mutate known patterns.

Run from wow-viewer/data-harvester/ (fast, CPU-only — safe to run directly):

    uv run python scripts/spec103_make_synthetic_adts.py --output ../output/spec103/synthetic

The printed dotnet/capture commands are for the USER to run (AGENTS RULE 0).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

GRID = np.mgrid[0:257, 0:257].astype(np.float32) / 256.0  # (v, u) in [0, 1]
DEFAULT_PATTERNS = ("plane", "ridge", "crater", "plateau", "hills", "valley", "terraces", "saddle", "dunes", "basin")


def _pattern_height(name: str, amplitude: float, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, float]]:
    v, u = GRID[0], GRID[1]
    angle = float(rng.uniform(0.0, 2.0 * np.pi))
    x = (u - 0.5) * np.cos(angle) + (v - 0.5) * np.sin(angle)
    y = -(u - 0.5) * np.sin(angle) + (v - 0.5) * np.cos(angle)
    offset = float(rng.uniform(-0.75, 0.75) * amplitude)
    params: dict[str, float] = {"angle_rad": angle, "base_offset": offset}
    if name == "plane":
        slope = float(rng.uniform(0.25, 1.0)); cross = float(rng.uniform(-0.35, 0.35))
        field = slope * x + cross * y
        params.update(slope=slope, cross_slope=cross)
    if name == "ridge":
        width = float(rng.uniform(0.10, 0.36)); power = float(rng.uniform(1.2, 3.5))
        field = np.exp(-np.square(x / width)) ** power + 0.12 * np.sin(y * rng.uniform(4.0, 12.0))
        params.update(width=width, power=power)
    elif name == "crater":
        cx, cy = float(rng.uniform(-0.18, 0.18)), float(rng.uniform(-0.18, 0.18))
        radius = np.sqrt(np.square(x - cx) + np.square(y - cy)); rim_radius = float(rng.uniform(0.18, 0.38)); rim_width = float(rng.uniform(0.035, 0.09))
        field = np.exp(-np.square((radius - rim_radius) / rim_width)) - rng.uniform(0.4, 0.9) * np.clip(1.0 - radius / rim_radius, 0.0, 1.0) ** rng.uniform(1.1, 2.5)
        params.update(center_x=cx, center_y=cy, rim_radius=rim_radius, rim_width=rim_width)
    elif name == "plateau":
        width = float(rng.uniform(0.16, 0.38)); feather = float(rng.uniform(18.0, 60.0))
        field = 1.0 / (1.0 + np.exp(-(width - np.maximum(np.abs(x), np.abs(y))) * feather))
        params.update(width=width, feather=feather)
    elif name == "hills":
        field = np.zeros_like(u)
        for _ in range(5):
            frequency, phase, weight = rng.uniform(1.5, 11.0), rng.uniform(0.0, 2*np.pi), rng.uniform(0.15, 1.0)
            a = rng.uniform(0.0, 2*np.pi)
            field += weight * np.sin((x*np.cos(a) + y*np.sin(a)) * frequency + phase)
    elif name == "valley":
        width = float(rng.uniform(0.09, 0.28)); floor = float(rng.uniform(0.05, 0.22))
        field = np.square(np.tanh(np.abs(x) / width)) + floor * np.sin(y * rng.uniform(3.0, 10.0))
        params.update(width=width, floor_wobble=floor)
    elif name == "terraces":
        steps = float(rng.integers(4, 13)); softness = float(rng.uniform(8.0, 30.0))
        field = np.floor((x + 0.5) * steps) / steps + 0.08 * np.tanh(np.sin(y * softness))
        params.update(steps=steps, softness=softness)
    elif name == "saddle":
        curvature = float(rng.uniform(1.0, 4.0)); field = curvature * (np.square(x) - np.square(y))
        params.update(curvature=curvature)
    elif name == "dunes":
        frequency = float(rng.uniform(8.0, 24.0)); warp = float(rng.uniform(1.0, 8.0))
        field = np.sin(x * frequency + np.sin(y * warp)) + 0.25 * np.sin(x * frequency * 2.2)
        params.update(frequency=frequency, warp=warp)
    elif name == "basin":
        cx, cy = float(rng.uniform(-0.18, 0.18)), float(rng.uniform(-0.18, 0.18)); width = float(rng.uniform(0.16, 0.42))
        field = -np.exp(-(np.square(x-cx) + np.square(y-cy)) / (2*width*width)) + 0.25 * np.sqrt(np.square(x) + np.square(y))
        params.update(center_x=cx, center_y=cy, width=width)
    elif name != "plane":
        raise ValueError(f"unknown pattern {name!r}")
    field = np.asarray(field, dtype=np.float32)
    span = max(float(field.max() - field.min()), 1e-6)
    return (offset + (field - field.min()) / span * amplitude).astype(np.float32), params


def main() -> int:
    ap = argparse.ArgumentParser(description="Spec 103 synthetic known-height tile author")
    ap.add_argument("--output", type=Path, default=Path("../output/spec103/synthetic"))
    ap.add_argument("--patterns", nargs="*", default=list(DEFAULT_PATTERNS))
    ap.add_argument("--amplitudes", nargs="*", type=float, default=[60.0, 180.0],
                    help="world-unit height amplitudes; each (pattern, amplitude) pair becomes one tile")
    ap.add_argument("--variants-per-pattern", type=int, default=16, help="independent parameter samples per pattern/amplitude")
    ap.add_argument("--seed", type=int, default=103)
    ap.add_argument("--map-name", default="synth108")
    ap.add_argument("--base-tile", type=int, nargs=2, default=(0, 0), metavar=("X", "Y"))
    ap.add_argument("--tiles-per-row", type=int, default=32)
    ap.add_argument(
        "--source-license",
        default="UNSPECIFIED",
        help="operator-declared license for the generated terrain source; never inferred",
    )
    ap.add_argument(
        "--source-rights-assertion",
        default="UNSPECIFIED",
        help="operator assertion identifying their authority to use the generated source",
    )
    ap.add_argument(
        "--lighting-source",
        choices=("authored", "lit"),
        default="authored",
        help="capture profile source; LIT captures are private-BYOD evidence",
    )
    lit_source = ap.add_mutually_exclusive_group()
    lit_source.add_argument("--lit-file", type=Path)
    lit_source.add_argument("--lit-client-root", type=Path)
    ap.add_argument(
        "--lit-virtual-path",
        default="World/Maps/Azeroth/lights.lit",
        help="archive path used with --lit-client-root",
    )
    args = ap.parse_args()
    if args.lighting_source == "lit" and args.lit_file is None and args.lit_client_root is None:
        ap.error("--lighting-source lit requires --lit-file or --lit-client-root")

    out_root = args.output.resolve()
    heights_dir = out_root / "known_heights"
    inference_dir = out_root / "patch_inputs"
    adt_blank_dir = out_root / "adt_blank"
    adt_patched_dir = out_root / "adt_patched"
    captures_dir = out_root / "captures"
    for d in (heights_dir, inference_dir):
        d.mkdir(parents=True, exist_ok=True)

    if args.variants_per_pattern < 1 or args.tiles_per_row < 1:
        ap.error("--variants-per-pattern and --tiles-per-row must be positive")
    specs = [(p, a, variant) for p in args.patterns for a in args.amplitudes for variant in range(args.variants_per_pattern)]
    manifest = []
    generate_cmds: list[str] = []
    capture_cmds: list[str] = []

    x0, y0 = args.base_tile
    per_row = args.tiles_per_row
    for index, (pattern, amplitude, variant) in enumerate(specs):
        # every other coordinate: no adjacency, so seam stitching never fires
        tile_x = x0 + (index % per_row) * 2
        tile_y = y0 + (index // per_row) * 2
        if tile_x > 63 or tile_y > 63:
            raise ValueError("synthetic grid exceeds valid 0..63 ADT coordinates; reduce variants or change layout")
        tile_name = f"{args.map_name}_{tile_x}_{tile_y}"

        sample_seed = int(args.seed + index * 7919)
        height, shape_params = _pattern_height(pattern, amplitude, np.random.default_rng(sample_seed))
        npy_path = heights_dir / f"{tile_name}.npy"
        np.save(npy_path, height)
        height_sha256 = hashlib.sha256(npy_path.read_bytes()).hexdigest()

        summary_dir = inference_dir / tile_name
        summary_dir.mkdir(parents=True, exist_ok=True)
        (summary_dir / "inference_summary.json").write_text(json.dumps({
            "tile_name": tile_name,
            "predicted_height_257_path": str(npy_path.resolve()).replace("\\", "/"),
        }, indent=2), encoding="utf-8")

        manifest.append({
            "tile_name": tile_name, "map": args.map_name, "tile_x": tile_x, "tile_y": tile_y,
            "pattern": pattern, "variant": variant, "sample_seed": sample_seed, "shape_parameters": shape_params, "amplitude": amplitude,
            "height_npy": str(npy_path.resolve()).replace("\\", "/"),
            "height_sha256": height_sha256,
            "terrain_source_origin": "analytic_generated",
            "terrain_source_license": args.source_license,
            "terrain_source_rights_assertion": args.source_rights_assertion,
            "height_min": float(height.min()), "height_max": float(height.max()),
        })
        generate_cmds.append(
            f"dotnet run --project tools/inspect/WowViewer.Tool.Inspect -c Release -- "
            f"map generate-blank --tile-x {tile_x} --tile-y {tile_y} --map-name {args.map_name} "
            f"--format lk --output-dir \"{adt_blank_dir}\""
        )
        lighting_args = "--lighting-source authored"
        if args.lighting_source == "lit" and args.lit_file is not None:
            lighting_args = f'--lighting-source lit --lit-file "{args.lit_file.resolve()}"'
        elif args.lighting_source == "lit" and args.lit_client_root is not None:
            lighting_args = (
                f'--lighting-source lit --lit-client-root "{args.lit_client_root.resolve()}" '
                f'--lit-virtual-path "{args.lit_virtual_path}"'
            )
        capture_cmds.append(
            f"dotnet run --project tools/capture/WowViewer.Tool.Capture -c Release -- "
            f"render --client-root \"{adt_patched_dir}\" --tile-name {tile_name} "
            f"--output \"{captures_dir / (tile_name + '.png')}\" --resolution 256 "
            f"--game-time 0.35 {lighting_args}"
        )

    (out_root / "synthetic_manifest.json").write_text(json.dumps({
        "schema": "spec103-synthetic-manifest-v1",
        "map": args.map_name,
        "provenance": {
            "generator": "spec103_make_synthetic_adts.py",
            "terrain_source_origin": "analytic_generated",
            "terrain_source_license": args.source_license,
            "terrain_source_rights_assertion": args.source_rights_assertion,
            "license_policy": "operator_declared_never_inferred",
        },
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
    print("# 3. capture canonical one-tile top-down orthographic renders (GPU).")
    print("#    Each PNG receives a hash-bound .lighting.json evidence sidecar.")
    print("#    Or skip capture and use an explicitly authored --lighting-time variant in step 4:")
    for cmd in capture_cmds:
        print(cmd)
    print()
    print("# 4. assemble the 13-channel store (fast, CPU):")
    print(f"uv run python scripts/spec103_build_synthetic_store.py --manifest \"{out_root / 'synthetic_manifest.json'}\" "
          f"--minimap-dir \"{captures_dir}\" --output ../output/datasets/spec103/synthetic_v1.zarr")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
