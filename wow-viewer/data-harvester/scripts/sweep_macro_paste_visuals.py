"""Run macro paste visual sweeps for Spec 076 review.

This is a thin orchestrator around analyze_fractal_raw_components.py. It exists
so macro paste/scar grouping can be reviewed visually across parameter settings
instead of relying on one count or one radius.
"""

from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_OUTPUT_ROOT = _PROJECT_ROOT / "output" / "analysis" / "full-map-fractal-brush-library" / "macro_visual_sweep"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run macro paste/scar visual sweep outputs.")
    parser.add_argument("--build", required=True)
    parser.add_argument("--map", required=True)
    parser.add_argument("--tile-limit", type=int, default=0)
    parser.add_argument("--strip-tiles", type=int, default=8)
    parser.add_argument("--strip-overlap-alpha-tiles", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--close-radii", default="8,16,32")
    parser.add_argument("--min-areas", default="4096")
    parser.add_argument("--min-footprint", type=int, default=64)
    parser.add_argument("--downsample-factor", type=int, default=8)
    parser.add_argument("--max-regions-per-layer", type=int, default=500)
    parser.add_argument("--visualize-composite-signal", action="store_true")
    parser.add_argument("--output-root", type=Path, default=_DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    radii = _parse_int_list(str(args.close_radii))
    min_areas = _parse_int_list(str(args.min_areas))
    runs: list[dict[str, Any]] = []
    for radius in radii:
        for min_area in min_areas:
            run_name = f"close{radius}_area{min_area}_foot{int(args.min_footprint)}"
            run_root = output_root / run_name
            cmd = [
                sys.executable,
                str(Path(__file__).resolve().parent / "analyze_fractal_raw_components.py"),
                "--builds",
                str(args.build),
                "--maps",
                str(args.map),
                "--tile-limit",
                str(int(args.tile_limit)),
                "--strip-tiles",
                str(int(args.strip_tiles)),
                "--strip-overlap-alpha-tiles",
                str(int(args.strip_overlap_alpha_tiles)),
                "--threshold",
                str(float(args.threshold)),
                "--macro-pastes",
                "--macro-close-radius",
                str(radius),
                "--macro-min-area",
                str(min_area),
                "--macro-min-footprint",
                str(int(args.min_footprint)),
                "--macro-downsample-factor",
                str(int(args.downsample_factor)),
                "--max-regions-per-layer",
                str(int(args.max_regions_per_layer)),
                "--output-root",
                str(run_root),
                "--no-overlay",
                "--visualize-macro",
            ]
            if bool(args.visualize_composite_signal):
                cmd.append("--visualize-composite-signal")
            print("Running", run_name, flush=True)
            subprocess.run(cmd, check=True)
            run_summary = _read_summary(run_root)
            runs.append(
                {
                    "run_name": run_name,
                    "run_root": str(run_root),
                    "close_radius": radius,
                    "min_area": min_area,
                    "target_count": int(run_summary.get("target_count", 0)),
                    "region_count": _region_count(run_summary),
                    "macro_review": _macro_review_path(run_root),
                }
            )
    summary = {"output_root": str(output_root), "runs": runs}
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_index(output_root, runs)
    print("Macro visual sweep complete", flush=True)
    print(f"  output_root: {output_root}", flush=True)
    print(f"  runs: {len(runs)}", flush=True)


def _parse_int_list(value: str) -> list[int]:
    items = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not items:
        raise ValueError("Expected at least one integer")
    return items


def _read_summary(run_root: Path) -> dict[str, Any]:
    path = run_root / "summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _region_count(summary: dict[str, Any]) -> int:
    return int(sum(int(target.get("region_count", 0)) for target in summary.get("targets", [])))


def _macro_review_path(run_root: Path) -> str:
    for target_dir in sorted(run_root.glob("*_tile*")):
        review = target_dir / "segments_raw" / "macro_review" / "index.html"
        if review.exists():
            return str(review)
    return ""


def _write_index(output_root: Path, runs: list[dict[str, Any]]) -> None:
    rows = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Macro Paste Visual Sweep</title></head><body>",
        "<h1>Spec 076 Macro Paste Visual Sweep</h1>",
        f"<p>Output root: <code>{html.escape(str(output_root))}</code></p>",
        "<table border='1' cellpadding='6'>",
        "<tr><th>Run</th><th>Close Radius</th><th>Min Area</th><th>Regions</th><th>Review</th></tr>",
    ]
    for run in runs:
        review = Path(str(run.get("macro_review", "")))
        link = ""
        if review.exists():
            rel = html.escape(str(review.relative_to(output_root).as_posix()))
            link = f"<a href='{rel}'>macro_review</a>"
        rows.append(
            f"<tr><td>{html.escape(str(run.get('run_name', '')))}</td>"
            f"<td>{int(run.get('close_radius', 0))}</td>"
            f"<td>{int(run.get('min_area', 0))}</td>"
            f"<td>{int(run.get('region_count', 0))}</td><td>{link}</td></tr>"
        )
    rows.append("</table></body></html>")
    (output_root / "index.html").write_text("\n".join(rows), encoding="utf-8")


if __name__ == "__main__":
    main()
