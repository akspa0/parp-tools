from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze PM4 research reports and highlight tiles with strong coordinate splits."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--reports-json", type=Path, help="Existing JSON report file from Pm4Research.Cli scan-dir/export-json")
    source.add_argument("--pm4-dir", type=Path, help="Directory of PM4 files to scan via Pm4Research.Cli")
    parser.add_argument(
        "--cli-project",
        type=Path,
        default=Path("src/Pm4Research.Cli/Pm4Research.Cli.csproj"),
        help="Path to the Pm4Research.Cli project used when --pm4-dir is supplied",
    )
    parser.add_argument("--top", type=int, default=12, help="Number of top entries to print per section")
    parser.add_argument(
        "--min-count",
        type=int,
        default=32,
        help="Minimum vector count required before a quadrant split is considered interesting",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reports = load_reports(args)
    if not reports:
        print("No reports found.", file=sys.stderr)
        return 1

    print_overview(reports)
    print_vector_rankings(reports, args.top)
    print_mprl_rankings(reports, args.top)
    print_quadrant_hotspots(reports, args.top, args.min_count)
    print_unrecognized_chunks(reports, args.top)
    return 0


def load_reports(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.reports_json is not None:
        return normalize_reports(json.loads(args.reports_json.read_text(encoding="utf-8")))

    with tempfile.NamedTemporaryFile(prefix="pm4-scan-", suffix=".json", delete=False) as handle:
        temp_path = Path(handle.name)

    try:
        command = [
            "dotnet",
            "run",
            "--project",
            str(args.cli_project),
            "--",
            "scan-dir",
            "--input",
            str(args.pm4_dir),
            "--output",
            str(temp_path),
        ]
        subprocess.run(command, check=True)
        return normalize_reports(json.loads(temp_path.read_text(encoding="utf-8")))
    finally:
        temp_path.unlink(missing_ok=True)


def normalize_reports(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [report for report in payload if isinstance(report, dict)]
    if isinstance(payload, dict):
        return [payload]
    return []


def print_overview(reports: list[dict[str, Any]]) -> None:
    non_empty_msvt = sum(1 for report in reports if vector_count(report, "Msvt") > 0)
    non_empty_mspv = sum(1 for report in reports if vector_count(report, "Mspv") > 0)
    non_empty_mscn = sum(1 for report in reports if vector_count(report, "Mscn") > 0)
    non_empty_mprl = sum(1 for report in reports if report.get("Mprl", {}).get("TotalCount", 0) > 0)

    print("PM4 Report Overview")
    print("===================")
    print(f"reports: {len(reports)}")
    print(f"non-empty MSPV: {non_empty_mspv}")
    print(f"non-empty MSVT: {non_empty_msvt}")
    print(f"non-empty MSCN: {non_empty_mscn}")
    print(f"non-empty MPRL: {non_empty_mprl}")
    print()


def print_vector_rankings(reports: list[dict[str, Any]], top: int) -> None:
    for key in ("Mspv", "Msvt", "Mscn", "MprlPositions"):
        ranked = sorted(reports, key=lambda report: vector_count(report, key), reverse=True)
        print(f"Top {top} {key} Counts")
        print("-" * (len(key) + 16))
        for report in ranked[:top]:
            count = vector_count(report, key)
            if count <= 0:
                continue
            print(f"{count:6d}  {report.get('SourcePath', '<unknown>')}")
        print()


def print_mprl_rankings(reports: list[dict[str, Any]], top: int) -> None:
    ranked = sorted(reports, key=lambda report: report.get("Mprl", {}).get("TotalCount", 0), reverse=True)
    print(f"Top {top} MPRL Counts")
    print("-----------------")
    for report in ranked[:top]:
        summary = report.get("Mprl", {})
        count = summary.get("TotalCount", 0)
        if count <= 0:
            continue
        floor_min = summary.get("FloorMin")
        floor_max = summary.get("FloorMax")
        print(
            f"{count:6d}  floors={floor_min}..{floor_max}  {report.get('SourcePath', '<unknown>')}"
        )
    print()


def print_quadrant_hotspots(reports: list[dict[str, Any]], top: int, min_count: int) -> None:
    hotspots: list[tuple[float, int, str, str, str, dict[str, Any]]] = []
    for report in reports:
        source = report.get("SourcePath", "<unknown>")
        for key in ("Mspv", "Msvt", "Mscn", "MprlPositions"):
            summary = report.get(key, {})
            count = summary.get("Count", 0)
            if count < min_count:
                continue
            for quadrant in summary.get("Quadrants", []):
                cells = [
                    quadrant.get("LowLow", 0),
                    quadrant.get("LowHigh", 0),
                    quadrant.get("HighLow", 0),
                    quadrant.get("HighHigh", 0),
                ]
                max_cell = max(cells)
                min_cell = min(cells)
                dominant_share = max_cell / count if count else 0.0
                spread = max_cell - min_cell
                hotspots.append((dominant_share, spread, key, quadrant.get("Plane", "?"), source, quadrant))

    hotspots.sort(key=lambda item: (item[0], item[1]), reverse=True)
    print(f"Top {top} Quadrant Hotspots")
    print("-------------------------")
    for dominant_share, spread, key, plane, source, quadrant in hotspots[:top]:
        print(
            f"share={dominant_share:0.3f} spread={spread:4d} {key}/{plane} "
            f"ll={quadrant.get('LowLow', 0)} lh={quadrant.get('LowHigh', 0)} "
            f"hl={quadrant.get('HighLow', 0)} hh={quadrant.get('HighHigh', 0)} {source}"
        )
    print()


def print_unrecognized_chunks(reports: list[dict[str, Any]], top: int) -> None:
    counter: Counter[str] = Counter()
    for report in reports:
        counter.update(report.get("UnrecognizedChunks", []))

    print(f"Top {top} Unrecognized Chunks")
    print("--------------------------")
    for chunk, count in counter.most_common(top):
        print(f"{count:6d}  {chunk}")
    print()


def vector_count(report: dict[str, Any], key: str) -> int:
    summary = report.get(key, {})
    if not isinstance(summary, dict):
        return 0
    value = summary.get("Count", 0)
    return int(value) if isinstance(value, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())