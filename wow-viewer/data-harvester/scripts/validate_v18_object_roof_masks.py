"""Validate learned/metadata object-roof mask outputs on bounded anchors."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate object-roof mask prediction outputs.")
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--build", type=str, default=None)
    parser.add_argument("--map", type=str, default=None)
    parser.add_argument("--tile-x", type=int, default=None)
    parser.add_argument("--tile-y", type=int, default=None)
    parser.add_argument("--min-mask-coverage", type=float, default=0.005)
    parser.add_argument("--min-top-family-score", type=float, default=0.20)
    parser.add_argument("--require-non-empty", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise RuntimeError(f"Missing predictions file: {path}")
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        out.append(dict(json.loads(text)))
    return out


def _filter(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if args.build is not None and str(row.get("build", "")) != str(args.build):
            continue
        if args.map is not None and str(row.get("map", "")) != str(args.map):
            continue
        if args.tile_x is not None and int(row.get("tile_x", -1) or -1) != int(args.tile_x):
            continue
        if args.tile_y is not None and int(row.get("tile_y", -1) or -1) != int(args.tile_y):
            continue
        out.append(row)
    return out


def main() -> None:
    args = _parse_args()
    pred_dir = Path(args.pred_dir)
    rows = _read_rows(pred_dir / "predictions.jsonl")
    rows = _filter(rows, args)
    if not rows:
        raise RuntimeError("No prediction rows matched validation filter")

    issues: list[str] = []
    non_empty = 0
    passing_tiles = 0
    for row in rows:
        cov = float(row.get("mask_mean", 0.0) or 0.0)
        top_score = float(row.get("top_family_score", 0.0) or 0.0)
        mask_sum = float(row.get("mask_sum", 0.0) or 0.0)
        if mask_sum > 0.0:
            non_empty += 1
        ok_cov = cov >= float(args.min_mask_coverage)
        ok_family = top_score >= float(args.min_top_family_score)
        if ok_cov and ok_family:
            passing_tiles += 1

    if bool(args.require_non_empty) and non_empty <= 0:
        issues.append("all_masks_empty")
    if passing_tiles <= 0:
        issues.append("no_tiles_passed_thresholds")

    report = {
        "status": "pass" if not issues else "fail",
        "tiles_checked": len(rows),
        "non_empty_tiles": int(non_empty),
        "passing_tiles": int(passing_tiles),
        "min_mask_coverage": float(args.min_mask_coverage),
        "min_top_family_score": float(args.min_top_family_score),
        "issues": issues,
        "validated_at": datetime.now(timezone.utc).isoformat(),
    }
    (pred_dir / "mask_validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

