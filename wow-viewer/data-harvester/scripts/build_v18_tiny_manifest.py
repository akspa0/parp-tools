from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
_SRC_ROOT = _PROJECT_ROOT / "data-harvester" / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from harvester.v16_curation import DIFFICULTY_BUCKETS, load_curation_rows, write_rows_parquet

_DEFAULT_DATASET_DIR = _PROJECT_ROOT / "output" / "datasets" / "v18"
_DEFAULT_CURATION_ROOT = _DEFAULT_DATASET_DIR / "curation"
_DEFAULT_SOURCE_RUN = "v18_focus_terrain_v1"
_DEFAULT_RUN_NAME = "v18_focus_tiny_v1"
_DEFAULT_BUILDS = ("0_5_3_3368", "3_3_5_12340")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Derive a super-tiny focused V18 curation manifest from an existing kept_tiles pool."
    )
    p.add_argument(
        "--source-manifest",
        type=Path,
        default=_DEFAULT_CURATION_ROOT / _DEFAULT_SOURCE_RUN,
        help="Source focused manifest directory or kept_tiles.parquet path.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Explicit output directory. Defaults to output/datasets/v18/curation/<run-name>/",
    )
    p.add_argument("--run-name", default=_DEFAULT_RUN_NAME)
    p.add_argument(
        "--builds",
        nargs="+",
        default=list(_DEFAULT_BUILDS),
        help="Focused builds to keep in the tiny manifest.",
    )
    p.add_argument(
        "--samples-per-bucket-per-build",
        type=int,
        default=3,
        help="Hard cap per build/difficulty bucket stratum. Use 0 to disable the cap.",
    )
    p.add_argument(
        "--fraction-per-bucket-per-build",
        type=float,
        default=1.0,
        help="Additional fractional cap per build/difficulty bucket stratum in (0, 1]. Use 0 to disable.",
    )
    p.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Deterministic seed used for map round-robin ordering inside each stratum.",
    )
    return p.parse_args()


def _normalize_bucket(value: Any) -> str:
    bucket = str(value or "").strip().lower()
    if bucket not in DIFFICULTY_BUCKETS:
        raise RuntimeError(f"Unexpected difficulty_bucket '{value}' in source manifest.")
    return bucket


def _row_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -float(row.get("quality_score", 0.0) or 0.0),
        -float(row.get("usefulness_score", 0.0) or 0.0),
        -float(row.get("difficulty_score", 0.0) or 0.0),
        str(row.get("map", "")),
        int(row.get("tile_id", -1)),
    )


def _resolve_target_count(
    available: int,
    samples_per_bucket_per_build: int,
    fraction_per_bucket_per_build: float,
) -> int:
    target = int(available)
    if samples_per_bucket_per_build > 0:
        target = min(target, int(samples_per_bucket_per_build))
    if fraction_per_bucket_per_build > 0.0:
        target = min(target, max(1, int(math.ceil(available * float(fraction_per_bucket_per_build)))))
    return max(0, target)


def _select_diverse_rows(rows: list[dict[str, Any]], target: int, seed: int) -> list[dict[str, Any]]:
    if target <= 0 or not rows:
        return []
    ranked_rows = sorted(rows, key=_row_sort_key)
    if target >= len(ranked_rows):
        return ranked_rows

    rows_by_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ranked_rows:
        rows_by_map[str(row.get("map", ""))].append(row)

    map_names = sorted(
        rows_by_map,
        key=lambda name: (
            _row_sort_key(rows_by_map[name][0]),
            name,
        ),
    )
    if map_names:
        start = int(seed) % len(map_names)
        map_names = map_names[start:] + map_names[:start]

    chosen: list[dict[str, Any]] = []
    while len(chosen) < target:
        progressed = False
        for map_name in map_names:
            queue = rows_by_map[map_name]
            if not queue:
                continue
            chosen.append(queue.pop(0))
            progressed = True
            if len(chosen) >= target:
                break
        if not progressed:
            break
    return chosen


def _load_source_summary(source_manifest: Path) -> dict[str, Any]:
    summary_path = source_manifest / "summary.json" if source_manifest.is_dir() else source_manifest.parent / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    return {}


def build_tiny_manifest_rows(
    rows: list[dict[str, Any]],
    *,
    builds: list[str],
    samples_per_bucket_per_build: int,
    fraction_per_bucket_per_build: float,
    sample_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    allowed_builds = set(builds)
    kept_rows = [dict(row) for row in rows if bool(row.get("keep", True)) and str(row.get("build", "")) in allowed_builds]
    if not kept_rows:
        raise RuntimeError("No kept rows matched the requested builds.")

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    source_bucket_counts: dict[str, int] = {bucket: 0 for bucket in DIFFICULTY_BUCKETS}
    build_bucket_counts: dict[str, dict[str, int]] = {
        build: {bucket: 0 for bucket in DIFFICULTY_BUCKETS} for build in builds
    }
    source_build_bucket_counts: dict[str, dict[str, int]] = {
        build: {bucket: 0 for bucket in DIFFICULTY_BUCKETS} for build in builds
    }
    for row in kept_rows:
        build = str(row.get("build", ""))
        bucket = _normalize_bucket(row.get("difficulty_bucket"))
        grouped[(build, bucket)].append(row)
        source_bucket_counts[bucket] += 1
        source_build_bucket_counts[build][bucket] += 1

    selected: list[dict[str, Any]] = []
    selected_bucket_counts: dict[str, int] = {bucket: 0 for bucket in DIFFICULTY_BUCKETS}
    strata_summary: list[dict[str, Any]] = []

    for build_idx, build in enumerate(builds):
        for bucket_idx, bucket in enumerate(DIFFICULTY_BUCKETS):
            stratum_rows = grouped.get((build, bucket), [])
            available = len(stratum_rows)
            target = _resolve_target_count(
                available,
                samples_per_bucket_per_build=samples_per_bucket_per_build,
                fraction_per_bucket_per_build=fraction_per_bucket_per_build,
            )
            chosen = _select_diverse_rows(
                stratum_rows,
                target=target,
                seed=int(sample_seed) + (build_idx * len(DIFFICULTY_BUCKETS)) + bucket_idx,
            )
            selected.extend(chosen)
            selected_bucket_counts[bucket] += len(chosen)
            build_bucket_counts[build][bucket] = len(chosen)
            strata_summary.append(
                {
                    "build": build,
                    "difficulty_bucket": bucket,
                    "available_rows": int(available),
                    "selected_rows": int(len(chosen)),
                    "source_maps": int(len({str(row.get('map', '')) for row in stratum_rows})),
                    "selected_maps": int(len({str(row.get('map', '')) for row in chosen})),
                }
            )

    selected.sort(
        key=lambda row: (
            builds.index(str(row.get("build", ""))),
            int(row.get("difficulty_rank", DIFFICULTY_BUCKETS.index(_normalize_bucket(row.get("difficulty_bucket"))))),
            str(row.get("map", "")),
            int(row.get("tile_id", -1)),
        )
    )
    summary = {
        "source_kept_tiles": int(len(kept_rows)),
        "source_difficulty_bucket_counts": source_bucket_counts,
        "source_build_bucket_counts": source_build_bucket_counts,
        "difficulty_bucket_counts": selected_bucket_counts,
        "build_bucket_counts": build_bucket_counts,
        "selection_recipe": {
            "strategy": "tiny_bucket_balanced",
            "samples_per_bucket_per_build": int(samples_per_bucket_per_build),
            "fraction_per_bucket_per_build": float(fraction_per_bucket_per_build),
            "map_diversity_round_robin": True,
            "per_stratum": strata_summary,
        },
    }
    return selected, summary


def main() -> None:
    args = _parse_args()
    if int(args.samples_per_bucket_per_build) < 0:
        raise SystemExit("--samples-per-bucket-per-build must be >= 0")
    if not (0.0 <= float(args.fraction_per_bucket_per_build) <= 1.0):
        raise SystemExit("--fraction-per-bucket-per-build must be in [0.0, 1.0]")
    if int(args.samples_per_bucket_per_build) == 0 and float(args.fraction_per_bucket_per_build) == 0.0:
        raise SystemExit("Set at least one of --samples-per-bucket-per-build or --fraction-per-bucket-per-build.")

    source_manifest = Path(args.source_manifest)
    rows = load_curation_rows(source_manifest)
    source_summary = _load_source_summary(source_manifest)
    selected, selection_summary = build_tiny_manifest_rows(
        rows,
        builds=[str(build) for build in args.builds],
        samples_per_bucket_per_build=int(args.samples_per_bucket_per_build),
        fraction_per_bucket_per_build=float(args.fraction_per_bucket_per_build),
        sample_seed=int(args.sample_seed),
    )

    output_dir = args.output_dir or (_DEFAULT_CURATION_ROOT / str(args.run_name))
    output_dir.mkdir(parents=True, exist_ok=True)

    write_rows_parquet(output_dir / "tiles.parquet", selected)
    write_rows_parquet(output_dir / "kept_tiles.parquet", selected)
    (output_dir / "tiles.jsonl").write_text(
        "\n".join(json.dumps(row) for row in selected) + "\n",
        encoding="utf-8",
    )

    quality_scores = [float(row.get("quality_score", 0.0) or 0.0) for row in selected]
    usefulness_scores = [float(row.get("usefulness_score", 0.0) or 0.0) for row in selected]
    summary = {
        "profile": str(source_summary.get("profile", _DEFAULT_SOURCE_RUN)),
        "canonical_profile": str(source_summary.get("canonical_profile", source_summary.get("profile", _DEFAULT_SOURCE_RUN))),
        "builds": [str(build) for build in args.builds],
        "dataset_dir": str(source_summary.get("dataset_dir", _DEFAULT_DATASET_DIR)),
        "sample_seed": int(args.sample_seed),
        "tile_count": int(len(selected)),
        "kept_tiles": int(len(selected)),
        "rejected_tiles": 0,
        "keep_ratio": float(1.0 if selected else 0.0),
        "quality_score_mean_kept": float(np.mean(quality_scores)) if quality_scores else 0.0,
        "usefulness_score_mean_kept": float(np.mean(usefulness_scores)) if usefulness_scores else 0.0,
        "source_manifest": str(source_manifest),
        "source_run_name": source_manifest.name if source_manifest.is_dir() else source_manifest.parent.name,
        "selection_profile": str(args.run_name),
    }
    summary.update(selection_summary)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {output_dir / 'summary.json'}", flush=True)
    print(f"Wrote {output_dir / 'kept_tiles.parquet'}", flush=True)
    print(
        f"tiny_profile={args.run_name} kept={len(selected)} source_kept={selection_summary['source_kept_tiles']} "
        f"buckets={selection_summary['difficulty_bucket_counts']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
