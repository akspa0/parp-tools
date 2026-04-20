from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

from train_v9 import (
    DEFAULT_DEV_EVAL_BLOCK_SIZE,
    DEFAULT_MAX_ABS_WDL_DELTA,
    DEFAULT_MAX_MEAN_WDL_DELTA,
    DEFAULT_MIN_HEIGHT_RANGE,
    DEFAULT_MIN_MINIMAP_GRADIENT,
    DEFAULT_MIN_MINIMAP_VARIANCE,
    DEFAULT_SEED,
    V9SampleEntry,
    audit_entries,
    load_cache_manifest,
    select_diverse_eval_entries,
    utc_now_iso,
    write_json,
)


DEFAULT_HOLDOUT_SIZE = 64
DEFAULT_DETAIL_CANDIDATE_FRACTION = 0.35
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[4] / "output" / "ml-training" / "v9-dev-eval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a detail-heavy development-evaluation holdout manifest and a non-overlapping training remainder manifest from a v9 tensor cache.",
    )
    parser.add_argument("cache_manifest", help="Path to a v9_tensor_cache_manifest.json file.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory that will receive the holdout/remainder manifests and summary.")
    parser.add_argument("--holdout-size", type=int, default=DEFAULT_HOLDOUT_SIZE, help="Number of accepted entries reserved for the dev-eval holdout.")
    parser.add_argument("--detail-candidate-fraction", type=float, default=DEFAULT_DETAIL_CANDIDATE_FRACTION,
                        help="Top fraction of accepted entries considered as detail-heavy candidates before diversity selection.")
    parser.add_argument("--detail-candidate-limit", type=int, default=None,
                        help="Optional absolute cap on the detail-heavy candidate pool before diversity selection.")
    parser.add_argument("--diversity-block-size", type=int, default=DEFAULT_DEV_EVAL_BLOCK_SIZE,
                        help="Tile block size used when spreading the holdout across map regions.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--require-wdl", action=argparse.BooleanOptionalAction, default=True,
                        help="Require WDL priors in accepted entries. Leave enabled when using dev_wdl_mae_improvement selection.")
    parser.add_argument("--require-minimap", action=argparse.BooleanOptionalAction, default=True,
                        help="Require minimap RGB in accepted entries.")
    parser.add_argument("--min-height-range", type=float, default=DEFAULT_MIN_HEIGHT_RANGE)
    parser.add_argument("--min-minimap-variance", type=float, default=DEFAULT_MIN_MINIMAP_VARIANCE)
    parser.add_argument("--min-minimap-gradient", type=float, default=DEFAULT_MIN_MINIMAP_GRADIENT)
    parser.add_argument("--max-mean-wdl-delta", type=float, default=DEFAULT_MAX_MEAN_WDL_DELTA)
    parser.add_argument("--max-abs-wdl-delta", type=float, default=DEFAULT_MAX_ABS_WDL_DELTA)
    parser.add_argument("--holdout-manifest-name", default="v9_dev_eval_holdout_manifest.json")
    parser.add_argument("--remainder-manifest-name", default="v9_training_remainder_manifest.json")
    return parser.parse_args()


def load_manifest_payload(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sample_key(entry: V9SampleEntry) -> str:
    return f"{entry.dataset_key}:{entry.tile_name}"


def summarize_entries(entries: Sequence[V9SampleEntry]) -> dict[str, float | int]:
    if not entries:
        return {
            "count": 0,
            "detail_energy_mean": 0.0,
            "detail_energy_max": 0.0,
            "height_range_mean": 0.0,
            "minimap_gradient_mean": 0.0,
        }

    count = len(entries)
    return {
        "count": count,
        "detail_energy_mean": float(sum(entry.detail_energy for entry in entries) / count),
        "detail_energy_max": float(max(entry.detail_energy for entry in entries)),
        "height_range_mean": float(sum(entry.height_range for entry in entries) / count),
        "minimap_gradient_mean": float(sum(entry.minimap_gradient for entry in entries) / count),
    }


def build_subset_manifest(
    *,
    base_manifest: dict[str, Any],
    subset_entries: Sequence[dict[str, Any]],
    source_manifest: Path,
    split_name: str,
    selection: dict[str, Any],
) -> dict[str, Any]:
    manifest = dict(base_manifest)
    manifest.update(
        {
            "schema_version": "v9-native-tensor-cache.v2",
            "created_at_utc": utc_now_iso(),
            "source_cache_manifest": str(source_manifest),
            "split_name": split_name,
            "selection": selection,
            "processed": len(subset_entries),
            "skipped": 0,
            "entries": list(subset_entries),
        }
    )
    return manifest


def main() -> None:
    args = parse_args()
    if args.holdout_size < 1:
        raise SystemExit("--holdout-size must be at least 1.")
    if not 0.0 < args.detail_candidate_fraction <= 1.0:
        raise SystemExit("--detail-candidate-fraction must be greater than 0 and at most 1.")
    if args.detail_candidate_limit is not None and args.detail_candidate_limit < args.holdout_size:
        raise SystemExit("--detail-candidate-limit must be at least as large as --holdout-size.")
    if args.diversity_block_size < 1:
        raise SystemExit("--diversity-block-size must be at least 1.")

    cache_manifest_path = Path(args.cache_manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_manifest = load_manifest_payload(cache_manifest_path)
    raw_entries = list(base_manifest.get("entries", []))
    entries = load_cache_manifest(cache_manifest_path)
    audited, sane_entries, rejection_counts = audit_entries(
        entries=entries,
        require_wdl=args.require_wdl,
        require_minimap=args.require_minimap,
        min_height_range=args.min_height_range,
        min_minimap_variance=args.min_minimap_variance,
        min_minimap_gradient=args.min_minimap_gradient,
        max_mean_wdl_delta=args.max_mean_wdl_delta,
        max_abs_wdl_delta=args.max_abs_wdl_delta,
    )

    if len(sane_entries) <= args.holdout_size:
        raise SystemExit(
            f"Need more than {args.holdout_size} accepted entries to carve a non-overlapping holdout; found {len(sane_entries)}."
        )

    ranked_detail_entries = sorted(
        sane_entries,
        key=lambda entry: (
            entry.detail_energy,
            entry.minimap_gradient,
            entry.height_range,
            entry.minimap_variance,
            -entry.hole_coverage,
        ),
        reverse=True,
    )
    candidate_count = max(args.holdout_size, int(math.ceil(len(ranked_detail_entries) * args.detail_candidate_fraction)))
    if args.detail_candidate_limit is not None:
        candidate_count = min(candidate_count, args.detail_candidate_limit)
    detail_candidates = ranked_detail_entries[:candidate_count]
    holdout_entries = select_diverse_eval_entries(detail_candidates, args.holdout_size, args.diversity_block_size, args.seed)
    holdout_keys = {sample_key(entry) for entry in holdout_entries}
    remainder_entries = [entry for entry in sane_entries if sample_key(entry) not in holdout_keys]
    if not remainder_entries:
        raise SystemExit("Holdout consumed the entire accepted pool; choose a smaller holdout size.")

    raw_by_key = {
        f"{str(entry.get('dataset_key', ''))}:{str(entry.get('tile_name', ''))}": entry
        for entry in raw_entries
    }
    holdout_manifest_entries = [raw_by_key[sample_key(entry)] for entry in holdout_entries]
    remainder_manifest_entries = [raw_by_key[sample_key(entry)] for entry in remainder_entries]

    selection = {
        "seed": args.seed,
        "holdout_size": args.holdout_size,
        "detail_candidate_fraction": args.detail_candidate_fraction,
        "detail_candidate_limit": args.detail_candidate_limit,
        "diversity_block_size": args.diversity_block_size,
        "require_wdl": args.require_wdl,
        "require_minimap": args.require_minimap,
        "min_height_range": args.min_height_range,
        "min_minimap_variance": args.min_minimap_variance,
        "min_minimap_gradient": args.min_minimap_gradient,
        "max_mean_wdl_delta": args.max_mean_wdl_delta,
        "max_abs_wdl_delta": args.max_abs_wdl_delta,
    }

    holdout_manifest = build_subset_manifest(
        base_manifest=base_manifest,
        subset_entries=holdout_manifest_entries,
        source_manifest=cache_manifest_path,
        split_name="dev_eval_holdout",
        selection=selection,
    )
    remainder_manifest = build_subset_manifest(
        base_manifest=base_manifest,
        subset_entries=remainder_manifest_entries,
        source_manifest=cache_manifest_path,
        split_name="training_remainder",
        selection=selection,
    )

    holdout_manifest_path = output_dir / args.holdout_manifest_name
    remainder_manifest_path = output_dir / args.remainder_manifest_name
    write_json(holdout_manifest_path, holdout_manifest)
    write_json(remainder_manifest_path, remainder_manifest)

    detail_cutoff = float(detail_candidates[-1].detail_energy) if detail_candidates else 0.0
    summary = {
        "schema_version": "v9-dev-eval-holdout-summary.v1",
        "created_at_utc": utc_now_iso(),
        "source_cache_manifest": str(cache_manifest_path),
        "holdout_manifest": str(holdout_manifest_path),
        "remainder_manifest": str(remainder_manifest_path),
        "source_entries": len(entries),
        "accepted_entries": len(sane_entries),
        "rejected_entries": len(entries) - len(sane_entries),
        "rejection_counts": rejection_counts,
        "candidate_pool": summarize_entries(detail_candidates),
        "holdout": summarize_entries(holdout_entries),
        "remainder": summarize_entries(remainder_entries),
        "detail_energy_cutoff": detail_cutoff,
        "selection": selection,
        "holdout_tiles": [entry.sample_key for entry in holdout_entries],
    }
    write_json(output_dir / "v9_dev_eval_holdout_summary.json", summary)

    print(
        f"Accepted {len(sane_entries)} / {len(entries)} entries | detail candidates {len(detail_candidates)} | "
        f"holdout {len(holdout_entries)} | remainder {len(remainder_entries)}"
    )
    print(f"Detail cutoff: {detail_cutoff:.6f}")
    print(f"Holdout manifest: {holdout_manifest_path}")
    print(f"Remainder manifest: {remainder_manifest_path}")


if __name__ == "__main__":
    main()