from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
import sys

if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from harvester.v16_curation import DIFFICULTY_BUCKETS, write_rows_parquet  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build V18 refined trainer manifest from deduped canvas candidates.")
    parser.add_argument("--deduped-candidates", type=Path, required=True, help="Path to candidates_deduped.jsonl/.json or directory containing it.")
    parser.add_argument("--run-name", type=str, default="v18_refined_manifest")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--profile", type=str, default="normal_terrain_v18_refined_v1")
    parser.add_argument("--max-clusters", type=int, default=0, help="If >0, keep only top-N clusters by quality score.")
    parser.add_argument("--max-variants-per-cluster", type=int, default=2, help="Per-cluster cap for balanced candidate selection.")
    parser.add_argument("--max-tiles", type=int, default=0, help="If >0, cap final unique tile rows.")
    parser.add_argument("--min-score-mean", type=float, default=0.12)
    parser.add_argument("--min-transition-mean", type=float, default=0.85)
    parser.add_argument("--min-hard-mean", type=float, default=0.40)
    parser.add_argument("--min-train-mask-mean", type=float, default=1.10)
    parser.add_argument("--min-tile-coverage-count", type=int, default=1)
    parser.add_argument("--keep-noncanonical", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _resolve_deduped_path(path: Path) -> Path:
    if path.is_file():
        return path
    cand = path / "candidates_deduped.jsonl"
    if cand.exists():
        return cand
    cand_json = path / "candidates_deduped.json"
    if cand_json.exists():
        return cand_json
    raise FileNotFoundError(f"No deduped candidates file under: {path}")


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    return list(payload.get("rows", []))


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, float(value))))


def _candidate_quality(row: dict[str, Any]) -> dict[str, float | str]:
    score_mean = float(row.get("score_mean", 0.0))
    hard_mean = float(row.get("hard_mean", 0.0))
    transition_mean = float(row.get("transition_mean", 0.0))
    train_mask_mean = float(row.get("train_mask_mean", 0.0))

    score_n = _clamp01(score_mean / 0.45)
    hard_n = _clamp01(hard_mean / 2.0)
    transition_n = _clamp01(transition_mean / 2.0)
    train_mask_n = _clamp01(train_mask_mean / 4.0)

    deformation_richness = _clamp01((0.65 * hard_n) + (0.35 * transition_n))
    normal_coverage = _clamp01((0.50 * train_mask_n) + 0.50)
    terrain_validity = _clamp01((0.60 * train_mask_n) + (0.40 * hard_n))
    painted_signal = transition_n
    minimap_usefulness = _clamp01((0.65 * transition_n) + (0.35 * score_n))
    usefulness_score = _clamp01(
        (0.25 * deformation_richness)
        + (0.20 * normal_coverage)
        + (0.20 * terrain_validity)
        + (0.15 * painted_signal)
        + (0.20 * minimap_usefulness)
    )
    difficulty_score = _clamp01((0.55 * deformation_richness) + (0.25 * painted_signal) + (0.20 * normal_coverage))
    pathology_pressure = _clamp01(max(0.0, 0.50 - terrain_validity) + max(0.0, 0.45 - minimap_usefulness))

    if pathology_pressure >= 0.35 and difficulty_score >= 0.30:
        difficulty_bucket = "pathological"
    elif difficulty_score >= 0.62 and usefulness_score >= 0.45:
        difficulty_bucket = "hard"
    elif difficulty_score >= 0.35 or usefulness_score >= 0.40:
        difficulty_bucket = "medium"
    else:
        difficulty_bucket = "easy"

    return {
        "quality_score": usefulness_score,
        "usefulness_score": usefulness_score,
        "difficulty_score": difficulty_score,
        "difficulty_bucket": difficulty_bucket,
        "difficulty_rank": int(DIFFICULTY_BUCKETS.index(difficulty_bucket)),
        "score_deformation_richness": deformation_richness,
        "score_normal_coverage": normal_coverage,
        "score_terrain_validity": terrain_validity,
        "score_painted_signal": painted_signal,
        "score_minimap_target_usefulness": minimap_usefulness,
        "score_pathology_pressure": pathology_pressure,
    }


def _candidate_passes(row: dict[str, Any], args: argparse.Namespace) -> bool:
    if int(row.get("tile_coverage_count", 0)) < int(args.min_tile_coverage_count):
        return False
    if float(row.get("score_mean", 0.0)) < float(args.min_score_mean):
        return False
    if float(row.get("transition_mean", 0.0)) < float(args.min_transition_mean):
        return False
    if float(row.get("hard_mean", 0.0)) < float(args.min_hard_mean):
        return False
    if float(row.get("train_mask_mean", 0.0)) < float(args.min_train_mask_mean):
        return False
    if not bool(args.keep_noncanonical) and not bool(row.get("is_canonical", False)):
        return False
    return True


def _cluster_balanced_candidates(rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_cluster: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        cluster_id = str(row.get("cluster_id", f"cluster_single_{int(row.get('candidate_id', -1)):08d}"))
        by_cluster.setdefault(cluster_id, []).append(row)

    for members in by_cluster.values():
        members.sort(
            key=lambda r: (
                int(r.get("variant_rank", 999999)),
                -float(r.get("quality_score", 0.0)),
                -float(r.get("score_mean", 0.0)),
                int(r.get("candidate_id", 0)),
            )
        )

    cluster_entries: list[tuple[str, list[dict[str, Any]], float]] = []
    for cluster_id, members in by_cluster.items():
        kept = [m for m in members if _candidate_passes(m, args)]
        if not kept:
            continue
        kept = kept[: max(1, int(args.max_variants_per_cluster))]
        cluster_size = int(max(int(m.get("cluster_size", 1)) for m in members))
        cluster_weight = float(1.0 / math.sqrt(max(1, cluster_size)))
        for row in kept:
            row["cluster_balance_weight"] = cluster_weight
        cluster_quality = float(max(float(m.get("quality_score", 0.0)) for m in kept))
        cluster_entries.append((cluster_id, kept, cluster_quality))

    cluster_entries.sort(key=lambda item: (item[2], item[0]), reverse=True)
    if int(args.max_clusters) > 0:
        cluster_entries = cluster_entries[: int(args.max_clusters)]

    # Round-robin emit to avoid cluster dominance by size.
    selected: list[dict[str, Any]] = []
    queues = {cid: list(rows_) for cid, rows_, _q in cluster_entries}
    ordered_cluster_ids = [cid for cid, _rows, _q in cluster_entries]
    while True:
        progressed = False
        for cid in ordered_cluster_ids:
            queue = queues[cid]
            if not queue:
                continue
            selected.append(queue.pop(0))
            progressed = True
        if not progressed:
            break

    cluster_counts = {cid: sum(1 for row in selected if str(row.get("cluster_id", "")) == cid) for cid in ordered_cluster_ids}
    evidence = {
        "clusters_available": len(by_cluster),
        "clusters_selected": len(ordered_cluster_ids),
        "cluster_selected_candidate_counts": cluster_counts,
        "max_variants_per_cluster": int(args.max_variants_per_cluster),
    }
    return selected, evidence


def _selection_hash(rows: list[dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for row in rows:
        key = "|".join(
            [
                str(row.get("build", "")),
                str(row.get("tile_id", "")),
                str(row.get("map", "")),
                str(row.get("tile_x", "")),
                str(row.get("tile_y", "")),
                str(row.get("cluster_ids", "")),
            ]
        )
        h.update(key.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _difficulty_bucket_from_rank(rank: int) -> str:
    rank_i = max(0, min(len(DIFFICULTY_BUCKETS) - 1, int(rank)))
    return str(DIFFICULTY_BUCKETS[rank_i])


def _tile_rows_from_candidates(selected_candidates: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tile_rows: dict[tuple[str, int], dict[str, Any]] = {}
    skipped_no_tile_id = 0
    max_tiles = int(args.max_tiles)
    reached_cap = False

    raw_tile_refs = 0
    selected_tile_refs = 0
    unique_raw_tiles: set[tuple[str, int]] = set()

    for row in selected_candidates:
        coverage = list(row.get("tile_coverage", []))
        raw_tile_refs += len(coverage)
        for cov in coverage:
            build = str(row.get("build", ""))
            tile_id = int(cov.get("tile_id", -1))
            if build and tile_id >= 0:
                unique_raw_tiles.add((build, tile_id))

    for row in selected_candidates:
        quality = _candidate_quality(row)
        row.update(quality)
        coverage = list(row.get("tile_coverage", []))
        for cov in coverage:
            build = str(row.get("build", ""))
            tile_id = int(cov.get("tile_id", -1))
            if not build or tile_id < 0:
                skipped_no_tile_id += 1
                continue
            key = (build, tile_id)
            if key not in tile_rows and max_tiles > 0 and len(tile_rows) >= max_tiles:
                reached_cap = True
                break
            selected_tile_refs += 1
            map_name = str(row.get("map", ""))
            tile_x = int(cov.get("tile_x", -1))
            tile_y = int(cov.get("tile_y", -1))

            existing = tile_rows.get(key)
            if existing is None:
                tile_rows[key] = {
                    "profile": str(args.profile),
                    "keep": True,
                    "build": build,
                    "tile_id": tile_id,
                    "map": map_name,
                    "tile_x": tile_x,
                    "tile_y": tile_y,
                    "quality_score": float(quality["quality_score"]),
                    "usefulness_score": float(quality["usefulness_score"]),
                    "difficulty_score": float(quality["difficulty_score"]),
                    "difficulty_rank": int(quality["difficulty_rank"]),
                    "score_deformation_richness": float(quality["score_deformation_richness"]),
                    "score_normal_coverage": float(quality["score_normal_coverage"]),
                    "score_terrain_validity": float(quality["score_terrain_validity"]),
                    "score_painted_signal": float(quality["score_painted_signal"]),
                    "score_minimap_target_usefulness": float(quality["score_minimap_target_usefulness"]),
                    "score_pathology_pressure": float(quality["score_pathology_pressure"]),
                    "what_plate": False,
                    "reject_reason": None,
                    "_source_candidate_ids": [int(row.get("candidate_id", -1))],
                    "_source_clusters": {str(row.get("cluster_id", ""))},
                    "_source_cluster_weights": [float(row.get("cluster_balance_weight", 1.0))],
                }
            else:
                existing["quality_score"] = float(max(existing["quality_score"], float(quality["quality_score"])))
                existing["usefulness_score"] = float(max(existing["usefulness_score"], float(quality["usefulness_score"])))
                existing["difficulty_score"] = float(max(existing["difficulty_score"], float(quality["difficulty_score"])))
                existing["difficulty_rank"] = int(max(existing["difficulty_rank"], int(quality["difficulty_rank"])))
                existing["score_deformation_richness"] = float(max(existing["score_deformation_richness"], float(quality["score_deformation_richness"])))
                existing["score_normal_coverage"] = float(max(existing["score_normal_coverage"], float(quality["score_normal_coverage"])))
                existing["score_terrain_validity"] = float(max(existing["score_terrain_validity"], float(quality["score_terrain_validity"])))
                existing["score_painted_signal"] = float(max(existing["score_painted_signal"], float(quality["score_painted_signal"])))
                existing["score_minimap_target_usefulness"] = float(
                    max(existing["score_minimap_target_usefulness"], float(quality["score_minimap_target_usefulness"]))
                )
                existing["score_pathology_pressure"] = float(max(existing["score_pathology_pressure"], float(quality["score_pathology_pressure"])))
                existing["_source_candidate_ids"].append(int(row.get("candidate_id", -1)))
                existing["_source_clusters"].add(str(row.get("cluster_id", "")))
                existing["_source_cluster_weights"].append(float(row.get("cluster_balance_weight", 1.0)))
        if reached_cap:
            break

    out_rows: list[dict[str, Any]] = []
    for (_build, _tile_id), row in sorted(tile_rows.items(), key=lambda item: (item[0][0], item[0][1])):
        row["difficulty_bucket"] = _difficulty_bucket_from_rank(int(row["difficulty_rank"]))
        clusters = sorted(c for c in row.pop("_source_clusters") if c)
        candidate_ids = sorted(cid for cid in row.pop("_source_candidate_ids") if cid >= 0)
        weights = row.pop("_source_cluster_weights")
        row["source_candidate_ids"] = ",".join(str(v) for v in candidate_ids)
        row["source_cluster_ids"] = ",".join(clusters)
        row["source_cluster_count"] = int(len(clusters))
        row["cluster_balance_weight_mean"] = float(np.mean(weights)) if weights else 1.0
        out_rows.append(row)

    raw_duplicate_ratio = 1.0 - (float(len(unique_raw_tiles)) / float(raw_tile_refs)) if raw_tile_refs > 0 else 0.0
    selected_duplicate_ratio = 1.0 - (float(len(out_rows)) / float(selected_tile_refs)) if selected_tile_refs > 0 else 0.0
    evidence = {
        "raw_tile_refs": int(raw_tile_refs),
        "selected_tile_refs": int(selected_tile_refs),
        "unique_raw_tiles": int(len(unique_raw_tiles)),
        "unique_selected_tiles": int(len(out_rows)),
        "duplicate_ratio_raw_refs": float(raw_duplicate_ratio),
        "duplicate_ratio_selected_refs": float(selected_duplicate_ratio),
        "skipped_no_tile_id_refs": int(skipped_no_tile_id),
        "tile_cap_reached": bool(reached_cap),
    }
    return out_rows, evidence


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def main() -> None:
    args = _parse_args()
    input_path = _resolve_deduped_path(args.deduped_candidates)
    run_name = str(args.run_name)
    output_dir = args.output_dir or (Path("../output/tmp") / run_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(input_path)
    if not rows:
        raise RuntimeError(f"No rows loaded from {input_path}")

    selected_candidates, cluster_evidence = _cluster_balanced_candidates(rows, args)
    tile_rows, tile_evidence = _tile_rows_from_candidates(selected_candidates, args)
    if not tile_rows:
        raise RuntimeError("Refined manifest is empty after quality gates and selection.")

    write_rows_parquet(output_dir / "kept_tiles.parquet", tile_rows)
    write_rows_parquet(output_dir / "tiles.parquet", tile_rows)
    _write_jsonl(output_dir / "tiles.jsonl", tile_rows)
    _write_jsonl(output_dir / "selected_candidates.jsonl", selected_candidates)

    # Trainer compatibility snapshot with explicit per-bucket counts.
    bucket_counts: dict[str, int] = {}
    build_counts: dict[str, int] = {}
    cluster_counts: dict[str, int] = {}
    for row in tile_rows:
        bucket = str(row.get("difficulty_bucket", "unbucketed"))
        build = str(row.get("build", "unknown"))
        bucket_counts[bucket] = int(bucket_counts.get(bucket, 0) + 1)
        build_counts[build] = int(build_counts.get(build, 0) + 1)
        for cluster_id in str(row.get("source_cluster_ids", "")).split(","):
            cluster_id = cluster_id.strip()
            if not cluster_id:
                continue
            cluster_counts[cluster_id] = int(cluster_counts.get(cluster_id, 0) + 1)

    summary = {
        "profile": str(args.profile),
        "run_name": run_name,
        "input_path": str(input_path),
        "candidate_rows_in": int(len(rows)),
        "candidate_rows_selected": int(len(selected_candidates)),
        "kept_tiles": int(len(tile_rows)),
        "selection_hash": _selection_hash(tile_rows),
        "cluster_distribution": {
            "clusters_available": int(cluster_evidence["clusters_available"]),
            "clusters_selected": int(cluster_evidence["clusters_selected"]),
            "cluster_selected_candidate_counts": cluster_evidence["cluster_selected_candidate_counts"],
            "cluster_tile_counts": dict(sorted(cluster_counts.items(), key=lambda item: item[0])),
        },
        "duplicate_ratio_metrics": tile_evidence,
        "bucket_counts": dict(sorted(bucket_counts.items())),
        "build_counts": dict(sorted(build_counts.items())),
        "selection_knobs": {
            "max_clusters": int(args.max_clusters),
            "max_variants_per_cluster": int(args.max_variants_per_cluster),
            "max_tiles": int(args.max_tiles),
            "min_score_mean": float(args.min_score_mean),
            "min_transition_mean": float(args.min_transition_mean),
            "min_hard_mean": float(args.min_hard_mean),
            "min_train_mask_mean": float(args.min_train_mask_mean),
            "min_tile_coverage_count": int(args.min_tile_coverage_count),
            "keep_noncanonical": bool(args.keep_noncanonical),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "config.snapshot.json").write_text(json.dumps(vars(args), indent=2, default=str, sort_keys=True), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Wrote {output_dir / 'kept_tiles.parquet'}")


if __name__ == "__main__":
    main()
