from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REQUIRED_ARRAYS = ("minimap_rgb_256", "height_257", "height_17")
DEFAULT_OUTPUT = Path("output/ml-training/v10_curated/curated_v10_training_manifest.json")


@dataclass(slots=True)
class Candidate:
    shard_path: Path
    tile_name: str
    dataset_key: str
    source_manifest: Path | None
    source_schema: str
    source_entry: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Curate NPZ shards for v10 terrain training from native v10 and legacy v9 manifests.")
    parser.add_argument("inputs", nargs="+", help="Manifest JSON files, NPZ files, or directories containing either.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--report", default="")
    parser.add_argument("--max-per-dataset", type=int, default=0)
    parser.add_argument("--max-total", type=int, default=0)
    parser.add_argument("--min-height-range", type=float, default=1.0)
    parser.add_argument("--min-minimap-variance", type=float, default=1.0e-6)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--include-flat", action="store_true", help="Keep flat/blank height shards.")
    parser.add_argument("--include-blank-minimap", action="store_true", help="Keep near-blank minimap shards.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    input_paths = [Path(value).resolve() for value in args.inputs]
    candidates = list(discover_candidates(input_paths))

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    signal_counts: Counter[str] = Counter()
    array_counts: Counter[str] = Counter()

    for candidate in candidates:
        status = inspect_candidate(candidate, args)
        if status["accepted"]:
            entry = status["entry"]
            accepted.append(entry)
            signal_counts.update(entry["available_signals"])
            array_counts.update(entry["array_names"])
        else:
            rejected.append(status["rejection"])

    preselection_accepted_count = len(accepted)
    accepted = select_balanced(accepted, args.max_per_dataset, args.max_total, rng)
    accepted.sort(key=lambda item: (item["dataset_key"], item["tile_name"], item["shard_path"].lower()))
    selected_signal_counts = Counter()
    selected_array_counts = Counter()
    for entry in accepted:
        selected_signal_counts.update(entry["available_signals"])
        selected_array_counts.update(entry["array_names"])

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "v10-training-curated-manifest.v1",
        "inputs": [str(path) for path in input_paths],
        "required_arrays": list(REQUIRED_ARRAYS),
        "filters": {
            "min_height_range": args.min_height_range,
            "min_minimap_variance": args.min_minimap_variance,
            "include_flat": args.include_flat,
            "include_blank_minimap": args.include_blank_minimap,
            "max_per_dataset": args.max_per_dataset,
            "max_total": args.max_total,
            "seed": args.seed,
        },
        "entries": accepted,
    }
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    report_path = Path(args.report).resolve() if args.report else output_path.with_name(output_path.stem + "_report.json")
    report = build_report(
        candidates,
        accepted,
        rejected,
        signal_counts,
        array_counts,
        selected_signal_counts,
        selected_array_counts,
        preselection_accepted_count,
    )
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("v10 shard curation report")
    print(f"Candidates: {len(candidates)}")
    print(f"Accepted: {len(accepted)}")
    print(f"Rejected: {len(rejected)}")
    print(f"Manifest: {output_path}")
    print(f"Report: {report_path}")


def discover_candidates(inputs: Iterable[Path]) -> Iterable[Candidate]:
    for path in inputs:
        if path.is_dir():
            for manifest in sorted(path.rglob("*.json")):
                yield from discover_manifest_candidates(manifest)
            for npz_path in sorted(path.rglob("*.npz")):
                yield Candidate(npz_path, npz_path.stem, infer_dataset_key(npz_path), None, "directory-npz", {})
            continue

        if path.suffix.lower() == ".npz" and path.is_file():
            yield Candidate(path, path.stem, infer_dataset_key(path), None, "single-npz", {})
            continue

        if path.suffix.lower() == ".json" and path.is_file():
            yield from discover_manifest_candidates(path)


def discover_manifest_candidates(manifest_path: Path) -> Iterable[Candidate]:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return

    schema = str(payload.get("schema_version") or payload.get("SchemaVersion") or "")
    entries = payload.get("entries") or payload.get("Entries") or []
    if not isinstance(entries, list):
        return

    base_dir = manifest_path.parent
    for entry in entries:
        if not isinstance(entry, dict):
            continue

        shard_value = (
            entry.get("shard_path")
            or entry.get("ShardPath")
            or entry.get("npz_path")
            or entry.get("NpzPath")
        )
        if not isinstance(shard_value, str) or not shard_value.lower().endswith(".npz"):
            continue

        shard_path = Path(shard_value)
        if not shard_path.is_absolute():
            shard_path = (base_dir / shard_path).resolve()

        tile_name = str(entry.get("tile_name") or entry.get("TileName") or shard_path.stem)
        dataset_key = str(entry.get("dataset_key") or entry.get("DatasetKey") or infer_dataset_key(shard_path))
        yield Candidate(shard_path, tile_name, dataset_key, manifest_path, schema, entry)


def inspect_candidate(candidate: Candidate, args: argparse.Namespace) -> dict[str, Any]:
    if not candidate.shard_path.exists():
        return reject(candidate, "missing_shard")

    try:
        with np.load(candidate.shard_path, allow_pickle=False) as shard:
            array_names = sorted(shard.files)
            missing = [name for name in REQUIRED_ARRAYS if name not in shard.files]
            if missing:
                return reject(candidate, "missing_required_arrays", missing=missing, array_names=array_names)

            minimap = np.asarray(shard["minimap_rgb_256"])
            height = np.asarray(shard["height_257"], dtype=np.float32)
            height17 = np.asarray(shard["height_17"], dtype=np.float32)
            if minimap.shape != (256, 256, 3):
                return reject(candidate, "bad_minimap_shape", shape=list(minimap.shape), array_names=array_names)
            if height.shape != (257, 257):
                return reject(candidate, "bad_height_257_shape", shape=list(height.shape), array_names=array_names)
            if height17.shape != (17, 17):
                return reject(candidate, "bad_height_17_shape", shape=list(height17.shape), array_names=array_names)

            height_min = finite_float(np.nanmin(height))
            height_max = finite_float(np.nanmax(height))
            height_range = height_max - height_min
            minimap_variance = finite_float(np.var(minimap.astype(np.float32) / 255.0))
            minimap_gradient = compute_minimap_gradient(minimap)

            if not args.include_flat and height_range < args.min_height_range:
                return reject(candidate, "height_range_below_threshold", height_range=height_range, array_names=array_names)
            if not args.include_blank_minimap and minimap_variance < args.min_minimap_variance:
                return reject(candidate, "minimap_variance_below_threshold", minimap_variance=minimap_variance, array_names=array_names)

            available_signals = infer_available_signals(array_names, candidate.source_entry)
            quality_score = score_candidate(height_range, minimap_variance, minimap_gradient, available_signals)
            entry = {
                "dataset_key": candidate.dataset_key,
                "tile_name": candidate.tile_name,
                "shard_path": str(candidate.shard_path),
                "source_manifest": str(candidate.source_manifest) if candidate.source_manifest else "",
                "source_schema": candidate.source_schema,
                "height_min": height_min,
                "height_max": height_max,
                "height_range": height_range,
                "minimap_variance": minimap_variance,
                "minimap_gradient": minimap_gradient,
                "quality_score": quality_score,
                "array_names": array_names,
                "available_signals": available_signals,
            }
            return {"accepted": True, "entry": entry}
    except Exception as exc:
        return reject(candidate, "unreadable_npz", error=str(exc))


def reject(candidate: Candidate, reason: str, **extra: Any) -> dict[str, Any]:
    rejection = {
        "dataset_key": candidate.dataset_key,
        "tile_name": candidate.tile_name,
        "shard_path": str(candidate.shard_path),
        "source_manifest": str(candidate.source_manifest) if candidate.source_manifest else "",
        "reason": reason,
    }
    rejection.update(extra)
    return {"accepted": False, "rejection": rejection}


def select_balanced(entries: list[dict[str, Any]], max_per_dataset: int, max_total: int, rng: random.Random) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        grouped[entry["dataset_key"]].append(entry)

    selected: list[dict[str, Any]] = []
    for dataset_entries in grouped.values():
        dataset_entries.sort(key=lambda item: (-float(item["quality_score"]), item["tile_name"]))
        if max_per_dataset > 0:
            dataset_entries = dataset_entries[:max_per_dataset]
        selected.extend(dataset_entries)

    if max_total > 0 and len(selected) > max_total:
        rng.shuffle(selected)
        selected.sort(key=lambda item: (-float(item["quality_score"]), item["dataset_key"], item["tile_name"]))
        selected = selected[:max_total]

    return selected


def infer_available_signals(array_names: list[str], source_entry: dict[str, Any]) -> list[str]:
    from_entry = source_entry.get("AvailableSignals") or source_entry.get("available_signals")
    signals = set(str(signal) for signal in from_entry) if isinstance(from_entry, list) else set()
    for name in array_names:
        if name == "metadata.json":
            continue
        signals.add(name)
    return sorted(signals)


def infer_dataset_key(path: Path) -> str:
    parent = path.parent.name
    if parent.lower() == "shards" and path.parent.parent.name:
        parent = path.parent.parent.name
    return parent or "dataset"


def compute_minimap_gradient(minimap: np.ndarray) -> float:
    rgb = minimap.astype(np.float32) / 255.0
    dx = np.abs(rgb[:, 1:, :] - rgb[:, :-1, :]).mean()
    dy = np.abs(rgb[1:, :, :] - rgb[:-1, :, :]).mean()
    return finite_float(dx + dy)


def score_candidate(height_range: float, minimap_variance: float, minimap_gradient: float, signals: list[str]) -> float:
    signal_bonus = min(1.5, len(signals) / 20.0)
    return math.log1p(max(0.0, height_range)) + (minimap_variance * 20.0) + (minimap_gradient * 10.0) + signal_bonus


def finite_float(value: Any) -> float:
    result = float(value)
    return result if math.isfinite(result) else 0.0


def build_report(
    candidates: list[Candidate],
    accepted: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    preselection_signal_counts: Counter[str],
    preselection_array_counts: Counter[str],
    selected_signal_counts: Counter[str],
    selected_array_counts: Counter[str],
    preselection_accepted_count: int,
) -> dict[str, Any]:
    accepted_by_dataset = Counter(entry["dataset_key"] for entry in accepted)
    rejected_by_reason = Counter(item["reason"] for item in rejected)
    source_schemas = Counter(candidate.source_schema for candidate in candidates)
    return {
        "schema_version": "v10-training-curation-report.v1",
        "candidate_count": len(candidates),
        "accepted_count": len(accepted),
        "preselection_accepted_count": preselection_accepted_count,
        "rejected_count": len(rejected),
        "source_schemas": dict(sorted(source_schemas.items())),
        "accepted_by_dataset": dict(sorted(accepted_by_dataset.items())),
        "rejected_by_reason": dict(sorted(rejected_by_reason.items())),
        "signal_counts": dict(sorted(selected_signal_counts.items())),
        "array_counts": dict(sorted(selected_array_counts.items())),
        "preselection_signal_counts": dict(sorted(preselection_signal_counts.items())),
        "preselection_array_counts": dict(sorted(preselection_array_counts.items())),
        "rejected_examples": rejected[:100],
    }


if __name__ == "__main__":
    main()
