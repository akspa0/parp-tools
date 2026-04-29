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
DEFAULT_MAX_SELECTED_FRACTION = 0.5
DEFAULT_PATTERN_DICTIONARIES = (
    Path("output/build-validation/v10-wave2-hybrid-proof/brush_dictionary.json"),
    Path("output/build-validation/v10-wave2-mcal-brushes/mcal_brush_dictionary.json"),
    Path("output/build-validation/v10-wave2-mcal-compositions/mcal_composition_dictionary.json"),
    Path("output/build-validation/v10-wave2-height-profiles/height_profile_dictionary.json"),
    Path("output/build-validation/v10-wave2-prefab-cells-large/8x8/prefab_cell_dictionary.json"),
    Path("output/build-validation/v10-wave2-prefab-cells-large/12x12/prefab_cell_dictionary.json"),
    Path("output/build-validation/v10-wave2-prefab-cells-large/16x16/prefab_cell_dictionary.json"),
)


@dataclass(slots=True)
class Candidate:
    shard_path: Path
    tile_name: str
    dataset_key: str
    era_tag: str
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
    parser.add_argument(
        "--max-selected-fraction",
        type=float,
        default=DEFAULT_MAX_SELECTED_FRACTION,
        help=(
            "Maximum fraction of accepted shards to keep after quality filtering. "
            "Default 0.5 keeps the curated set at or below half of the valid pool; use 0 to disable."
        ),
    )
    parser.add_argument("--min-height-range", type=float, default=1.0)
    parser.add_argument("--min-minimap-variance", type=float, default=1.0e-6)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--include-flat", action="store_true", help="Keep flat/blank height shards.")
    parser.add_argument("--include-blank-minimap", action="store_true", help="Keep near-blank minimap shards.")
    parser.add_argument(
        "--pattern-dictionary",
        action="append",
        default=[],
        help="Optional Wave 2 pattern dictionary JSON to attach as per-tile curation hints. May be repeated.",
    )
    parser.add_argument(
        "--no-default-pattern-dictionaries",
        action="store_true",
        help="Do not auto-load known local Wave 2 pattern dictionary outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    input_paths = [Path(value).resolve() for value in args.inputs]
    candidates = list(discover_candidates(input_paths))
    pattern_index, pattern_sources = load_pattern_index(
        [Path(value).resolve() for value in args.pattern_dictionary],
        include_defaults=not args.no_default_pattern_dictionaries,
    )

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    signal_counts: Counter[str] = Counter()
    array_counts: Counter[str] = Counter()

    for candidate in candidates:
        status = inspect_candidate(candidate, args, pattern_index)
        if status["accepted"]:
            entry = status["entry"]
            accepted.append(entry)
            signal_counts.update(entry["available_signals"])
            array_counts.update(entry["array_names"])
        else:
            rejected.append(status["rejection"])

    preselection_accepted_count = len(accepted)
    accepted = select_balanced(accepted, args.max_per_dataset, args.max_total, args.max_selected_fraction, rng)
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
            "max_selected_fraction": args.max_selected_fraction,
            "seed": args.seed,
        },
        "pattern_dictionary_sources": [str(path) for path in pattern_sources],
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
        pattern_sources,
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
                dataset_key = infer_dataset_key(npz_path)
                yield Candidate(npz_path, npz_path.stem, dataset_key, infer_era_tag(dataset_key, npz_path), None, "directory-npz", {})
            continue

        if path.suffix.lower() == ".npz" and path.is_file():
            dataset_key = infer_dataset_key(path)
            yield Candidate(path, path.stem, dataset_key, infer_era_tag(dataset_key, path), None, "single-npz", {})
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
        era_tag = infer_era_tag(dataset_key, shard_path, entry)
        yield Candidate(shard_path, tile_name, dataset_key, era_tag, manifest_path, schema, entry)


def inspect_candidate(candidate: Candidate, args: argparse.Namespace, pattern_index: dict[str, Counter[str]]) -> dict[str, Any]:
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
            pattern_detection = build_pattern_detection(candidate, pattern_index)
            if pattern_detection:
                available_signals.append("pattern_detection")
                available_signals = sorted(set(available_signals))
            quality_score = score_candidate(height_range, minimap_variance, minimap_gradient, available_signals, pattern_detection)
            entry = {
                "dataset_key": candidate.dataset_key,
                "era_tag": candidate.era_tag,
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
            if pattern_detection:
                entry["pattern_detection"] = pattern_detection
            return {"accepted": True, "entry": entry}
    except Exception as exc:
        return reject(candidate, "unreadable_npz", error=str(exc))


def reject(candidate: Candidate, reason: str, **extra: Any) -> dict[str, Any]:
    rejection = {
        "dataset_key": candidate.dataset_key,
        "era_tag": candidate.era_tag,
        "tile_name": candidate.tile_name,
        "shard_path": str(candidate.shard_path),
        "source_manifest": str(candidate.source_manifest) if candidate.source_manifest else "",
        "reason": reason,
    }
    rejection.update(extra)
    return {"accepted": False, "rejection": rejection}


def select_balanced(
    entries: list[dict[str, Any]],
    max_per_dataset: int,
    max_total: int,
    max_selected_fraction: float,
    rng: random.Random,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        grouped[entry["dataset_key"]].append(entry)

    selected: list[dict[str, Any]] = []
    for dataset_entries in grouped.values():
        dataset_entries.sort(key=lambda item: (-float(item["quality_score"]), item["tile_name"]))
        if max_per_dataset > 0:
            dataset_entries = dataset_entries[:max_per_dataset]
        selected.extend(dataset_entries)

    target_count = len(selected)
    if 0.0 < max_selected_fraction < 1.0:
        target_count = min(target_count, max(1, int(math.floor(len(entries) * max_selected_fraction))))
    if max_total > 0:
        target_count = min(target_count, max_total)
    if len(selected) > target_count:
        selected = select_era_balanced(selected, target_count, rng)

    return selected


def select_era_balanced(entries: list[dict[str, Any]], target_count: int, rng: random.Random) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        era_tag = str(entry.get("era_tag") or "unknown")
        grouped[f"{era_tag}::{entry['dataset_key']}"].append(entry)

    ordered_groups: list[tuple[str, list[dict[str, Any]]]] = []
    for key, group_entries in grouped.items():
        rng.shuffle(group_entries)
        group_entries.sort(
            key=lambda item: (
                1 if item.get("pattern_detection") else 0,
                float(item["quality_score"]),
                item["tile_name"],
            ),
            reverse=True,
        )
        ordered_groups.append((key, group_entries))

    ordered_groups.sort(key=lambda item: item[0])
    selected: list[dict[str, Any]] = []
    while len(selected) < target_count and any(group for _, group in ordered_groups):
        for _, group in ordered_groups:
            if not group:
                continue
            selected.append(group.pop(0))
            if len(selected) >= target_count:
                break
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


def infer_era_tag(dataset_key: str, path: Path, source_entry: dict[str, Any] | None = None) -> str:
    if source_entry:
        for key in ("era_tag", "EraTag", "build_key", "BuildKey", "build", "Build"):
            value = source_entry.get(key)
            if isinstance(value, str) and value.strip():
                return normalize_era_tag(value)

    if "__" in dataset_key:
        prefix = dataset_key.split("__", 1)[0]
        if prefix:
            return normalize_era_tag(prefix)

    for part in reversed(path.parts):
        if "__" in part:
            return normalize_era_tag(part.split("__", 1)[0])

    return "unknown"


def normalize_era_tag(value: str) -> str:
    return value.strip().replace(".", "_").replace("-", "_").lower() or "unknown"


def compute_minimap_gradient(minimap: np.ndarray) -> float:
    rgb = minimap.astype(np.float32) / 255.0
    dx = np.abs(rgb[:, 1:, :] - rgb[:, :-1, :]).mean()
    dy = np.abs(rgb[1:, :, :] - rgb[:-1, :, :]).mean()
    return finite_float(dx + dy)


def score_candidate(
    height_range: float,
    minimap_variance: float,
    minimap_gradient: float,
    signals: list[str],
    pattern_detection: dict[str, Any] | None,
) -> float:
    signal_bonus = min(1.5, len(signals) / 20.0)
    pattern_bonus = 0.0
    if pattern_detection:
        pattern_bonus = min(1.0, float(pattern_detection.get("total_hits", 0)) / 16.0)
    return math.log1p(max(0.0, height_range)) + (minimap_variance * 20.0) + (minimap_gradient * 10.0) + signal_bonus + pattern_bonus


def load_pattern_index(extra_paths: list[Path], include_defaults: bool) -> tuple[dict[str, Counter[str]], list[Path]]:
    source_paths: list[Path] = []
    if include_defaults:
        source_paths.extend(Path(path).resolve() for path in DEFAULT_PATTERN_DICTIONARIES)
    source_paths.extend(extra_paths)

    index: dict[str, Counter[str]] = defaultdict(Counter)
    loaded_sources: list[Path] = []
    for path in source_paths:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        category = infer_pattern_category(payload, path)
        for tile_name in extract_pattern_tile_names(payload):
            normalized = normalize_tile_key(tile_name)
            if normalized:
                index[normalized][category] += 1
        loaded_sources.append(path)

    return index, loaded_sources


def infer_pattern_category(payload: dict[str, Any], path: Path) -> str:
    schema = str(payload.get("schema_version") or "").lower()
    name = path.name.lower()
    if "prefab-cell" in schema or "prefab_cell" in name:
        return "prefab_cell"
    if "height-profile" in schema or "height_profile" in name:
        return "height_profile"
    if "mcal-composition" in schema or "mcal_composition" in name:
        return "mcal_composition"
    if "mcal-brush" in schema or "mcal_brush" in name:
        return "mcal_brush"
    if "brush_dictionary" in name:
        return "anchor_brush"
    if "mcly" in schema or "mclay" in name:
        return "mcly_palette"
    return path.stem


def extract_pattern_tile_names(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key in ("tile_name", "TileName"):
            item = value.get(key)
            if isinstance(item, str):
                yield item
        examples = value.get("examples")
        if isinstance(examples, list):
            for item in examples:
                yield from extract_pattern_tile_names(item)
        example_chunks = value.get("example_chunks")
        if isinstance(example_chunks, list):
            for item in example_chunks:
                yield from extract_pattern_tile_names(item)
        example_tiles = value.get("example_tiles")
        if isinstance(example_tiles, list):
            for item in example_tiles:
                if isinstance(item, str):
                    yield item
                else:
                    yield from extract_pattern_tile_names(item)
        dictionary = value.get("dictionary")
        if isinstance(dictionary, list):
            for item in dictionary:
                yield from extract_pattern_tile_names(item)
        patterns = value.get("patterns")
        if isinstance(patterns, list):
            for item in patterns:
                yield from extract_pattern_tile_names(item)
        labels = value.get("labels")
        if isinstance(labels, list):
            for item in labels:
                yield from extract_pattern_tile_names(item)
        return

    if isinstance(value, list):
        for item in value:
            yield from extract_pattern_tile_names(item)


def build_pattern_detection(candidate: Candidate, pattern_index: dict[str, Counter[str]]) -> dict[str, Any] | None:
    counters = Counter()
    for key in {normalize_tile_key(candidate.tile_name), normalize_tile_key(candidate.shard_path.stem)}:
        if key in pattern_index:
            counters.update(pattern_index[key])

    if not counters:
        return None

    return {
        "schema_version": "v10-pattern-detection-hints.v1",
        "categories": dict(sorted(counters.items())),
        "category_count": len(counters),
        "total_hits": int(sum(counters.values())),
    }


def normalize_tile_key(value: str) -> str:
    result = value.strip().replace("\\", "/").split("/")[-1].lower()
    if result.endswith(".npz"):
        result = result[:-4]
    if result.endswith("_v10"):
        result = result[:-4]
    return result


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
    pattern_sources: list[Path],
) -> dict[str, Any]:
    accepted_by_dataset = Counter(entry["dataset_key"] for entry in accepted)
    accepted_by_era = Counter(str(entry.get("era_tag") or "unknown") for entry in accepted)
    pattern_category_counts: Counter[str] = Counter()
    pattern_sample_count = 0
    for entry in accepted:
        pattern_detection = entry.get("pattern_detection")
        if not isinstance(pattern_detection, dict):
            continue
        pattern_sample_count += 1
        categories = pattern_detection.get("categories")
        if isinstance(categories, dict):
            pattern_category_counts.update({str(key): int(value) for key, value in categories.items()})
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
        "accepted_by_era": dict(sorted(accepted_by_era.items())),
        "rejected_by_reason": dict(sorted(rejected_by_reason.items())),
        "signal_counts": dict(sorted(selected_signal_counts.items())),
        "array_counts": dict(sorted(selected_array_counts.items())),
        "preselection_signal_counts": dict(sorted(preselection_signal_counts.items())),
        "preselection_array_counts": dict(sorted(preselection_array_counts.items())),
        "pattern_dictionary_sources": [str(path) for path in pattern_sources],
        "pattern_sample_count": pattern_sample_count,
        "pattern_category_counts": dict(sorted(pattern_category_counts.items())),
        "rejected_examples": rejected[:100],
    }


if __name__ == "__main__":
    main()
