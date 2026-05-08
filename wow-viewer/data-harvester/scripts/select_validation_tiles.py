import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np


def compute_gradient(rgb: np.ndarray) -> float:
    luminance = rgb.mean(axis=2)
    dx = np.diff(luminance, axis=1)
    dy = np.diff(luminance, axis=0)
    gx = dx[:-1, :]
    gy = dy[:, :-1]
    return float(np.sqrt(gx * gx + gy * gy).mean())


def compute_score(npz_path: Path) -> dict:
    with np.load(npz_path) as data:
        height = data["height_257"] if "height_257" in data else None
        minimap = data["minimap_rgb_256"] if "minimap_rgb_256" in data else None
        holes = data["hole_mask_16"] if "hole_mask_16" in data else None

        height_range = float(np.max(height) - np.min(height)) if height is not None else 0.0
        minimap_variance = 0.0
        minimap_gradient = 0.0
        if minimap is not None:
            minimap_norm = minimap.astype(np.float32) / 255.0
            minimap_variance = float(np.var(minimap_norm))
            minimap_gradient = compute_gradient(minimap_norm)

        hole_coverage = float(np.mean(holes.astype(np.float32))) if holes is not None else 0.0

        height_score = min(height_range / 64.0, 4.0)
        minimap_score = min(minimap_gradient / 0.02, 3.0) + min(minimap_variance / 0.01, 3.0)
        hole_penalty = min(hole_coverage * 2.0, 1.0)
        score = height_score + minimap_score - hole_penalty

        return {
            "height_range": height_range,
            "minimap_variance": minimap_variance,
            "minimap_gradient": minimap_gradient,
            "hole_coverage": hole_coverage,
            "score": score,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Select validation tiles from harvested NPZ shards")
    parser.add_argument("input_root", help="Root directory containing shards/<build>/<map>/*.npz")
    parser.add_argument("--output-json", required=True, help="Output JSON selection summary")
    parser.add_argument("--copy-dir", required=True, help="Directory to copy selected NPZ files into")
    parser.add_argument("--per-bucket", type=int, default=5, help="Random samples per complexity bucket per build")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed")
    args = parser.parse_args()

    root = Path(args.input_root)
    copy_dir = Path(args.copy_dir)
    copy_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    by_build: dict[str, list[dict]] = {}

    for npz_path in sorted(root.glob("*/*/*.npz")):
        rel = npz_path.relative_to(root)
        if len(rel.parts) < 3:
            continue
        build, map_name = rel.parts[0], rel.parts[1]
        metrics = compute_score(npz_path)
        by_build.setdefault(build, []).append({
            "build": build,
            "map": map_name,
            "path": str(npz_path),
            **metrics,
        })

    selections: list[dict] = []
    summaries: list[dict] = []

    for build, items in sorted(by_build.items()):
        ranked = sorted(items, key=lambda item: item["score"])
        count = len(ranked)
        if count == 0:
            continue

        low_cut = count // 3
        mid_cut = (count * 2) // 3
        buckets = {
            "low": ranked[:low_cut],
            "medium": ranked[low_cut:mid_cut],
            "high": ranked[mid_cut:],
        }

        for bucket_name, bucket_items in buckets.items():
            if not bucket_items:
                continue
            chosen = rng.sample(bucket_items, k=min(args.per_bucket, len(bucket_items)))
            bucket_dir = copy_dir / build / bucket_name
            bucket_dir.mkdir(parents=True, exist_ok=True)

            for item in chosen:
                src = Path(item["path"])
                dst = bucket_dir / src.name
                shutil.copy2(src, dst)
                item["copied_to"] = str(dst)
                item["bucket"] = bucket_name
                selections.append(item)

            summaries.append({
                "build": build,
                "bucket": bucket_name,
                "available": len(bucket_items),
                "selected": len(chosen),
                "min_score": min(entry["score"] for entry in bucket_items),
                "max_score": max(entry["score"] for entry in bucket_items),
            })

    output = {
        "schema_version": "validation-selection.v1",
        "input_root": str(root),
        "copy_dir": str(copy_dir),
        "per_bucket": args.per_bucket,
        "seed": args.seed,
        "summaries": summaries,
        "selections": selections,
    }

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Selected {len(selections)} validation tiles -> {args.output_json}")


if __name__ == "__main__":
    main()
