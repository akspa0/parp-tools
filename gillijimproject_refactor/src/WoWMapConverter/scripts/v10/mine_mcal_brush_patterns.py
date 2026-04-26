#!/usr/bin/env python3
"""
Wave 2: Object-Anchored MCAL Brush Pattern Mining

Extracts 3D brush patterns by correlating alpha mask compositions with object placements.
These are spatial relationships between alpha textures and object assets — the actual
"3D brushes" used by artists to decorate and sculpt terrain.

A 3D brush pattern is defined as:
  - An alpha mask signature (spatial texture pattern)
  - A set of object asset paths that commonly co-occur with that signature
  - Spatial offsets between the alpha pattern center and object placement points
  - Height/Z relationships (objects placed on, above, or below terrain features)

The script mines these patterns by:
  1. Loading per-tile alpha packs + object placement catalogs (MDDF/MODF)
  2. Extracting alpha patches around each object placement
  3. Clustering patches that share similar object type signatures
  4. Building a dictionary of object-anchored brush patterns

Usage:
    python mine_mcal_brush_patterns.py \
        --input-dir /path/to/npz/tiles \
        --placement-dir /path/to/placement/json \
        --output-dir /path/to/output \
        --context-radius 64 \
        --dictionary-size 128
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine object-anchored 3D brush patterns from MCAL alpha + placement data"
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing Wave 1 .npz tile files")
    parser.add_argument("--placement-dir", required=True, help="Directory containing per-tile placement JSON (MDDF/MODF)")
    parser.add_argument("--output-dir", required=True, help="Directory to write brush dictionary and reports")
    parser.add_argument("--context-radius", type=int, default=64, help="Pixel radius around object to extract alpha context (64=half tile, 128=full tile)")
    parser.add_argument("--dictionary-size", type=int, default=128, help="Target number of 3D brush patterns")
    parser.add_argument("--min-occurrences", type=int, default=3, help="Minimum occurrences for a pattern to be kept")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def load_npz_files(input_dir: str) -> list[Path]:
    root = Path(input_dir)
    files = sorted(root.rglob("*.npz"))
    if not files:
        raise ValueError(f"No .npz files found in {input_dir}")
    print(f"Found {len(files)} .npz files", file=sys.stderr)
    return files


def load_placement_data(placement_dir: str, tile_name: str) -> list[dict[str, Any]]:
    """Load MDDF/MODF placement records for a tile."""
    # Try multiple naming conventions
    root = Path(placement_dir)
    candidates = [
        root / f"{tile_name}_placements.json",
        root / f"{tile_name}.json",
        root / tile_name / "placements.json",
    ]

    for path in candidates:
        if path.exists():
            with open(path) as f:
                data = json.load(f)
                if isinstance(data, dict):
                    # Flatten dict of lists
                    placements = []
                    for key, items in data.items():
                        if isinstance(items, list):
                            for item in items:
                                item["_source"] = key  # mddf or modf
                                placements.append(item)
                    return placements
                elif isinstance(data, list):
                    return data

    return []


def extract_full_tile_alpha(npz_path: Path) -> tuple[NDArray[np.float32], dict[str, Any]] | None:
    """Load the full 256x256x4 alpha pack from a Wave 1 .npz file."""
    data = np.load(npz_path)
    alpha_key = "mcal_alpha_pack_256"
    if alpha_key not in data:
        return None

    alpha = data[alpha_key].astype(np.float32)
    if alpha.shape != (256, 256, 4):
        return None

    meta = {"tile_name": str(npz_path.stem)}
    if "mcly_layer_mask" in data:
        meta["layer_mask"] = data["mcly_layer_mask"].astype(bool)
    if "mcly_texture_ids" in data:
        meta["texture_ids"] = data["mcly_texture_ids"].astype(np.int32)
    if "height_257" in data:
        meta["heightmap"] = data["height_257"].astype(np.float32)

    return alpha, meta


def world_to_tile_uv(world_x: float, world_y: float, tile_x: int, tile_y: int) -> tuple[float, float]:
    """Convert WoW world coordinates to tile-local UV (0-1)."""
    # WoW tile size = 533.3333 units
    tile_size = 533.3333
    # Tile origin in world space
    origin_x = tile_x * tile_size
    origin_y = tile_y * tile_size

    local_x = world_x - origin_x
    local_y = world_y - origin_y

    u = local_x / tile_size
    v = local_y / tile_size

    return u, v


def extract_alpha_context(
    alpha_tile: NDArray[np.float32],
    heightmap: NDArray[np.float32] | None,
    world_x: float,
    world_y: float,
    world_z: float,
    tile_x: int,
    tile_y: int,
    radius: int,
) -> dict[str, Any] | None:
    """
    Extract alpha context around an object placement point.
    Returns the alpha patch, height context, and spatial relationships.
    """
    u, v = world_to_tile_uv(world_x, world_y, tile_x, tile_y)

    # Convert to pixel coordinates (0-255)
    px = int(u * 255)
    py = int(v * 255)

    if px < 0 or px >= 256 or py < 0 or py >= 256:
        return None

    # Extract patch bounds
    x0 = max(0, px - radius)
    x1 = min(256, px + radius)
    y0 = max(0, py - radius)
    y1 = min(256, py + radius)

    patch = alpha_tile[y0:y1, x0:x1, :]

    # If patch is too small, skip
    if patch.shape[0] < radius or patch.shape[1] < radius:
        return None

    # Sample height at placement point
    height_at_point = None
    if heightmap is not None:
        height_at_point = float(heightmap[min(py, 256), min(px, 256)])

    # Compute patch statistics per layer
    layer_stats = []
    for layer in range(4):
        layer_patch = patch[:, :, layer]
        layer_stats.append({
            "mean": float(np.mean(layer_patch)),
            "std": float(np.std(layer_patch)),
            "max": float(np.max(layer_patch)),
            "entropy": compute_entropy(layer_patch),
        })

    # Dominant layer
    dominant_layer = int(np.argmax([s["mean"] for s in layer_stats]))

    return {
        "patch": patch,
        "center_px": (px, py),
        "height_at_point": height_at_point,
        "layer_stats": layer_stats,
        "dominant_layer": dominant_layer,
        "patch_shape": patch.shape,
    }


def compute_entropy(patch: NDArray[np.float32], bins: int = 16) -> float:
    """Compute spatial entropy."""
    hist, _ = np.histogram(patch.flatten(), bins=bins, range=(0.0, 1.0))
    probs = hist.astype(np.float32) / (hist.sum() + 1e-8)
    return float(-np.sum(probs * np.log2(probs + 1e-8)))


def classify_object_type(asset_path: str) -> str:
    """Classify an object by its asset path into a broad category."""
    path_lower = asset_path.lower()

    categories = {
        "tree": ["tree", "bush", "shrub", "fern", "plant", "palm", "oak", "pine", "birch"],
        "rock": ["rock", "stone", "boulder", "pebble", "cliff", "crag"],
        "building": ["building", "house", "hut", "tower", "barrack", "inn", "shop"],
        "structure": ["bridge", "wall", "fence", "gate", "arch", "pillar", "platform"],
        "detail": ["flower", "grass", "mushroom", "twig", "root", "leaf", "stick"],
        "wmo": [".wmo"],
    }

    for category, keywords in categories.items():
        if any(kw in path_lower for kw in keywords):
            return category

    return "other"


def extract_brush_instances(
    npz_path: Path,
    placement_dir: str,
    context_radius: int,
) -> list[dict[str, Any]]:
    """
    Extract all object-anchored brush instances from a tile.
    Each instance is an alpha context patch + object metadata.
    """
    result = extract_full_tile_alpha(npz_path)
    if result is None:
        return []

    alpha_tile, meta = result
    tile_name = meta["tile_name"]
    heightmap = meta.get("heightmap")

    # Parse tile coordinates from name (e.g., "Azeroth_30_40" or "development_16_32")
    tile_x, tile_y = parse_tile_coords(tile_name)
    if tile_x is None or tile_y is None:
        print(f"  Could not parse tile coords from {tile_name}, skipping", file=sys.stderr)
        return []

    placements = load_placement_data(placement_dir, tile_name)
    if not placements:
        return []

    instances = []
    for placement in placements:
        asset_path = placement.get("path", placement.get("model_path", placement.get("name", "")))
        if not asset_path:
            continue

        pos = placement.get("position", placement.get("pos", {}))
        if isinstance(pos, list) and len(pos) >= 3:
            world_x, world_y, world_z = pos[0], pos[1], pos[2]
        elif isinstance(pos, dict):
            world_x = pos.get("x", 0.0)
            world_y = pos.get("y", 0.0)
            world_z = pos.get("z", 0.0)
        else:
            continue

        context = extract_alpha_context(
            alpha_tile, heightmap,
            world_x, world_y, world_z,
            tile_x, tile_y,
            context_radius,
        )
        if context is None:
            continue

        instances.append({
            "tile_name": tile_name,
            "asset_path": asset_path,
            "object_type": classify_object_type(asset_path),
            "world_position": (world_x, world_y, world_z),
            "alpha_context": context["patch"],
            "dominant_layer": context["dominant_layer"],
            "layer_stats": context["layer_stats"],
            "height_at_point": context["height_at_point"],
            "object_scale": placement.get("scale", 1.0),
            "object_rotation": placement.get("rotation", placement.get("orient", 0.0)),
        })

    return instances


def parse_tile_coords(tile_name: str) -> tuple[int | None, int | None]:
    """Parse tile coordinates from filename like 'MapName_XX_YY' or 'MapName_XX_YY_tex0'."""
    import re
    # Match patterns like Azeroth_30_40, development_16_32, etc.
    match = re.search(r"_(\d+)_(\d+)(?:_|$)", tile_name)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def compute_brush_feature_vector(instance: dict[str, Any]) -> NDArray[np.float32]:
    """Compute a feature vector for clustering brush instances."""
    patch = instance["alpha_context"]
    blended = np.max(patch, axis=2)

    # Spatial statistics
    mean_val = float(np.mean(blended))
    std_val = float(np.std(blended))
    entropy = compute_entropy(blended)

    # Per-layer presence
    layer_presence = np.array([instance["layer_stats"][i]["mean"] for i in range(4)], dtype=np.float32)

    # Frequency features (capture repeating patterns)
    fft = np.fft.fft2(blended)
    fft_mag = np.abs(fft)
    fft_low = float(np.mean(fft_mag[:8, :8]))
    fft_high = float(np.mean(fft_mag[32:, 32:]))

    # Object type one-hot
    obj_types = ["tree", "rock", "building", "structure", "detail", "wmo", "other"]
    type_onehot = np.zeros(len(obj_types), dtype=np.float32)
    obj_type = instance["object_type"]
    if obj_type in obj_types:
        type_onehot[obj_types.index(obj_type)] = 1.0

    # Height relationship
    height_rel = instance.get("height_at_point", 0.0) or 0.0
    scale = float(instance.get("object_scale", 1.0))

    features = np.concatenate([
        np.array([mean_val, std_val, entropy, fft_low, fft_high, height_rel, scale], dtype=np.float32),
        layer_presence,
        type_onehot,
    ])

    return features


def cluster_brush_patterns(
    instances: list[dict[str, Any]],
    dictionary_size: int,
    min_occurrences: int,
    seed: int,
) -> dict[str, Any]:
    """
    Cluster object-anchored brush instances into 3D brush pattern types.
    """
    if len(instances) < dictionary_size:
        print(f"Warning: only {len(instances)} instances, reducing dictionary size", file=sys.stderr)
        dictionary_size = max(1, len(instances) // 2)

    print(f"Computing features for {len(instances)} brush instances...", file=sys.stderr)

    features = np.stack([compute_brush_feature_vector(inst) for inst in instances])

    # Normalize
    feat_mean = features.mean(axis=0)
    feat_std = features.std(axis=0) + 1e-8
    X_norm = (features - feat_mean) / feat_std

    # K-means++
    print(f"Clustering into {dictionary_size} 3D brush patterns...", file=sys.stderr)
    rng = np.random.default_rng(seed)
    centroids = kmeans_pp_init(X_norm, dictionary_size, rng)
    labels, centroids = lloyd_iterations(X_norm, centroids, max_iter=100)

    # Build dictionary
    dictionary = []
    for i in range(dictionary_size):
        mask = labels == i
        cluster_size = int(mask.sum())
        if cluster_size < min_occurrences:
            continue

        cluster_instances = [instances[j] for j in np.where(mask)[0]]

        # Aggregate alpha stamp (mean of all contexts in cluster)
        cluster_patches = [inst["alpha_context"] for inst in cluster_instances]
        # Normalize patch sizes to the smallest common shape
        min_h = min(p.shape[0] for p in cluster_patches)
        min_w = min(p.shape[1] for p in cluster_patches)
        normalized_patches = [p[:min_h, :min_w, :] for p in cluster_patches]
        stamp = np.stack(normalized_patches).mean(axis=0)

        # Aggregate object types
        type_counts: dict[str, int] = {}
        asset_counts: dict[str, int] = {}
        for inst in cluster_instances:
            t = inst["object_type"]
            type_counts[t] = type_counts.get(t, 0) + 1
            asset = inst["asset_path"]
            asset_counts[asset] = asset_counts.get(asset, 0) + 1

        # Top associated assets
        top_assets = sorted(asset_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        # Height relationship stats
        heights = [inst["height_at_point"] for inst in cluster_instances if inst["height_at_point"] is not None]
        height_mean = float(np.mean(heights)) if heights else 0.0
        height_std = float(np.std(heights)) if heights else 0.0

        dictionary.append({
            "pattern_id": i,
            "stamp": stamp.astype(np.float32),
            "cluster_size": cluster_size,
            "dominant_object_type": max(type_counts, key=type_counts.get),
            "object_type_distribution": type_counts,
            "top_assets": top_assets,
            "height_mean": height_mean,
            "height_std": height_std,
            "mean_layer_presence": [float(np.mean([inst["layer_stats"][j]["mean"] for inst in cluster_instances])) for j in range(4)],
        })

    return {
        "dictionary": dictionary,
        "total_instances": len(instances),
        "dictionary_size": len(dictionary),
        "feature_mean": feat_mean.astype(np.float32),
        "feature_std": feat_std.astype(np.float32),
    }


def kmeans_pp_init(data: NDArray[np.float32], k: int, rng: np.random.Generator) -> NDArray[np.float32]:
    n_samples, _ = data.shape
    centroids = np.zeros((k, data.shape[1]), dtype=np.float32)
    centroids[0] = data[rng.integers(n_samples)]

    for i in range(1, k):
        dists = np.min(np.sum((data[:, None, :] - centroids[:i][None, :, :]) ** 2, axis=2), axis=1)
        probs = dists / (dists.sum() + 1e-8)
        chosen = rng.choice(n_samples, p=probs)
        centroids[i] = data[chosen]

    return centroids


def lloyd_iterations(data: NDArray[np.float32], centroids: NDArray[np.float32], max_iter: int) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
    for _ in range(max_iter):
        dists = np.sum((data[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        labels = np.argmin(dists, axis=1).astype(np.int32)

        new_centroids = np.zeros_like(centroids)
        for i in range(centroids.shape[0]):
            mask = labels == i
            if mask.sum() > 0:
                new_centroids[i] = data[mask].mean(axis=0)
            else:
                new_centroids[i] = data[np.random.randint(len(data))]

        if np.allclose(centroids, new_centroids, atol=1e-5):
            break
        centroids = new_centroids

    return labels, centroids


def save_dictionary(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    dictionary = result["dictionary"]
    stamps = np.stack([d["stamp"] for d in dictionary])

    np.savez(
        output_dir / "object_anchored_brush_dictionary.npz",
        stamps=stamps.astype(np.float32),
        pattern_ids=np.array([d["pattern_id"] for d in dictionary], dtype=np.int32),
        cluster_sizes=np.array([d["cluster_size"] for d in dictionary], dtype=np.int32),
        feature_mean=result["feature_mean"],
        feature_std=result["feature_std"],
    )

    # JSON sidecar
    json_meta = []
    for d in dictionary:
        json_meta.append({
            "pattern_id": d["pattern_id"],
            "cluster_size": d["cluster_size"],
            "dominant_object_type": d["dominant_object_type"],
            "object_type_distribution": d["object_type_distribution"],
            "top_assets": d["top_assets"],
            "height_mean": d["height_mean"],
            "height_std": d["height_std"],
            "mean_layer_presence": d["mean_layer_presence"],
        })

    with open(output_dir / "object_anchored_brush_dictionary.json", "w") as f:
        json.dump({
            "total_instances": result["total_instances"],
            "dictionary_size": result["dictionary_size"],
            "patterns": json_meta,
        }, f, indent=2)

    print(f"Saved {result['dictionary_size']} object-anchored 3D brush patterns to {output_dir}", file=sys.stderr)


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    npz_files = load_npz_files(args.input_dir)

    all_instances: list[dict[str, Any]] = []

    for npz_path in npz_files:
        instances = extract_brush_instances(
            npz_path,
            args.placement_dir,
            args.context_radius,
        )
        all_instances.extend(instances)

        # Subsample if too many
        if len(all_instances) > args.dictionary_size * 100:
            rng.shuffle(all_instances)
            all_instances = all_instances[:args.dictionary_size * 100]

    if len(all_instances) < args.dictionary_size:
        print(f"ERROR: Only {len(all_instances)} instances found, need at least {args.dictionary_size}", file=sys.stderr)
        return 1

    print(f"Total brush instances after extraction: {len(all_instances)}", file=sys.stderr)

    result = cluster_brush_patterns(
        all_instances,
        args.dictionary_size,
        args.min_occurrences,
        args.seed,
    )
    save_dictionary(result, Path(args.output_dir))

    return 0


if __name__ == "__main__":
    sys.exit(main())
