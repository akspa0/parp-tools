#!/usr/bin/env python3
"""
Wave 2: Anchor-Aware MCAL Brush Pattern Mining

Extracts 3D brush patterns by correlating alpha mask compositions with either:
    - object placements
    - terrain-mesh shape alone

This captures both kinds of artist-authored prefab structure:
    - object-anchored relationships between alpha textures and placed assets
    - terrain-anchored relationships where the alpha composition repeats because the
        terrain mesh shape repeats, even with no nearby objects

The script mines these patterns by:
    1. Loading per-tile alpha packs plus optional placement catalogs
    2. Extracting alpha patches around object placements and/or terrain-shape anchors
    3. Clustering patches that share similar alpha, terrain-shape, and anchor features
    4. Building a dictionary of reusable 3D brush patterns

Usage:
    python mine_mcal_brush_patterns.py \
        --input-dir /path/to/npz/tiles \
        --placement-dir /path/to/placement/json \
        --output-dir /path/to/output \
        --context-radius 64 \
        --dictionary-size 128
"""

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


WORLD_TILE_SIZE = 533.33333
WORLD_MAP_ORIGIN = 32.0 * WORLD_TILE_SIZE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine object- and terrain-anchored 3D brush patterns from MCAL alpha, terrain mesh, and optional placement data"
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing Wave 1 .npz tile files")
    parser.add_argument("--placement-dir", help="Directory containing per-tile placement JSON (MDDF/MODF)")
    parser.add_argument("--output-dir", required=True, help="Directory to write brush dictionary and reports")
    parser.add_argument("--context-radius", type=int, default=64, help="Pixel radius around each anchor to extract alpha context")
    parser.add_argument("--dictionary-size", type=int, default=128, help="Target number of 3D brush patterns")
    parser.add_argument("--min-occurrences", type=int, default=3, help="Minimum occurrences for a pattern to be kept")
    parser.add_argument("--anchor-mode", choices=["objects", "terrain", "hybrid"], default="hybrid", help="Which anchor families to mine")
    parser.add_argument("--terrain-samples-per-tile", type=int, default=128, help="Maximum number of terrain-only anchors to keep per tile when terrain or hybrid mode is active")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def load_npz_files(input_dir: str) -> list[Path]:
    root = Path(input_dir)
    files = sorted(root.rglob("*.npz"))
    if not files:
        raise ValueError(f"No .npz files found in {input_dir}")
    print(f"Found {len(files)} .npz files", file=sys.stderr)
    return files


def load_placement_data(placement_dir: str | None, tile_name: str) -> list[dict[str, Any]]:
    """Load MDDF/MODF placement records for a tile."""
    if not placement_dir:
        return []

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


def load_npz_tensor(data: np.lib.npyio.NpzFile, key: str) -> NDArray[Any] | None:
    """Load a tensor from an NPZ file, handling both direct arrays and embedded .npy bytes."""
    if key not in data:
        return None

    value = data[key]
    if isinstance(value, np.ndarray):
        return value

    if isinstance(value, (bytes, bytearray)):
        raw = bytes(value)
        if not raw.startswith((b"\x93NUMPY", b"?NUMPY")):
            return None

        major = raw[6]
        if major == 1:
            header_len = int.from_bytes(raw[8:10], byteorder="little", signed=False)
            header_offset = 10
        elif major in (2, 3):
            header_len = int.from_bytes(raw[8:12], byteorder="little", signed=False)
            header_offset = 12
        else:
            return None

        header_bytes = raw[header_offset:header_offset + header_len]
        header = ast.literal_eval(header_bytes.decode("latin1").strip())
        dtype = np.dtype(header["descr"])
        shape = tuple(header["shape"])
        payload = raw[header_offset + header_len:]
        tensor = np.frombuffer(payload, dtype=dtype)
        return tensor.reshape(shape, order="F" if header.get("fortran_order") else "C")

    return None


def extract_full_tile_alpha(npz_path: Path) -> tuple[NDArray[np.float32], dict[str, Any]] | None:
    """Load the full 256x256x4 alpha pack from a Wave 1 .npz file."""
    data = np.load(npz_path)
    alpha_key = "mcal_alpha_pack_256"
    alpha_tensor = load_npz_tensor(data, alpha_key)
    if alpha_tensor is None:
        return None

    alpha = alpha_tensor.astype(np.float32)
    if alpha.ndim != 3 or alpha.shape[2] != 4:
        return None

    meta = {"tile_name": str(npz_path.stem)}
    layer_mask = load_npz_tensor(data, "mcly_layer_mask")
    if layer_mask is not None:
        meta["layer_mask"] = layer_mask.astype(bool)
    texture_ids = load_npz_tensor(data, "mcly_texture_ids")
    if texture_ids is not None:
        meta["texture_ids"] = texture_ids.astype(np.int32)
    heightmap = load_npz_tensor(data, "height_257")
    if heightmap is not None:
        meta["heightmap"] = heightmap.astype(np.float32)

    return alpha, meta


def world_to_tile_uv(world_x: float, world_y: float, tile_x: int, tile_y: int) -> tuple[float, float]:
    """Convert WoW world coordinates to tile-local UV (0-1)."""
    candidates = [
        ((world_x / WORLD_TILE_SIZE) - tile_x, (world_y / WORLD_TILE_SIZE) - tile_y),
        (((WORLD_MAP_ORIGIN - world_y) / WORLD_TILE_SIZE) - tile_x,
         ((WORLD_MAP_ORIGIN - world_x) / WORLD_TILE_SIZE) - tile_y),
    ]

    best_uv = (float("nan"), float("nan"))
    best_score = float("-inf")
    for u, v in candidates:
        if u < -0.25 or u > 1.25 or v < -0.25 or v > 1.25:
            continue

        score = -(abs(u - 0.5) + abs(v - 0.5))
        if score > best_score:
            best_score = score
            best_uv = (u, v)

    return best_uv


def compute_terrain_stats(
    heightmap: NDArray[np.float32] | None,
    u: float,
    v: float,
    alpha_width: int,
    alpha_height: int,
    radius: int,
) -> tuple[float | None, dict[str, float]]:
    default_stats = {
        "relief": 0.0,
        "slope_mean": 0.0,
        "slope_std": 0.0,
        "curvature_mean": 0.0,
        "curvature_abs_mean": 0.0,
        "roughness": 0.0,
    }
    if heightmap is None:
        return None, default_stats

    height_height, height_width = heightmap.shape[:2]
    hx = min(max(int(round(u * (height_width - 1))), 0), height_width - 1)
    hy = min(max(int(round(v * (height_height - 1))), 0), height_height - 1)

    scale_x = max(1, int(round(radius * ((height_width - 1) / max(1, alpha_width - 1)))))
    scale_y = max(1, int(round(radius * ((height_height - 1) / max(1, alpha_height - 1)))))

    x0 = max(0, hx - scale_x)
    x1 = min(height_width, hx + scale_x + 1)
    y0 = max(0, hy - scale_y)
    y1 = min(height_height, hy + scale_y + 1)
    height_patch = heightmap[y0:y1, x0:x1]
    if height_patch.size == 0:
        return None, default_stats

    gx, gy = np.gradient(height_patch.astype(np.float32))
    slope = np.hypot(gx, gy)
    dxx = np.gradient(gx, axis=0)
    dyy = np.gradient(gy, axis=1)
    curvature = dxx + dyy

    stats = {
        "relief": float(np.max(height_patch) - np.min(height_patch)),
        "slope_mean": float(np.mean(slope)),
        "slope_std": float(np.std(slope)),
        "curvature_mean": float(np.mean(curvature)),
        "curvature_abs_mean": float(np.mean(np.abs(curvature))),
        "roughness": float(np.std(height_patch)),
    }
    return float(heightmap[hy, hx]), stats


def classify_terrain_signature(stats: dict[str, float]) -> str:
    relief = stats.get("relief", 0.0)
    slope_mean = stats.get("slope_mean", 0.0)
    curvature_mean = stats.get("curvature_mean", 0.0)
    roughness = stats.get("roughness", 0.0)

    if relief < 0.75 and slope_mean < 0.08:
        return "flat"
    if curvature_mean > 0.05 and slope_mean > 0.12:
        return "ridge"
    if curvature_mean < -0.05 and slope_mean > 0.12:
        return "basin"
    if slope_mean > 0.22:
        return "slope"
    if roughness > 1.0:
        return "rough"
    return "undulating"


def extract_alpha_context_at_uv(
    alpha_tile: NDArray[np.float32],
    heightmap: NDArray[np.float32] | None,
    u: float,
    v: float,
    radius: int,
) -> dict[str, Any] | None:
    if np.isnan(u) or np.isnan(v):
        return None

    alpha_height, alpha_width = alpha_tile.shape[:2]

    # Convert to alpha-grid pixel coordinates using the stored resolution.
    px = int(u * (alpha_width - 1))
    py = int(v * (alpha_height - 1))

    if px < 0 or px >= alpha_width or py < 0 or py >= alpha_height:
        return None

    # Extract patch bounds
    x0 = max(0, px - radius)
    x1 = min(alpha_width, px + radius)
    y0 = max(0, py - radius)
    y1 = min(alpha_height, py + radius)

    patch = alpha_tile[y0:y1, x0:x1, :]

    # Keep only full-diameter patches so downstream clustering and serialization
    # operate on a consistent tensor shape.
    if patch.shape[0] < (radius * 2) or patch.shape[1] < (radius * 2):
        return None

    height_at_point, terrain_stats = compute_terrain_stats(
        heightmap,
        u,
        v,
        alpha_width,
        alpha_height,
        radius,
    )

    layer_stats = []
    for layer in range(4):
        layer_patch = patch[:, :, layer]
        layer_stats.append({
            "mean": float(np.mean(layer_patch)),
            "std": float(np.std(layer_patch)),
            "max": float(np.max(layer_patch)),
            "entropy": compute_entropy(layer_patch),
        })

    dominant_layer = int(np.argmax([s["mean"] for s in layer_stats]))

    return {
        "patch": patch,
        "center_px": (px, py),
        "height_at_point": height_at_point,
        "layer_stats": layer_stats,
        "dominant_layer": dominant_layer,
        "patch_shape": patch.shape,
        "terrain_stats": terrain_stats,
    }


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
    return extract_alpha_context_at_uv(alpha_tile, heightmap, u, v, radius)


def compute_entropy(patch: NDArray[np.float32], bins: int = 16) -> float:
    """Compute spatial entropy."""
    hist, _ = np.histogram(patch.flatten(), bins=bins, range=(0.0, 1.0))
    probs = hist.astype(np.float32) / (hist.sum() + 1e-8)
    return float(-np.sum(probs * np.log2(probs + 1e-8)))


def compute_alpha_energy(patch: NDArray[np.float32]) -> float:
    blended = np.max(patch, axis=2)
    return float(np.std(blended) + compute_entropy(blended))


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
    placement_dir: str | None,
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
            "anchor_type": "object",
            "terrain_signature": classify_terrain_signature(context["terrain_stats"]),
            "terrain_stats": context["terrain_stats"],
            "world_position": (world_x, world_y, world_z),
            "alpha_context": context["patch"],
            "dominant_layer": context["dominant_layer"],
            "layer_stats": context["layer_stats"],
            "height_at_point": context["height_at_point"],
            "object_scale": placement.get("scale", 1.0),
            "object_rotation": placement.get("rotation", placement.get("orient", 0.0)),
            "sample_score": compute_alpha_energy(context["patch"]),
        })

    return instances


def extract_terrain_brush_instances(
    npz_path: Path,
    context_radius: int,
    terrain_samples_per_tile: int,
) -> list[dict[str, Any]]:
    """Extract terrain-anchored brush instances from alpha plus terrain mesh shape alone."""
    if terrain_samples_per_tile <= 0:
        return []

    result = extract_full_tile_alpha(npz_path)
    if result is None:
        return []

    alpha_tile, meta = result
    heightmap = meta.get("heightmap")
    if heightmap is None:
        return []

    tile_name = meta["tile_name"]
    alpha_height, alpha_width = alpha_tile.shape[:2]
    stride = max(context_radius * 2, 32)
    candidates: list[tuple[float, dict[str, Any]]] = []

    for py in range(context_radius, alpha_height - context_radius, stride):
        for px in range(context_radius, alpha_width - context_radius, stride):
            u = px / float(alpha_width - 1)
            v = py / float(alpha_height - 1)
            context = extract_alpha_context_at_uv(alpha_tile, heightmap, u, v, context_radius)
            if context is None:
                continue

            terrain_stats = context["terrain_stats"]
            layer_presence = [float(stat["mean"]) for stat in context["layer_stats"]]
            alpha_energy = compute_alpha_energy(context["patch"])
            terrain_energy = (
                terrain_stats["relief"]
                + terrain_stats["slope_mean"]
                + terrain_stats["curvature_abs_mean"]
                + terrain_stats["roughness"]
            )
            layer_contrast = float(np.std(np.array(layer_presence, dtype=np.float32)))
            score = alpha_energy + layer_contrast + (0.1 * terrain_energy)
            if score <= 0.05:
                continue

            candidates.append((
                score,
                {
                    "tile_name": tile_name,
                    "asset_path": "__terrain__",
                    "object_type": "terrain",
                    "anchor_type": "terrain",
                    "terrain_signature": classify_terrain_signature(terrain_stats),
                    "terrain_stats": terrain_stats,
                    "world_position": None,
                    "alpha_context": context["patch"],
                    "dominant_layer": context["dominant_layer"],
                    "layer_stats": context["layer_stats"],
                    "height_at_point": context["height_at_point"],
                    "object_scale": 1.0,
                    "object_rotation": 0.0,
                    "sample_score": score,
                },
            ))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return [instance for _, instance in candidates[:terrain_samples_per_tile]]


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
    low_y = min(8, fft_mag.shape[0])
    low_x = min(8, fft_mag.shape[1])
    fft_low = float(np.mean(fft_mag[:low_y, :low_x]))

    high_y = max(1, fft_mag.shape[0] // 2)
    high_x = max(1, fft_mag.shape[1] // 2)
    fft_high = float(np.mean(fft_mag[high_y:, high_x:]))

    # Object type one-hot
    obj_types = ["tree", "rock", "building", "structure", "detail", "wmo", "other", "terrain"]
    type_onehot = np.zeros(len(obj_types), dtype=np.float32)
    obj_type = instance["object_type"]
    if obj_type in obj_types:
        type_onehot[obj_types.index(obj_type)] = 1.0

    anchor_types = ["object", "terrain"]
    anchor_onehot = np.zeros(len(anchor_types), dtype=np.float32)
    anchor_type = instance.get("anchor_type", "object")
    if anchor_type in anchor_types:
        anchor_onehot[anchor_types.index(anchor_type)] = 1.0

    # Height relationship
    height_rel = instance.get("height_at_point", 0.0) or 0.0
    scale = float(instance.get("object_scale", 1.0))
    terrain_stats = instance.get("terrain_stats") or {}
    terrain_features = np.array([
        float(terrain_stats.get("relief", 0.0)),
        float(terrain_stats.get("slope_mean", 0.0)),
        float(terrain_stats.get("slope_std", 0.0)),
        float(terrain_stats.get("curvature_mean", 0.0)),
        float(terrain_stats.get("curvature_abs_mean", 0.0)),
        float(terrain_stats.get("roughness", 0.0)),
        float(instance.get("sample_score", 0.0)),
    ], dtype=np.float32)

    features = np.concatenate([
        np.array([mean_val, std_val, entropy, fft_low, fft_high, height_rel, scale], dtype=np.float32),
        terrain_features,
        layer_presence,
        type_onehot,
        anchor_onehot,
    ])

    return features


def cluster_brush_patterns(
    instances: list[dict[str, Any]],
    dictionary_size: int,
    min_occurrences: int,
    seed: int,
) -> dict[str, Any]:
    """
    Cluster anchor-aware brush instances into 3D brush pattern types.
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
        anchor_counts: dict[str, int] = {}
        asset_counts: dict[str, int] = {}
        terrain_signature_counts: dict[str, int] = {}
        for inst in cluster_instances:
            t = inst["object_type"]
            type_counts[t] = type_counts.get(t, 0) + 1
            anchor = inst.get("anchor_type", "object")
            anchor_counts[anchor] = anchor_counts.get(anchor, 0) + 1
            terrain_signature = inst.get("terrain_signature", "unknown")
            terrain_signature_counts[terrain_signature] = terrain_signature_counts.get(terrain_signature, 0) + 1
            asset = inst["asset_path"]
            if asset != "__terrain__":
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
            "dominant_anchor_type": max(anchor_counts, key=anchor_counts.get),
            "dominant_object_type": max(type_counts, key=type_counts.get),
            "anchor_type_distribution": anchor_counts,
            "object_type_distribution": type_counts,
            "terrain_signature_distribution": terrain_signature_counts,
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
    npz_payload = {
        "stamps": stamps.astype(np.float32),
        "pattern_ids": np.array([d["pattern_id"] for d in dictionary], dtype=np.int32),
        "cluster_sizes": np.array([d["cluster_size"] for d in dictionary], dtype=np.int32),
        "feature_mean": result["feature_mean"],
        "feature_std": result["feature_std"],
    }

    np.savez(output_dir / "brush_dictionary.npz", **npz_payload)
    np.savez(output_dir / "object_anchored_brush_dictionary.npz", **npz_payload)

    # JSON sidecar
    json_meta = []
    for d in dictionary:
        json_meta.append({
            "pattern_id": d["pattern_id"],
            "cluster_size": d["cluster_size"],
            "dominant_anchor_type": d["dominant_anchor_type"],
            "dominant_object_type": d["dominant_object_type"],
            "anchor_type_distribution": d["anchor_type_distribution"],
            "object_type_distribution": d["object_type_distribution"],
            "terrain_signature_distribution": d["terrain_signature_distribution"],
            "top_assets": d["top_assets"],
            "height_mean": d["height_mean"],
            "height_std": d["height_std"],
            "mean_layer_presence": d["mean_layer_presence"],
        })

    json_payload = {
        "total_instances": result["total_instances"],
        "dictionary_size": result["dictionary_size"],
        "patterns": json_meta,
    }
    with open(output_dir / "brush_dictionary.json", "w") as f:
        json.dump(json_payload, f, indent=2)
    with open(output_dir / "object_anchored_brush_dictionary.json", "w") as f:
        json.dump(json_payload, f, indent=2)

    print(f"Saved {result['dictionary_size']} anchor-aware 3D brush patterns to {output_dir}", file=sys.stderr)


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    npz_files = load_npz_files(args.input_dir)

    all_instances: list[dict[str, Any]] = []

    for npz_path in npz_files:
        if args.anchor_mode in {"objects", "hybrid"}:
            all_instances.extend(extract_brush_instances(
                npz_path,
                args.placement_dir,
                args.context_radius,
            ))

        if args.anchor_mode in {"terrain", "hybrid"}:
            all_instances.extend(extract_terrain_brush_instances(
                npz_path,
                args.context_radius,
                args.terrain_samples_per_tile,
            ))

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
