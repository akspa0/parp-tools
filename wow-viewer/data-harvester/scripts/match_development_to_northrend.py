"""Cross-Map Visual Similarity Matcher: Development Prototypes -> Northrend (Spec 265).

Matches 2,252 development minimap tiles against authentic 3.0.1 Northrend minimaps
using normalized multi-scale perceptual feature embeddings to discover prototype lineages
and transfer authentic ground-truth WDL lattices and terrain elevations.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


def load_and_embed_minimaps(
    image_paths: List[Path],
    thumb_size: int = 32,
) -> Tuple[np.ndarray, List[str]]:
    """Load and compute normalized multi-scale color/gradient embeddings for minimaps."""
    features = []
    stems = []

    for idx, p in enumerate(image_paths):
        try:
            with Image.open(p) as img:
                img_rgb = img.convert("RGB")
                # 32x32 color thumbnail
                thumb = img_rgb.resize((thumb_size, thumb_size), Image.Resampling.BILINEAR)
                arr = np.array(thumb, dtype=np.float32) / 255.0

                # Color channels + Sobel spatial gradients
                gray = np.mean(arr, axis=-1)
                gy, gx = np.gradient(gray)

                feat = np.concatenate([arr.flatten(), gx.flatten(), gy.flatten()])
                norm = np.linalg.norm(feat)
                if norm > 1e-4:
                    feat /= norm
                else:
                    feat = np.zeros_like(feat)

                features.append(feat)
                stems.append(p.stem)
        except Exception as e:
            print(f"[WARN] Error reading {p.name}: {e}")

    return np.array(features, dtype=np.float32), stems


def main() -> int:
    t0 = time.time()
    dev_dir = Path("../test_data/original_development/World/Textures/Minimap")
    if not dev_dir.is_dir():
        dev_dir = Path("test_data/original_development/World/Textures/Minimap")

    nr_dir = Path("../output/northrend_minimaps_301")
    if not nr_dir.is_dir():
        nr_dir = Path("output/northrend_minimaps_301")

    dev_files = sorted(dev_dir.glob("development_*.png"))
    nr_files = sorted(nr_dir.glob("Northrend_*.png"))

    print("=================================================================")
    print("Cross-Map Visual Similarity Matcher (Development -> Northrend 3.0.1)")
    print(f"Development Minimap Tiles: {len(dev_files)}")
    print(f"Northrend Minimap Tiles:   {len(nr_files)}")
    print("=================================================================")

    if not dev_files or not nr_files:
        print("[ERROR] Missing minimap files.")
        return 1

    print("Extracting feature embeddings for Development tiles...")
    dev_feats, dev_stems = load_and_embed_minimaps(dev_files)

    print("Extracting feature embeddings for Northrend tiles...")
    nr_feats, nr_stems = load_and_embed_minimaps(nr_files)

    print("Computing cross-map correlation matrix...")
    sim_matrix = np.dot(dev_feats, nr_feats.T)  # Shape (N_dev, N_nr)

    # Load Northrend WDL lattice data
    wdl_path = Path("../output/northrend_wdl_301.npz")
    if not wdl_path.is_file():
        wdl_path = Path("output/northrend_wdl_301.npz")
    wdl_data = np.load(wdl_path) if wdl_path.is_file() else None

    nr_wdl_map: Dict[Tuple[int, int], np.ndarray] = {}
    if wdl_data is not None:
        t_xy = wdl_data["tile_xy"]
        outer = wdl_data["outer"]
        for i in range(len(t_xy)):
            nr_wdl_map[(int(t_xy[i, 0]), int(t_xy[i, 1]))] = outer[i]

    matches = []
    threshold = 0.82  # Visual similarity threshold

    print(f"\nEvaluating visual matches (similarity >= {threshold:.2f})...")
    for i, dev_stem in enumerate(dev_stems):
        best_j = int(np.argmax(sim_matrix[i]))
        score = float(sim_matrix[i, best_j])
        nr_stem = nr_stems[best_j]

        if score >= threshold:
            # Parse dev (x, y)
            d_parts = dev_stem.split("_")
            dx, dy = int(d_parts[1]), int(d_parts[2])

            # Parse nr (x, y)
            n_parts = nr_stem.split("_")
            nx, ny = int(n_parts[1]), int(n_parts[2])

            has_wdl = (nx, ny) in nr_wdl_map
            wdl_span = float(np.ptp(nr_wdl_map[(nx, ny)])) if has_wdl else 0.0

            matches.append({
                "development_tile": dev_stem,
                "dev_x": dx,
                "dev_y": dy,
                "northrend_tile": nr_stem,
                "nr_x": nx,
                "nr_y": ny,
                "similarity": score,
                "has_northrend_wdl": has_wdl,
                "wdl_span_yards": wdl_span,
            })

    # Sort matches by similarity score descending
    matches.sort(key=lambda m: m["similarity"], reverse=True)

    print(f"Discovered {len(matches)} strong matches between Development and Northrend!")
    print("\nTop 20 Discovered Prototypes:")
    print(f"{'Development':<22} | {'Northrend':<20} | {'Sim':<6} | {'WDL Span (yds)':<15}")
    print("-" * 70)
    for m in matches[:20]:
        print(f"{m['development_tile']:<22} | {m['northrend_tile']:<20} | {m['similarity']:.4f} | {m['wdl_span_yards']:.1f}")

    # Check development_16_33 specifically
    for i, dev_stem in enumerate(dev_stems):
        if dev_stem == "development_16_33":
            best_j = int(np.argmax(sim_matrix[i]))
            score = float(sim_matrix[i, best_j])
            nr_stem = nr_stems[best_j]
            print(f"\nSpecific Check: development_16_33 best match is {nr_stem} with score {score:.4f}")

    # Save matching ledger to JSON
    out_json = Path("../output/development_to_northrend_matches.json")
    if not out_json.parent.is_dir():
        out_json = Path("output/development_to_northrend_matches.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(matches, indent=2))
    print(f"\nSaved full lineage matches to: {out_json}")
    print(f"Elapsed: {time.time() - t0:.2f}s")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
