"""Visualize V18 paste candidates overlaid on tile minimaps."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import zarr

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

OUT = Path("../output/v18/pastes/v18_full_corpus_v5")
ZARR_BASE = Path("../output/datasets/v16")

# Palette for drawing bboxes (BGR for OpenCV-style, RGB for PIL)
COLORS = [
    (255, 0, 0),    # red
    (0, 255, 0),    # green
    (0, 0, 255),    # blue
    (255, 255, 0),  # yellow
    (255, 0, 255),  # magenta
    (0, 255, 255),  # cyan
    (255, 128, 0),  # orange
    (128, 0, 255),  # purple
]


def main():
    # Load candidates
    candidates = []
    with open(OUT / "candidates.jsonl") as f:
        for line in f:
            candidates.append(json.loads(line))
    print(f"Loaded {len(candidates)} candidates")

    # Group by build + tile_id
    by_tile: dict[tuple[str, int], list[dict]] = {}
    for c in candidates:
        key = (c["build"], c["tile_id"])
        by_tile.setdefault(key, []).append(c)

    print(f"Candidates span {len(by_tile)} tiles")

    # Group by build for Zarr access
    by_build: dict[str, list] = {}
    for (build, tid), cands in by_tile.items():
        by_build.setdefault(build, []).append((tid, cands))

    vis_dir = OUT / "visualization"
    vis_dir.mkdir(parents=True, exist_ok=True)

    tile_count = 0
    for build, tile_list in sorted(by_build.items()):
        zarr_path = ZARR_BASE / f"{build}.zarr"
        if not zarr_path.exists():
            continue
        store = zarr.storage.LocalStore(str(zarr_path), read_only=True)
        root = zarr.open_group(store=store, mode="r")
        minimaps = root["minimap_rgb"]

        for tid, cands in tile_list:
            if tid >= minimaps.shape[0]:
                continue
            mm = minimaps[tid].astype(np.float32) / 255.0
            # Make an RGB overlay (copy 3 times)
            overlay = np.clip(mm, 0.0, 1.0).copy()
            if overlay.ndim == 2:
                overlay = np.stack([overlay] * 3, axis=-1)
            elif overlay.shape[-1] != 3:
                overlay = overlay[..., :3]

            # Draw bboxes
            for i, c in enumerate(cands):
                x0, y0, x1, y1 = c["tile_local_bbox"]
                color = COLORS[i % len(COLORS)]
                score = c.get("score_mean", 0.0)
                area = c.get("component_area", 0)
                label = f"{score:.2f}a{area}"
                thickness = 2
                # top/bottom edges
                overlay[y0:y0 + thickness, x0:x1 + 1] = np.array(color) / 255.0
                overlay[y1 - thickness + 1:y1 + 1, x0:x1 + 1] = np.array(color) / 255.0
                # left/right edges
                overlay[y0:y1 + 1, x0:x0 + thickness] = np.array(color) / 255.0
                overlay[y0:y1 + 1, x1 - thickness + 1:x1 + 1] = np.array(color) / 255.0

            # Save tile visualization
            tile_name = f"{build}_tid{tid:05d}_{len(cands)}cands.png"
            _save_png(overlay, vis_dir / tile_name)
            tile_count += 1

            # Limit to 50 tiles for a quick check
            if tile_count >= 50:
                break
        if tile_count >= 50:
            break

    print(f"Saved {tile_count} tile visualizations to {vis_dir}")

    # Also make an atlas of candidate crops
    crop_dir = vis_dir / "crops"
    crop_dir.mkdir(exist_ok=True)
    crops_saved = 0
    for build, tile_list in sorted(by_build.items()):
        zarr_path = ZARR_BASE / f"{build}.zarr"
        if not zarr_path.exists():
            continue
        store = zarr.storage.LocalStore(str(zarr_path), read_only=True)
        root = zarr.open_group(store=store, mode="r")
        minimaps = root["minimap_rgb"]
        alphas = root["alpha_256"]
        for tid, cands in tile_list:
            if tid >= minimaps.shape[0]:
                continue
            mm = minimaps[tid].astype(np.float32) / 255.0
            for i, c in enumerate(cands):
                x0, y0, x1, y1 = c["tile_local_bbox"]
                crop = np.clip(mm[y0:y1 + 1, x0:x1 + 1], 0.0, 1.0)
                if crop.ndim == 2:
                    crop = np.stack([crop] * 3, axis=-1)
                fname = f"{build}_tid{tid:05d}_cid{c['candidate_id']:06d}.png"
                _save_png(crop, crop_dir / fname)
                crops_saved += 1
                if crops_saved >= 200:
                    break
            if crops_saved >= 200:
                break
        if crops_saved >= 200:
            break
    print(f"Saved {crops_saved} crop visualizations to {crop_dir}")

    # Generate a simple HTML report
    html_parts = ["<html><body><h1>V18 Paste Candidates</h1>"]
    html_parts.append(f"<p>{len(candidates)} candidates, {len(by_tile)} tiles</p>")
    html_parts.append("<h2>Sample tile overlays</h2>")
    for png in sorted(vis_dir.glob("*_*_*cands.png")):
        html_parts.append(f'<img src="{png.name}" width="512"><br>')
    html_parts.append("<h2>Sample crops</h2>")
    for png in sorted(crop_dir.glob("*.png"))[:50]:
        html_parts.append(f'<img src="crops/{png.name}" width="128">')
    html_parts.append("</body></html>")
    (vis_dir / "report.html").write_text("\n".join(html_parts), encoding="utf-8")
    print(f"Report: {vis_dir / 'report.html'}")


def _save_png(rgb: np.ndarray, path: Path):
    from PIL import Image
    img = Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8), mode="RGB")
    img.save(path)


if __name__ == "__main__":
    main()
