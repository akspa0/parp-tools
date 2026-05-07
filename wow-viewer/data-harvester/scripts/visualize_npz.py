"""
Visualization and verification tool for wow-viewer NPZ tile shards.

Usage:
    uv run python scripts/visualize_npz.py <npz_file_or_dir> [--output-dir <dir>] [--quilt]

    Single tile: renders heightmap + minimap + normals + shadow + liquid
    Directory:   renders each .npz file, optionally quilts all tiles into a map overview
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize wow-viewer NPZ tile shards")
    parser.add_argument("input", help="NPZ file or directory of NPZ files")
    parser.add_argument("--output-dir", "-o", default=None, help="Output directory for PNGs")
    parser.add_argument("--quilt", action="store_true", help="Stitch tiles into a full-map quilt")
    parser.add_argument("--no-individual", action="store_true", help="Skip per-tile renders")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output_dir) if args.output_dir else Path("visualized")
    out_dir.mkdir(parents=True, exist_ok=True)

    if in_path.is_file():
        npz_files = [in_path]
    elif in_path.is_dir():
        npz_files = sorted(in_path.glob("*.npz"))
    else:
        print(f"Error: '{in_path}' is not a file or directory")
        sys.exit(1)

    if not npz_files:
        print(f"No NPZ files found in '{in_path}'")
        sys.exit(1)

    print(f"Processing {len(npz_files)} NPZ files...")

    tiles = []
    for npz_path in npz_files:
        try:
            tile_data = read_npz(npz_path)
            tiles.append(tile_data)
            if not args.no_individual:
                render_tile(tile_data, out_dir)
        except Exception as e:
            print(f"  SKIP {npz_path.name}: {e}")

    if args.quilt and tiles:
        render_quilt(tiles, out_dir)

    print(f"Done. Output in: {out_dir}")


# ---------------------------------------------------------------------------
# NPZ reading
# ---------------------------------------------------------------------------

def read_npz(path: Path) -> dict:
    """Load an NPZ and normalize arrays to a common dictionary."""
    data = np.load(path)
    entry = {"path": path}

    key_map = {
        "height_257": "height",
        "mcnr_normal_xyz": "normals",
        "mcsh_shadow_mask_256": "shadow",
        "mclq_surface_height": "liquid_height",
        "mclq_type_mask": "liquid_type",
        "mcal_alpha_pack_256": "alpha",
        "mcly_texture_ids": "texture_ids",
        "mcly_layer_mask": "layer_mask",
        "hole_mask_16": "holes",
        "minimap_rgb_256": "minimap",
    }

    for npz_key, entry_key in key_map.items():
        if npz_key in data:
            entry[entry_key] = data[npz_key]

    # Parse tile coordinates from filename (e.g., kalimdor_32_32.npz)
    stem = path.stem
    parts = stem.split("_")
    try:
        nums = [int(p) for p in parts if p.lstrip("-").isdigit()]
        if len(nums) >= 2:
            entry["tile_x"] = nums[-2]
            entry["tile_y"] = nums[-1]
        entry["map_name"] = "_".join(parts[:-2]) if len(nums) >= 2 else stem
    except (ValueError, IndexError):
        entry["map_name"] = stem
        entry["tile_x"] = 0
        entry["tile_y"] = 0

    return entry


# ---------------------------------------------------------------------------
# Per-tile rendering
# ---------------------------------------------------------------------------

def render_tile(tile: dict, out_dir: Path):
    """Render all available signals for a single tile."""
    stem = tile["path"].stem

    plots = []
    if "height" in tile:
        plots.append(("height", tile["height"], "viridis", "Height (Z)"))
    if "minimap" in tile:
        plots.append(("minimap", tile["minimap"], None, "Minimap RGB"))
    if "normals" in tile:
        plots.append(("normals", tile["normals"], None, "Normals XYZ"))
    if "shadow" in tile:
        plots.append(("shadow", tile["shadow"], "gray", "Shadow Mask"))
    if "liquid_height" in tile:
        plots.append(("liquid", tile["liquid_height"], "Blues", "Liquid Height"))
    if "alpha" in tile:
        plots.append(("alpha", tile["alpha"], "gray", "Alpha Pack"))
    if "holes" in tile:
        plots.append(("holes", np.asarray(tile["holes"], dtype=float), "gray_r", "Hole Mask"))

    n = len(plots)
    if n == 0:
        return

    cols = min(3, n)
    rows = math.ceil(n / cols)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"  WARN: matplotlib not available, skipping renders for {stem}")
        return

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 4.5), squeeze=False)

    for idx, (name, arr, cmap, title) in enumerate(plots):
        ax = axes[idx // cols][idx % cols]
        _render_signal(ax, name, arr, cmap, title)

    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].axis("off")

    fig.suptitle(stem, fontsize=10)
    plt.tight_layout()
    out_path = out_dir / f"{stem}.png"
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path.name}")


def _render_signal(ax, name: str, arr: np.ndarray, cmap, title: str):
    """Render a single signal onto an axis."""
    if name == "minimap":
        ax.imshow(arr)
    elif name == "normals":
        # Map XYZ to RGB: R=X (right), G=Y (up), B=Z (forward)
        rgb = (arr * 0.5 + 0.5).clip(0, 1)
        ax.imshow(rgb)
    elif name == "alpha":
        # Show first 3 channels as RGB
        if arr.ndim == 3 and arr.shape[2] >= 3:
            ax.imshow(arr[:, :, :3])
        else:
            ax.imshow(arr[:, :, 0] if arr.ndim == 3 else arr, cmap=cmap or "gray")
    elif name == "texture_ids":
        ax.imshow(arr[:, :, 0] if arr.ndim == 3 else arr, cmap="tab20")
    elif name == "layer_mask":
        mask = arr.astype(float) if arr.ndim == 2 else arr[:, :, 0].astype(float)
        ax.imshow(mask, cmap="gray")
    elif name == "liquid_type":
        ax.imshow(arr.astype(float) if arr.ndim == 2 else arr[:, :, 0].astype(float), cmap="tab10")
    else:
        ax.imshow(arr, cmap=cmap or "viridis")

    ax.set_title(title, fontsize=8)
    ax.axis("off")


# ---------------------------------------------------------------------------
# Map quilt
# ---------------------------------------------------------------------------

def render_quilt(tiles: list, out_dir: Path):
    """Stitch all tiles into a full-map quilt."""
    valid = [t for t in tiles if "tile_x" in t and "tile_y" in t]
    if not valid:
        print("  No tiles with coordinate info; skipping quilt")
        return

    min_x = min(t["tile_x"] for t in valid)
    max_x = max(t["tile_x"] for t in valid)
    min_y = min(t["tile_y"] for t in valid)
    max_y = max(t["tile_y"] for t in valid)

    tile_size = 256

    quilt_signals = ["height", "shadow", "liquid_height"]
    map_name = valid[0].get("map_name", "quilt")

    for signal in quilt_signals:
        quilt = np.full(((max_y - min_y + 1) * tile_size,
                         (max_x - min_x + 1) * tile_size), np.nan, dtype=np.float32)

        for t in valid:
            if signal not in t:
                continue
            tx = t["tile_x"] - min_x
            ty = t["tile_y"] - min_y
            arr = t[signal]
            # Resize to tile_size if needed
            if arr.shape[0] != tile_size or arr.shape[1] != tile_size:
                # Use simple downsampling
                fx = arr.shape[1] / tile_size
                fy = arr.shape[0] / tile_size
                arr_resized = np.zeros((tile_size, tile_size), dtype=np.float32)
                for y in range(tile_size):
                    for x in range(tile_size):
                        sx = int(x * fx)
                        sy = int(y * fy)
                        arr_resized[y, x] = arr[sy, sx]
                arr = arr_resized

            y0 = ty * tile_size
            y1 = y0 + tile_size
            x0 = tx * tile_size
            x1 = x0 + tile_size

            h = min(arr.shape[0], tile_size)
            w = min(arr.shape[1], tile_size)
            quilt[y0:y0 + h, x0:x0 + w] = arr[:h, :w] if arr.ndim == 2 else arr[:h, :w, 0]

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print(f"  WARN: matplotlib not available, skipping quilt for {signal}")
            return

        fig, ax = plt.subplots(figsize=(16, 16))
        valid_mask = ~np.isnan(quilt)
        vmin = np.nanmin(quilt) if valid_mask.any() else 0
        vmax = np.nanmax(quilt) if valid_mask.any() else 1
        masked = np.ma.masked_where(np.isnan(quilt), quilt)
        cmap = "Blues" if signal == "liquid_height" else "viridis" if signal == "height" else "gray"
        ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
        ax.set_title(f"{map_name} — {signal}", fontsize=14)
        ax.axis("off")

        out_path = out_dir / f"{map_name}_quilt_{signal}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Quilt: {out_path.name}")

    # Also quilt minimap if available
    quilt_signals_rgb = ["minimap"]
    for signal in quilt_signals_rgb:
        quilt = np.zeros(((max_y - min_y + 1) * tile_size,
                          (max_x - min_x + 1) * tile_size, 3), dtype=np.uint8)

        has_any = False
        for t in valid:
            if signal not in t:
                continue
            has_any = True
            tx = t["tile_x"] - min_x
            ty = t["tile_y"] - min_y
            arr = t[signal]

            if arr.shape[0] != tile_size or arr.shape[1] != tile_size:
                arr_rgb = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                fx = arr.shape[1] / tile_size
                fy = arr.shape[0] / tile_size
                for y in range(tile_size):
                    for x in range(tile_size):
                        sx = int(x * fx)
                        sy = int(y * fy)
                        arr_rgb[y, x] = arr[sy, sx]
                arr = arr_rgb

            y0 = ty * tile_size
            x0 = tx * tile_size
            y1 = min(y0 + tile_size, quilt.shape[0])
            x1 = min(x0 + tile_size, quilt.shape[1])
            h = y1 - y0
            w = x1 - x0
            quilt[y0:y1, x0:x1] = arr[:h, :w]

        if not has_any:
            continue

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return

        fig, ax = plt.subplots(figsize=(16, 16))
        ax.imshow(quilt, origin="upper")
        ax.set_title(f"{map_name} — {signal}", fontsize=14)
        ax.axis("off")
        out_path = out_dir / f"{map_name}_quilt_{signal}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Quilt: {out_path.name}")


if __name__ == "__main__":
    main()
