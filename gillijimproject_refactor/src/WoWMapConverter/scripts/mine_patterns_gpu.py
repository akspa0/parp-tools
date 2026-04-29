#!/usr/bin/env python3
"""
GPU-accelerated tileset pattern mining.

Reads PNG tilesets exported by `harvest-tileset-blps`, runs FFT-based
periodicity detection and autocorrelation on GPU via PyTorch, and outputs
a pattern library JSON for downstream MCAL/MCLY reconstruction.

Usage:
    python mine_patterns_gpu.py \
        --manifest /path/to/harvest_manifest.json \
        --output-dir /path/to/output \
        [--device cuda] [--limit 200]
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image
from numpy.typing import NDArray


@dataclass
class PatternStamp:
    texture_name: str
    width: int
    height: int
    tile_size_x: int
    tile_size_y: int
    periodicity_score: float
    dominant_frequencies: list[float] = field(default_factory=list)
    pattern_scale_hint: str = "micro"
    edge_behavior: str = "uniform"
    mean_luminance: float = 0.0
    luminance_stddev: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU-accelerated tileset pattern mining")
    parser.add_argument("--manifest", required=True, help="Path to harvest_manifest.json")
    parser.add_argument("--output-dir", required=True, help="Output directory for pattern_library.json")
    parser.add_argument("--device", default="cuda", help="Torch device (cuda, cpu)")
    parser.add_argument("--limit", type=int, default=0, help="Max textures to process (0 = all)")
    parser.add_argument("--min-periodicity", type=float, default=0.15, help="Minimum periodicity score to keep")
    return parser.parse_args()


def load_image_rgba(path: str) -> tuple[NDArray, int, int]:
    img = Image.open(path).convert("RGBA")
    w, h = img.size
    rgba = np.array(img, dtype=np.float32)
    return rgba, w, h


def rgba_to_grayscale(rgba: NDArray) -> NDArray:
    return 0.299 * rgba[:, :, 0] + 0.587 * rgba[:, :, 1] + 0.114 * rgba[:, :, 2]


def compute_autocorrelation_gpu(gray: torch.Tensor) -> torch.Tensor:
    """Compute 2D autocorrelation via FFT convolution theorem: R = IFFT(|FFT|^2)"""
    gray = gray - gray.mean()
    fft = torch.fft.fft2(gray)
    power = fft.abs() ** 2
    acorr = torch.fft.ifft2(power).real
    acorr = acorr / (acorr[0, 0] + 1e-8)
    acorr = torch.fft.ifftshift(acorr)
    half_h = acorr.shape[0] // 2
    half_w = acorr.shape[1] // 2
    return acorr[:half_h, :half_w]


def detect_tile_size(acorr: torch.Tensor, min_search: int = 4, min_score: float = 0.3) -> tuple[int, int, float]:
    """Detect repeating tile size from autocorrelation peaks."""
    h, w = acorr.shape
    best_x, best_y, best_score = 1, 1, 0.0
    for dy in range(min_search, h):
        for dx in range(min_search, w):
            val = acorr[dy, dx].item()
            if val > min_score:
                score = val * (1.0 - abs(dx - dy) / max(dx, dy, 1))
                if score > best_score:
                    best_score = score
                    best_x = dx
                    best_y = dy
    return best_x, best_y, best_score


def extract_dominant_frequencies(gray_gpu: torch.Tensor, top_k: int = 5) -> list[float]:
    """Extract top-K dominant frequency magnitudes from FFT."""
    fft = torch.fft.fft2(gray_gpu)
    fft_shifted = torch.fft.fftshift(fft)
    magnitude = fft_shifted.abs()
    flat = magnitude.flatten()
    top_vals = flat.topk(top_k + 1).values.cpu().numpy().tolist()
    return top_vals[1:] if len(top_vals) > 1 else top_vals


def classify_pattern_scale(tile_x: int, tile_y: int, periodicity: float) -> str:
    if periodicity < 0.15:
        return "micro"
    mx = max(tile_x, tile_y)
    if mx <= 16:
        return "micro"
    if mx <= 48:
        return "meso"
    return "macro"


def classify_edge_behavior(acorr: torch.Tensor) -> str:
    h, w = acorr.shape
    cx, cy = w // 2, h // 2
    center_vals = acorr[cy - 2:cy + 2, cx - 2:cx + 2].mean().item()
    edge_vals_top = acorr[:2, :].mean().item()
    edge_vals_bot = acorr[-2:, :].mean().item()
    edge_vals_left = acorr[:, :2].mean().item()
    edge_vals_right = acorr[:, -2:].mean().item()
    edge_avg = (edge_vals_top + edge_vals_bot + edge_vals_left + edge_vals_right) / 4.0
    if center_vals > edge_avg * 1.2:
        return "center_highlight"
    if edge_avg > center_vals * 1.2:
        return "edge_darkening"
    return "uniform"


def analyze_texture(rgba: NDArray, name: str, device: torch.device) -> PatternStamp:
    gray = rgba_to_grayscale(rgba)
    gray_t = torch.from_numpy(gray).to(device)

    acorr = compute_autocorrelation_gpu(gray_t)
    tile_x, tile_y, periodicity = detect_tile_size(acorr)
    freqs = extract_dominant_frequencies(gray_t)
    pattern_scale = classify_pattern_scale(tile_x, tile_y, periodicity)
    edge = classify_edge_behavior(acorr)
    mean_lum = float(gray.mean())
    std_lum = float(gray.std())

    return PatternStamp(
        texture_name=name,
        width=rgba.shape[1],
        height=rgba.shape[0],
        tile_size_x=tile_x,
        tile_size_y=tile_y,
        periodicity_score=periodicity,
        dominant_frequencies=freqs,
        pattern_scale_hint=pattern_scale,
        edge_behavior=edge,
        mean_luminance=mean_lum,
        luminance_stddev=std_lum,
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"Error: manifest not found: {manifest_path}")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    entries = manifest.get("entries", [])
    print(f"Loaded {len(entries)} harvest entries")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    limit = args.limit if args.limit > 0 else len(entries)
    min_periodicity = args.min_periodicity
    patterns: list[dict] = []
    processed = 0
    errors = 0

    for entry in entries[:limit]:
        png_path = entry.get("png_path", "")
        name = entry.get("name", "")
        design_kit = entry.get("design_kit", "")
        era_tag = entry.get("era_tag", "")

        if not Path(png_path).exists():
            continue

        try:
            rgba, w, h = load_image_rgba(png_path)
            if w < 16 or h < 16:
                continue

            stamp = analyze_texture(rgba, name, device)

            if stamp.periodicity_score >= min_periodicity:
                patterns.append({
                    "texture_name": stamp.texture_name,
                    "width": stamp.width,
                    "height": stamp.height,
                    "tile_size_x": stamp.tile_size_x,
                    "tile_size_y": stamp.tile_size_y,
                    "periodicity_score": stamp.periodicity_score,
                    "dominant_frequencies": stamp.dominant_frequencies,
                    "pattern_scale_hint": stamp.pattern_scale_hint,
                    "edge_behavior": stamp.edge_behavior,
                    "mean_luminance": stamp.mean_luminance,
                    "luminance_stddev": stamp.luminance_stddev,
                    "design_kit": design_kit,
                    "era_tag": era_tag,
                })

            processed += 1
            if processed % 200 == 0:
                print(f"  Processed {processed}/{limit}...")

        except Exception as exc:
            errors += 1
            if errors <= 5:
                print(f"  Error processing {name}: {exc}")

    patterns.sort(key=lambda p: p["periodicity_score"], reverse=True)

    output = {
        "schema_version": "v10-gpu-patterns.v1",
        "total_processed": processed,
        "total_errors": errors,
        "total_patterns": len(patterns),
        "min_periodicity": min_periodicity,
        "device": str(device),
        "patterns": patterns,
    }

    output_path = output_dir / "pattern_library_gpu.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nDone: {processed} processed, {errors} errors, {len(patterns)} patterns")
    print(f"Written: {output_path}")

    if patterns:
        print("\n=== Top Patterns (by periodicity score) ===")
        for p in patterns[:20]:
            print(f"  {p['design_kit']}/{p['texture_name']}: tile={p['tile_size_x']}x{p['tile_size_y']} scale={p['pattern_scale_hint']} score={p['periodicity_score']:.4f}")


if __name__ == "__main__":
    import sys
    main()
