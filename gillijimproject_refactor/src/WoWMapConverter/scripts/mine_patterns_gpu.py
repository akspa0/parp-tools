#!/usr/bin/env python3
"""
GPU-accelerated tileset pattern mining with WoWEdit-style brush previews.

Usage:
    I:/parp/parp-tools/gillijimproject_refactor/.venv-train/Scripts/python.exe \
        mine_patterns_gpu.py \
        --manifest i:/parp/parp-tools/output/ml-training/v10_tileset_pngs/harvest_manifest.json \
        --output-dir i:/parp/parp-tools/output/ml-training/v10_tileset_patterns \
        --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
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
    pattern_scale_hint: str = "noise"
    edge_behavior: str = "uniform"
    mean_luminance: float = 0.0
    luminance_stddev: float = 0.0
    design_kit: str = ""
    era_tag: str = ""
    brush_path: str = ""
    stamp_path: str = ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GPU tileset pattern miner with brush previews")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--min-periodicity", type=float, default=0.15)
    p.add_argument("--collage-cols", type=int, default=6)
    p.add_argument("--collage-rows", type=int, default=10)
    p.add_argument("--brush-size", type=int, default=128)
    p.add_argument("--stamp-size", type=int, default=128)
    p.add_argument("--no-previews", action="store_true")
    return p.parse_args()


def load_rgba(path: str) -> tuple[NDArray, int, int]:
    img = Image.open(path).convert("RGBA")
    return np.array(img, dtype=np.float32), img.size[0], img.size[1]


def to_gray(rgba: NDArray) -> NDArray:
    return 0.299 * rgba[:, :, 0] + 0.587 * rgba[:, :, 1] + 0.114 * rgba[:, :, 2]


def autocorr_gpu(gray_t: torch.Tensor) -> torch.Tensor:
    gray_t = gray_t - gray_t.mean()
    f = torch.fft.fft2(gray_t)
    pwr = f.abs() ** 2
    a = torch.fft.ifft2(pwr).real
    a = a / (a[0, 0] + 1e-8)
    a = torch.fft.ifftshift(a)
    return a[:a.shape[0] // 2, :a.shape[1] // 2]


def detect_tile(ac: torch.Tensor, min_srch: int = 4) -> tuple[int, int, float]:
    h, w = ac.shape
    bx, by, bs = 1, 1, 0.0
    for dy in range(min_srch, h):
        for dx in range(min_srch, w):
            v = ac[dy, dx].item()
            if v > 0.25:
                pn = 1.0 / (1.0 + 0.003 * max(dx, dy))
                s = v * (1.0 - abs(dx - dy) / max(dx, dy, 1)) * pn
                if s > bs:
                    bs, bx, by = s, dx, dy
    return bx, by, bs


def top_freqs(gray_t: torch.Tensor, k: int = 5) -> list[float]:
    f = torch.fft.fftshift(torch.fft.fft2(gray_t))
    return f.abs().flatten().topk(k + 1).values.cpu().tolist()[1:]


def classify_scale(tx: int, ty: int, per: float) -> str:
    if per < 0.15: return "noise"
    m = max(tx, ty)
    if m <= 16: return "micro"
    if m <= 48: return "meso"
    if m <= 128: return "macro"
    return "tilable"


def classify_edge(ac: torch.Tensor) -> str:
    h, w = ac.shape
    if h < 4 or w < 4: return "uniform"
    c = ac[h // 4:h // 4 + h // 2, w // 4:w // 4 + w // 2].mean().item()
    e = float(ac[:2, :].mean().item() + ac[-2:, :].mean().item() + ac[:, :2].mean().item() + ac[:, -2:].mean().item()) / 4
    if c > e * 1.2: return "center_highlight"
    if e > c * 1.2: return "edge_darkening"
    return "uniform"


def render_brush(rgba: NDArray, tx: int, ty: int, out_size: int) -> Image.Image:
    """Render a WoWEdit-style grayscale 3D brush thumbnail from stamp."""
    if tx <= 4 or ty <= 4:
        return None

    stamp = rgba[:min(ty, rgba.shape[0]), :min(tx, rgba.shape[1])].copy()
    gray = to_gray(stamp)
    gmin, gmax = gray.min(), gray.max()
    if gmax > gmin:
        gray = (gray - gmin) / (gmax - gmin)

    # Bump-map shading (light from upper-left, Photoshop 2003 era style)
    h, w = gray.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            dx = gray[y, x + 1] - gray[y, x - 1]
            dy = gray[y + 1, x] - gray[y - 1, x]
            shade = max(0.45, 1.0 - abs(dx) * 0.6 - abs(dy) * 0.4)
            b = int(np.clip(96 + 159 * (gray[y, x] - 0.5) * 2 * shade, 20, 235))
            out[y, x] = [b, b + 3, b + 3]

    img = Image.fromarray(out)
    img = img.resize((out_size, out_size), Image.LANCZOS)

    # Card: dark frame, 2003-era brush panel
    card = Image.new("RGB", (out_size + 20, out_size + 20), (24, 24, 28))
    card.paste((48, 48, 52), [0, 0, card.width, 1])
    card.paste((48, 48, 52), [0, 0, 1, card.height])
    card.paste(img, (10, 10))
    return card


def build_collage(stamps: list[PatternStamp], out_dir: Path, cols: int, rows: int, brush_sz: int) -> None:
    """Grid collage of top brush patterns for Discord posting."""
    top = [s for s in stamps[:cols * rows] if s.brush_path and os.path.exists(s.brush_path)]
    if not top:
        return

    card = brush_sz + 20
    cw = cols * card
    ch = rows * card + 28
    canvas = Image.new("RGB", (cw, ch), (16, 16, 20))
    try:
        font = ImageFont.truetype("consola.ttf", 10)
    except Exception:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(canvas)

    draw.rectangle([0, 0, cw, 22], fill=(28, 28, 34))
    draw.text((6, 4), "WoWEdit Brush Panel  —  tileset pattern library v2",
              fill=(180, 180, 180), font=font)

    for idx, s in enumerate(top):
        r, c = idx // cols, idx % cols
        x, y = c * card, r * card + 24
        try:
            b = Image.open(s.brush_path).convert("RGB")
            canvas.paste(b, (x, y))
        except Exception:
            canvas.paste(Image.new("RGB", (card, card), (32, 32, 36)), (x, y))
        label = s.texture_name[:16]
        draw.text((x + 2, y + card - 15), label, fill=(140, 140, 140), font=font)
        info = f"t{s.tile_size_x}x{s.tile_size_y} p{s.periodicity_score:.0%}"
        draw.text((x + 2, y + card - 27), info, fill=(100, 100, 100), font=font)

    p = out_dir / "brush_panel_collage.png"
    canvas.save(p)
    print(f"Collage: {p}")


def analyze(rgba: NDArray, name: str, kit: str, era: str, dev: torch.device,
            out_dir: Path, brush_sz: int, stamp_sz: int, no_prev: bool) -> PatternStamp:
    gray = to_gray(rgba)
    gt = torch.from_numpy(gray).to(dev)

    ac = autocorr_gpu(gt)
    tx, ty, per = detect_tile(ac)
    fr = top_freqs(gt)
    sc = classify_scale(tx, ty, per)
    ed = classify_edge(ac)

    s = PatternStamp(texture_name=name, width=rgba.shape[1], height=rgba.shape[0],
                     tile_size_x=tx, tile_size_y=ty, periodicity_score=per,
                     dominant_frequencies=fr, pattern_scale_hint=sc, edge_behavior=ed,
                     mean_luminance=float(gray.mean()), luminance_stddev=float(gray.std()),
                     design_kit=kit, era_tag=era)

    if not no_prev and per >= 0.15 and tx > 4 and ty > 4:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)

        # Save raw stamp
        sp = out_dir / "stamps" / f"{safe}.png"
        sp.parent.mkdir(parents=True, exist_ok=True)
        stamp_arr = rgba[:min(ty, rgba.shape[0]), :min(tx, rgba.shape[1])].astype(np.uint8)
        Image.fromarray(stamp_arr).resize((stamp_sz, stamp_sz), Image.LANCZOS).save(sp)
        s.stamp_path = str(sp)

        # Render brush preview
        bp = out_dir / "brushes" / f"{safe}.png"
        bp.parent.mkdir(parents=True, exist_ok=True)
        brush = render_brush(rgba, tx, ty, brush_sz)
        if brush:
            brush.save(bp)
            s.brush_path = str(bp)

    return s


def main() -> None:
    args = parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}")

    with open(args.manifest) as f:
        mf = json.load(f)
    entries = mf.get("entries", [])
    print(f"Loaded {len(entries)} entries")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    limit = args.limit if args.limit > 0 else len(entries)
    patterns: list[PatternStamp] = []
    done, errs = 0, 0

    for e in entries[:limit]:
        pp = e.get("png_path", "")
        if not pp or not os.path.exists(pp):
            continue
        try:
            rgba, w, h = load_rgba(pp)
            if w < 16 or h < 16:
                continue
            s = analyze(rgba, e.get("name", ""), e.get("design_kit", ""),
                       e.get("era_tag", ""), dev, out, args.brush_size,
                       args.stamp_size, args.no_previews)
            if s.periodicity_score >= args.min_periodicity:
                patterns.append(s)
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{limit}...")
        except Exception as ex:
            errs += 1
            if errs <= 5:
                print(f"  Err {e.get('name', '?')}: {ex}")

    patterns.sort(key=lambda x: x.periodicity_score, reverse=True)

    # Top 30
    print(f"\n=== Top 30 ===")
    for pp in patterns[:30]:
        b = " [B]" if pp.brush_path else ""
        print(f"  {pp.design_kit}/{pp.texture_name}: "
              f"t{pp.tile_size_x}x{pp.tile_size_y} s{pp.periodicity_score:.4f} "
              f"{pp.pattern_scale_hint} {pp.edge_behavior}{b}")

    # JSON
    out_json = []
    for pp in patterns:
        out_json.append({
            "texture_name": pp.texture_name, "design_kit": pp.design_kit,
            "era_tag": pp.era_tag, "width": pp.width, "height": pp.height,
            "tile_size_x": pp.tile_size_x, "tile_size_y": pp.tile_size_y,
            "periodicity_score": pp.periodicity_score,
            "dominant_frequencies": pp.dominant_frequencies,
            "pattern_scale_hint": pp.pattern_scale_hint,
            "edge_behavior": pp.edge_behavior,
            "mean_luminance": pp.mean_luminance,
            "luminance_stddev": pp.luminance_stddev,
            "brush_path": pp.brush_path, "stamp_path": pp.stamp_path,
        })

    jp = out / "pattern_library_gpu.json"
    with open(jp, "w") as f:
        json.dump({"schema_version": "v10-gpu-patterns.v2",
                   "total_processed": done, "total_errors": errs,
                   "total_patterns": len(patterns),
                   "min_periodicity": args.min_periodicity,
                   "device": str(dev), "patterns": out_json}, f, indent=2)

    # Collage
    if not args.no_previews:
        build_collage(patterns, out, args.collage_cols, args.collage_rows, args.brush_size)

    print(f"\nDone: {done} done, {errs} errs, {len(patterns)} patterns saved to {jp}")


if __name__ == "__main__":
    main()
