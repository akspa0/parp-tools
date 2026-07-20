#!/usr/bin/env python3
"""Process ek.jpg: cut into 64x64 tiles, omit white/black tiles, save for inference."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Increase decompression bomb limit for large images
Image.MAX_IMAGE_PIXELS = 300_000_000

SRC = Path(__file__).resolve().parent.parent.parent / "map-references" / "ek.jpg"
OUT_DIR = Path("output/ek_tiles_64")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TILE_SIZE = 64
WHITE_THRESH = 250  # mean pixel value > this = white
BLACK_THRESH = 5    # mean pixel value < this = black

def main():
    print(f"Loading {SRC}...")
    with Image.open(SRC) as img:
        img = img.convert("RGB")
        w, h = img.size
        print(f"Image size: {w}x{h}")
        
        # Calculate grid
        tiles_x = w // TILE_SIZE
        tiles_y = h // TILE_SIZE
        print(f"Grid: {tiles_x} x {tiles_y} = {tiles_x * tiles_y} tiles")
        
        saved = 0
        skipped_white = 0
        skipped_black = 0
        
        for ty in range(tiles_y):
            for tx in range(tiles_x):
                left = tx * TILE_SIZE
                top = ty * TILE_SIZE
                right = left + TILE_SIZE
                bottom = top + TILE_SIZE
                
                tile = img.crop((left, top, right, bottom))
                arr = np.asarray(tile, dtype=np.uint8)
                mean_val = arr.mean()
                
                if mean_val > WHITE_THRESH:
                    skipped_white += 1
                    continue
                if mean_val < BLACK_THRESH:
                    skipped_black += 1
                    continue
                
                # Save tile
                out_path = OUT_DIR / f"ek_{ty:04d}_{tx:04d}.png"
                tile.save(out_path)
                saved += 1
            
            if ty % 50 == 0:
                print(f"  Row {ty}/{tiles_y} - saved {saved}, skipped white {skipped_white}, black {skipped_black}")
        
        print(f"\nDone! Saved {saved} tiles to {OUT_DIR}")
        print(f"Skipped: {skipped_white} white, {skipped_black} black")

if __name__ == "__main__":
    main()