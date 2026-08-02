"""Measure the authored minimap's water colour so the liquid palette can be calibrated.

`MinimapLiquidPalette.PreAlpha053` was set by eye off a comparison screenshot. This reads the real
number off the authored tiles that `synthetic-minimap --authored-reference` writes.

Usage:
    python wow-viewer/tools/scripts/measure_authored_water.py <dir-with-*_authored.png>

Water is isolated as the blue-dominant pixels (B clearly above R), then reported as the median so a
few shoreline/foam pixels cannot drag the estimate. Feed the result straight back in as
`--water-color r,g,b`.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image


def measure(path: Path) -> np.ndarray | None:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    # Water in these minimaps is blue-dominant and not near-black; terrain is red/green dominant.
    water = (blue > red + 0.12) & (blue > 0.15)
    if water.sum() < 32:
        return None
    return np.median(rgb[water], axis=0)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    root = Path(sys.argv[1])
    tiles = sorted(root.rglob("*_authored.png")) or sorted(root.rglob("*.png"))
    if not tiles:
        print(f"No PNGs found under {root}")
        return 1

    samples = []
    for tile in tiles:
        result = measure(tile)
        if result is None:
            continue
        samples.append(result)
        print(f"  {tile.name}: {result[0]:.3f},{result[1]:.3f},{result[2]:.3f}")

    if not samples:
        print("No tile contained enough water pixels to measure.")
        return 1

    overall = np.median(np.stack(samples), axis=0)
    print(f"\n{len(samples)} tile(s) with water.")
    print(f"Median authored water: {overall[0]:.3f},{overall[1]:.3f},{overall[2]:.3f}")
    print(f"\n  --water-color {overall[0]:.2f},{overall[1]:.2f},{overall[2]:.2f}")
    print(
        "\nNote: this is the COMPOSITED colour (water over terrain). If the synthesized water still "
        "reads too dark after using it, raise opacity via --water-color r,g,b,<opacity>."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
