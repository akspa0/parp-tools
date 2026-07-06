"""FR-009: minimap cleaning CLI (User Story 2).

Reads a minimap NPZ (key ``minimap_rgb``) and an object-mask NPZ (key
``object_precise_mask``), writes the cleaned 256x256x3 float32 RGB NPZ.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from harvester.v24.clean_minimap import clean_minimap  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minimap", required=True)
    parser.add_argument("--object-mask", required=True)
    parser.add_argument("--no-object-minimap", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with np.load(args.minimap) as data:
        key = "minimap_rgb" if "minimap_rgb" in data else "minimap"
        minimap = np.asarray(data[key])

    with np.load(args.object_mask) as data:
        key = "object_precise_mask" if "object_precise_mask" in data else "mask"
        mask = np.asarray(data[key])

    rendered = None
    if args.no_object_minimap:
        with np.load(args.no_object_minimap) as data:
            key = "no_object_minimap" if "no_object_minimap" in data else "minimap"
            rendered = np.asarray(data[key])

    cleaned, meta = clean_minimap(minimap, mask, rendered)
    np.savez(
        args.output,
        cleaned_minimap=cleaned,
        cleaned_minimap_unavailable=np.array(meta["cleaned_minimap_unavailable"]),
    )
    print(f"wrote {args.output} (source={meta['source']}, unavailable={meta['cleaned_minimap_unavailable']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
