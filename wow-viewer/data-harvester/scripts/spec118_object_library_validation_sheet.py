"""Render a labeled contact sheet from a Spec 118 object-mask library zarr store.

A visual "is the capture actually right?" check for build_object_library.py's
--from-harvest-stream output: samples objects from assets.parquet/capture_rgb/capture_mask,
stratified across asset_type (wmo vs mdx/m2) so both are represented, and draws each as an
[textured image | mask] pair with its asset path and capture status. Sampling favors entries
with real mask coverage (not blank captures) as a tiebreaker, but does not filter by success --
a failed/blank capture shows up as a blank panel, on purpose, so failures are visible here rather
than silently excluded.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np


def _mask_coverage(mask_row: np.ndarray) -> float:
    return float((np.asarray(mask_row) > 0).mean())


def _sample_indices(rows: list[dict], per_type: int, seed: int) -> list[int]:
    """Up to `per_type` indices per asset_type, weighted toward higher mask coverage
    (via reservoir-free top-K-of-random-subset) so the sheet isn't dominated by empty captures,
    while still being a genuine sample, not a cherry-picked best-case."""
    rng = random.Random(seed)
    by_type: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        by_type.setdefault(str(row.get("asset_type", "unknown")), []).append(i)

    selected: list[int] = []
    for _asset_type, indices in sorted(by_type.items()):
        pool = indices[:]
        rng.shuffle(pool)
        selected.extend(pool[: per_type])
    return selected


def _composite(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """[image | mask-as-rgb] side by side, with a thin white separator."""
    h, w = image_rgb.shape[:2]
    mask_rgb = np.stack([mask, mask, mask], axis=-1).astype(np.uint8)
    sep = np.full((h, max(1, w // 32), 3), 255, dtype=np.uint8)
    return np.concatenate([image_rgb, sep, mask_rgb], axis=1)


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pyarrow.parquet as pq
    import zarr

    ap = argparse.ArgumentParser(description="Render a Spec 118 object-mask library validation contact sheet")
    ap.add_argument("--store", required=True, type=Path, help="path to the <run-name>.zarr directory")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--per-type", type=int, default=16, help="objects sampled per asset_type (default 16)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    group = zarr.open_group(str(args.store), mode="r")
    rows = pq.read_table(args.store / "assets.parquet").to_pylist()
    if "capture_rgb" not in group or "capture_mask" not in group:
        raise SystemExit(f"store has no capture_rgb/capture_mask arrays: {args.store}")

    n = group["capture_rgb"].shape[0]
    if len(rows) != n:
        print(f"WARNING: assets.parquet has {len(rows)} rows but capture_rgb has {n} — using min()")
        n = min(n, len(rows))
        rows = rows[:n]

    indices = _sample_indices(rows, args.per_type, args.seed)
    if not indices:
        raise SystemExit("no entries to sample")

    type_counts: dict[str, int] = {}
    for row in rows:
        t = str(row.get("asset_type", "unknown"))
        type_counts[t] = type_counts.get(t, 0) + 1
    status_counts: dict[str, int] = {}
    for row in rows:
        s = str(row.get("capture_status", "unknown"))
        status_counts[s] = status_counts.get(s, 0) + 1

    cols = 4
    sample_rows = math.ceil(len(indices) / cols)
    fig, axes = plt.subplots(sample_rows, cols, figsize=(cols * 4.2, sample_rows * 2.6))
    axes = np.asarray(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")

    coverages: list[float] = []
    for ax, idx in zip(axes, indices, strict=False):
        row = rows[idx]
        image_rgb = np.asarray(group["capture_rgb"][idx])
        mask = np.asarray(group["capture_mask"][idx])
        coverage = _mask_coverage(mask)
        coverages.append(coverage)
        try:
            composite = _composite(image_rgb, mask)
            ax.imshow(composite)
        except Exception as exc:  # noqa: BLE001 - one bad entry must not kill the sheet
            ax.text(0.5, 0.5, f"ERR\n{exc}", ha="center", va="center", fontsize=6, color="red")

        asset_path = str(row.get("original_asset_path", "?"))
        short_name = asset_path.replace("\\", "/").rsplit("/", 1)[-1]
        title = f"[{row.get('asset_type', '?')}] {short_name}\nstatus={row.get('capture_status', '?')} mask_cov={coverage:.1%}"
        ax.set_title(title, fontsize=7)

    mean_cov = (sum(coverages) / len(coverages)) if coverages else 0.0
    type_summary = ", ".join(f"{k}={v}" for k, v in sorted(type_counts.items()))
    status_summary = ", ".join(f"{k}={v}" for k, v in sorted(status_counts.items()))
    fig.suptitle(
        f"Spec 118 object library — {args.store.name}   |   {n} total entries ({type_summary})   |   "
        f"{status_summary}   |   sampled {len(indices)}, mean mask coverage {mean_cov:.1%}",
        fontsize=10, y=0.998,
    )

    out = args.output or (args.store.parent / f"validation-sheet-{args.store.stem}.png")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"wrote {out}  ({len(indices)} panels | total entries={n} | {type_summary} | {status_summary})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
