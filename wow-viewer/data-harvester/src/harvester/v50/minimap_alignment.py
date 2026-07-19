"""Authored↔detail minimap registration analysis (Spec 113 US1 gate, T006/T008).

The SR pair (authored client LR, our detail HR) is only valid if the two images are spatially
registered — same tile bounds, same orientation. This codebase has real orientation history (the
solar-direction north/south reversals, the open GLB Y-mirror bug), so alignment is measured, never
assumed (``contracts/detail-render-contract.md`` §gate):

- every tile in the sample is scored under all 8 dihedral transforms plus a small translation
  search (normalized cross-correlation on mean-pooled, luma, zero-mean images);
- the gate passes only as ``pass_identity`` or ``pass_with_transform`` (ONE transform wins for
  every sampled tile within tolerance); a per-tile-varying winner is ``fail_inconsistent`` and
  halts the spec — misaligned pairs are worse than no pairs.

Also computes the SC-001 detail-gain metric: high-frequency energy (mean |Laplacian|) of the
detail HR versus a bicubic upscale of the material-average 256 render — proving the detail render
actually contains real texel detail rather than upsampled flat colors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DIHEDRAL_TRANSFORMS = (
    "identity", "rot90", "rot180", "rot270", "flip_h", "flip_v", "transpose", "anti_transpose",
)

# NCC above this (under the winning transform, best translation) counts as registered.
DEFAULT_NCC_TOLERANCE = 0.55
DEFAULT_TRANSLATION_RADIUS = 4
DEFAULT_OFFSET_TOLERANCE = 1


def apply_dihedral(image: np.ndarray, transform: str) -> np.ndarray:
    """Apply one of the 8 square-symmetry transforms to an HxW(xC) image."""
    if transform == "identity":
        return image
    if transform == "rot90":
        return np.rot90(image, k=1, axes=(0, 1))
    if transform == "rot180":
        return np.rot90(image, k=2, axes=(0, 1))
    if transform == "rot270":
        return np.rot90(image, k=3, axes=(0, 1))
    if transform == "flip_h":
        return image[:, ::-1]
    if transform == "flip_v":
        return image[::-1, :]
    if transform == "transpose":
        return np.swapaxes(image, 0, 1)
    if transform == "anti_transpose":
        return np.rot90(np.swapaxes(image, 0, 1), k=2, axes=(0, 1))
    raise ValueError(f"unknown transform {transform!r}")


def _luma(rgb: np.ndarray) -> np.ndarray:
    return np.asarray(rgb, dtype=np.float64).mean(axis=2) if rgb.ndim == 3 else np.asarray(rgb, dtype=np.float64)


def mean_pool(image: np.ndarray, factor: int) -> np.ndarray:
    h, w = image.shape[:2]
    return image[: h - h % factor, : w - w % factor].reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if denom <= 1e-12:
        return 0.0
    return float((a * b).sum() / denom)


def _ncc_with_translation(reference: np.ndarray, candidate: np.ndarray, radius: int) -> tuple[float, tuple[int, int]]:
    best = -1.0
    best_offset = (0, 0)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            # Compare only the valid overlap. np.roll would wrap unrelated pixels across the tile
            # edge and can manufacture a high registration score for the wrong offset.
            if dy >= 0:
                ref_y, candidate_y = slice(dy, reference.shape[0]), slice(0, reference.shape[0] - dy)
            else:
                ref_y, candidate_y = slice(0, reference.shape[0] + dy), slice(-dy, reference.shape[0])
            if dx >= 0:
                ref_x, candidate_x = slice(dx, reference.shape[1]), slice(0, reference.shape[1] - dx)
            else:
                ref_x, candidate_x = slice(0, reference.shape[1] + dx), slice(-dx, reference.shape[1])
            score = _ncc(reference[ref_y, ref_x], candidate[candidate_y, candidate_x])
            if score > best:
                best = score
                best_offset = (dy, dx)
    return best, best_offset


def apply_translation(image: np.ndarray, offset: tuple[int, int] | list[int]) -> np.ndarray:
    """Shift an image by ``(dy, dx)`` without edge wrap, replicating the nearest valid edge.

    The alignment search operates at authored-LR resolution. Callers applying the correction to HR
    must scale the offset first (the pair-set builder does this using the declared SR scale).
    """
    dy, dx = int(offset[0]), int(offset[1])
    source_y = np.clip(np.arange(image.shape[0]) - dy, 0, image.shape[0] - 1)
    source_x = np.clip(np.arange(image.shape[1]) - dx, 0, image.shape[1] - 1)
    return image[source_y[:, None], source_x[None, :], ...]


def register_tile(authored_rgb: np.ndarray, detail_rgb: np.ndarray, *, translation_radius: int = DEFAULT_TRANSLATION_RADIUS) -> dict:
    """Best dihedral transform + translation registering the detail HR onto the authored LR."""
    authored = _luma(authored_rgb)
    detail = _luma(detail_rgb)
    factor = detail.shape[0] // authored.shape[0]
    if factor > 1:
        detail = mean_pool(detail, factor)

    per_transform = {}
    for transform in DIHEDRAL_TRANSFORMS:
        candidate = apply_dihedral(detail, transform)
        if candidate.shape != authored.shape:
            continue  # non-square inputs exclude the swapping transforms
        score, offset = _ncc_with_translation(authored, candidate, translation_radius)
        per_transform[transform] = {"ncc": score, "offset": offset}

    best_transform = max(per_transform, key=lambda t: per_transform[t]["ncc"])
    return {
        "best_transform": best_transform,
        "ncc": per_transform[best_transform]["ncc"],
        "offset": list(per_transform[best_transform]["offset"]),
        "per_transform": {t: round(v["ncc"], 4) for t, v in per_transform.items()},
    }


def high_frequency_energy(rgb: np.ndarray) -> float:
    """Mean absolute Laplacian of the luma — the SC-001 detail measure."""
    luma = _luma(rgb)
    lap = (
        -4.0 * luma[1:-1, 1:-1]
        + luma[:-2, 1:-1] + luma[2:, 1:-1] + luma[1:-1, :-2] + luma[1:-1, 2:]
    )
    return float(np.abs(lap).mean())


def bicubic_upscale(rgb: np.ndarray, size: int) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.fromarray(np.asarray(rgb, dtype=np.uint8)).resize((size, size), Image.BICUBIC))


def evaluate_gate(
    per_tile: list[dict],
    *,
    ncc_tolerance: float = DEFAULT_NCC_TOLERANCE,
    offset_tolerance: int = DEFAULT_OFFSET_TOLERANCE,
) -> dict:
    """The US1 verdict. ``fail_inconsistent`` on a per-tile-varying winner OR a sub-tolerance
    aggregate — never a silent per-tile fixup."""
    winners = {entry["best_transform"] for entry in per_tile}
    nccs = np.array([entry["ncc"] for entry in per_tile], dtype=np.float64)
    offsets = np.asarray([entry["offset"] for entry in per_tile], dtype=np.int64)
    transform_consistent = len(winners) == 1
    global_offset = (
        np.rint(np.median(offsets, axis=0)).astype(np.int64) if len(offsets) else np.array([0, 0])
    )
    offset_consistent = bool(
        len(offsets)
        and (np.abs(offsets - global_offset) <= offset_tolerance).all()
    )
    consistent = transform_consistent and offset_consistent
    within = bool((nccs >= ncc_tolerance).all()) if len(nccs) else False
    if consistent and within:
        transform = next(iter(winners))
        is_identity = transform == "identity" and bool((global_offset == 0).all())
        gate = "pass_identity" if is_identity else "pass_with_transform"
        corrective = None if is_identity else transform
        corrective_offset = [0, 0] if is_identity else global_offset.tolist()
    else:
        gate = "fail_inconsistent"
        corrective = None
        corrective_offset = None
    residuals = 1.0 - nccs
    return {
        "gate": gate,
        "best_transform_global": next(iter(winners)) if transform_consistent else None,
        "best_offset_global": global_offset.tolist() if len(offsets) else None,
        "transform_is_consistent": transform_consistent,
        "offset_is_consistent": offset_consistent,
        "offset_tolerance": offset_tolerance,
        "corrective_transform": corrective,
        "corrective_offset_lr": corrective_offset,
        "ncc_p50": float(np.percentile(nccs, 50)) if len(nccs) else 0.0,
        "ncc_p05": float(np.percentile(nccs, 5)) if len(nccs) else 0.0,
        "residual_p50": float(np.percentile(residuals, 50)) if len(residuals) else 1.0,
        "residual_p95": float(np.percentile(residuals, 95)) if len(residuals) else 1.0,
        "ncc_tolerance": ncc_tolerance,
    }


def analyze_store(store_path: Path, *, sample: int, ncc_tolerance: float = DEFAULT_NCC_TOLERANCE) -> dict:
    import pyarrow.parquet as pq
    import zarr

    group = zarr.open_group(str(store_path), mode="r")
    index = pq.read_table(store_path / "index.parquet").to_pylist()
    for required in ("minimap_rgb_authored", "minimap_rgb_1024", "minimap_rgb"):
        if required not in group:
            raise ValueError(f"store lacks {required!r}; run the Spec 112 rebuild (and the detail render) first")
    if group.attrs.get("minimap_rgb_1024_render_mode") != "detail":
        raise ValueError(
            "store minimap_rgb_1024 is not provenance-marked as render_mode=detail; "
            "run the Spec 113 detail rebuild before alignment"
        )
    if sample <= 0:
        raise ValueError(f"sample must be positive, got {sample}")

    # Coverage is high, so probe a deterministic spread first and stop once the requested sample is
    # full. This avoids reading every 1024x1024 row merely to discover candidate presence.
    probe_count = min(len(index), max(sample * 4, sample))
    spread = np.linspace(0, len(index) - 1, probe_count, dtype=np.int64).tolist() if index else []
    probe_order = list(dict.fromkeys(spread + list(range(len(index)))))
    chosen = []
    for row in probe_order:
        if np.asarray(group["minimap_rgb_authored"][row]).any() and np.asarray(group["minimap_rgb_1024"][row]).any():
            chosen.append(row)
            if len(chosen) >= sample:
                break
    if not chosen:
        raise ValueError("no tile carries both an authored minimap and a 1024 render")

    per_tile = []
    detail_hf, bicubic_hf = [], []
    for row in chosen:
        authored = np.asarray(group["minimap_rgb_authored"][row])
        detail = np.asarray(group["minimap_rgb_1024"][row])
        result = register_tile(authored, detail)
        per_tile.append({
            "map": str(index[row]["map"]), "tile_x": int(index[row]["tile_x"]), "tile_y": int(index[row]["tile_y"]),
            **result,
        })
        detail_hf.append(high_frequency_energy(detail))
        base_256 = np.asarray(group["minimap_rgb"][row])
        if base_256.any():
            bicubic_hf.append(high_frequency_energy(bicubic_upscale(base_256, detail.shape[0])))

    verdict = evaluate_gate(per_tile, ncc_tolerance=ncc_tolerance)
    detail_gain = (float(np.mean(detail_hf)) / float(np.mean(bicubic_hf))) if bicubic_hf else None
    return {
        "schema": "v113-alignment-report-v1",
        "store": str(store_path.resolve()),
        "sample_size": len(chosen),
        "sample_tiles": [
            {"map": entry["map"], "tile_x": entry["tile_x"], "tile_y": entry["tile_y"]}
            for entry in per_tile
        ],
        "per_tile": per_tile,
        **verdict,
        "sc001_detail_gain": detail_gain,
        "sc001_detail_hf_mean": float(np.mean(detail_hf)) if detail_hf else None,
        "sc001_bicubic_hf_mean": float(np.mean(bicubic_hf)) if bicubic_hf else None,
    }


def analyze_stores(
    store_paths: list[Path],
    *,
    sample_per_store: int,
    ncc_tolerance: float = DEFAULT_NCC_TOLERANCE,
) -> dict:
    """Run one gate across both in-scope maps so a transform cannot vary by map unnoticed."""
    if not store_paths:
        raise ValueError("at least one --store is required")
    reports = [
        analyze_store(path, sample=sample_per_store, ncc_tolerance=ncc_tolerance)
        for path in store_paths
    ]
    per_tile = [entry for report in reports for entry in report["per_tile"]]
    verdict = evaluate_gate(per_tile, ncc_tolerance=ncc_tolerance)
    sample_size = sum(report["sample_size"] for report in reports)

    def weighted_mean(field: str) -> float | None:
        available = [
            (float(report[field]), int(report["sample_size"]))
            for report in reports
            if report.get(field) is not None
        ]
        weight = sum(count for _, count in available)
        return sum(value * count for value, count in available) / weight if weight else None

    detail_hf = weighted_mean("sc001_detail_hf_mean")
    bicubic_hf = weighted_mean("sc001_bicubic_hf_mean")
    return {
        "schema": "v113-alignment-report-v1",
        "stores": [str(path.resolve()) for path in store_paths],
        "sample_size": sample_size,
        "sample_per_store": sample_per_store,
        "sample_tiles": [
            {"map": entry["map"], "tile_x": entry["tile_x"], "tile_y": entry["tile_y"]}
            for entry in per_tile
        ],
        "per_tile": per_tile,
        **verdict,
        "sc001_detail_gain": detail_hf / bicubic_hf if detail_hf is not None and bicubic_hf else None,
        "sc001_detail_hf_mean": detail_hf,
        "sc001_bicubic_hf_mean": bicubic_hf,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Spec 113 US1 gate: authored<->detail registration + SC-001 detail gain")
    ap.add_argument("--store", action="append", required=True, type=Path, dest="stores")
    ap.add_argument("--sample", type=int, default=60, help="sample count per store")
    ap.add_argument("--ncc-tolerance", type=float, default=DEFAULT_NCC_TOLERANCE)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    report = analyze_stores(
        args.stores, sample_per_store=args.sample, ncc_tolerance=args.ncc_tolerance
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"alignment: gate={report['gate']} transform={report['best_transform_global']} "
          f"ncc_p50={report['ncc_p50']:.3f} ncc_p05={report['ncc_p05']:.3f} "
          f"sc001_detail_gain={report['sc001_detail_gain']}")
    if report["gate"] == "fail_inconsistent":
        print("US1 GATE FAILED: no single transform registers all sampled tiles -- do NOT build pairs (spec halts).")
        return 1
    return 0
