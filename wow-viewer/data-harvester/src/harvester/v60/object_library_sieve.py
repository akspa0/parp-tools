"""Build and validate v60 object-sieve controls from the real v50 object library.

The old v60 object controls used procedural ellipses and the failed real-mask experiment used
per-tile placement projections.  This module uses the Spec 118/119 object-library contract
instead: every placed object is a real ``capture_rgb``/``capture_mask`` pair from the 0.5.3
library.  The source library and the control corpus are read-only inputs; the derived corpus is a
new immutable run directory.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "v60-object-library-sieve-v1"
VALIDATION_SCHEMA = "v60-object-library-sieve-validation-v1"
INPUT_SIGNAL = "objectified_terrain_shadow_256"
CLEAN_SIGNAL = "terrain_shadow_256"
MASK_SIGNAL = "object_contamination_mask_256"
INSTANCE_SIGNAL = "object_instance_id_256"
PIXELS = 256
PLACEMENT_REGIMES = ("none", "sparse", "dense", "overlap", "boundary_crossing")
SIGNAL_SHAPES = {INPUT_SIGNAL: (PIXELS, PIXELS), CLEAN_SIGNAL: (PIXELS, PIXELS), MASK_SIGNAL: (PIXELS, PIXELS), INSTANCE_SIGNAL: (PIXELS, PIXELS)}


class ObjectLibrarySieveError(ValueError):
    """Raised when a source corpus or object library violates the derived contract."""


def _hash_bytes(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_float(array: np.ndarray) -> str:
    canonical = np.ascontiguousarray(np.asarray(array, dtype="<f4"))
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _stable_seed(*parts: object) -> int:
    value = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little") & 0xFFFFFFFF


def _asset_family(normalized_path: str) -> str:
    parts = [part for part in normalized_path.replace("\\", "/").lower().split("/") if part]
    if len(parts) >= 2:
        return "/".join(parts[:2])
    return parts[0] if parts else "unknown"


def _family_split(family: str, seed: int) -> str:
    # A stable five-way partition prevents the same library family appearing in both sides.
    return "validation" if _stable_seed("library-family-split", seed, family) % 5 == 0 else "train"


def _crop_capture(rgb: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    visible = np.asarray(mask) > 0
    if visible.ndim != 2 or rgb.ndim != 3 or rgb.shape[:2] != visible.shape or rgb.shape[2] != 3:
        raise ObjectLibrarySieveError("object capture RGB/mask shapes are incompatible")
    ys, xs = np.nonzero(visible)
    if len(xs) == 0:
        raise ObjectLibrarySieveError("blank object capture cannot be placed")
    pad = 2
    y0, y1 = max(0, int(ys.min()) - pad), min(visible.shape[0], int(ys.max()) + pad + 1)
    x0, x1 = max(0, int(xs.min()) - pad), min(visible.shape[1], int(xs.max()) + pad + 1)
    return np.asarray(rgb[y0:y1, x0:x1], dtype=np.uint8), visible[y0:y1, x0:x1]


def _transform_capture(
    rgb: np.ndarray,
    mask: np.ndarray,
    *,
    max_extent: int,
    angle_degrees: float,
) -> tuple[np.ndarray, np.ndarray]:
    from PIL import Image

    rgb, mask = _crop_capture(rgb, mask)
    height, width = rgb.shape[:2]
    scale = max_extent / max(height, width)
    scaled_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    rgb_image = Image.fromarray(rgb, mode="RGB").resize(scaled_size, Image.Resampling.BICUBIC)
    # BOX retains any source foreground covered by a destination pixel when reducing a thin
    # silhouette. NEAREST can erase narrow object parts completely before the placement step.
    mask_image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L").resize(
        scaled_size, Image.Resampling.BOX
    )
    rgb_image = rgb_image.rotate(angle_degrees, resample=Image.Resampling.BICUBIC, expand=True)
    mask_image = mask_image.rotate(angle_degrees, resample=Image.Resampling.BILINEAR, expand=True)
    transformed_rgb = np.asarray(rgb_image, dtype=np.uint8)
    transformed_mask = np.asarray(mask_image, dtype=np.uint8) > 0
    if transformed_mask.sum() == 0:
        # Very thin one-pixel library silhouettes can disappear under a nearest-neighbour
        # rotation at a small target extent. Preserve the exact source silhouette rather than
        # dropping the object or silently emitting an empty target.
        fallback_rgb = Image.fromarray(rgb, mode="RGB").resize(scaled_size, Image.Resampling.BICUBIC)
        fallback_mask = Image.fromarray((mask.astype(np.uint8) * 255), mode="L").resize(
            scaled_size, Image.Resampling.BOX
        )
        transformed_rgb = np.asarray(fallback_rgb, dtype=np.uint8)
        transformed_mask = np.asarray(fallback_mask, dtype=np.uint8) > 0
    if transformed_mask.sum() == 0:
        raise ObjectLibrarySieveError("object transform erased the capture mask")
    return transformed_rgb, transformed_mask


def _place(
    clean: np.ndarray,
    objectified: np.ndarray,
    union_mask: np.ndarray,
    instance_ids: np.ndarray,
    patch_rgb: np.ndarray,
    patch_mask: np.ndarray,
    *,
    centre_x: float,
    centre_y: float,
    instance_id: int,
) -> None:
    patch_height, patch_width = patch_mask.shape
    left = int(round(centre_x - (patch_width / 2.0)))
    top = int(round(centre_y - (patch_height / 2.0)))
    x0, x1 = max(0, left), min(PIXELS, left + patch_width)
    y0, y1 = max(0, top), min(PIXELS, top + patch_height)
    if x1 <= x0 or y1 <= y0:
        return
    patch_x0, patch_y0 = x0 - left, y0 - top
    patch_x1, patch_y1 = patch_x0 + (x1 - x0), patch_y0 + (y1 - y0)
    local_mask = patch_mask[patch_y0:patch_y1, patch_x0:patch_x1]
    if not local_mask.any():
        return
    rgb = patch_rgb[patch_y0:patch_y1, patch_x0:patch_x1].astype(np.float32) / 255.0
    luma = np.clip((rgb * np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)).sum(axis=2), 0.0, 1.0)
    clean_view = clean[y0:y1, x0:x1]
    input_view = objectified[y0:y1, x0:x1]
    input_view[local_mask] = (clean_view[local_mask] * 0.20) + (luma[local_mask] * 0.80)
    union_mask[y0:y1, x0:x1][local_mask] = 1.0
    instance_ids[y0:y1, x0:x1][local_mask] = instance_id


def _load_library(library: Path, *, blank_threshold: float, split_seed: int) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    import pyarrow.parquet as pq
    import zarr

    group = zarr.open_group(str(library), mode="r")
    for name in ("capture_rgb", "capture_mask"):
        if name not in group:
            raise ObjectLibrarySieveError(f"object library is missing {name}: {library}")
    assets_path = library / "assets.parquet"
    index_path = library / "index.parquet"
    if not assets_path.is_file() or not index_path.is_file():
        raise ObjectLibrarySieveError(f"object library is missing assets.parquet or index.parquet: {library}")
    assets = pq.read_table(assets_path).to_pylist()
    rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(assets):
        if row.get("capture_status") != "captured":
            continue
        mask = np.asarray(group["capture_mask"][row_index])
        coverage = float((mask > 0).mean())
        if coverage < blank_threshold:
            continue
        normalized_path = str(row.get("normalized_asset_path", ""))
        family = _asset_family(normalized_path)
        rows.append({
            **row,
            "_row_index": row_index,
            "_coverage": coverage,
            "_library_family": family,
            "_library_split": _family_split(family, split_seed),
        })
    if not rows:
        raise ObjectLibrarySieveError("object library has no non-blank captured rows")
    provenance = {
        "schema": str(group.attrs.get("schema", "")),
        "run_name": str(group.attrs.get("run_name", "")),
        "entry_count": int(group.attrs.get("entry_count", len(assets))),
        "source_store": str(library.resolve()),
        "assets_sha256": _hash_bytes(assets_path),
        "index_sha256": _hash_bytes(index_path),
        "blank_threshold": blank_threshold,
        "eligible_row_count": len(rows),
    }
    return group, rows, provenance


def _choose_rows(
    rows: list[dict[str, Any]], *, split: str, count: int, rng: np.random.Generator
) -> list[dict[str, Any]]:
    eligible = [row for row in rows if row["_library_split"] == ("validation" if split != "train" else "train")]
    if not eligible:
        raise ObjectLibrarySieveError(f"object library has no rows for split {split!r}")
    indices = rng.integers(0, len(eligible), size=count)
    return [eligible[int(index)] for index in indices]


def _placement_centres(regime: str, count: int, rng: np.random.Generator, sizes: list[tuple[int, int]]) -> list[tuple[float, float]]:
    if regime == "boundary_crossing":
        centres: list[tuple[float, float]] = []
        for index, (height, width) in enumerate(sizes):
            if index % 4 == 0:
                centres.append((-0.18 * width, float(rng.uniform(0.25 * PIXELS, 0.75 * PIXELS))))
            elif index % 4 == 1:
                centres.append((PIXELS + (0.18 * width), float(rng.uniform(0.25 * PIXELS, 0.75 * PIXELS))))
            elif index % 4 == 2:
                centres.append((float(rng.uniform(0.25 * PIXELS, 0.75 * PIXELS)), -0.18 * height))
            else:
                centres.append((float(rng.uniform(0.25 * PIXELS, 0.75 * PIXELS)), PIXELS + (0.18 * height)))
        return centres
    if regime == "overlap":
        centre_x = float(rng.uniform(0.38 * PIXELS, 0.62 * PIXELS))
        centre_y = float(rng.uniform(0.38 * PIXELS, 0.62 * PIXELS))
        return [
            (centre_x + float(rng.uniform(-14.0, 14.0)), centre_y + float(rng.uniform(-14.0, 14.0)))
            for _ in sizes
        ]
    return [
        (float(rng.uniform(0.10 * PIXELS, 0.90 * PIXELS)), float(rng.uniform(0.10 * PIXELS, 0.90 * PIXELS)))
        for _ in sizes
    ]


def _regime_count(regime: str) -> int:
    return {"none": 0, "sparse": 1, "dense": 4, "overlap": 4, "boundary_crossing": 4}[regime]


def build_object_library_sieve_corpus(
    *,
    control_corpus: Path,
    object_library: Path,
    output: Path,
    samples_per_terrain: int = 1,
    seed: int = 6001,
    blank_threshold: float = 0.01,
) -> dict[str, Any]:
    """Materialize a deterministic library-derived object-sieve corpus."""
    if samples_per_terrain < 1:
        raise ObjectLibrarySieveError("samples_per_terrain must be at least 1")
    if output.exists():
        raise ObjectLibrarySieveError(f"refusing to overwrite existing output: {output}")

    from harvester.v60.control_corpus import load_control_manifest, validate_control_corpus

    control_report = validate_control_corpus(control_corpus)
    if not control_report["valid"]:
        raise ObjectLibrarySieveError(
            "control corpus is invalid; fix it before deriving library overlays: "
            + "; ".join(control_report["failures"][:4])
        )
    control_manifest = load_control_manifest(control_corpus)
    library_group, library_rows, library_provenance = _load_library(
        object_library, blank_threshold=blank_threshold, split_seed=seed
    )
    output.mkdir(parents=True)
    manifest_rows: list[dict[str, Any]] = []
    regimes = list(PLACEMENT_REGIMES)

    for control_row in control_manifest["rows"]:
        with np.load(control_corpus / str(control_row["npz"]), allow_pickle=False) as payload:
            clean = np.asarray(payload[CLEAN_SIGNAL], dtype=np.float32)
            optional_height = np.asarray(payload["height_257"], dtype=np.float32) if "height_257" in payload else None
            optional_normals = np.asarray(payload["mcnr_normal_xyz"], dtype=np.float32) if "mcnr_normal_xyz" in payload else None
        split = str(control_row.get("split", "train"))
        for regime in regimes:
            sample_count = 1 if regime == "none" else samples_per_terrain
            for sample in range(sample_count):
                row_seed = _stable_seed(seed, control_row["row_id"], regime, sample)
                rng = np.random.default_rng(row_seed)
                placement_count = _regime_count(regime)
                effective_split = "validation" if regime == "boundary_crossing" else split
                selected = _choose_rows(library_rows, split=effective_split, count=placement_count, rng=rng)
                objectified = clean.copy()
                union_mask = np.zeros((PIXELS, PIXELS), dtype=np.float32)
                instance_ids = np.zeros((PIXELS, PIXELS), dtype=np.uint16)
                patches: list[tuple[np.ndarray, np.ndarray]] = []
                placement_specs: list[dict[str, Any]] = []
                for instance_id, library_row in enumerate(selected, start=1):
                    if regime == "sparse":
                        max_extent = int(rng.integers(20, 64))
                    elif regime == "dense":
                        max_extent = int(rng.integers(14, 48))
                    elif regime == "overlap":
                        max_extent = int(rng.integers(28, 64))
                    else:
                        max_extent = int(rng.integers(24, 72))
                    angle = float(rng.uniform(-180.0, 180.0))
                    try:
                        patch_rgb, patch_mask = _transform_capture(
                            np.asarray(library_group["capture_rgb"][library_row["_row_index"]]),
                            np.asarray(library_group["capture_mask"][library_row["_row_index"]]),
                            max_extent=max_extent,
                            angle_degrees=angle,
                        )
                    except ObjectLibrarySieveError as exc:
                        raise ObjectLibrarySieveError(
                            f"unable to transform library object {library_row.get('library_id', '')} "
                            f"for {control_row['row_id']} {regime}: {exc}"
                        ) from exc
                    patches.append((patch_rgb, patch_mask))
                    placement_specs.append({
                        "instance_id": instance_id,
                        "library_id": str(library_row.get("library_id", "")),
                        "asset_path": str(library_row.get("normalized_asset_path", "")),
                        "asset_type": str(library_row.get("asset_type", "unknown")),
                        "library_family": str(library_row["_library_family"]),
                        "library_row_index": int(library_row["_row_index"]),
                        "capture_mask_coverage": float(library_row["_coverage"]),
                        "max_extent": max_extent,
                        "rotation_degrees": angle,
                    })
                centres = _placement_centres(regime, placement_count, rng, [(p.shape[0], p.shape[1]) for _, p in patches])
                for instance_id, ((patch_rgb, patch_mask), (centre_x, centre_y)) in enumerate(zip(patches, centres, strict=True), start=1):
                    _place(
                        clean,
                        objectified,
                        union_mask,
                        instance_ids,
                        patch_rgb,
                        patch_mask,
                        centre_x=centre_x,
                        centre_y=centre_y,
                        instance_id=instance_id,
                    )
                    placement_specs[instance_id - 1].update({"centre_x": centre_x, "centre_y": centre_y})
                if regime != "none" and union_mask.max() <= 0.0:
                    raise ObjectLibrarySieveError(f"{control_row['row_id']} {regime} produced an empty mask")
                if regime == "boundary_crossing" and not (
                    union_mask[:8].any() or union_mask[-8:].any() or union_mask[:, :8].any() or union_mask[:, -8:].any()
                ):
                    raise ObjectLibrarySieveError(f"{control_row['row_id']} boundary_crossing did not touch a tile edge")

                row_id = f"{control_row['row_id']}-libobj-{regime}-s{sample:02d}"
                npz_name = f"{row_id}.npz"
                arrays: dict[str, np.ndarray] = {
                    INPUT_SIGNAL: np.asarray(objectified, dtype=np.float32),
                    CLEAN_SIGNAL: np.asarray(clean, dtype=np.float32),
                    MASK_SIGNAL: np.asarray(union_mask, dtype=np.float32),
                    INSTANCE_SIGNAL: np.asarray(instance_ids, dtype=np.uint16),
                }
                if optional_height is not None:
                    arrays["height_257"] = optional_height
                if optional_normals is not None:
                    arrays["mcnr_normal_xyz"] = optional_normals
                np.savez_compressed(output / npz_name, **arrays)
                manifest_rows.append({
                    "row_id": row_id,
                    "source_kind": "real_v50_object_library_composite",
                    "source_control_row_id": str(control_row["row_id"]),
                    "terrain_control_family": str(control_row.get("control_family", "")),
                    "terrain_complexity_bucket": str(control_row.get("complexity_bucket", "")),
                    "split": effective_split,
                    "placement_regime": regime,
                    "placement_count": placement_count,
                    "input": INPUT_SIGNAL,
                    "targets": [CLEAN_SIGNAL, MASK_SIGNAL],
                    "instance_signal": INSTANCE_SIGNAL,
                    "input_shape": list(SIGNAL_SHAPES[INPUT_SIGNAL]),
                    "target_shapes": {CLEAN_SIGNAL: list(SIGNAL_SHAPES[CLEAN_SIGNAL]), MASK_SIGNAL: list(SIGNAL_SHAPES[MASK_SIGNAL])},
                    "object_instance_count": int(instance_ids.max()),
                    "object_coverage": float((union_mask >= 0.5).mean()),
                    "library_object_ids": [spec["library_id"] for spec in placement_specs],
                    "object_instances": placement_specs,
                    "input_sha256": _hash_float(objectified),
                    "terrain_target_sha256": _hash_float(clean),
                    "contamination_target_sha256": _hash_float(union_mask),
                    "npz": npz_name,
                })

    manifest = {
        "schema": SCHEMA,
        "source_policy": "real_v50_object_library_over_project_control_terrain",
        "generator": "harvester.v60.object_library_sieve",
        "seed": seed,
        "samples_per_terrain": samples_per_terrain,
        "blank_threshold": blank_threshold,
        "source_control_manifest": str((control_corpus / "control_manifest.json").resolve()),
        "source_control_schema": control_manifest["schema"],
        "source_library": library_provenance,
        "signal_contract": [INPUT_SIGNAL, CLEAN_SIGNAL, MASK_SIGNAL, INSTANCE_SIGNAL],
        "row_count": len(manifest_rows),
        "terrain_row_count": len(control_manifest["rows"]),
        "object_families": sorted({str(spec["library_family"]) for row in manifest_rows for spec in row["object_instances"]}),
        "placement_regimes": regimes,
        "alignment_policy": "inherits_control_subcell_offsets; object_centres_are_screen_space_decimal",
        "rows": manifest_rows,
    }
    (output / "object_library_sieve_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"schema": SCHEMA, "output": str(output), "row_count": len(manifest_rows), "source_library_rows": len(library_rows)}


def load_object_library_sieve_manifest(corpus_root: str | Path) -> dict[str, Any]:
    path = Path(corpus_root) / "object_library_sieve_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"object-library sieve manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"expected schema {SCHEMA!r}, got {manifest.get('schema')!r}")
    if int(manifest.get("row_count", -1)) != len(manifest.get("rows", [])):
        raise ValueError("object-library sieve row_count does not match rows")
    return manifest


def validate_object_library_sieve_corpus(corpus_root: str | Path) -> dict[str, Any]:
    root = Path(corpus_root)
    manifest = load_object_library_sieve_manifest(root)
    failures: list[str] = []
    regime_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    library_ids: set[str] = set()
    family_splits: dict[str, str] = {}
    coverage_by_regime: dict[str, list[float]] = {}
    boundary_touch_counts: dict[str, int] = {}
    for position, row in enumerate(manifest["rows"]):
        prefix = f"row[{position}]"
        regime = str(row.get("placement_regime", ""))
        split = str(row.get("split", ""))
        if regime not in PLACEMENT_REGIMES:
            failures.append(f"{prefix}: invalid placement_regime {regime!r}")
        if split not in {"train", "validation", "test"}:
            failures.append(f"{prefix}: invalid split {split!r}")
        regime_counts[regime] = regime_counts.get(regime, 0) + 1
        split_counts[split] = split_counts.get(split, 0) + 1
        npz_name = row.get("npz")
        if not isinstance(npz_name, str) or not (root / npz_name).is_file():
            failures.append(f"{prefix}: missing NPZ {npz_name!r}")
            continue
        try:
            with np.load(root / npz_name, allow_pickle=False) as payload:
                arrays = {name: np.asarray(payload[name]) for name in SIGNAL_SHAPES}
        except (OSError, ValueError, KeyError) as exc:
            failures.append(f"{prefix}: unable to read NPZ: {exc}")
            continue
        for name, array in arrays.items():
            if array.shape != SIGNAL_SHAPES[name]:
                failures.append(f"{prefix}: {name} shape {array.shape} != {SIGNAL_SHAPES[name]}")
            if not np.isfinite(array.astype(np.float32)).all():
                failures.append(f"{prefix}: {name} contains non-finite values")
        for name in (INPUT_SIGNAL, CLEAN_SIGNAL, MASK_SIGNAL):
            if arrays[name].min() < -1e-6 or arrays[name].max() > 1.000001:
                failures.append(f"{prefix}: {name} is outside [0, 1]")
        instances = arrays[INSTANCE_SIGNAL]
        mask = arrays[MASK_SIGNAL]
        if np.any((instances > 0) != (mask >= 0.5)):
            failures.append(f"{prefix}: instance-ID occupancy differs from contamination mask")
        instance_count = int(instances.max())
        if int(row.get("object_instance_count", -1)) != instance_count:
            failures.append(f"{prefix}: object_instance_count does not match instance map")
        coverage = float((mask >= 0.5).mean())
        coverage_by_regime.setdefault(regime, []).append(coverage)
        if regime == "none" and (coverage != 0.0 or instance_count != 0 or not np.array_equal(arrays[INPUT_SIGNAL], arrays[CLEAN_SIGNAL])):
            failures.append(f"{prefix}: none row is not an exact clean negative")
        if regime != "none" and coverage <= 0.0:
            failures.append(f"{prefix}: {regime} row has an empty contamination mask")
        if regime == "boundary_crossing":
            touches = bool((mask[:8] >= 0.5).any() or (mask[-8:] >= 0.5).any() or (mask[:, :8] >= 0.5).any() or (mask[:, -8:] >= 0.5).any())
            if not touches:
                failures.append(f"{prefix}: boundary_crossing does not touch a tile boundary")
            else:
                boundary_touch_counts[regime] = boundary_touch_counts.get(regime, 0) + 1
        for spec in row.get("object_instances", []):
            library_id = str(spec.get("library_id", ""))
            family = str(spec.get("library_family", ""))
            if not library_id or not family:
                failures.append(f"{prefix}: object instance lacks library identity/family")
            if library_id:
                library_ids.add(library_id)
            if family:
                prior = family_splits.setdefault(family, split)
                if prior != split:
                    failures.append(f"{prefix}: library family {family!r} crosses {prior!r}/{split!r}")
    missing_regimes = sorted(set(PLACEMENT_REGIMES) - set(regime_counts))
    if missing_regimes:
        failures.append(f"missing placement regimes {missing_regimes}")
    return {
        "schema": VALIDATION_SCHEMA,
        "corpus_root": str(root),
        "row_count": len(manifest["rows"]),
        "regime_counts": dict(sorted(regime_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "library_object_count": len(library_ids),
        "library_family_count": len(family_splits),
        "mean_mask_coverage_by_regime": {key: float(np.mean(value)) for key, value in sorted(coverage_by_regime.items())},
        "boundary_touch_counts": dict(sorted(boundary_touch_counts.items())),
        "failures": failures,
        "valid": not failures,
    }


__all__ = [
    "CLEAN_SIGNAL",
    "INPUT_SIGNAL",
    "INSTANCE_SIGNAL",
    "MASK_SIGNAL",
    "ObjectLibrarySieveError",
    "PLACEMENT_REGIMES",
    "SCHEMA",
    "build_object_library_sieve_corpus",
    "load_object_library_sieve_manifest",
    "validate_object_library_sieve_corpus",
]
