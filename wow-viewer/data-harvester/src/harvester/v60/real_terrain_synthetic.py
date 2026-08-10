"""Build a clean-signal corpus from real-client terrain synthetic observations.

This bridge is intentionally separate from authored-minimap transfer.  It consumes the current
harvest NPZ contract's ``terrain_shadow_256`` plus independently harvested ``height_257`` target,
and labels the rows ``real_terrain_synthetic`` so actual map geometry can be tested without
pretending that authored RGB has passed albedo normalization.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from harvester.v60.clean_signal_corpus import (
    CORPUS_SCHEMA,
    array_sha256,
    validate_clean_signal_corpus,
)
from harvester.v60.clean_signal_inputs import IMAGE_SHAPE, build_clean_observation
from harvester.v60.clean_signal_targets import decompose_relative_height

REAL_TERRAIN_SYNTHETIC_SCHEMA = "v7-clean-signal-real-terrain-synthetic-v1"
SOURCE_KIND = "real_terrain_synthetic"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _inventory_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(_file_sha256(path).encode("ascii"))
    return f"sha256:{digest.hexdigest()}"


def _metadata(payload: Any, path: Path) -> dict[str, Any]:
    if "metadata.json" not in payload:
        return {"tile_name": path.stem}
    raw = payload["metadata.json"]
    if isinstance(raw, np.ndarray):
        raw = raw.item()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if not isinstance(raw, str):
        raise ValueError(f"{path}: metadata.json must contain UTF-8 JSON text")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: metadata.json must contain an object")
    return value


def _slug(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


def _row_identity(metadata: dict[str, Any], path: Path) -> tuple[str, str, str]:
    build = str(metadata.get("build_key") or metadata.get("build") or "unknown")
    map_name = str(metadata.get("map_name") or metadata.get("map") or "unknown")
    tile_x = metadata.get("tile_x")
    tile_y = metadata.get("tile_y")
    if tile_x is None or tile_y is None:
        tile_name = str(metadata.get("tile_name") or path.stem)
        row_id = f"{SOURCE_KIND}-{_slug(build)}-{_slug(tile_name)}"
    else:
        row_id = f"{SOURCE_KIND}-{_slug(build)}-{_slug(map_name)}-{int(tile_x):02d}-{int(tile_y):02d}"
    return row_id, row_id, f"{build}:{map_name}"


def _source_paths(input_root: str | Path) -> list[Path]:
    root = Path(input_root)
    if not root.is_dir():
        raise FileNotFoundError(f"real terrain synthetic input directory not found: {root}")
    paths = sorted(path for path in root.glob("*.npz") if path.is_file())
    if not paths:
        raise ValueError(f"real terrain synthetic input directory has no NPZ rows: {root}")
    return paths


def real_terrain_synthetic_build_plan(
    input_root: str | Path,
    *,
    confidence_value: float = 1.0,
    seed: int = 7137,
) -> dict[str, Any]:
    """Return a no-write plan for a real-terrain synthetic bridge corpus."""

    paths = _source_paths(input_root)
    if not np.isfinite(confidence_value) or not 0.0 <= confidence_value <= 1.0:
        raise ValueError("confidence_value must be finite and within [0, 1]")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    families: set[str] = set()
    for path in paths:
        with np.load(path, allow_pickle=False) as payload:
            metadata = _metadata(payload, path)
        _, _, family = _row_identity(metadata, path)
        families.add(family)
    return {
        "schema": REAL_TERRAIN_SYNTHETIC_SCHEMA,
        "corpus_schema": CORPUS_SCHEMA,
        "source_root": str(Path(input_root).resolve()),
        "source_npz_count": len(paths),
        "source_inventory_sha256": _inventory_sha256(paths),
        "source_kind": SOURCE_KIND,
        "input_signal": "terrain_shadow_256",
        "target_signal": "height_257",
        "families": sorted(families),
        "family_count": len(families),
        "split_mode": "within_family",
        "split_seed": seed,
        "confidence_status": "measured",
        "confidence_value": float(confidence_value),
        "forbidden_signals_seen": [],
        "dry_run": True,
    }


def build_real_terrain_synthetic_corpus(
    input_root: str | Path,
    output_root: str | Path,
    *,
    confidence_value: float = 1.0,
    seed: int = 7137,
) -> dict[str, Any]:
    """Materialize the bridge corpus atomically without mutating source NPZs."""

    plan = real_terrain_synthetic_build_plan(
        input_root,
        confidence_value=confidence_value,
        seed=seed,
    )
    paths = _source_paths(input_root)
    output = Path(output_root)
    partial = output.with_name(f"{output.name}.partial")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing clean-signal corpus: {output}")
    if partial.exists():
        raise FileExistsError(f"refusing to reuse existing partial clean-signal corpus: {partial}")
    partial.mkdir(parents=True)
    confidence = np.full(IMAGE_SHAPE, confidence_value, dtype=np.float32)
    pending: list[dict[str, Any]] = []
    try:
        for path in paths:
            with np.load(path, allow_pickle=False) as payload:
                metadata = _metadata(payload, path)
                if "terrain_shadow_256" not in payload or "height_257" not in payload:
                    raise ValueError(f"{path}: requires terrain_shadow_256 and height_257")
                shadow = np.asarray(payload["terrain_shadow_256"], dtype=np.float32)
                height = np.asarray(payload["height_257"], dtype=np.float32)
            if shadow.shape != IMAGE_SHAPE:
                raise ValueError(f"{path}: terrain_shadow_256 shape {shadow.shape} != {IMAGE_SHAPE}")
            if height.shape != (257, 257):
                raise ValueError(f"{path}: height_257 shape {height.shape} != (257, 257)")
            if not np.isfinite(shadow).all() or not np.isfinite(height).all():
                raise ValueError(f"{path}: source arrays contain non-finite values")
            if float(shadow.min()) < 0.0 or float(shadow.max()) > 1.0:
                raise ValueError(f"{path}: terrain_shadow_256 is outside [0, 1]")
            row_id, source_group_id, family = _row_identity(metadata, path)
            provenance = {
                "operation": "real_terrain_synthetic_observation_v1",
                "source_signal": "terrain_shadow_256",
                "target_signal": "height_257",
                "source_npz": str(path.resolve()),
                "source_npz_sha256": _file_sha256(path),
                "source_build": str(metadata.get("build_key") or metadata.get("build") or "unknown"),
                "source_map": str(metadata.get("map_name") or metadata.get("map") or "unknown"),
                "artifact_status": "fresh",
                "inference_target_reads": [],
            }
            package = build_clean_observation(
                shadow,
                confidence,
                "measured",
                provenance=provenance,
            )
            target = decompose_relative_height(height)
            arrays = {**package.arrays(), **target.arrays}
            pending.append(
                {
                    "row_id": row_id,
                    "source_group_id": source_group_id,
                    "family": family,
                    "metadata": metadata,
                    "provenance": provenance,
                    "arrays": arrays,
                    "source_path": path,
                }
            )

        by_family: dict[str, list[dict[str, Any]]] = {}
        for item in pending:
            by_family.setdefault(item["family"], []).append(item)
        rows: list[dict[str, Any]] = []
        for family, family_items in sorted(by_family.items()):
            family_items.sort(key=lambda item: item["row_id"])
            if len(family_items) < 2:
                raise ValueError(f"within-family bridge split requires at least two rows for {family!r}")
            digest = hashlib.sha256(f"{seed}:{family}".encode()).digest()
            validation_index = int.from_bytes(digest[:8], "little") % len(family_items)
            for index, item in enumerate(family_items):
                relative_npz = Path("rows") / f"{len(rows):06d}-{_slug(item['row_id'])}.npz"
                npz_path = partial / relative_npz
                npz_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(npz_path, **item["arrays"])
                metadata = item["metadata"]
                rows.append(
                    {
                        "row_id": item["row_id"],
                        "source_kind": SOURCE_KIND,
                        "source_group_id": item["source_group_id"],
                        "family": family,
                        "complexity_bucket": "real_observation",
                        "variant": index,
                        "split": "validation" if index == validation_index else "train",
                        "npz": relative_npz.as_posix(),
                        "confidence_status": "measured",
                        "observation_status": "accepted",
                        "observation_provenance": item["provenance"],
                        "forbidden_signals": [],
                        "array_hashes": {
                            name: array_sha256(array) for name, array in item["arrays"].items()
                        },
                        "map": str(metadata.get("map_name") or metadata.get("map") or "unknown"),
                        "tile_x": metadata.get("tile_x"),
                        "tile_y": metadata.get("tile_y"),
                    }
                )
        rows.sort(key=lambda row: row["row_id"])
        manifest = {
            "schema": CORPUS_SCHEMA,
            "row_count": len(rows),
            "split_mode": "within_family",
            "source_real_terrain_synthetic_root": str(Path(input_root).resolve()),
            "source_real_terrain_synthetic_inventory_sha256": plan["source_inventory_sha256"],
            "source_schema": REAL_TERRAIN_SYNTHETIC_SCHEMA,
            "required_families": sorted({str(row["family"]) for row in rows}),
            "forbidden_signals_seen": [],
            "builder": "harvester.v60.real_terrain_synthetic.build_real_terrain_synthetic_corpus",
            "confidence_status": "measured",
            "confidence_value": float(confidence_value),
            "rows": rows,
        }
        (partial / "clean_signal_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        partial.replace(output)
    except Exception:
        raise
    validation = validate_clean_signal_corpus(output)
    if not validation["valid"]:
        raise ValueError(f"published bridge corpus failed validation: {validation['failures'][:8]}")
    return {
        **plan,
        "dry_run": False,
        "output_root": str(output.resolve()),
        "manifest": str((output / "clean_signal_manifest.json").resolve()),
        "row_count": len(rows),
        "validation": validation,
    }


__all__ = [
    "REAL_TERRAIN_SYNTHETIC_SCHEMA",
    "SOURCE_KIND",
    "build_real_terrain_synthetic_corpus",
    "real_terrain_synthetic_build_plan",
]
