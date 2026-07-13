"""Numeric and visual helpers for auditing every Spec 102 dataset signal."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import zarr
from PIL import Image, ImageDraw, ImageFont

from harvester.spec102.m0 import STRICT_OBJECT_TARGET_KEY, strict_object_target_256
from harvester.spec102.strict_target_contract import REQUIRED_STRICT_OBJECT_TARGET_VERSION

AUDIT_SCHEMA = "spec102-dataset-signal-audit-v2"
M0_AUDITED_SIGNAL_KEYS = (
    "minimap_rgb",
    STRICT_OBJECT_TARGET_KEY,
    "object_geometry_visible_top_elevation_257",
    "object_geometry_visible_terrain_elevation_257",
    "object_geometry_visible_source_257",
    "liquid_mask_256",
    "liquid_height_256",
    "mcnk_flags_16",
    "normal_xyz_257",
    "height_257",
)


def sha256_file(path: Path) -> str:
    """Return the content hash used to bind an audit to its exact inputs."""
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_audited_signal_rows(
    row_numbers: list[int],
    signals: dict[str, np.ndarray],
) -> dict[int, str]:
    """Hash every stored M0 signal per numeric-store row in a canonical order."""
    missing = [name for name in M0_AUDITED_SIGNAL_KEYS if name not in signals]
    if missing:
        raise RuntimeError(f"cannot fingerprint missing M0 signal(s): {', '.join(missing)}")
    row_count = len(row_numbers)
    if len(set(row_numbers)) != row_count:
        raise RuntimeError("cannot fingerprint duplicate numeric-store rows")
    if any(np.asarray(signals[name]).shape[0] != row_count for name in M0_AUDITED_SIGNAL_KEYS):
        raise RuntimeError("cannot fingerprint M0 signals with inconsistent batch rows")
    fingerprints: dict[int, str] = {}
    for offset, row_number in enumerate(row_numbers):
        digest = sha256()
        digest.update(b"spec102-m0-audited-signal-row-v1\0")
        digest.update(int(row_number).to_bytes(8, byteorder="little", signed=True))
        for name in M0_AUDITED_SIGNAL_KEYS:
            value = np.ascontiguousarray(np.asarray(signals[name][offset]))
            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(value.dtype.str.encode("ascii"))
            digest.update(b"\0")
            digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
            digest.update(value.tobytes(order="C"))
        fingerprints[int(row_number)] = digest.hexdigest()
    return fingerprints


def combine_audited_signal_row_fingerprints(row_fingerprints: dict[int, str]) -> str:
    """Return the scope fingerprint independent of audit sharding or batch order."""
    digest = sha256()
    digest.update(b"spec102-m0-audited-signal-scope-v1\0")
    for row_number in sorted(row_fingerprints):
        digest.update(int(row_number).to_bytes(8, byteorder="little", signed=True))
        try:
            digest.update(bytes.fromhex(row_fingerprints[row_number]))
        except ValueError as error:
            raise RuntimeError(f"invalid audited signal row fingerprint for row {row_number}") from error
    return digest.hexdigest()


def _read_rows(array: object, rows: list[int]) -> np.ndarray:
    if not rows:
        raise RuntimeError("cannot fingerprint an empty M0 scope")
    if rows[-1] - rows[0] + 1 == len(rows):
        return np.asarray(array[rows[0]:rows[-1] + 1])
    return np.asarray(array.oindex[np.asarray(rows, dtype=np.int64)])


def current_audited_signal_fingerprint(
    store: Path,
    *,
    scoped_rows: list[int],
    batch_size: int = 16,
) -> str:
    """Recompute the full audited-signal content binding before CUDA starts."""
    if batch_size < 1:
        raise ValueError("M0 audit fingerprint batch_size must be positive")
    rows = sorted(int(row) for row in scoped_rows)
    if len(set(rows)) != len(rows):
        raise RuntimeError("M0 scope has duplicate rows for signal fingerprinting")
    try:
        group = zarr.open_group(str(store), mode="r")
    except Exception as error:
        raise RuntimeError(f"cannot open numeric store for M0 signal fingerprint: {error}") from error
    missing = [name for name in M0_AUDITED_SIGNAL_KEYS if name not in group]
    if missing:
        raise RuntimeError(f"numeric store lacks audited M0 signal(s): {', '.join(missing)}")
    row_fingerprints: dict[int, str] = {}
    for start in range(0, len(rows), batch_size):
        selected = rows[start:start + batch_size]
        batch = {name: _read_rows(group[name], selected) for name in M0_AUDITED_SIGNAL_KEYS}
        row_fingerprints.update(fingerprint_audited_signal_rows(selected, batch))
    return combine_audited_signal_row_fingerprints(row_fingerprints)


def validate_m0_training_audit(
    report_path: Path,
    *,
    store: Path,
    split_manifest: Path,
    expected_scope: dict,
    scoped_rows: list[int],
) -> dict:
    """Refuse M0 CUDA unless its narrow audited build-local contract still matches."""
    if not report_path.is_file():
        raise RuntimeError(f"M0 requires --signal-audit-report: missing file {report_path}")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"M0 cannot read signal-audit report {report_path}: {error}") from error
    if report.get("schema") != AUDIT_SCHEMA:
        raise RuntimeError("M0 signal-audit report has an unsupported schema")
    if report.get("safe_for_m0_build_local_training") is not True:
        failures = report.get("hard_failures") or ["audit did not certify the dataset"]
        raise RuntimeError(f"M0 blocked by signal audit: {failures[0]}")
    provenance = report.get("object_target_provenance") or {}
    if provenance.get("build_local_strict_target_accepted") is not True:
        raise RuntimeError("M0 blocked: audit did not explicitly accept the 3.3.5 strict geometry target scope")
    if provenance.get("terrain_occlusion_clipped") is not True:
        raise RuntimeError("M0 blocked: strict target audit lacks terrain-Z clipping proof")
    if provenance.get("per_pixel_object_top_elevation") is not True:
        raise RuntimeError("M0 blocked: strict target audit lacks per-pixel object-elevation proof")
    if provenance.get("target_version") != REQUIRED_STRICT_OBJECT_TARGET_VERSION:
        raise RuntimeError(
            "M0 blocked: strict target audit does not bind the required object-geometry target version"
        )
    if provenance.get("liquid_evidence_dry_only") is not True:
        raise RuntimeError("M0 blocked: strict target audit lacks Dry liquid-evidence proof")
    if report.get("m0_training_scope") != expected_scope:
        raise RuntimeError("M0 signal-audit report was produced for a different build-local scope")

    try:
        expected = {
            "store": str(store.resolve()),
            "split_manifest": str(split_manifest.resolve()),
            "store_contract_sha256": sha256_file(store / "contract.json"),
            "store_index_sha256": sha256_file(store / "index.parquet"),
            "split_manifest_sha256": sha256_file(split_manifest),
        }
    except OSError as error:
        raise RuntimeError(f"M0 cannot bind signal audit to current inputs: {error}") from error
    mismatches = [
        key for key, expected_value in expected.items()
        if report.get(key) != expected_value
    ]
    if mismatches:
        raise RuntimeError(
            "M0 signal-audit report is stale or bound to different inputs: "
            + ", ".join(mismatches)
        )
    expected_fingerprint = report.get("scoped_signal_fingerprint")
    if not isinstance(expected_fingerprint, str):
        raise RuntimeError("M0 signal-audit report lacks the scoped signal-content fingerprint")
    current_fingerprint = current_audited_signal_fingerprint(store, scoped_rows=scoped_rows)
    if current_fingerprint != expected_fingerprint:
        raise RuntimeError("M0 signal-audit report is stale: audited signal content changed")
    return report


@dataclass(frozen=True)
class PlacementTerrainRelation:
    pixel_x: int
    pixel_y: int
    object_top: float
    terrain_height: float
    top_source: str

    @property
    def clearance(self) -> float:
        return self.object_top - self.terrain_height


def _outside(value: float) -> float:
    return max(0.0, -value) + max(0.0, value - 1.0)


def project_placement_to_terrain(
    placement: dict,
    *,
    tile_x: int,
    tile_y: int,
    terrain_height_257: np.ndarray,
) -> PlacementTerrainRelation | None:
    """Mirror the builder's projection candidates and compare known object top to terrain."""
    x = float(placement["posX"])
    y = float(placement["posY"])
    z = float(placement["posZ"])
    tile_size = 533.33333
    origin = 17066.666
    candidates = (
        ((x / tile_size) - tile_x, (z / tile_size) - tile_y, y, "position_y"),
        (((origin - z) / tile_size) - tile_x, ((origin - x) / tile_size) - tile_y, y, "position_y"),
        ((x / tile_size) - tile_x, (y / tile_size) - tile_y, z, "position_z"),
        (((origin - y) / tile_size) - tile_x, ((origin - x) / tile_size) - tile_y, z, "position_z"),
    )
    in_range = [
        candidate for candidate in candidates
        if -0.25 <= candidate[0] <= 1.25 and -0.25 <= candidate[1] <= 1.25
    ]
    if not in_range:
        return None
    # Mirror TryProjectPlacementToTilePixel: candidates must fit the expanded
    # tile and the one nearest the tile centre wins; stable min keeps C# order.
    u, v, position_top, axis_source = min(
        in_range,
        key=lambda item: abs(item[0] - 0.5) + abs(item[1] - 0.5),
    )
    px = int(np.clip(np.rint(u * 256.0), 0, 256))
    py = int(np.clip(np.rint(v * 256.0), 0, 256))
    terrain = float(np.asarray(terrain_height_257)[py, px])
    bounds_valid = (
        float(placement.get("bbMinX", 0.0)) < float(placement.get("bbMaxX", 0.0))
        and float(placement.get("bbMinY", 0.0)) < float(placement.get("bbMaxY", 0.0))
    )
    if bounds_valid:
        if axis_source == "position_z":
            object_top = float(placement["bbMaxZ"])
            top_source = "bounds_z"
        else:
            object_top = float(placement["bbMaxY"])
            top_source = "bounds_y"
    else:
        object_top = position_top
        top_source = axis_source
    return PlacementTerrainRelation(px, py, object_top, terrain, top_source)


def _normalize_scalar(value: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    data = np.asarray(value, dtype=np.float32)
    valid = np.isfinite(data)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    result = np.zeros(data.shape, dtype=np.uint8)
    if not np.any(valid):
        return result
    lo, hi = np.percentile(data[valid], (2.0, 98.0))
    if hi <= lo:
        hi = lo + 1.0
    result[valid] = np.clip((data[valid] - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
    return result


def _resize_256(value: np.ndarray, *, resample: Image.Resampling = Image.Resampling.NEAREST) -> Image.Image:
    image = Image.fromarray(value)
    return image.resize((256, 256), resample=resample)


def _overlay(rgb: np.ndarray, mask: np.ndarray, colour: tuple[int, int, int]) -> np.ndarray:
    source = np.asarray(rgb, dtype=np.float32).copy()
    alpha = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)[..., None] * 0.65
    return np.clip(source * (1.0 - alpha) + np.asarray(colour, dtype=np.float32) * alpha, 0, 255).astype(np.uint8)


def render_signal_panel(samples: list[dict], *, split: str, source_label: str) -> Image.Image:
    if not samples:
        raise ValueError("signal panel requires samples")
    size = 256
    header = 75
    footer = 31
    row_height = size + footer
    headings = (
        "RAW MINIMAP RGB",
        "STRICT TERRAIN-VISIBLE OBJECT TARGET",
        "OBJECT OVERLAY (magenta)",
        "LIQUID MASK",
        "LIQUID HEIGHT (masked)",
        "TERRAIN HEIGHT (per-tile)",
        "NATIVE NORMAL XYZ",
        "MCNK FLAGS bits 0/6/15",
    )
    canvas = Image.new("RGB", (size * len(headings), header + row_height * len(samples)), (18, 18, 22))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((8, 6), f"Spec 102 dataset signal audit | split={split} | {source_label}", fill="white", font=font)
    draw.text((8, 22), "These are stored numeric tensors. Height views are display-normalized per tile; training reads raw floats.", fill=(205, 205, 210), font=font)
    draw.text((8, 38), "Object=white/magenta. Liquid=white/cyan. Normal RGB maps signed XYZ. Flags: R=0x0001 G=0x0040 B=0x8000.", fill=(235, 215, 150), font=font)
    for column, heading in enumerate(headings):
        draw.text((column * size + 6, 58), heading, fill="white", font=font)

    for index, sample in enumerate(samples):
        y = header + index * row_height
        rgb = np.asarray(sample["minimap_rgb"], dtype=np.uint8)
        strict = strict_object_target_256(sample[STRICT_OBJECT_TARGET_KEY])
        liquid = np.asarray(sample["liquid_mask_256"]) > 127
        liquid_height = _normalize_scalar(sample["liquid_height_256"], liquid)
        terrain_height = _normalize_scalar(sample["height_257"])
        normals = np.asarray(sample["normal_xyz_257"], dtype=np.int16)
        normal_rgb = np.clip((normals + 127) * (255.0 / 254.0), 0, 255).astype(np.uint8)
        flags = np.asarray(sample["mcnk_flags_16"], dtype=np.int64)
        flag_rgb = np.stack(
            [flags & 0x0001, (flags & 0x0040) >> 6, (flags & 0x8000) >> 15], axis=-1
        ).astype(np.uint8) * 255
        views = (
            Image.fromarray(rgb, "RGB"),
            Image.fromarray((np.clip(strict, 0, 1) * 255).astype(np.uint8), "L").convert("RGB"),
            Image.fromarray(_overlay(rgb, strict, (255, 0, 255)), "RGB"),
            Image.fromarray(liquid.astype(np.uint8) * 255, "L").convert("RGB"),
            Image.fromarray(liquid_height, "L").convert("RGB"),
            _resize_256(terrain_height, resample=Image.Resampling.BILINEAR).convert("RGB"),
            _resize_256(normal_rgb, resample=Image.Resampling.NEAREST),
            _resize_256(flag_rgb, resample=Image.Resampling.NEAREST),
        )
        for column, view in enumerate(views):
            canvas.paste(view, (column * size, y))
        row = sample["metadata"]
        label = (
            f"row={row['row']} {row['build']}/{row['map']} tile=({row['tile_x']},{row['tile_y']})  "
            f"strict-object={float((strict > 0.5).mean()):.3f} liquid={float(liquid.mean()):.3f} "
            f"relief={float(np.ptp(sample['height_257'])):.2f} source={row.get('liquid_source')}"
        )
        draw.rectangle((0, y + size, canvas.width, y + row_height), fill=(18, 18, 22))
        draw.text((8, y + size + 9), label, fill=(235, 235, 235), font=font)
    return canvas
