"""Prefab-aware, map-canvas corpus curation for Spec 103 Phase 3B.

ADT tiles are provenance pages, not analysis units.  This module consumes
Spec 076 full-map canvas/region evidence, derives deterministic terrain-art
prefab families and map-global placement context, then selects the smallest
set of tiles that covers the observed prefab vocabulary.

All alpha, terrain, MCLY, object, and liquid values are *curation evidence*.
They never become inference inputs for the image-only Spec 103 deployment
contract.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

from harvester.fractal_segments import segment_canvas_regions

ALPHA_TILE_SIZE = 256
CHUNK_PIXEL_SIZE = 16
EVIDENCE_SCHEMA_VERSION = "spec103-prefab-evidence-v1"

DEFAULT_REGION_LABELS = frozenset(
    {
        "accepted_candidate",
        "fractal_member",
        "composite_chonker",
        "one_off_detail",
        "rectangle_page",
        "macro_paste",
        "blocky_paste",
        "rejected_unknown",
    }
)

TRANSFORMS: tuple[tuple[str, Any], ...] = (
    ("identity", lambda value: value),
    ("rotate_90", lambda value: np.rot90(value, 1)),
    ("rotate_180", lambda value: np.rot90(value, 2)),
    ("rotate_270", lambda value: np.rot90(value, 3)),
    ("mirror_x", np.fliplr),
    ("mirror_y", np.flipud),
    ("mirror_diag", lambda value: np.transpose(value)),
    ("mirror_anti_diag", lambda value: np.fliplr(np.flipud(np.transpose(value)))),
)


LEDGER_SCHEMA = pa.schema(
    [
        ("schema_version", pa.string()),
        ("evidence_state", pa.string()),
        ("build", pa.string()),
        ("map", pa.string()),
        ("tile_id", pa.int64()),
        ("tile_x", pa.int32()),
        ("tile_y", pa.int32()),
        ("store_row", pa.int64()),
        ("chunk_keys", pa.list_(pa.string())),
        ("region_pixel_count_in_tile", pa.int64()),
        ("tile_fraction", pa.float64()),
        ("placement_id", pa.string()),
        ("prefab_family_id", pa.string()),
        ("family_source", pa.string()),
        ("source_region_id", pa.string()),
        ("region_evidence_source", pa.string()),
        ("layer_slot", pa.int32()),
        ("layer_idx", pa.int32()),
        ("curation_label", pa.string()),
        ("bbox_x", pa.int64()),
        ("bbox_y", pa.int64()),
        ("bbox_w", pa.int64()),
        ("bbox_h", pa.int64()),
        ("crosses_adt_boundary", pa.bool_()),
        ("transform_to_canonical", pa.string()),
        ("canonical_alpha_hash", pa.string()),
        ("alpha_evidence_status", pa.string()),
        ("alpha_missing_reason", pa.string()),
        ("multiscale_occupancy", pa.list_(pa.float64())),
        ("multiscale_transition", pa.list_(pa.float64())),
        ("arrangement_class", pa.string()),
        ("cellular_signature", pa.string()),
        ("cellular_ring_sector_counts", pa.list_(pa.int32())),
        ("same_family_neighbor_ids", pa.list_(pa.string())),
        ("neighbor_placement_ids", pa.list_(pa.string())),
        ("parent_placement_ids", pa.list_(pa.string())),
        ("child_placement_ids", pa.list_(pa.string())),
        ("height_mean", pa.float64()),
        ("height_std", pa.float64()),
        ("height_range", pa.float64()),
        ("normal_mean_xyz", pa.list_(pa.float64())),
        ("mcly_texture_ids", pa.list_(pa.int32())),
        ("mcly_texture_paths", pa.list_(pa.string())),
        ("tileset_variant_id", pa.string()),
        ("tileset_anomaly_candidate_ids", pa.list_(pa.string())),
        ("tileset_anomaly_ids", pa.list_(pa.string())),
        ("object_evidence_status", pa.string()),
        ("object_overlap", pa.float64()),
        ("object_distance_px", pa.float64()),
        ("object_asset_paths", pa.list_(pa.string())),
        ("object_instance_count", pa.int32()),
        ("object_spatial_signature", pa.string()),
        ("liquid_evidence_status", pa.string()),
        ("liquid_overlap", pa.float64()),
        ("evidence_completeness", pa.float64()),
        ("missing_evidence", pa.list_(pa.string())),
        ("source_store", pa.string()),
        ("source_canvas", pa.string()),
        ("source_canvas_sha256", pa.string()),
        ("source_store_evidence_sha256", pa.string()),
        ("source_adt_path", pa.string()),
        ("source_identity", pa.string()),
    ]
)


COVERAGE_SCHEMA = pa.schema(
    [
        ("schema_version", pa.string()),
        ("build", pa.string()),
        ("map", pa.string()),
        ("tile_id", pa.int64()),
        ("tile_x", pa.int32()),
        ("tile_y", pa.int32()),
        ("store_row", pa.int64()),
        ("clean_eligible", pa.bool_()),
        ("clean_reason", pa.string()),
        ("prefab_family_ids", pa.list_(pa.string())),
        ("placement_ids", pa.list_(pa.string())),
        ("transforms", pa.list_(pa.string())),
        ("tileset_variant_ids", pa.list_(pa.string())),
        ("tileset_anomaly_ids", pa.list_(pa.string())),
        ("arrangement_classes", pa.list_(pa.string())),
        ("coverage_tokens", pa.list_(pa.string())),
        ("coverage_weight", pa.float64()),
        ("evidence_completeness", pa.float64()),
        ("selected", pa.bool_()),
        ("selection_reason", pa.string()),
        ("representative_tile_key", pa.string()),
        ("split", pa.string()),
    ]
)


@dataclass(frozen=True, slots=True)
class PrefabCurationConfig:
    thumbnail_size: int = 16
    alpha_threshold: float = 0.05
    family_hamming_radius: int = 4
    neighbor_radii: tuple[float, ...] = (256.0, 1024.0, 4096.0)
    max_neighbors: int = 32
    global_tileset_rarity: float = 0.01
    local_tileset_rarity: float = 0.02
    min_family_tileset_support: int = 2
    max_selected_tiles: int | None = None
    background_per_map_regime: int = 1


@dataclass(slots=True)
class CanvasSource:
    build: str
    map_name: str
    path: Path
    root: zarr.Group
    layout: dict[str, Any]
    evidence_artifacts: dict[str, str]
    evidence_sha256: str


@dataclass(slots=True)
class StoreSource:
    path: Path
    root: zarr.Group
    index_rows: list[dict[str, Any]]
    decoded_metadata: dict[int, dict[str, Any]]
    source_metadata: dict[int, dict[str, Any]]
    placements: dict[int, list[dict[str, Any]]]
    evidence_artifacts: dict[str, str]
    evidence_sha256: str


@dataclass(frozen=True, slots=True)
class AlphaCropEvidence:
    """A real alpha crop or an explicit reason why no crop may be inferred."""

    values: np.ndarray | None
    status: str
    missing_reason: str


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: str | Path) -> str:
    """Hash a directory with framed relative paths and per-file content hashes."""
    root = Path(path)
    if root.is_file():
        return sha256_file(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    digest = hashlib.sha256(b"wowviewer-evidence-tree-v1\0")
    for file_path in sorted(value for value in root.rglob("*") if value.is_file()):
        relative = file_path.relative_to(root).as_posix().encode("utf-8")
        content_digest = bytes.fromhex(sha256_file(file_path))
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(content_digest)
    return digest.hexdigest()


def _artifact_evidence(root: Path, relative_paths: Sequence[str]) -> tuple[dict[str, str], str]:
    artifacts: dict[str, str] = {}
    for relative in relative_paths:
        artifact = root / relative
        artifacts[relative] = sha256_tree(artifact) if artifact.exists() else "missing"
    payload = json.dumps(artifacts, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return artifacts, hashlib.sha256(payload).hexdigest()


def stable_id(prefix: str, *parts: Any, length: int = 24) -> str:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return f"{prefix}_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:length]}"


def read_parquet_rows(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(path)
        rows.extend(pq.read_table(path).to_pylist())
    return rows


def derive_regions_for_empty_scopes(
    region_rows: list[dict[str, Any]],
    canvases: dict[tuple[str, str], CanvasSource],
    *,
    config: PrefabCurationConfig,
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    """Use the existing Spec 076 map segmenter only where a scope has zero catalog rows."""
    observed_scopes = {
        (
            str(row.get("build", "")),
            str(row.get("map_name", row.get("map", ""))),
        )
        for row in region_rows
        if str(row.get("build", ""))
        and str(row.get("map_name", row.get("map", "")))
    }
    combined = list(region_rows)
    derived_scopes: list[tuple[str, str]] = []
    for scope, canvas in sorted(canvases.items()):
        if scope in observed_scopes:
            continue
        if "alpha_256" not in canvas.root:
            raise ValueError(
                f"Region catalog is empty for {scope}, and {canvas.path} has no alpha_256 "
                "array for the map-wide fallback"
            )
        if "tile_id_256" not in canvas.root:
            raise ValueError(
                f"Region catalog is empty for {scope}, and {canvas.path} has no tile_id_256 "
                "provenance array; run the Spec 076 canvas/segmentation step first"
            )
        derived = segment_canvas_regions(
            canvas.root,
            threshold=float(config.alpha_threshold),
            min_area=16,
            min_atomic_footprint_px=8,
            curation_mode="default",
            max_regions_per_layer=None,
            catalog_rows=None,
        )
        if not derived:
            raise ValueError(
                f"Region catalog is empty for {scope}, and deterministic map-wide alpha "
                f"segmentation found no regions above threshold {config.alpha_threshold}; "
                "refusing to invent zero-mask pattern families"
            )
        for region in derived:
            row = asdict(region)
            row["region_evidence_source"] = "spec103_canvas_segmentation_fallback_v1"
            row["provenance"] = {
                **dict(row.get("provenance") or {}),
                "derived_by": "spec103_canvas_segmentation_fallback_v1",
                "canvas_path": str(canvas.path.resolve()),
                "canvas_evidence_sha256": canvas.evidence_sha256,
                "alpha_threshold": float(config.alpha_threshold),
                "min_area": 16,
            }
            combined.append(row)
        derived_scopes.append(scope)
    return combined, derived_scopes


def discover_canvas_sources(analysis_root: str | Path) -> dict[tuple[str, str], CanvasSource]:
    """Discover canvases by their own layout metadata, never directory-name parsing."""
    result: dict[tuple[str, str], CanvasSource] = {}
    for path in sorted(Path(analysis_root).rglob("canvas.zarr")):
        root = zarr.open_group(str(path), mode="r")
        layout = dict(root.attrs.get("layout", {}))
        build = str(layout.get("build", ""))
        map_name = str(layout.get("map_name", ""))
        if not build or not map_name:
            continue
        key = (build, map_name)
        existing = result.get(key)
        if existing is not None and existing.path != path:
            raise ValueError(f"Multiple canvases for build/map {key}: {existing.path} and {path}")
        artifacts, evidence_sha256 = _artifact_evidence(
            path,
            ("zarr.json", ".zgroup", ".zattrs", ".zmetadata", "alpha_256"),
        )
        result[key] = CanvasSource(
            build, map_name, path, root, layout, artifacts, evidence_sha256
        )
    return result


def discover_region_paths(analysis_root: str | Path) -> list[Path]:
    return _discover_authoritative_parquets(
        Path(analysis_root), ("fractal_regions.parquet", "regions.parquet")
    )


def discover_member_paths(analysis_root: str | Path) -> list[Path]:
    """Choose the highest-authority available member catalog.

    Raw near-component membership is intentionally not auto-selected.  It may
    still be passed explicitly for historical comparison.
    """
    return _discover_authoritative_parquets(
        Path(analysis_root),
        ("prefab_members.parquet", "fractal_region_members.parquet"),
    )


def _discover_authoritative_parquets(root: Path, names: Sequence[str]) -> list[Path]:
    """Prefer higher-authority files independently for each build/map scope."""
    selected: list[Path] = []
    covered_by_higher: set[tuple[str, str]] = set()
    for name in names:
        level_paths = sorted(root.rglob(name))
        level_scopes: set[tuple[str, str]] = set()
        for path in level_paths:
            scopes = _parquet_scope_keys(path)
            # Unknown scope cannot safely be suppressed; normalization still detects conflicts.
            if not scopes or scopes.difference(covered_by_higher):
                selected.append(path)
            level_scopes.update(scopes)
        covered_by_higher.update(level_scopes)
    return selected


def _parquet_scope_keys(path: Path) -> set[tuple[str, str]]:
    names = set(pq.ParquetFile(path).schema_arrow.names)
    build_column = "build" if "build" in names else None
    map_column = "map_name" if "map_name" in names else ("map" if "map" in names else None)
    if build_column is None or map_column is None:
        return set()
    table = pq.read_table(path, columns=[build_column, map_column]).to_pydict()
    return {
        (str(build), str(map_name))
        for build, map_name in zip(
            table[build_column], table[map_column], strict=True
        )
        if str(build) and str(map_name)
    }


def load_store_sources(store_paths: Iterable[str | Path]) -> tuple[list[StoreSource], list[dict[str, Any]]]:
    sources: list[StoreSource] = []
    combined: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for raw_path in store_paths:
        path = Path(raw_path)
        index_path = path / "index.parquet"
        if not index_path.exists():
            raise FileNotFoundError(f"Missing store index: {index_path}")
        root = zarr.open_group(str(path), mode="r")
        rows = pq.read_table(index_path).to_pylist()
        evidence_artifacts, evidence_sha256 = _artifact_evidence(
            path,
            (
                "zarr.json",
                ".zgroup",
                ".zattrs",
                ".zmetadata",
                "index.parquet",
                "decoded_metadata.parquet",
                "placements.parquet",
                "object_precise_mask",
                "liquid_mask",
                "mcly_texture_ids",
                "mcly_layer_mask",
            ),
        )
        index_sha256 = evidence_artifacts["index.parquet"]
        for row_index, raw in enumerate(rows):
            row = dict(raw)
            build = str(row.get("build") or path.stem.removesuffix(".zarr"))
            map_name = str(row.get("map", ""))
            tile_id = int(row.get("tile_id", row_index))
            key = (build, map_name, tile_id)
            if key in seen:
                raise ValueError(f"Duplicate store tile identity: {key}")
            seen.add(key)
            row.update(
                {
                    "build": build,
                    "map": map_name,
                    "tile_id": tile_id,
                    "store_row": int(row_index),
                    "source_store": str(path.resolve()),
                    "source_index_sha256": index_sha256,
                    "source_store_evidence_sha256": evidence_sha256,
                    "_store_source_index": len(sources),
                }
            )
            combined.append(row)
        decoded_metadata, source_metadata = _load_decoded_metadata(path)
        placements = _load_placements(path)
        sources.append(
            StoreSource(
                path.resolve(),
                root,
                rows,
                decoded_metadata,
                source_metadata,
                placements,
                evidence_artifacts,
                evidence_sha256,
            )
        )
    combined.sort(key=_tile_sort_key)
    return sources, combined


def canonical_alpha_signature(
    alpha: np.ndarray,
    *,
    size: int = 16,
    threshold: float = 0.05,
) -> tuple[str, str, str, np.ndarray]:
    """Return D4-invariant binary identity and the transform to canonical form."""
    normalized = _normalized_thumbnail(alpha, size=size)
    binary = normalized > float(threshold)
    candidates: list[tuple[bytes, int, str, np.ndarray]] = []
    for order, (name, transform) in enumerate(TRANSFORMS):
        variant = np.ascontiguousarray(transform(binary), dtype=np.uint8)
        packed = np.packbits(variant.reshape(-1)).tobytes()
        candidates.append((packed, order, name, variant))
    packed, _order, name, canonical = min(candidates, key=lambda item: (item[0], item[1]))
    bit_hex = packed.hex()
    digest = hashlib.sha256(packed).hexdigest()
    return f"alpha_{digest[:24]}", name, bit_hex, canonical


def cluster_d4_signatures(
    rows: list[dict[str, Any]],
    *,
    hamming_radius: int,
) -> None:
    """Assign stable families to rows lacking trusted upstream membership.

    Locality-sensitive bands generate candidates; actual grouping uses Hamming
    distance over canonical thumbnail bits.  Connected-component IDs are based
    on the smallest member hash and therefore independent of input order.
    """
    pending = [
        idx
        for idx, row in enumerate(rows)
        if not str(row.get("prefab_family_id", ""))
        and str(row.get("alpha_evidence_status", "present")) == "present"
    ]
    if not pending:
        return
    parent = {idx: idx for idx in pending}

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    bit_count = max(1, len(str(rows[pending[0]].get("canonical_alpha_bits", ""))) * 4)
    bands = max(1, int(hamming_radius) + 1)
    band_width = math.ceil(bit_count / bands)
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx in sorted(pending, key=lambda item: str(rows[item].get("canonical_alpha_bits", ""))):
        raw = str(rows[idx].get("canonical_alpha_bits", ""))
        if not raw:
            continue
        value = int(raw, 16)
        candidates: set[int] = set()
        for band in range(bands):
            shift = band * band_width
            width = min(band_width, bit_count - shift)
            if width <= 0:
                continue
            key = (band, (value >> shift) & ((1 << width) - 1))
            candidates.update(buckets[key])
        for other in candidates:
            other_value = int(str(rows[other]["canonical_alpha_bits"]), 16)
            if (value ^ other_value).bit_count() <= int(hamming_radius):
                union(idx, other)
        for band in range(bands):
            shift = band * band_width
            width = min(band_width, bit_count - shift)
            if width > 0:
                buckets[(band, (value >> shift) & ((1 << width) - 1))].append(idx)

    groups: dict[int, list[int]] = defaultdict(list)
    for idx in pending:
        groups[find(idx)].append(idx)
    for members in groups.values():
        hashes = sorted(str(rows[idx].get("canonical_alpha_hash", "")) for idx in members)
        family_id = stable_id("prefab_d4", hashes[0], len(members))
        for idx in members:
            rows[idx]["prefab_family_id"] = family_id
            rows[idx]["family_source"] = "d4_hamming_fallback"


def write_typed_parquet(path: str | Path, rows: list[dict[str, Any]], schema: pa.Schema) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    normalized = [{field.name: row.get(field.name) for field in schema} for row in rows]
    pq.write_table(pa.Table.from_pylist(normalized, schema=schema), output)


def normalize_placements(
    region_rows: list[dict[str, Any]],
    member_rows: list[dict[str, Any]],
    canvases: dict[tuple[str, str], CanvasSource],
    *,
    config: PrefabCurationConfig,
    included_labels: set[str] | frozenset[str] = DEFAULT_REGION_LABELS,
) -> list[dict[str, Any]]:
    """Merge region/member evidence and derive canonical D4 placement identity."""
    regions: dict[str, dict[str, Any]] = {}
    for raw in region_rows:
        region_id = str(raw.get("region_id", ""))
        if not region_id:
            continue
        normalized = _normalize_row(raw)
        previous = regions.get(region_id)
        if previous is not None and _region_conflict(previous, normalized):
            raise ValueError(f"Conflicting rows for region_id={region_id}")
        regions[region_id] = normalized

    bases: list[dict[str, Any]] = []
    regions_with_membership: set[str] = set()
    for member in member_rows:
        region_id = str(member.get("region_id", member.get("source_region_id", "")))
        merged = {**regions.get(region_id, {}), **_normalize_row(member)}
        merged["region_id"] = region_id
        bases.append(merged)
        if region_id:
            regions_with_membership.add(region_id)
    # A membership catalog is an annotation, never a filter. Preserve one-offs,
    # composites, rejected research states, and any other unmatched region rows.
    bases.extend(
        region for region_id, region in regions.items() if region_id not in regions_with_membership
    )

    deduped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for raw in bases:
        label = str(raw.get("curation_label", "unclassified"))
        if label in {"raw_component", "too_small_unique"}:
            continue
        if included_labels and label not in included_labels and label != "unclassified":
            continue
        build = str(raw.get("build", ""))
        map_name = str(raw.get("map_name", raw.get("map", "")))
        bbox = _bbox(raw.get("bbox_xywh", (0, 0, 0, 0)))
        layer_slot = int(raw.get("layer_slot", raw.get("layer_idx", 0)) or 0)
        layer_idx = int(raw.get("layer_idx", layer_slot) or layer_slot)
        if not build or not map_name or bbox[2] <= 0 or bbox[3] <= 0:
            continue
        region_id = str(raw.get("region_id") or stable_id("region", build, map_name, layer_idx, bbox))
        key = (build, map_name, region_id, layer_slot, bbox)
        if key in deduped:
            existing_family, _existing_source = _upstream_family(deduped[key])
            candidate_family, _candidate_source = _upstream_family(raw)
            if existing_family and candidate_family and existing_family != candidate_family:
                raise ValueError(
                    f"Conflicting prefab membership for region_id={region_id}: "
                    f"{existing_family!r} versus {candidate_family!r}"
                )
            continue
        canvas = canvases.get((build, map_name))
        alpha_evidence = _read_alpha_crop(canvas, bbox, layer_slot)
        if alpha_evidence.values is not None:
            alpha_hash, transform, bit_hex, canonical = canonical_alpha_signature(
                alpha_evidence.values,
                size=config.thumbnail_size,
                threshold=config.alpha_threshold,
            )
            occupancy, transition = multiscale_alpha_descriptor(
                canonical.astype(np.float32), levels=(2, 4, 8)
            )
        else:
            alpha_hash, transform, bit_hex = "", "", ""
            occupancy, transition = [], []
        upstream_family, family_source = _upstream_family(raw)
        missing_evidence = list(raw.get("missing_evidence", []) or [])
        if alpha_evidence.status != "present":
            missing_evidence.append(f"alpha:{alpha_evidence.status}")
        placement_id = stable_id(
            "placement", build, map_name, region_id, layer_idx, bbox, length=28
        )
        row = {
            **raw,
            "build": build,
            "map_name": map_name,
            "region_id": region_id,
            "region_evidence_source": str(
                raw.get("region_evidence_source") or "spec076_region_catalog"
            ),
            "placement_id": placement_id,
            "prefab_family_id": upstream_family,
            "family_source": family_source,
            "layer_slot": layer_slot,
            "layer_idx": layer_idx,
            "bbox_xywh": bbox,
            "canonical_alpha_hash": alpha_hash,
            "canonical_alpha_bits": bit_hex,
            "transform_to_canonical": transform,
            "alpha_evidence_status": alpha_evidence.status,
            "alpha_missing_reason": alpha_evidence.missing_reason,
            "multiscale_occupancy": occupancy,
            "multiscale_transition": transition,
            "source_canvas": str(canvas.path.resolve()) if canvas is not None else "",
            "source_canvas_sha256": canvas.evidence_sha256 if canvas is not None else "",
            "missing_evidence": sorted(set(missing_evidence)),
        }
        deduped[key] = row

    rows = sorted(
        deduped.values(),
        key=lambda item: (
            str(item["build"]),
            str(item["map_name"]),
            int(item["layer_idx"]),
            tuple(item["bbox_xywh"]),
            str(item["region_id"]),
        ),
    )
    cluster_d4_signatures(rows, hamming_radius=config.family_hamming_radius)
    counts = Counter(
        str(row["prefab_family_id"]) for row in rows if str(row["prefab_family_id"])
    )
    for row in rows:
        family = str(row["prefab_family_id"])
        row["family_member_count"] = int(counts[family]) if family else 0
    return rows


def multiscale_alpha_descriptor(
    alpha: np.ndarray, *, levels: Sequence[int] = (2, 4, 8)
) -> tuple[list[float], list[float]]:
    """Describe occupancy and transitions without depending on ADT page boundaries."""
    binary = np.asarray(alpha, dtype=np.float32) > 0.5
    occupancy: list[float] = []
    transitions: list[float] = []
    for level in levels:
        pooled = _pool_binary(binary, int(level))
        occupancy.extend(float(value) for value in pooled.reshape(-1))
        if pooled.size:
            horizontal = np.abs(np.diff(pooled, axis=1)).mean() if pooled.shape[1] > 1 else 0.0
            vertical = np.abs(np.diff(pooled, axis=0)).mean() if pooled.shape[0] > 1 else 0.0
            transitions.append(float((horizontal + vertical) * 0.5))
        else:
            transitions.append(0.0)
    return occupancy, transitions


def add_map_composition_features(
    placements: list[dict[str, Any]], *, config: PrefabCurationConfig
) -> None:
    """Attach map-global neighbour/cellular/containment features in canvas coordinates."""
    by_map: dict[tuple[str, str], list[int]] = defaultdict(list)
    for idx, row in enumerate(placements):
        by_map[(str(row["build"]), str(row["map_name"]))].append(idx)

    for indices in by_map.values():
        centers = np.asarray([_bbox_center(placements[idx]["bbox_xywh"]) for idx in indices])
        tree = cKDTree(centers) if len(indices) else None
        family_indices: dict[str, list[int]] = defaultdict(list)
        for local_idx, global_idx in enumerate(indices):
            family = str(placements[global_idx]["prefab_family_id"])
            if family:
                family_indices[family].append(local_idx)

        for local_idx, global_idx in enumerate(indices):
            row = placements[global_idx]
            center = centers[local_idx]
            radius = float(max(config.neighbor_radii))
            local_neighbors = (
                tree.query_ball_point(center, radius) if tree is not None else []
            )
            local_neighbors = [value for value in local_neighbors if value != local_idx]
            local_neighbors.sort(
                key=lambda value: (
                    float(np.linalg.norm(centers[value] - center)),
                    str(placements[indices[value]]["placement_id"]),
                )
            )
            local_neighbors = local_neighbors[: int(config.max_neighbors)]

            family = str(row["prefab_family_id"])
            same_local = (
                [value for value in family_indices[family] if value != local_idx]
                if family
                else []
            )
            same_local.sort(
                key=lambda value: (
                    float(np.linalg.norm(centers[value] - center)),
                    str(placements[indices[value]]["placement_id"]),
                )
            )
            same_local = same_local[: int(config.max_neighbors)]
            same_vectors = [centers[value] - center for value in same_local]
            ring_sector = _ring_sector_counts(same_vectors, config.neighbor_radii)
            arrangement = _arrangement_class(same_vectors)
            cellular_signature = stable_id(
                "cell", ring_sector, arrangement, _quantized_vectors(same_vectors), length=20
            )

            bbox = _bbox(row["bbox_xywh"])
            parents: list[str] = []
            children: list[str] = []
            cross_layer: list[str] = []
            for candidate_local in local_neighbors:
                candidate = placements[indices[candidate_local]]
                candidate_bbox = _bbox(candidate["bbox_xywh"])
                if _contains(candidate_bbox, bbox):
                    parents.append(str(candidate["placement_id"]))
                if _contains(bbox, candidate_bbox):
                    children.append(str(candidate["placement_id"]))
                if int(candidate["layer_idx"]) != int(row["layer_idx"]) and _bbox_distance(
                    bbox, candidate_bbox
                ) <= CHUNK_PIXEL_SIZE:
                    cross_layer.append(str(candidate["placement_id"]))

            row.update(
                {
                    "neighbor_placement_ids": [
                        str(placements[indices[value]]["placement_id"])
                        for value in local_neighbors
                    ],
                    "same_family_neighbor_ids": [
                        str(placements[indices[value]]["placement_id"])
                        for value in same_local
                    ],
                    "parent_placement_ids": sorted(set(parents)),
                    "child_placement_ids": sorted(set(children)),
                    "cross_layer_neighbor_ids": sorted(set(cross_layer)),
                    "cellular_ring_sector_counts": ring_sector,
                    "arrangement_class": arrangement,
                    "cellular_signature": cellular_signature,
                }
            )


def add_tileset_context(
    placements: list[dict[str, Any]],
    canvases: dict[tuple[str, str], CanvasSource],
    stores: list[StoreSource],
    tile_rows: list[dict[str, Any]],
    *,
    config: PrefabCurationConfig,
) -> None:
    """Resolve tile-local MTEX IDs to paths and retain rare copied-texture evidence."""
    cells, map_counts, map_totals = _build_map_texture_cells(canvases, stores, tile_rows)

    for row in placements:
        map_key = (str(row["build"]), str(row["map_name"]))
        x, y, w, h = _bbox(row["bbox_xywh"])
        x0, y0 = x // CHUNK_PIXEL_SIZE, y // CHUNK_PIXEL_SIZE
        x1 = max(x0 + 1, math.ceil((x + w) / CHUNK_PIXEL_SIZE))
        y1 = max(y0 + 1, math.ceil((y + h) / CHUNK_PIXEL_SIZE))
        texture_counts: Counter[str] = Counter()
        for cy in range(y0, y1):
            for cx in range(x0, x1):
                texture_counts.update(cells.get(map_key, {}).get((cx, cy), set()))
        textures = set(texture_counts)
        row["mcly_texture_paths"] = sorted(textures)
        row["mcly_texture_ids"] = sorted(
            int(value) for value in row.get("mcly_texture_ids", []) if int(value) >= 0
        )
        variant_parts = [f"{row['build']}:{value}" for value in row["mcly_texture_paths"]]
        if not variant_parts:
            variant_parts = [f"{row['build']}:id:{value}" for value in row["mcly_texture_ids"]]
        row["tileset_variant_id"] = stable_id("tileset", variant_parts, length=20)
        local_counts: Counter[str] = Counter()
        local_total = 0
        pad = 8
        for cy in range(y0 - pad, y1 + pad):
            for cx in range(x0 - pad, x1 + pad):
                if x0 <= cx < x1 and y0 <= cy < y1:
                    continue
                values = cells.get(map_key, {}).get((cx, cy), set())
                local_counts.update(values)
                local_total += len(values)
        row["_tileset_local_counts"] = local_counts
        row["_tileset_local_total"] = local_total

    family_texture_support: dict[tuple[str, str], int] = Counter()
    for row in placements:
        family = str(row["prefab_family_id"])
        if not family:
            continue
        for texture in set(row["mcly_texture_paths"]):
            family_texture_support[(family, texture)] += 1

    for row in placements:
        map_key = (str(row["build"]), str(row["map_name"]))
        total = max(1, int(map_totals[map_key]))
        candidates: list[str] = []
        confirmed: list[str] = []
        family = str(row["prefab_family_id"])
        for texture in row["mcly_texture_paths"]:
            prevalence = float(map_counts[map_key][texture]) / float(total)
            local_total = max(1, int(row["_tileset_local_total"]))
            local_prevalence = float(row["_tileset_local_counts"][texture]) / float(local_total)
            if prevalence <= float(config.global_tileset_rarity) or local_prevalence <= float(
                config.local_tileset_rarity
            ):
                candidates.append(texture)
                support = family_texture_support[(family, texture)] if family else 0
                if family and support >= int(config.min_family_tileset_support):
                    confirmed.append(texture)
        row["tileset_anomaly_candidate_ids"] = sorted(candidates)
        row["tileset_anomaly_ids"] = sorted(confirmed)


def explode_pattern_evidence_ledger(
    placements: list[dict[str, Any]],
    canvases: dict[tuple[str, str], CanvasSource],
    stores: list[StoreSource],
    tile_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Explode map placements to tile/chunk evidence without making tiles analysis units."""
    tiles_by_map: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in tile_rows:
        tiles_by_map[(str(row["build"]), str(row["map"]))].append(row)
    signal_cache = _TileSignalCache(stores)
    ledger: list[dict[str, Any]] = []

    for placement in placements:
        map_key = (str(placement["build"]), str(placement["map_name"]))
        canvas = canvases.get(map_key)
        if canvas is None:
            continue
        layout = canvas.layout
        min_x = int(layout.get("min_tile_x", 0))
        min_y = int(layout.get("min_tile_y", 0))
        bbox = _bbox(placement["bbox_xywh"])
        for tile in tiles_by_map.get(map_key, []):
            origin_x = (int(tile.get("tile_x", -1)) - min_x) * ALPHA_TILE_SIZE
            origin_y = (int(tile.get("tile_y", -1)) - min_y) * ALPHA_TILE_SIZE
            intersection = _intersection(bbox, (origin_x, origin_y, ALPHA_TILE_SIZE, ALPHA_TILE_SIZE))
            if intersection is None:
                continue
            ix, iy, iw, ih = intersection
            local = (ix - origin_x, iy - origin_y, iw, ih)
            chunk_keys = _chunk_keys(local)
            object_status, object_overlap, object_distance = signal_cache.mask_context(
                tile, "object_precise_mask", local
            )
            liquid_status, liquid_overlap, _ = signal_cache.mask_context(tile, "liquid_mask", local)
            source_index = int(tile["_store_source_index"])
            store = stores[source_index]
            placement_rows = store.placements.get(int(tile["tile_id"]), [])
            related_placement_rows, object_spatial_signature = _object_spatial_context(
                placement_rows,
                tile_x=int(tile.get("tile_x", -1)),
                tile_y=int(tile.get("tile_y", -1)),
                local_bbox=local,
                max_distance_px=CHUNK_PIXEL_SIZE,
            )
            object_assets = sorted(
                {
                    str(value.get("asset_path", ""))
                    for value in related_placement_rows
                    if value.get("asset_path")
                }
            )
            missing = list(placement.get("missing_evidence", []))
            if object_status == "missing":
                missing.append("object_mask")
            if liquid_status == "missing":
                missing.append("liquid_mask")
            if not placement.get("mcly_texture_paths"):
                missing.append("tileset_paths")
            metadata = store.source_metadata.get(int(tile["tile_id"]), {})
            expected = 6
            completeness = float(expected - min(expected, len(set(missing)))) / float(expected)
            pixels = int(iw * ih)
            row = {
                "schema_version": EVIDENCE_SCHEMA_VERSION,
                "evidence_state": (
                    "recovered_evidence"
                    if placement.get("alpha_evidence_status") == "present"
                    else "missing_alpha_evidence"
                ),
                "build": map_key[0],
                "map": map_key[1],
                "tile_id": int(tile["tile_id"]),
                "tile_x": int(tile.get("tile_x", -1)),
                "tile_y": int(tile.get("tile_y", -1)),
                "store_row": int(tile["store_row"]),
                "chunk_keys": chunk_keys,
                "region_pixel_count_in_tile": pixels,
                "tile_fraction": float(pixels) / float(ALPHA_TILE_SIZE * ALPHA_TILE_SIZE),
                "placement_id": str(placement["placement_id"]),
                "prefab_family_id": str(placement["prefab_family_id"]),
                "family_source": str(placement["family_source"]),
                "source_region_id": str(placement["region_id"]),
                "region_evidence_source": str(
                    placement.get("region_evidence_source", "spec076_region_catalog")
                ),
                "layer_slot": int(placement["layer_slot"]),
                "layer_idx": int(placement["layer_idx"]),
                "curation_label": str(placement.get("curation_label", "unclassified")),
                "bbox_x": int(bbox[0]),
                "bbox_y": int(bbox[1]),
                "bbox_w": int(bbox[2]),
                "bbox_h": int(bbox[3]),
                "crosses_adt_boundary": _crosses_adt_boundary(bbox),
                "transform_to_canonical": str(placement["transform_to_canonical"]),
                "canonical_alpha_hash": str(placement["canonical_alpha_hash"]),
                "alpha_evidence_status": str(placement.get("alpha_evidence_status", "")),
                "alpha_missing_reason": str(placement.get("alpha_missing_reason", "")),
                "multiscale_occupancy": [float(value) for value in placement["multiscale_occupancy"]],
                "multiscale_transition": [float(value) for value in placement["multiscale_transition"]],
                "arrangement_class": str(placement.get("arrangement_class", "isolated")),
                "cellular_signature": str(placement.get("cellular_signature", "")),
                "cellular_ring_sector_counts": [int(value) for value in placement.get("cellular_ring_sector_counts", [])],
                "same_family_neighbor_ids": list(placement.get("same_family_neighbor_ids", [])),
                "neighbor_placement_ids": list(placement.get("neighbor_placement_ids", [])),
                "parent_placement_ids": list(placement.get("parent_placement_ids", [])),
                "child_placement_ids": list(placement.get("child_placement_ids", [])),
                "height_mean": _optional_float(placement.get("height_mean")),
                "height_std": _optional_float(placement.get("height_std")),
                "height_range": _optional_float(placement.get("height_range")),
                "normal_mean_xyz": _float_list(placement.get("normal_mean_xyz", [])),
                "mcly_texture_ids": [int(value) for value in placement.get("mcly_texture_ids", [])],
                "mcly_texture_paths": list(placement.get("mcly_texture_paths", [])),
                "tileset_variant_id": str(placement.get("tileset_variant_id", "")),
                "tileset_anomaly_candidate_ids": list(placement.get("tileset_anomaly_candidate_ids", [])),
                "tileset_anomaly_ids": list(placement.get("tileset_anomaly_ids", [])),
                "object_evidence_status": object_status,
                "object_overlap": object_overlap,
                "object_distance_px": object_distance,
                "object_asset_paths": object_assets,
                "object_instance_count": int(len(related_placement_rows)),
                "object_spatial_signature": object_spatial_signature,
                "liquid_evidence_status": liquid_status,
                "liquid_overlap": liquid_overlap,
                "evidence_completeness": completeness,
                "missing_evidence": sorted(set(missing)),
                "source_store": str(tile["source_store"]),
                "source_canvas": str(canvas.path.resolve()),
                "source_canvas_sha256": canvas.evidence_sha256,
                "source_store_evidence_sha256": store.evidence_sha256,
                "source_adt_path": str(metadata.get("source_adt_path", "")),
                "source_identity": stable_id(
                    "source",
                    tile["source_index_sha256"],
                    store.evidence_sha256,
                    canvas.evidence_sha256,
                    metadata.get("source_adt_path", ""),
                    placement["region_id"],
                    object_spatial_signature,
                ),
            }
            ledger.append(row)
    ledger.sort(key=lambda row: (_tile_sort_key(row), row["placement_id"], row["layer_idx"]))
    return ledger


def aggregate_tile_pattern_coverage(
    ledger: list[dict[str, Any]],
    tile_rows: list[dict[str, Any]],
    clean_rows: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Create one deterministic tile row while preserving memberships in the ledger."""
    evidence_by_tile: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in ledger:
        evidence_by_tile[(str(row["build"]), str(row["map"]), int(row["tile_id"]))].append(row)

    clean_by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    if clean_rows is not None:
        for row in clean_rows:
            clean_by_key[
                (str(row.get("build", "")), str(row.get("map", "")), int(row["tile_id"]))
            ] = row

    coverage: list[dict[str, Any]] = []
    for tile in tile_rows:
        key = (str(tile["build"]), str(tile["map"]), int(tile["tile_id"]))
        rows = evidence_by_tile.get(key, [])
        clean = clean_by_key.get(key)
        clean_eligible = bool(clean.get("keep", False)) if clean_rows is not None and clean else clean_rows is None
        clean_reason = (
            str(clean.get("reason", "kept"))
            if clean is not None
            else ("not_in_clean_manifest" if clean_rows is not None else "eligible_no_clean_manifest")
        )
        families = sorted(
            {str(row["prefab_family_id"]) for row in rows if str(row["prefab_family_id"])}
        )
        placements = sorted({str(row["placement_id"]) for row in rows})
        transforms = sorted({str(row["transform_to_canonical"]) for row in rows})
        variants = sorted({str(row["tileset_variant_id"]) for row in rows if row["tileset_variant_id"]})
        anomalies = sorted({value for row in rows for value in row["tileset_anomaly_ids"]})
        arrangements = sorted({str(row["arrangement_class"]) for row in rows})
        tokens: set[str] = set()
        for row in rows:
            family = str(row["prefab_family_id"])
            if not family:
                tokens.add(
                    "alpha-state:"
                    f"{row['build']}:{row['map']}:{int(row['layer_idx'])}:"
                    f"{row.get('alpha_evidence_status', 'unknown')}"
                )
                continue
            tokens.add(f"family:{family}")
            transform = str(row["transform_to_canonical"])
            if transform and transform != "identity":
                tokens.add(f"transform:{family}:{transform}")
            variant = str(row["tileset_variant_id"])
            if variant:
                tokens.add(f"tileset:{family}:{variant}")
            arrangement = str(row["arrangement_class"])
            tokens.add(f"composition:{family}:{arrangement}")
            relief = _relief_bucket(row.get("height_std"))
            tokens.add(f"relief:{family}:{relief}")
            for anomaly in row["tileset_anomaly_ids"]:
                tokens.add(f"anomaly:{family}:{anomaly}")
            if int(row.get("object_instance_count", 0)) > 0:
                tokens.add(f"objects:{family}:{row.get('object_spatial_signature', '')}")
        if not tokens:
            regime = str((clean or {}).get("height_regime", "unknown"))
            tokens.add(f"background:{key[0]}:{key[1]}:{regime}")
        completeness = float(np.mean([row["evidence_completeness"] for row in rows])) if rows else 0.0
        coverage.append(
            {
                "schema_version": EVIDENCE_SCHEMA_VERSION,
                "build": key[0],
                "map": key[1],
                "tile_id": key[2],
                "tile_x": int(tile.get("tile_x", -1)),
                "tile_y": int(tile.get("tile_y", -1)),
                "store_row": int(tile["store_row"]),
                "clean_eligible": clean_eligible,
                "clean_reason": clean_reason,
                "prefab_family_ids": families,
                "placement_ids": placements,
                "transforms": transforms,
                "tileset_variant_ids": variants,
                "tileset_anomaly_ids": anomalies,
                "arrangement_classes": arrangements,
                "coverage_tokens": sorted(tokens),
                "coverage_weight": float(sum(_token_weight(token) for token in tokens)),
                "evidence_completeness": completeness,
                "selected": False,
                "selection_reason": "not_selected",
                "representative_tile_key": "",
                "split": "excluded",
            }
        )
    coverage.sort(key=_tile_sort_key)
    return coverage


def select_representative_tiles(
    coverage: list[dict[str, Any]],
    *,
    config: PrefabCurationConfig,
    val_maps: set[str] | None = None,
) -> tuple[set[int], set[str]]:
    """Deterministic set cover while retaining every eligible complete-holdout page."""
    validation_maps = set(val_maps or set())
    eligible = [idx for idx, row in enumerate(coverage) if bool(row["clean_eligible"])]
    required_tokens = {token for idx in eligible for token in coverage[idx]["coverage_tokens"]}
    reserved = {
        idx for idx in eligible if str(coverage[idx]["map"]) in validation_maps
    }
    selected: list[int] = sorted(reserved, key=lambda idx: _tile_sort_key(coverage[idx]))
    uncovered = set(required_tokens).difference(
        token for idx in selected for token in coverage[idx]["coverage_tokens"]
    )
    budget = config.max_selected_tiles
    while uncovered and (budget is None or len(selected) < int(budget)):
        best: tuple[Any, ...] | None = None
        best_idx: int | None = None
        for idx in eligible:
            if idx in selected:
                continue
            new_tokens = uncovered.intersection(coverage[idx]["coverage_tokens"])
            if not new_tokens:
                continue
            score = (
                -sum(_token_weight(token) for token in new_tokens),
                -sum(1 for token in new_tokens if token.startswith("family:")),
                -float(coverage[idx]["evidence_completeness"]),
                *_tile_sort_key(coverage[idx]),
            )
            if best is None or score < best:
                best = score
                best_idx = idx
        if best_idx is None:
            break
        selected.append(best_idx)
        uncovered.difference_update(coverage[best_idx]["coverage_tokens"])

    # Remove any tile whose tokens all remain covered elsewhere.
    token_counts = Counter(token for idx in selected for token in coverage[idx]["coverage_tokens"])
    for idx in reversed(selected.copy()):
        if idx in reserved:
            continue
        tokens = coverage[idx]["coverage_tokens"]
        if tokens and all(token_counts[token] > 1 for token in tokens):
            selected.remove(idx)
            for token in tokens:
                token_counts[token] -= 1

    selected_set = set(selected)
    for idx, row in enumerate(coverage):
        if not row["clean_eligible"]:
            row["selected"] = False
            row["selection_reason"] = f"clean_filter:{row['clean_reason']}"
            continue
        if idx in selected_set:
            row["selected"] = True
            row["selection_reason"] = (
                "complete_map_holdout" if idx in reserved else "representative_coverage"
            )
            row["representative_tile_key"] = _tile_key_text(row)
            continue
        best_rep = _best_representative(idx, selected_set, coverage)
        row["selected"] = False
        if best_rep is not None:
            row["selection_reason"] = "duplicate_coverage"
            row["representative_tile_key"] = _tile_key_text(coverage[best_rep])
        else:
            row["selection_reason"] = "budget_exhausted" if uncovered else "unrepresented"
    return selected_set, uncovered


def assign_group_safe_splits(
    coverage: list[dict[str, Any]], *, val_maps: set[str]
) -> dict[str, Any]:
    """Split selected tile<->prefab connected components without leakage."""
    selected = [idx for idx, row in enumerate(coverage) if bool(row["selected"])]
    parent = {idx: idx for idx in selected}

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    first_by_family: dict[str, int] = {}
    for idx in selected:
        for family in coverage[idx]["prefab_family_ids"]:
            previous = first_by_family.setdefault(str(family), idx)
            union(idx, previous)

    components: dict[int, list[int]] = defaultdict(list)
    for idx in selected:
        components[find(idx)].append(idx)
    for members in components.values():
        split = "val" if any(str(coverage[idx]["map"]) in val_maps for idx in members) else "train"
        for idx in members:
            coverage[idx]["split"] = split
    for idx, row in enumerate(coverage):
        if idx not in parent:
            row["split"] = "excluded"

    family_splits: dict[str, set[str]] = defaultdict(set)
    for idx in selected:
        for family in coverage[idx]["prefab_family_ids"]:
            family_splits[str(family)].add(str(coverage[idx]["split"]))
    leakage = [family for family, splits in family_splits.items() if len(splits) != 1]
    holdout_misses = [
        _tile_key_text(coverage[idx])
        for idx, row in enumerate(coverage)
        if str(row["map"]) in val_maps
        and bool(row["clean_eligible"])
        and (not bool(row["selected"]) or str(row["split"]) != "val")
    ]
    if leakage or holdout_misses:
        raise ValueError(
            f"Prefab split audit failed: leakage={leakage[:8]} holdout_misses={holdout_misses[:8]}"
        )
    return {
        "component_count": len(components),
        "selected_count": len(selected),
        "family_count": len(family_splits),
        "family_leakage_count": len(leakage),
        "holdout_miss_count": len(holdout_misses),
        "holdout_eligible_count": sum(
            bool(row["clean_eligible"]) and str(row["map"]) in val_maps for row in coverage
        ),
        "holdout_selected_count": sum(
            bool(row["selected"]) and str(row["map"]) in val_maps for row in coverage
        ),
        "split_counts": dict(Counter(coverage[idx]["split"] for idx in selected)),
    }


def build_curation_manifest(
    coverage: list[dict[str, Any]], clean_rows: list[dict[str, Any]] | None
) -> list[dict[str, Any]]:
    clean_by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in clean_rows or []:
        clean_by_key[(str(row.get("build", "")), str(row.get("map", "")), int(row["tile_id"]))] = row
    manifest: list[dict[str, Any]] = []
    for row in coverage:
        key = (str(row["build"]), str(row["map"]), int(row["tile_id"]))
        base = dict(clean_by_key.get(key, {}))
        base.update(
            {
                "tile_id": key[2],
                "build": key[0],
                "map": key[1],
                "tile_x": int(row["tile_x"]),
                "tile_y": int(row["tile_y"]),
                "keep": bool(row["selected"]),
                "reason": str(row["selection_reason"]),
                "partition": str(row["split"]),
                "prefab_family_ids": list(row["prefab_family_ids"]),
                "placement_ids": list(row["placement_ids"]),
                "coverage_tokens": list(row["coverage_tokens"]),
                "representative_tile_key": str(row["representative_tile_key"]),
                "evidence_completeness": float(row["evidence_completeness"]),
                "schema_version": EVIDENCE_SCHEMA_VERSION,
            }
        )
        manifest.append(base)
    manifest.sort(key=_tile_sort_key)
    validate_manifest_rows(manifest)
    return manifest


def validate_manifest_rows(rows: list[dict[str, Any]]) -> None:
    required = {"tile_id", "build", "map", "tile_x", "tile_y", "keep", "reason", "partition"}
    seen: set[tuple[str, str, int]] = set()
    for row in rows:
        missing = sorted(required.difference(row))
        if missing:
            raise ValueError(f"Curation manifest row missing {missing}")
        key = (str(row["build"]), str(row["map"]), int(row["tile_id"]))
        if key in seen:
            raise ValueError(f"Duplicate curation manifest tile: {key}")
        seen.add(key)
        if bool(row["keep"]) and str(row["partition"]) not in {"train", "val", "test"}:
            raise ValueError(f"Kept tile {key} has invalid partition={row['partition']!r}")


def resolve_manifest_rows(
    index_rows: list[dict[str, Any]],
    manifest_rows: list[dict[str, Any]],
    *,
    val_key: str,
    val_value: str,
) -> tuple[list[int], list[int], str]:
    """Resolve trainer row positions using explicit partitions when available."""
    manifest_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    family_partitions: dict[str, set[str]] = defaultdict(set)
    for row in manifest_rows:
        key = (str(row.get("build", "")), int(row["tile_id"]))
        if key in manifest_by_key:
            raise ValueError(f"Duplicate runtime curation-manifest key: {key}")
        manifest_by_key[key] = row
        if not bool(row.get("keep", False)):
            continue
        partition = str(row.get("partition", ""))
        for family in row.get("prefab_family_ids", []) or []:
            family_partitions[str(family)].add(partition)
    leakage = sorted(family for family, splits in family_partitions.items() if len(splits) > 1)
    if leakage:
        raise ValueError(
            "Prefab-family partition leakage in curation manifest: "
            + ", ".join(leakage[:8])
        )
    has_partition = any("partition" in row and str(row.get("partition")) for row in manifest_rows)
    train: list[int] = []
    val: list[int] = []
    for row_index, row in enumerate(index_rows):
        entry = manifest_by_key.get((str(row.get("build", "")), int(row["tile_id"])))
        if entry is None or not bool(entry.get("keep", False)):
            continue
        if has_partition:
            partition = str(entry.get("partition", "excluded"))
            if partition == "train":
                train.append(row_index)
            elif partition == "val":
                val.append(row_index)
        elif str(row.get(val_key, "")) == str(val_value):
            val.append(row_index)
        else:
            train.append(row_index)
    mode = "manifest_partition" if has_partition else f"legacy_holdout:{val_key}={val_value}"
    validate_source_group_split(index_rows, train, val)
    return train, val, mode


def validate_source_group_split(
    index_rows: list[dict[str, Any]],
    train_rows: Sequence[int],
    val_rows: Sequence[int],
) -> None:
    """Fail closed when time/color variants of one terrain source cross a holdout."""
    partitions: dict[str, set[str]] = defaultdict(set)
    for partition, positions in (("train", train_rows), ("val", val_rows)):
        for position in positions:
            row = index_rows[int(position)]
            source_group_id = str(row.get("source_group_id") or "").strip()
            if source_group_id:
                partitions[source_group_id].add(partition)
    leakage = sorted(group for group, values in partitions.items() if len(values) > 1)
    if leakage:
        raise ValueError(
            "Source-group partition leakage (time/color variants must stay together): "
            + ", ".join(leakage[:8])
        )


def run_prefab_curation(
    *,
    store_paths: Sequence[str | Path],
    analysis_root: str | Path,
    output_dir: str | Path,
    region_paths: Sequence[str | Path] = (),
    member_paths: Sequence[str | Path] = (),
    clean_manifest: str | Path | Sequence[str | Path] | None = None,
    val_maps: set[str] | None = None,
    config: PrefabCurationConfig | None = None,
) -> dict[str, Any]:
    """Run the complete CPU-side evidence and selection pipeline."""
    config = config or PrefabCurationConfig()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    analysis = Path(analysis_root)
    canvases = discover_canvas_sources(analysis)
    if not canvases:
        raise FileNotFoundError(f"No Spec 076 canvas.zarr found under {analysis}")
    resolved_regions = [Path(path) for path in region_paths] or discover_region_paths(analysis)
    if not resolved_regions:
        raise FileNotFoundError(f"No fractal region Parquet found under {analysis}")
    resolved_members = [Path(path) for path in member_paths] or discover_member_paths(analysis)
    region_rows = read_parquet_rows(resolved_regions)
    member_rows = read_parquet_rows(resolved_members)
    stores, tile_rows = load_store_sources(store_paths)
    clean_manifest_paths = _resolve_clean_manifest_paths(clean_manifest)
    clean_rows = _load_clean_manifest(clean_manifest_paths) if clean_manifest_paths else None
    region_rows, derived_region_scopes = derive_regions_for_empty_scopes(
        region_rows, canvases, config=config
    )

    placements = normalize_placements(region_rows, member_rows, canvases, config=config)
    if not placements:
        raise ValueError("No eligible prefab placements were produced")
    add_map_composition_features(placements, config=config)
    add_tileset_context(placements, canvases, stores, tile_rows, config=config)
    ledger = explode_pattern_evidence_ledger(placements, canvases, stores, tile_rows)
    coverage = aggregate_tile_pattern_coverage(ledger, tile_rows, clean_rows)
    validation_maps = set(val_maps or set())
    selected, uncovered = select_representative_tiles(
        coverage, config=config, val_maps=validation_maps
    )
    split_audit = assign_group_safe_splits(coverage, val_maps=validation_maps)
    manifest = build_curation_manifest(coverage, clean_rows)

    ledger_path = output / "pattern_evidence_ledger.parquet"
    coverage_path = output / "tile_pattern_coverage.parquet"
    manifest_path = output / "curation_manifest.parquet"
    write_typed_parquet(ledger_path, ledger, LEDGER_SCHEMA)
    write_typed_parquet(coverage_path, coverage, COVERAGE_SCHEMA)
    pq.write_table(pa.Table.from_pylist(manifest), manifest_path)

    source_files = [*resolved_regions, *resolved_members, *clean_manifest_paths]
    source_identities = {
        str(path.resolve()): sha256_file(path) for path in source_files if path.exists()
    }
    summary = {
        "schema": EVIDENCE_SCHEMA_VERSION,
        "analysis_root": str(analysis.resolve()),
        "stores": [
            {
                "path": str(source.path),
                "index_sha256": source.evidence_artifacts["index.parquet"],
                "evidence_sha256": source.evidence_sha256,
                "evidence_artifacts": source.evidence_artifacts,
                "tile_count": len(source.index_rows),
            }
            for source in stores
        ],
        "canvases": [
            {
                "build": canvas.build,
                "map": canvas.map_name,
                "path": str(canvas.path.resolve()),
                "evidence_sha256": canvas.evidence_sha256,
                "evidence_artifacts": canvas.evidence_artifacts,
            }
            for _key, canvas in sorted(canvases.items())
        ],
        "source_artifacts": source_identities,
        "region_count": len(region_rows),
        "derived_region_count": sum(
            str(row.get("region_evidence_source", ""))
            == "spec103_canvas_segmentation_fallback_v1"
            for row in region_rows
        ),
        "derived_region_scopes": [list(scope) for scope in derived_region_scopes],
        "member_count": len(member_rows),
        "placement_count": len(placements),
        "prefab_family_count": len(
            {row["prefab_family_id"] for row in placements if row["prefab_family_id"]}
        ),
        "ledger_row_count": len(ledger),
        "eligible_tile_count": sum(bool(row["clean_eligible"]) for row in coverage),
        "selected_tile_count": len(selected),
        "excluded_tile_count": len(coverage) - len(selected),
        "uncovered_tokens": sorted(uncovered),
        "split_audit": split_audit,
        "validation_maps": sorted(validation_maps),
        "config": {
            "thumbnail_size": config.thumbnail_size,
            "alpha_threshold": config.alpha_threshold,
            "family_hamming_radius": config.family_hamming_radius,
            "neighbor_radii": list(config.neighbor_radii),
            "max_neighbors": config.max_neighbors,
            "global_tileset_rarity": config.global_tileset_rarity,
            "local_tileset_rarity": config.local_tileset_rarity,
            "min_family_tileset_support": config.min_family_tileset_support,
            "max_selected_tiles": config.max_selected_tiles,
            "background_per_map_regime": config.background_per_map_regime,
        },
        "outputs": {
            "pattern_evidence_ledger": {
                "path": str(ledger_path),
                "sha256": sha256_file(ledger_path),
            },
            "tile_pattern_coverage": {
                "path": str(coverage_path),
                "sha256": sha256_file(coverage_path),
            },
            "curation_manifest": {
                "path": str(manifest_path),
                "sha256": sha256_file(manifest_path),
            },
        },
    }
    (output / "curation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


class _TileSignalCache:
    def __init__(self, stores: list[StoreSource], *, max_items: int = 64) -> None:
        self.stores = stores
        self.max_items = max_items
        self.cache: OrderedDict[tuple[int, int, str], tuple[np.ndarray, np.ndarray | None]] = OrderedDict()

    def mask_context(
        self, tile: dict[str, Any], array_name: str, local_bbox: tuple[int, int, int, int]
    ) -> tuple[str, float | None, float | None]:
        source_idx = int(tile["_store_source_index"])
        store_row = int(tile["store_row"])
        source = self.stores[source_idx]
        if array_name not in source.root:
            return "missing", None, None
        cache_key = (source_idx, store_row, array_name)
        cached = self.cache.get(cache_key)
        if cached is None:
            mask = np.asarray(source.root[array_name][store_row]) > 0.5
            distance = distance_transform_edt(~mask) if bool(mask.any()) else None
            self.cache[cache_key] = (mask, distance)
            self.cache.move_to_end(cache_key)
            while len(self.cache) > self.max_items:
                self.cache.popitem(last=False)
        else:
            self.cache.move_to_end(cache_key)
            mask, distance = cached
        x, y, w, h = local_bbox
        y1 = min(mask.shape[0], y + h)
        x1 = min(mask.shape[1], x + w)
        crop = mask[y:y1, x:x1]
        overlap = float(crop.mean()) if crop.size else 0.0
        if distance is None:
            return "present_zero", overlap, None
        distance_crop = distance[y:y1, x:x1]
        return "present", overlap, float(distance_crop.min()) if distance_crop.size else None


def _load_decoded_metadata(path: Path) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    decoded: dict[int, dict[str, Any]] = {}
    sources: dict[int, dict[str, Any]] = {}
    metadata_path = path / "decoded_metadata.parquet"
    if not metadata_path.exists():
        return decoded, sources
    for row in pq.read_table(metadata_path).to_pylist():
        tile_id = int(row.get("tile_id", -1))
        if tile_id < 0:
            continue
        payload: dict[str, Any] = {}
        try:
            payload = json.loads(str(row.get("decoded_metadata_json", "{}")))
        except (json.JSONDecodeError, TypeError):
            payload = {}
        decoded[tile_id] = payload
        sources[tile_id] = {
            "source_adt_path": str(row.get("source_adt_path", "")),
            "source_wdt_path": str(row.get("source_wdt_path", "")),
            "tile_name": str(row.get("tile_name", "")),
        }
    return decoded, sources


def _load_placements(path: Path) -> dict[int, list[dict[str, Any]]]:
    result: dict[int, list[dict[str, Any]]] = defaultdict(list)
    placement_path = path / "placements.parquet"
    if not placement_path.exists():
        return result
    for row in pq.read_table(placement_path).to_pylist():
        result[int(row.get("tile_id", -1))].append(row)
    for rows in result.values():
        rows.sort(key=lambda row: (str(row.get("asset_path", "")), int(row.get("instance_idx", -1))))
    return result


def _load_clean_manifest(
    path: str | Path | Sequence[str | Path] | None,
) -> list[dict[str, Any]] | None:
    if path is None:
        return None
    paths = [path] if isinstance(path, (str, Path)) else list(path)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for value in paths:
        resolved = Path(value)
        if resolved.is_dir():
            resolved = resolved / "curation_manifest.parquet"
        if not resolved.exists():
            raise FileNotFoundError(resolved)
        for row in pq.read_table(resolved).to_pylist():
            key = (
                str(row.get("build", "")),
                str(row.get("map", row.get("map_name", ""))),
                int(row["tile_id"]),
            )
            if key in seen:
                raise ValueError(f"Duplicate clean-manifest tile across inputs: {key}")
            seen.add(key)
            rows.append(row)
    return rows


def _resolve_clean_manifest_paths(
    path: str | Path | Sequence[str | Path] | None,
) -> list[Path]:
    if path is None:
        return []
    values = [path] if isinstance(path, (str, Path)) else list(path)
    resolved_paths: list[Path] = []
    for value in values:
        resolved = Path(value)
        if resolved.is_dir():
            resolved = resolved / "curation_manifest.parquet"
        if not resolved.exists():
            raise FileNotFoundError(resolved)
        resolved_paths.append(resolved)
    return resolved_paths


def _read_alpha_crop(
    source: CanvasSource | None,
    bbox: tuple[int, int, int, int],
    layer_slot: int,
) -> AlphaCropEvidence:
    if source is None:
        return AlphaCropEvidence(None, "missing_canvas", "no build/map canvas was discovered")
    if "alpha_256" not in source.root:
        return AlphaCropEvidence(
            None, "missing_alpha_array", "canvas has no alpha_256 array"
        )
    x, y, w, h = bbox
    array = source.root["alpha_256"]
    if len(array.shape) != 3:
        return AlphaCropEvidence(
            None,
            "invalid_alpha_shape",
            f"alpha_256 must be HxWxL, found {tuple(int(v) for v in array.shape)}",
        )
    if layer_slot < 0 or layer_slot >= int(array.shape[2]):
        return AlphaCropEvidence(
            None,
            "missing_layer_slot",
            f"layer slot {layer_slot} is outside 0..{int(array.shape[2]) - 1}",
        )
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > int(array.shape[1]) or y + h > int(
        array.shape[0]
    ):
        return AlphaCropEvidence(
            None,
            "invalid_alpha_bbox",
            f"bbox {bbox} is outside alpha canvas {tuple(int(v) for v in array.shape[:2])}",
        )
    max_samples = 64
    step = max(1, math.ceil(max(w, h) / max_samples))
    crop = np.asarray(array[y : y + h : step, x : x + w : step, layer_slot], dtype=np.float32)
    if crop.size == 0:
        return AlphaCropEvidence(None, "empty_alpha_crop", f"bbox {bbox} produced no samples")
    return AlphaCropEvidence(crop, "present", "")


def _upstream_family(row: dict[str, Any]) -> tuple[str, str]:
    for field, source in (
        ("prefab_family_id", "upstream_prefab_family"),
        ("family_id", "upstream_family_candidate"),
        ("cluster_id", "upstream_cluster_candidate"),
    ):
        value = str(row.get(field, ""))
        if value:
            return value, source
    return "", ""


def _build_map_texture_cells(
    canvases: dict[tuple[str, str], CanvasSource],
    stores: list[StoreSource],
    tile_rows: list[dict[str, Any]],
) -> tuple[
    dict[tuple[str, str], dict[tuple[int, int], set[str]]],
    dict[tuple[str, str], Counter[str]],
    Counter[tuple[str, str]],
]:
    cells: dict[tuple[str, str], dict[tuple[int, int], set[str]]] = defaultdict(dict)
    counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    totals: Counter[tuple[str, str]] = Counter()
    for tile in tile_rows:
        map_key = (str(tile["build"]), str(tile["map"]))
        canvas = canvases.get(map_key)
        if canvas is None:
            continue
        source = stores[int(tile["_store_source_index"])]
        if "mcly_texture_ids" not in source.root or "mcly_layer_mask" not in source.root:
            continue
        row_index = int(tile["store_row"])
        ids = np.asarray(source.root["mcly_texture_ids"][row_index], dtype=np.int32)
        mask = np.asarray(source.root["mcly_layer_mask"][row_index]) > 0
        names = [
            str(value).strip().lower().replace("\\", "/")
            for value in source.decoded_metadata.get(int(tile["tile_id"]), {}).get(
                "mcly_texture_names", []
            )
        ]
        min_x = int(canvas.layout.get("min_tile_x", 0))
        min_y = int(canvas.layout.get("min_tile_y", 0))
        origin_x = (int(tile["tile_x"]) - min_x) * 16
        origin_y = (int(tile["tile_y"]) - min_y) * 16
        for cy in range(min(16, ids.shape[0])):
            for cx in range(min(16, ids.shape[1])):
                active_paths: set[str] = set()
                for layer in range(min(4, ids.shape[2])):
                    if not bool(mask[cy, cx, layer]):
                        continue
                    texture_id = int(ids[cy, cx, layer])
                    if 0 <= texture_id < len(names) and names[texture_id]:
                        active_paths.add(f"{map_key[0]}:{names[texture_id]}")
                    elif texture_id >= 0:
                        active_paths.add(
                            f"{map_key[0]}:unresolved-tile-{int(tile['tile_id'])}-id-{texture_id}"
                        )
                if not active_paths:
                    continue
                coord = (origin_x + cx, origin_y + cy)
                cells[map_key][coord] = active_paths
                counts[map_key].update(active_paths)
                totals[map_key] += len(active_paths)
    return cells, counts, totals


def _placement_texture_paths(
    placement: dict[str, Any],
    canvases: dict[tuple[str, str], CanvasSource],
    tile_lookup: dict[tuple[str, str, int], dict[str, Any]],
    stores: list[StoreSource],
) -> set[str]:
    result: set[str] = set()
    build = str(placement["build"])
    map_name = str(placement["map_name"])
    canvas = canvases.get((build, map_name))
    if canvas is None:
        return result
    layout = canvas.layout
    min_x = int(layout.get("min_tile_x", 0))
    min_y = int(layout.get("min_tile_y", 0))
    bbox = _bbox(placement["bbox_xywh"])
    texture_ids: set[int] = set()
    for tile_key, tile in tile_lookup.items():
        if tile_key[:2] != (build, map_name):
            continue
        origin = (
            (int(tile["tile_x"]) - min_x) * ALPHA_TILE_SIZE,
            (int(tile["tile_y"]) - min_y) * ALPHA_TILE_SIZE,
            ALPHA_TILE_SIZE,
            ALPHA_TILE_SIZE,
        )
        intersection = _intersection(bbox, origin)
        if intersection is None:
            continue
        local_x = intersection[0] - origin[0]
        local_y = intersection[1] - origin[1]
        x0 = max(0, local_x // CHUNK_PIXEL_SIZE)
        y0 = max(0, local_y // CHUNK_PIXEL_SIZE)
        x1 = min(16, math.ceil((local_x + intersection[2]) / CHUNK_PIXEL_SIZE))
        y1 = min(16, math.ceil((local_y + intersection[3]) / CHUNK_PIXEL_SIZE))
        source = stores[int(tile["_store_source_index"])]
        row_idx = int(tile["store_row"])
        if "mcly_texture_ids" not in source.root or "mcly_layer_mask" not in source.root:
            continue
        ids = np.asarray(source.root["mcly_texture_ids"][row_idx, y0:y1, x0:x1], dtype=np.int32)
        mask = np.asarray(source.root["mcly_layer_mask"][row_idx, y0:y1, x0:x1]) > 0
        local_names = [
            str(value).strip().lower().replace("\\", "/")
            for value in source.decoded_metadata.get(int(tile["tile_id"]), {}).get("mcly_texture_names", [])
        ]
        for value in np.unique(ids[mask]):
            texture_id = int(value)
            if texture_id < 0:
                continue
            texture_ids.add(texture_id)
            if 0 <= texture_id < len(local_names) and local_names[texture_id]:
                result.add(f"{build}:{local_names[texture_id]}")
    placement["mcly_texture_ids"] = sorted(texture_ids)
    return result


def _object_spatial_context(
    rows: list[dict[str, Any]],
    *,
    tile_x: int,
    tile_y: int,
    local_bbox: tuple[int, int, int, int],
    max_distance_px: float,
) -> tuple[list[dict[str, Any]], str]:
    """Keep only region-adjacent instances and hash paired asset/pose offsets."""
    x, y, w, h = local_bbox
    center_x = x + w * 0.5
    center_y = y + h * 0.5
    paired: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    for row in rows:
        projected = _project_placement_to_tile(row, tile_x=tile_x, tile_y=tile_y)
        if projected is None:
            continue
        px, py = projected
        dx = max(float(x) - px, px - float(x + w), 0.0)
        dy = max(float(y) - py, py - float(y + h), 0.0)
        if math.hypot(dx, dy) > float(max_distance_px):
            continue
        pair = (
            str(row.get("instance_type", "")),
            str(row.get("asset_path", "")).strip().lower().replace("\\", "/"),
            int(round((px - center_x) * 4.0)),
            int(round((py - center_y) * 4.0)),
            round(float(row.get("rotX", 0.0) or 0.0), 1),
            round(float(row.get("rotY", 0.0) or 0.0), 1),
            round(float(row.get("rotZ", 0.0) or 0.0), 1),
            round(float(row.get("scale", 1.0) or 1.0), 3),
        )
        paired.append((pair, row))
    paired.sort(key=lambda item: item[0])
    if not paired:
        return [], "objects_none"
    signature = stable_id("objects_region", [item[0] for item in paired], length=20)
    return [item[1] for item in paired], signature


def _project_placement_to_tile(
    row: dict[str, Any], *, tile_x: int, tile_y: int
) -> tuple[float, float] | None:
    """Project classic placement coordinates with the existing four-mode dataset contract."""
    try:
        pos_x = float(row.get("posX", 0.0))
        pos_y = float(row.get("posY", 0.0))
        pos_z = float(row.get("posZ", 0.0))
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (pos_x, pos_y, pos_z)):
        return None
    map_origin = 17066.666
    tile_world = 533.33333
    candidates = (
        ((pos_x / tile_world) - tile_x, (pos_z / tile_world) - tile_y),
        (
            ((map_origin - pos_z) / tile_world) - tile_x,
            ((map_origin - pos_x) / tile_world) - tile_y,
        ),
        ((pos_x / tile_world) - tile_x, (pos_y / tile_world) - tile_y),
        (
            ((map_origin - pos_y) / tile_world) - tile_x,
            ((map_origin - pos_x) / tile_world) - tile_y,
        ),
    )
    best = min(
        candidates,
        key=lambda value: (
            max(0.0, -value[0])
            + max(0.0, value[0] - 1.0)
            + max(0.0, -value[1])
            + max(0.0, value[1] - 1.0)
        ),
    )
    if best[0] < 0.0 or best[0] > 1.0 or best[1] < 0.0 or best[1] > 1.0:
        return None
    return best[0] * (ALPHA_TILE_SIZE - 1), best[1] * (ALPHA_TILE_SIZE - 1)


def _pool_binary(binary: np.ndarray, level: int) -> np.ndarray:
    image = Image.fromarray(np.asarray(binary, dtype=np.uint8) * 255, mode="L")
    resized = np.asarray(image.resize((level, level), Image.Resampling.BOX), dtype=np.float32)
    return resized / 255.0


def _ring_sector_counts(vectors: list[np.ndarray], radii: Sequence[float]) -> list[int]:
    counts = np.zeros((len(radii), 8), dtype=np.int32)
    for vector in vectors:
        distance = float(np.linalg.norm(vector))
        ring = next((idx for idx, radius in enumerate(radii) if distance <= radius), None)
        if ring is None or distance == 0.0:
            continue
        angle = (math.atan2(float(vector[1]), float(vector[0])) + 2.0 * math.pi) % (2.0 * math.pi)
        sector = int(math.floor(angle / (2.0 * math.pi / 8.0))) % 8
        counts[ring, sector] = min(3, counts[ring, sector] + 1)
    return counts.reshape(-1).tolist()


def _arrangement_class(vectors: list[np.ndarray]) -> str:
    if not vectors:
        return "isolated"
    if len(vectors) == 1:
        return "pair"
    matrix = np.asarray(vectors, dtype=np.float64)
    covariance = np.cov(matrix.T) if len(vectors) > 1 else np.eye(2)
    eigenvalues = np.sort(np.linalg.eigvalsh(covariance))
    if eigenvalues[-1] > max(1e-9, eigenvalues[0]) * 6.0:
        return "chain"
    angles = np.asarray(
        [(math.atan2(float(v[1]), float(v[0])) + 2 * math.pi) % (2 * math.pi) for v in vectors]
    )
    sectors = len(set((angles / (2 * math.pi / 8)).astype(int).tolist()))
    radii = np.asarray([np.linalg.norm(v) for v in vectors])
    if len(vectors) >= 5 and sectors >= 6 and radii.mean() > 0 and radii.std() / radii.mean() < 0.3:
        return "ring"
    if len(vectors) >= 4 and sectors >= 4:
        return "radial_cluster"
    if len(vectors) >= 3 and sectors >= 3:
        return "branch"
    return "cluster"


def _quantized_vectors(vectors: list[np.ndarray]) -> list[list[int]]:
    if not vectors:
        return []
    scale = max(1.0, min(float(np.linalg.norm(vector)) for vector in vectors if np.linalg.norm(vector) > 0))
    return sorted(np.rint(np.asarray(vectors) / scale * 4.0).astype(np.int16).tolist())


def _bbox(value: Any) -> tuple[int, int, int, int]:
    values = list(value or [])
    if len(values) != 4:
        return (0, 0, 0, 0)
    return tuple(int(item) for item in values)  # type: ignore[return-value]


def _crosses_adt_boundary(bbox: tuple[int, int, int, int]) -> bool:
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return False
    return (
        x // ALPHA_TILE_SIZE != (x + w - 1) // ALPHA_TILE_SIZE
        or y // ALPHA_TILE_SIZE != (y + h - 1) // ALPHA_TILE_SIZE
    )


def _bbox_center(value: Any) -> tuple[float, float]:
    x, y, w, h = _bbox(value)
    return (x + w * 0.5, y + h * 0.5)


def _contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    ox, oy, ow, oh = outer
    ix, iy, iw, ih = inner
    return outer != inner and ox <= ix and oy <= iy and ox + ow >= ix + iw and oy + oh >= iy + ih


def _bbox_distance(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    dx = max(lx - (rx + rw), rx - (lx + lw), 0)
    dy = max(ly - (ry + rh), ry - (ly + lh), 0)
    return float(math.hypot(dx, dy))


def _intersection(
    left: tuple[int, int, int, int], right: tuple[int, int, int, int]
) -> tuple[int, int, int, int] | None:
    x0 = max(left[0], right[0])
    y0 = max(left[1], right[1])
    x1 = min(left[0] + left[2], right[0] + right[2])
    y1 = min(left[1] + left[3], right[1] + right[3])
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1 - x0, y1 - y0)


def _chunk_keys(local_bbox: tuple[int, int, int, int]) -> list[str]:
    x, y, w, h = local_bbox
    x0 = max(0, x // CHUNK_PIXEL_SIZE)
    y0 = max(0, y // CHUNK_PIXEL_SIZE)
    x1 = min(15, (x + max(1, w) - 1) // CHUNK_PIXEL_SIZE)
    y1 = min(15, (y + max(1, h) - 1) // CHUNK_PIXEL_SIZE)
    return [f"{chunk_x},{chunk_y}" for chunk_y in range(y0, y1 + 1) for chunk_x in range(x0, x1 + 1)]


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    provenance = result.get("provenance")
    if isinstance(provenance, str):
        try:
            result["provenance"] = json.loads(provenance)
        except json.JSONDecodeError:
            result["provenance"] = {"raw": provenance}
    return result


def _region_conflict(left: dict[str, Any], right: dict[str, Any]) -> bool:
    fields = ("build", "map_name", "layer_idx", "bbox_xywh")
    return any(left.get(field) != right.get(field) for field in fields)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _float_list(value: Any) -> list[float]:
    if value is None:
        return []
    return [float(item) for item in value]


def _relief_bucket(value: Any) -> str:
    parsed = _optional_float(value)
    if parsed is None:
        return "unknown"
    if parsed < 1.0:
        return "flat"
    if parsed < 10.0:
        return "rolling"
    return "steep"


def _token_weight(token: str) -> float:
    prefix = token.split(":", 1)[0]
    return {
        "family": 10.0,
        "anomaly": 8.0,
        "objects": 6.0,
        "transform": 4.0,
        "tileset": 3.0,
        "composition": 3.0,
        "relief": 2.0,
        "background": 1.0,
    }.get(prefix, 1.0)


def _best_representative(
    idx: int, selected: set[int], coverage: list[dict[str, Any]]
) -> int | None:
    tokens = set(coverage[idx]["coverage_tokens"])
    candidates: list[tuple[Any, ...]] = []
    for other in selected:
        overlap = tokens.intersection(coverage[other]["coverage_tokens"])
        if not overlap:
            continue
        candidates.append(
            (
                -sum(_token_weight(token) for token in overlap),
                -float(coverage[other]["evidence_completeness"]),
                *_tile_sort_key(coverage[other]),
                other,
            )
        )
    return min(candidates)[-1] if candidates else None


def _tile_key_text(row: dict[str, Any]) -> str:
    return f"{row['build']}:{row['map']}:{int(row['tile_x'])},{int(row['tile_y'])}:id{int(row['tile_id'])}"


def _normalized_thumbnail(alpha: np.ndarray, *, size: int) -> np.ndarray:
    source = np.asarray(alpha, dtype=np.float32)
    if source.ndim != 2:
        raise ValueError(f"alpha crop must be 2D, got {source.shape}")
    if source.size == 0:
        return np.zeros((size, size), dtype=np.float32)
    source = np.nan_to_num(source, nan=0.0, posinf=1.0, neginf=0.0)
    source = np.clip(source, 0.0, 1.0)
    h, w = source.shape
    scale = min(float(size) / max(1, w), float(size) / max(1, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    image = Image.fromarray((source * 255.0).astype(np.uint8), mode="L")
    resized = np.asarray(
        image.resize((new_w, new_h), Image.Resampling.BILINEAR), dtype=np.float32
    ) / 255.0
    canvas = np.zeros((size, size), dtype=np.float32)
    y = (size - new_h) // 2
    x = (size - new_w) // 2
    canvas[y : y + new_h, x : x + new_w] = resized
    return canvas


def _tile_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("build", "")),
        str(row.get("map", row.get("map_name", ""))),
        int(row.get("tile_y", -1)),
        int(row.get("tile_x", -1)),
        int(row.get("tile_id", -1)),
    )
