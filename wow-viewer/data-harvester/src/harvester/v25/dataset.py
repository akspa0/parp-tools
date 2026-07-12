"""V25 dataset: lean Zarr datastore for the terrain convergence model (Spec 102).

The V25 store is built fresh from the existing V18 substrate (terrain arrays),
the V22 enrichment (global tileset vocabulary + object placements), and — when
available — the V24 store's pre-computed cleaned minimaps.  Only the signals
the V25 model actually trains on are carried over; liquids, normals, holes,
MCNK flags, roof/visibility masks, and asset payload groups stay behind in
their source stores.

Per-tile arrays (all Blosc LZ4 clevel-1 compressed, 1-tile chunks):

==================  =========  ===============  =====================================
name                dtype      shape            role
==================  =========  ===============  =====================================
minimap_rgb         uint8      (256, 256, 3)    model input (raw RGB minimap)
clean_minimap_256   uint8      (256, 256, 3)    TerrainInpaintHead target
object_mask_256     float32    (256, 256)       ObjectMaskDecoder footprint target
height_257          float32    (257, 257)       Stage B height target
wdl_height_33       float32    (33, 33)         Stage A WDL prior target (stride-8)
alpha_256           uint8      (256, 256, 4)    MCAL fractal/alpha target
mcly_layer_mask     uint8      (16, 16, 4)      MCLY active-layer target
mcly_vocab_ids      int16      (16, 16, 4)      vocab-mapped tileset ids (-1 = none)
==================  =========  ===============  =====================================

Sidecar tables:

* ``index.parquet`` — row, tile_id, build, map, tile_x, tile_y, v18_row,
  clean_source, height_mean, height_std.
* ``placements.parquet`` — per-object ground truth (kind, name/unique/model id,
  asset path, position, rotation, scale) promoted from the V22 flat arrays.
* ``tileset_vocab.parquet`` — vocab_id -> global tileset id (+ path, count).
  The last vocab id is the out-of-vocabulary bucket.
* ``pm4_segments.parquet`` — optional; pre-parsed PM4 segment signal records
  attached via :func:`attach_pm4_segments`.  Python never parses raw ``.pm4``
  files (Spec 102 FR-102-402): records come from the C# export JSON.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr

from harvester.pm4_asset_matching.models import (
    Pm4Bounds3,
    Pm4SegmentAnchorSignals,
    Pm4SegmentHeightStats,
    Pm4SegmentSignalRecord,
    Pm4SegmentTopologyStats,
)

V25_DATASET_VERSION = "v25.2"

# Spec 102: lightweight Blosc LZ4 level-1 compression for every array.
DEFAULT_CODEC = zarr.codecs.BloscCodec(cname="lz4", clevel=1)

OOV_LABEL = "<oov>"


@dataclass(frozen=True)
class _ArraySpec:
    name: str
    dtype: np.dtype
    shape: tuple[int, ...]


def _arr(name: str, dtype, shape: tuple[int, ...]) -> _ArraySpec:
    return _ArraySpec(name=name, dtype=np.dtype(dtype), shape=shape)


V25_PER_TILE_SPECS: tuple[_ArraySpec, ...] = (
    _arr("minimap_rgb", np.uint8, (256, 256, 3)),
    _arr("clean_minimap_256", np.uint8, (256, 256, 3)),
    _arr("object_mask_256", np.float32, (256, 256)),
    _arr("height_257", np.float32, (257, 257)),
    _arr("wdl_height_33", np.float32, (33, 33)),
    _arr("alpha_256", np.uint8, (256, 256, 4)),
    _arr("mcly_layer_mask", np.uint8, (16, 16, 4)),
    _arr("mcly_vocab_ids", np.int16, (16, 16, 4)),
    # Liquid + chunk-flag loss signals (user-directed 2026-07-11): liquid areas
    # must be maskable out of height supervision, and era restoration needs
    # MH2O/MCLQ facts.  liquid_mask_256 is coverage scaled to 0-255.
    _arr("liquid_mask_256", np.uint8, (256, 256)),
    _arr("liquid_type_256", np.uint8, (256, 256)),
    _arr("liquid_height_256", np.float32, (256, 256)),
    _arr("mcnk_flags_16", np.int32, (16, 16)),
    # Full-signal completeness pass (user-directed 2026-07-11 — "every signal
    # we will ever need"): MCNR normals (int8, native precision; the validity
    # masks are derivable: checkerboard formula + nonzero magnitude), MCSH
    # shadows, renderer-truth object visibility (inpaint weighting), the
    # object-inpainted intended-ground heights, and per-vertex instance ids.
    _arr("normal_xyz_257", np.int8, (257, 257, 3)),
    _arr("shadow_mask_256", np.uint8, (256, 256)),
    _arr("object_visibility_256", np.uint8, (256, 256)),
    _arr("ground_intent_height_257", np.float32, (257, 257)),
    _arr("object_instance_mask", np.int32, (257, 257)),
)

# V18/V22 signals deliberately NOT carried into V25 (documented so the omission
# is a decision, not an oversight): holes_16 (inverted at the C# source per the
# V24 audit — corrupt until fixed upstream; never train on known-bad data),
# normal_mask + mcnr_mask_257 (derivable from stored normals: nonzero magnitude
# and the x%2==y%2 checkerboard), object_mask (superseded by
# object_precise_mask), object_roof_mask/confidence + object_filtered_mask +
# model_focus_mask + model_above_terrain_mask (deprecated diagnostics),
# mddf_mask/modf_mask (derivable from placements), models/, tilesets/ payload
# groups (paths ride along in the placement/vocab tables).


# ---------------------------------------------------------------------------
# Row selection
# ---------------------------------------------------------------------------


def select_rows(
    v18_index: pa.Table,
    maps: list[str] | None = None,
    curation_manifest: Path | None = None,
    difficulty_bucket: str | None = None,
    limit: int | None = None,
) -> list[int]:
    """Pick V18 rows for the build, honoring map filters and the curation manifest."""
    builds = v18_index["build"].to_pylist()
    tile_ids = v18_index["tile_id"].to_pylist()
    map_names = v18_index["map"].to_pylist()

    keep = [True] * len(tile_ids)
    if maps:
        wanted = {m.lower() for m in maps}
        keep = [k and (m.lower() in wanted) for k, m in zip(keep, map_names)]

    if curation_manifest is not None:
        manifest = pq.read_table(curation_manifest)
        kept_pairs: set[tuple[str, int]] = set()
        m_builds = manifest["build"].to_pylist()
        m_tiles = manifest["tile_id"].to_pylist()
        m_keep = manifest["keep"].to_pylist()
        m_bucket = (
            manifest["difficulty_bucket"].to_pylist()
            if "difficulty_bucket" in manifest.column_names
            else [None] * len(m_tiles)
        )
        for b, t, k, bucket in zip(m_builds, m_tiles, m_keep, m_bucket):
            if not k:
                continue
            if difficulty_bucket is not None and bucket != difficulty_bucket:
                continue
            kept_pairs.add((str(b), int(t)))
        keep = [
            k and ((str(b), int(t)) in kept_pairs)
            for k, b, t in zip(keep, builds, tile_ids)
        ]

    rows = [i for i, k in enumerate(keep) if k]
    if limit is not None:
        rows = rows[:limit]
    return rows


def _contiguous_runs(rows: list[int]) -> list[tuple[int, int]]:
    """Split sorted row indices into [lo, hi) runs for sequential Zarr slicing."""
    runs: list[tuple[int, int]] = []
    if not rows:
        return runs
    start = prev = rows[0]
    for r in rows[1:]:
        if r == prev + 1:
            prev = r
            continue
        runs.append((start, prev + 1))
        start = prev = r
    runs.append((start, prev + 1))
    return runs


# ---------------------------------------------------------------------------
# Derived signals
# ---------------------------------------------------------------------------


def object_mask_256_from_precise(precise_257: np.ndarray) -> np.ndarray:
    """Reduce the 257x257 corner-grid footprint to the 256x256 minimap cell grid.

    Each minimap cell takes the max of its four corner samples, preserving the
    fractional coverage values of ``object_precise_mask``.
    """
    m = np.asarray(precise_257, dtype=np.float32)
    corners = np.stack([m[:-1, :-1], m[1:, :-1], m[:-1, 1:], m[1:, 1:]])
    return corners.max(axis=0)


def wdl_height_33_from_257(height_257: np.ndarray) -> np.ndarray:
    """Stride-8 sampling — the same math as ``harvester.v25.prior.WdlDownsampler``."""
    return np.asarray(height_257, dtype=np.float32)[::8, ::8]


def _to_uint8_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb)
    if rgb.dtype == np.uint8:
        return rgb
    rgb = rgb.astype(np.float32)
    if rgb.max() <= 1.5:
        rgb = rgb * 255.0
    return np.clip(np.round(rgb), 0, 255).astype(np.uint8)


def _to_uint8_alpha(alpha: np.ndarray) -> np.ndarray:
    alpha = np.asarray(alpha)
    if alpha.dtype == np.uint8:
        return alpha
    alpha = alpha.astype(np.float32)
    if alpha.max() <= 1.5:
        alpha = alpha * 255.0
    return np.clip(np.round(alpha), 0, 255).astype(np.uint8)


def _to_uint8_unit(mask: np.ndarray) -> np.ndarray:
    """Quantize a [0, 1] coverage mask to uint8 0-255 (preserves fractions)."""
    mask = np.asarray(mask)
    if mask.dtype == np.uint8:
        return mask
    return np.clip(np.round(mask.astype(np.float32) * 255.0), 0, 255).astype(np.uint8)


def _to_int8_normals(normals: np.ndarray) -> np.ndarray:
    """Quantize [-1, 1] float normals back to MCNR's native int8 precision."""
    normals = np.asarray(normals)
    if normals.dtype == np.int8:
        return normals
    return np.clip(np.round(normals.astype(np.float32) * 127.0), -127, 127).astype(np.int8)


# ---------------------------------------------------------------------------
# Tileset vocabulary (era-scoped: keyed by (build, normalized tileset path))
# ---------------------------------------------------------------------------


def tileset_key(build: str, tileset_id: int, tileset_paths: list[str]) -> str:
    """Era-scoped key for a tileset: ``<build>:<normalized path>``.

    Tileset *content* changed across eras even when the path stayed the same
    (user-directed 2026-07-11: "the images are literally different between
    them, even though they will have the same names"), so every era keeps its
    own vocabulary entries — grass in 0.5.3 and grass in 3.3.5 are distinct
    texture concepts with distinct vocab ids.  When a store has no path table,
    fall back to a build-scoped id key.
    """
    if 0 <= tileset_id < len(tileset_paths):
        path = str(tileset_paths[tileset_id]).strip().lower().replace("\\", "/")
        if path:
            return f"{build}:{path}"
    return f"{build}#{tileset_id}"


def build_tileset_vocab(counts: dict[str, int], vocab_size: int) -> dict[str, int]:
    """Rank tileset keys by frequency and keep the top ``vocab_size - 1``.

    Returns a mapping tileset key -> vocab index.  Vocab index
    ``vocab_size - 1`` is reserved as the out-of-vocabulary bucket.
    """
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return {key: i for i, (key, _) in enumerate(ranked[: vocab_size - 1])}


def count_tileset_keys(
    tileset_ids_batches: list[np.ndarray],
    layer_mask_batches: list[np.ndarray],
    build: str,
    tileset_paths: list[str],
    counts: dict[str, int],
) -> None:
    """Accumulate per-key frequencies over active layers into ``counts``."""
    for ids, mask in zip(tileset_ids_batches, layer_mask_batches):
        active = (np.asarray(mask) > 0) & (np.asarray(ids) >= 0)
        vals, freq = np.unique(np.asarray(ids)[active], return_counts=True)
        for v, f in zip(vals.tolist(), freq.tolist()):
            key = tileset_key(build, int(v), tileset_paths)
            counts[key] = counts.get(key, 0) + int(f)


def map_tileset_ids(
    tileset_ids: np.ndarray,
    layer_mask: np.ndarray,
    vocab: dict[int, int],
    vocab_size: int,
) -> np.ndarray:
    """Map global tileset ids to vocab indices; -1 where the layer is inactive."""
    ids = np.asarray(tileset_ids, dtype=np.int64)
    out = np.full(ids.shape, -1, dtype=np.int16)
    active = (np.asarray(layer_mask) > 0) & (ids >= 0)
    oov = vocab_size - 1
    flat_ids = ids[active]
    mapped = np.array([vocab.get(int(t), oov) for t in flat_ids.tolist()], dtype=np.int16)
    out[active] = mapped
    return out


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def _as_source_list(value, count_hint: int | None = None) -> list[Path | None]:
    """Normalize a store argument to a list of optional paths.

    Accepts a single path, ``None``, or a sequence; the string ``"-"`` marks a
    missing entry when passing per-build lists on the CLI.
    """
    if value is None:
        items: list[Path | None] = [None] * (count_hint or 1)
    elif isinstance(value, (str, Path)):
        items = [Path(value)]
    else:
        items = [None if (v is None or str(v) == "-") else Path(v) for v in value]
    if count_hint is not None:
        if len(items) == 1 and count_hint > 1:
            raise ValueError(
                f"expected {count_hint} store entries (one per --v18-store, use '-' for none); got 1"
            )
        if len(items) != count_hint:
            raise ValueError(f"expected {count_hint} store entries, got {len(items)}")
    return items


class _BuildSource:
    """One (V18, V22, V24) source triple, opened and row-selected."""

    def __init__(
        self,
        v18_store: Path,
        v22_store: Path | None,
        v24_store: Path | None,
        maps: list[str] | None,
        curation_manifest: Path | None,
        difficulty_bucket: str | None,
        limit: int | None,
        height_repair_root: Path | None = None,
    ):
        self.v18_store = Path(v18_store)
        self.v18 = zarr.open_group(str(v18_store), mode="r")
        self.index = pq.read_table(self.v18_store / "index.parquet")
        self.rows = sorted(
            select_rows(
                self.index,
                maps=maps,
                curation_manifest=curation_manifest,
                difficulty_bucket=difficulty_bucket,
                limit=limit,
            )
        )
        self.runs = _contiguous_runs(self.rows)
        self.build = str(self.index["build"][0].as_py()) if self.index.num_rows else "unknown"
        self.has_no_object = "no_object_minimap" in self.v18

        # Mismatch-repair join: prefer corrected heights when the repair store
        # carries this build ("never process bad data when a repaired version
        # exists").
        self.repair = None
        if height_repair_root is not None:
            repair_group = Path(height_repair_root) / self.build
            if repair_group.exists():
                candidate = zarr.open_group(str(repair_group), mode="r")
                if "height_corrected_257" in candidate:
                    self.repair = candidate

        # Optional V24 join: v18_row -> v24 row with a pre-computed cleaned minimap.
        self.v24 = None
        self.v24_row_by_v18: dict[int, int] = {}
        if v24_store is not None and Path(v24_store).exists():
            v24 = zarr.open_group(str(v24_store), mode="r")
            if "cleaned_minimap_256" in v24:
                v24_index = pq.read_table(Path(v24_store) / "index.parquet")
                for i, v18_row in enumerate(v24_index["v18_row"].to_pylist()):
                    self.v24_row_by_v18[int(v18_row)] = i
                self.v24 = v24

        # Optional V22 join: tileset ids + placements + path tables.
        self.v22 = None
        self.v22_store = v22_store
        self.tileset_paths: list[str] = []
        self.model_paths: list[str] = []
        if v22_store is not None and Path(v22_store).exists():
            v22 = zarr.open_group(str(v22_store), mode="r")
            if "mcly_tileset_ids" in v22:
                self.v22 = v22
                if "tilesets" in v22 and "tileset_paths" in v22["tilesets"]:
                    self.tileset_paths = [str(p) for p in v22["tilesets"]["tileset_paths"][:]]
                if "models" in v22 and "model_paths" in v22["models"]:
                    self.model_paths = [str(p) for p in v22["models"]["model_paths"][:]]


def build_v25_dataset(
    v18_store,
    output: Path,
    v22_store=None,
    v24_store=None,
    maps: list[str] | None = None,
    curation_manifest: Path | None = None,
    difficulty_bucket: str | None = None,
    limit: int | None = None,
    vocab_size: int = 256,
    batch_rows: int = 64,
    overwrite: bool = False,
    progress_interval: int = 200,
    height_repair_root: Path | None = None,
    mismatch_report: Path | None = None,
) -> Path:
    """Build a fresh V25 Zarr store from one or more (V18, V22, V24) source triples.

    ``v18_store``/``v22_store``/``v24_store`` accept a single path or
    index-paired lists (entry i of each list belongs to the same build; use
    ``None``/``"-"`` for a missing V22/V24).  The tileset vocabulary is
    era-scoped — keyed by (build, normalized path) — because tileset content
    changed across eras even under identical names.  ``limit`` applies per
    build.

    ``curation_manifest`` both filters (keep==True) and is **baked in**: every
    manifest column (buckets, quality/usefulness/difficulty scores, coverage
    stats, profiles) is joined per tile into the output ``index.parquet``.
    ``height_repair_root`` points at a mismatch-repair store whose per-build
    ``height_corrected_257`` replaces raw heights.  ``mismatch_report`` joins
    per-tile ``mismatch_severity``/``mismatch_reason`` audit columns.
    """
    from harvester.v24.clean_minimap import clean_minimap

    output = Path(output)
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"{output} exists; pass overwrite=True to rebuild")
        import shutil

        shutil.rmtree(output)

    v18_list = _as_source_list(v18_store)
    if any(p is None for p in v18_list):
        raise ValueError("every source triple needs a V18 store")
    v22_list = _as_source_list(v22_store, count_hint=len(v18_list))
    v24_list = _as_source_list(v24_store, count_hint=len(v18_list))

    sources: list[_BuildSource] = []
    for v18_p, v22_p, v24_p in zip(v18_list, v22_list, v24_list):
        src = _BuildSource(
            v18_p, v22_p, v24_p, maps, curation_manifest, difficulty_bucket, limit,
            height_repair_root=height_repair_root,
        )
        if not src.rows:
            raise ValueError(f"row selection produced no tiles for {v18_p}")
        print(
            f"[v25-build] {src.build}: {len(src.rows)} tiles selected"
            f" (v24 cleans: {len(src.v24_row_by_v18)}, v22: {'yes' if src.v22 else 'no'}"
            f", repaired heights: {'yes' if src.repair is not None else 'no'})",
            flush=True,
        )
        sources.append(src)

    n = sum(len(s.rows) for s in sources)

    # Curation metadata bake-in: every manifest column rides into index.parquet.
    _JOIN_SKIP = {"build", "tile_id", "map", "tile_x", "tile_y", "height_std"}
    curation_cols: list[str] = []
    curation_by_tile: dict[tuple[str, int], dict] = {}
    if curation_manifest is not None:
        manifest = pq.read_table(curation_manifest)
        curation_cols = [c for c in manifest.column_names if c not in _JOIN_SKIP]
        m_rows = manifest.to_pylist()
        for r in m_rows:
            curation_by_tile[(str(r["build"]), int(r["tile_id"]))] = {
                c: r[c] for c in curation_cols
            }
        print(f"[v25-build] baking {len(curation_cols)} curation columns into the index", flush=True)

    mismatch_by_tile: dict[tuple[str, int], dict] = {}
    if mismatch_report is not None and Path(mismatch_report).exists():
        for r in pq.read_table(mismatch_report).to_pylist():
            mismatch_by_tile[(str(r["build"]), int(r["tile_id"]))] = {
                "mismatch_severity": r.get("mismatch_severity"),
                "mismatch_reason": r.get("mismatch_reason"),
            }
        print(f"[v25-build] mismatch audit joined for {len(mismatch_by_tile)} tiles", flush=True)

    # Pass 1: shared tileset vocabulary over path keys across every build.
    key_counts: dict[str, int] = {}
    for src in sources:
        if src.v22 is None:
            continue
        for lo, hi in src.runs:
            count_tileset_keys(
                [np.asarray(src.v22["mcly_tileset_ids"][lo:hi])],
                [np.asarray(src.v18["mcly_layer_mask"][lo:hi])],
                src.build,
                src.tileset_paths,
                key_counts,
            )
    vocab_by_key = build_tileset_vocab(key_counts, vocab_size) if key_counts else {}
    if vocab_by_key:
        print(
            f"[v25-build] shared tileset vocab: {len(vocab_by_key)} in-vocab keys (+1 OOV) "
            f"from {len(key_counts)} distinct tilesets",
            flush=True,
        )

    # Create output store.
    root = zarr.open_group(str(output), mode="w")
    arrays: dict[str, zarr.Array] = {}
    for spec in V25_PER_TILE_SPECS:
        arrays[spec.name] = root.create_array(
            spec.name,
            shape=(n, *spec.shape),
            chunks=(1, *spec.shape),
            dtype=spec.dtype,
            compressors=DEFAULT_CODEC,
        )

    index_rows: list[dict] = []
    placement_rows: list[dict] = []
    clean_source_counts: dict[str, int] = {}
    nonfinite_height_tiles = 0
    written = 0

    for src in sources:
        v18, v22, v24 = src.v18, src.v22, src.v24
        idx = src.index
        builds = idx["build"].to_pylist()
        tile_ids = idx["tile_id"].to_pylist()
        map_names = idx["map"].to_pylist()
        tile_xs = idx["tile_x"].to_pylist()
        tile_ys = idx["tile_y"].to_pylist()
        h_means = idx["height_mean"].to_pylist() if "height_mean" in idx.column_names else [0.0] * len(builds)
        h_stds = idx["height_std"].to_pylist() if "height_std" in idx.column_names else [0.0] * len(builds)

        # Per-build tileset id -> shared vocab index.
        vocab_by_tid: dict[int, int] = {}
        if v22 is not None and vocab_by_key:
            oov = vocab_size - 1
            n_tilesets = max(len(src.tileset_paths), int(np.asarray(v22["mcly_tileset_ids"][:]).max(initial=0)) + 1)
            for tid in range(n_tilesets):
                vocab_by_tid[tid] = vocab_by_key.get(
                    tileset_key(src.build, tid, src.tileset_paths), oov
                )

        has_liquid_mask = "liquid_mask" in v18
        has_liquid_type = "liquid_type_256" in v18
        has_liquid_height = "liquid_height" in v18
        has_mcnk_flags = "mcnk_flags_16" in v18
        has_normals = "normal_xyz" in v18
        has_shadow = "shadow_mask" in v18
        has_visibility = "object_visibility_mask" in v18
        has_ground_intent = "ground_intent_height_257" in v18
        has_instance = "object_instance_mask" in v18

        for lo, hi in src.runs:
            # Sequential slice reads (the V24 preload lesson, applied at build time).
            minimap_b = np.asarray(v18["minimap_rgb"][lo:hi])
            height_b = np.asarray(v18["height_257"][lo:hi])
            repaired_b = None
            if src.repair is not None:
                # The repair store is a sparse overlay: NaN everywhere except
                # the tiles it actually corrected. Merge per cell — corrected
                # where finite, raw elsewhere.
                corrected_b = np.asarray(src.repair["height_corrected_257"][lo:hi])
                finite = np.isfinite(corrected_b)
                height_b = np.where(finite, corrected_b, height_b)
                repaired_b = finite.any(axis=(1, 2))
            alpha_b = np.asarray(v18["alpha_256"][lo:hi])
            normal_b = np.asarray(v18["normal_xyz"][lo:hi]) if has_normals else None
            shadow_b = np.asarray(v18["shadow_mask"][lo:hi]) if has_shadow else None
            vis_b = np.asarray(v18["object_visibility_mask"][lo:hi]) if has_visibility else None
            gih_b = np.asarray(v18["ground_intent_height_257"][lo:hi]) if has_ground_intent else None
            inst_b = np.asarray(v18["object_instance_mask"][lo:hi]) if has_instance else None
            precise_b = np.asarray(v18["object_precise_mask"][lo:hi])
            mcly_mask_b = np.asarray(v18["mcly_layer_mask"][lo:hi])
            no_object_b = np.asarray(v18["no_object_minimap"][lo:hi]) if src.has_no_object else None
            liquid_mask_b = np.asarray(v18["liquid_mask"][lo:hi]) if has_liquid_mask else None
            liquid_type_b = np.asarray(v18["liquid_type_256"][lo:hi]) if has_liquid_type else None
            liquid_height_b = np.asarray(v18["liquid_height"][lo:hi]) if has_liquid_height else None
            mcnk_flags_b = np.asarray(v18["mcnk_flags_16"][lo:hi]) if has_mcnk_flags else None
            tileset_b = np.asarray(v22["mcly_tileset_ids"][lo:hi]) if v22 is not None else None
            mddf_off_b = np.asarray(v22["mddf_placement_offset"][lo:hi]) if v22 is not None and "mddf_placement_offset" in v22 else None
            mddf_cnt_b = np.asarray(v22["mddf_count"][lo:hi]) if v22 is not None and "mddf_count" in v22 else None
            modf_off_b = np.asarray(v22["modf_placement_offset"][lo:hi]) if v22 is not None and "modf_placement_offset" in v22 else None
            modf_cnt_b = np.asarray(v22["modf_count"][lo:hi]) if v22 is not None and "modf_count" in v22 else None

            for j, v18_row in enumerate(range(lo, hi)):
                out_row = written

                minimap = minimap_b[j]
                height = height_b[j].astype(np.float32)
                precise = precise_b[j]
                if not np.isfinite(height).all():
                    nonfinite_height_tiles += 1

                # Clean minimap: prefer the V24 pre-computed array, else compute once here.
                if v24 is not None and v18_row in src.v24_row_by_v18:
                    cleaned = np.asarray(v24["cleaned_minimap_256"][src.v24_row_by_v18[v18_row]])
                    clean_source = "v24_precomputed"
                else:
                    cleaned, meta = clean_minimap(
                        minimap,
                        precise,
                        no_object_minimap=no_object_b[j] if no_object_b is not None else None,
                    )
                    clean_source = meta["source"]
                clean_source_counts[clean_source] = clean_source_counts.get(clean_source, 0) + 1

                mcly_mask = (mcly_mask_b[j] > 0).astype(np.uint8)
                if tileset_b is not None:
                    vocab_ids = map_tileset_ids(tileset_b[j], mcly_mask, vocab_by_tid, vocab_size)
                else:
                    vocab_ids = np.full((16, 16, 4), -1, dtype=np.int16)

                arrays["minimap_rgb"][out_row] = _to_uint8_rgb(minimap)
                arrays["clean_minimap_256"][out_row] = _to_uint8_rgb(cleaned)
                arrays["object_mask_256"][out_row] = object_mask_256_from_precise(precise)
                arrays["height_257"][out_row] = height
                arrays["wdl_height_33"][out_row] = wdl_height_33_from_257(height)
                arrays["alpha_256"][out_row] = _to_uint8_alpha(alpha_b[j])
                arrays["mcly_layer_mask"][out_row] = mcly_mask
                arrays["mcly_vocab_ids"][out_row] = vocab_ids
                arrays["liquid_mask_256"][out_row] = (
                    _to_uint8_unit(liquid_mask_b[j]) if liquid_mask_b is not None
                    else np.zeros((256, 256), dtype=np.uint8)
                )
                arrays["liquid_type_256"][out_row] = (
                    liquid_type_b[j].astype(np.uint8) if liquid_type_b is not None
                    else np.zeros((256, 256), dtype=np.uint8)
                )
                arrays["liquid_height_256"][out_row] = (
                    liquid_height_b[j].astype(np.float32) if liquid_height_b is not None
                    else np.zeros((256, 256), dtype=np.float32)
                )
                arrays["mcnk_flags_16"][out_row] = (
                    mcnk_flags_b[j].astype(np.int32) if mcnk_flags_b is not None
                    else np.zeros((16, 16), dtype=np.int32)
                )
                arrays["normal_xyz_257"][out_row] = (
                    _to_int8_normals(normal_b[j]) if normal_b is not None
                    else np.zeros((257, 257, 3), dtype=np.int8)
                )
                arrays["shadow_mask_256"][out_row] = (
                    _to_uint8_unit(shadow_b[j]) if shadow_b is not None
                    else np.zeros((256, 256), dtype=np.uint8)
                )
                arrays["object_visibility_256"][out_row] = (
                    _to_uint8_unit(vis_b[j]) if vis_b is not None
                    else np.zeros((256, 256), dtype=np.uint8)
                )
                arrays["ground_intent_height_257"][out_row] = (
                    gih_b[j].astype(np.float32) if gih_b is not None else height
                )
                arrays["object_instance_mask"][out_row] = (
                    inst_b[j].astype(np.int32) if inst_b is not None
                    else np.full((257, 257), -1, dtype=np.int32)
                )

                tile_key = (str(builds[v18_row]), int(tile_ids[v18_row]))
                index_row = {
                    "row": out_row,
                    "tile_id": int(tile_ids[v18_row]),
                    "build": str(builds[v18_row]),
                    "map": str(map_names[v18_row]),
                    "tile_x": int(tile_xs[v18_row]),
                    "tile_y": int(tile_ys[v18_row]),
                    "v18_row": int(v18_row),
                    "clean_source": clean_source,
                    "height_mean": float(h_means[v18_row]),
                    "height_std": float(h_stds[v18_row]),
                    "height_repaired": bool(repaired_b[j]) if repaired_b is not None else False,
                }
                if curation_cols:
                    cur = curation_by_tile.get(tile_key, {})
                    for c in curation_cols:
                        index_row[c] = cur.get(c)
                if mismatch_by_tile:
                    mm = mismatch_by_tile.get(tile_key, {})
                    index_row["mismatch_severity"] = mm.get("mismatch_severity")
                    index_row["mismatch_reason"] = mm.get("mismatch_reason")
                index_rows.append(index_row)

                # Placements from the V22 flat arrays.
                if mddf_off_b is not None:
                    off, cnt = int(mddf_off_b[j]), int(mddf_cnt_b[j][0])
                    if cnt > 0:
                        data = np.asarray(v22["mddf_placement_data"][off : off + cnt])
                        mids = np.asarray(v22["mddf_model_ids"][off : off + cnt])
                        for k in range(cnt):
                            d = data[k]
                            mid = int(mids[k])
                            placement_rows.append(
                                {
                                    "row": out_row,
                                    "kind": "m2",
                                    "name_id": int(d[0]),
                                    "unique_id": int(d[1]),
                                    "model_id": mid,
                                    "asset_path": src.model_paths[mid] if 0 <= mid < len(src.model_paths) else "",
                                    "pos_x": float(d[2]),
                                    "pos_y": float(d[3]),
                                    "pos_z": float(d[4]),
                                    "rot_x": float(d[5]),
                                    "rot_y": float(d[6]),
                                    "rot_z": float(d[7]),
                                    "scale": float(d[8]),
                                }
                            )
                if modf_off_b is not None:
                    off, cnt = int(modf_off_b[j]), int(modf_cnt_b[j][0])
                    if cnt > 0:
                        data = np.asarray(v22["modf_placement_data"][off : off + cnt])
                        mids = np.asarray(v22["modf_model_ids"][off : off + cnt])
                        for k in range(cnt):
                            d = data[k]
                            mid = int(mids[k])
                            placement_rows.append(
                                {
                                    "row": out_row,
                                    "kind": "wmo",
                                    "name_id": int(d[0]),
                                    "unique_id": int(d[1]),
                                    "model_id": mid,
                                    "asset_path": src.model_paths[mid] if 0 <= mid < len(src.model_paths) else "",
                                    "pos_x": float(d[2]),
                                    "pos_y": float(d[3]),
                                    "pos_z": float(d[4]),
                                    "rot_x": float(d[5]),
                                    "rot_y": float(d[6]),
                                    "rot_z": float(d[7]),
                                    "scale": 1.0,
                                }
                            )

                written += 1
                if written % progress_interval == 0:
                    print(f"[v25-build] {written}/{n} tiles", flush=True)

    # Sidecar tables.
    pq.write_table(pa.Table.from_pylist(index_rows), output / "index.parquet")
    if placement_rows:
        pq.write_table(pa.Table.from_pylist(placement_rows), output / "placements.parquet")
    if vocab_by_key:
        inv = sorted(vocab_by_key.items(), key=lambda kv: kv[1])
        vocab_rows = []
        for key, vi in inv:
            # Era-scoped keys: "<build>:<path>" (or "<build>#<tid>" fallback).
            if ":" in key:
                key_build, key_path = key.split(":", 1)
            elif "#" in key:
                key_build, key_path = key.split("#", 1)
                key_path = f"#{key_path}"
            else:
                key_build, key_path = "", key
            vocab_rows.append(
                {
                    "vocab_id": vi,
                    "build": key_build,
                    "tileset_path": key_path,
                    "key": key,
                    "count": key_counts.get(key, 0),
                }
            )
        vocab_rows.append(
            {"vocab_id": vocab_size - 1, "build": "", "tileset_path": OOV_LABEL, "key": OOV_LABEL, "count": 0}
        )
        pq.write_table(pa.Table.from_pylist(vocab_rows), output / "tileset_vocab.parquet")

    root.attrs.update(
        {
            "v25_dataset_version": V25_DATASET_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "tile_count": n,
            "builds": sorted({s.build for s in sources}),
            "vocab_size": vocab_size,
            "source_v18_stores": [str(s.v18_store) for s in sources],
            "source_v22_stores": [str(s.v22_store) if s.v22 is not None else None for s in sources],
            "source_v24_stores": [str(p) if p is not None else None for p in v24_list],
            "signals": [s.name for s in V25_PER_TILE_SPECS],
            "clean_source_counts": clean_source_counts,
            "placement_count": len(placement_rows),
            "curation_manifest": str(curation_manifest) if curation_manifest else None,
            "curation_columns": curation_cols,
            "nonfinite_height_tiles": nonfinite_height_tiles,
            "height_repair_root": str(height_repair_root) if height_repair_root else None,
            "height_repaired_builds": [s.build for s in sources if s.repair is not None],
            "mismatch_report": str(mismatch_report) if mismatch_report else None,
        }
    )
    if nonfinite_height_tiles:
        print(
            f"[v25-build] WARNING: {nonfinite_height_tiles} tiles carry non-finite "
            f"heights — inspect the sources before training on this store",
            flush=True,
        )
    print(
        f"[v25-build] done: {n} tiles, {len(placement_rows)} placements, "
        f"clean sources {clean_source_counts} -> {output}",
        flush=True,
    )
    return output


# ---------------------------------------------------------------------------
# Training-side reader
# ---------------------------------------------------------------------------


class V25TileSource:
    """Reads a V25 store with the contiguous-preload pattern (FR-102-502).

    ``preload(rows)`` batch-reads every per-tile array through sorted
    contiguous slices, then ``load(row)`` is a dict lookup.  Random access
    without preload still works for ad-hoc inspection.
    """

    def __init__(self, store_path: Path):
        self.path = Path(store_path)
        self.root = zarr.open_group(str(self.path), mode="r")
        self.index = pq.read_table(self.path / "index.parquet")
        self._cache: dict[str, dict[int, np.ndarray]] = {}

        self.placements_by_row: dict[int, list[dict]] = {}
        placements_path = self.path / "placements.parquet"
        if placements_path.exists():
            table = pq.read_table(placements_path)
            cols = {name: table[name].to_pylist() for name in table.column_names}
            for i in range(table.num_rows):
                self.placements_by_row.setdefault(int(cols["row"][i]), []).append(
                    {name: cols[name][i] for name in table.column_names}
                )

        self.vocab_size = int(self.root.attrs.get("vocab_size", 256))

    def __len__(self) -> int:
        return self.index.num_rows

    def preload(self, rows: list[int]) -> None:
        wanted = sorted({int(r) for r in rows})
        runs = _contiguous_runs(wanted)
        for spec in V25_PER_TILE_SPECS:
            if spec.name not in self.root:
                continue  # store predates this signal; _read zero-fills
            cache = self._cache.setdefault(spec.name, {})
            arr = self.root[spec.name]
            for lo, hi in runs:
                block = np.asarray(arr[lo:hi])
                for j, row in enumerate(range(lo, hi)):
                    cache[row] = block[j]

    def _read(self, name: str, row: int) -> np.ndarray:
        cache = self._cache.get(name)
        if cache is not None and row in cache:
            return cache[row]
        if name not in self.root:
            # Store built before this signal existed — zero-fill at spec shape.
            spec = next(s for s in V25_PER_TILE_SPECS if s.name == name)
            return np.zeros(spec.shape, dtype=spec.dtype)
        return np.asarray(self.root[name][row])

    def load(self, row: int) -> dict:
        """Return the per-tile training record as float/int numpy arrays."""
        record = {
            "minimap": self._read("minimap_rgb", row).astype(np.float32) / 255.0,
            "clean_minimap": self._read("clean_minimap_256", row).astype(np.float32) / 255.0,
            "object_mask": self._read("object_mask_256", row).astype(np.float32),
            "height_257": self._read("height_257", row).astype(np.float32),
            "wdl_height_33": self._read("wdl_height_33", row).astype(np.float32),
            "alpha": self._read("alpha_256", row).astype(np.float32) / 255.0,
            "mcly_layer_mask": self._read("mcly_layer_mask", row).astype(np.int64),
            "mcly_vocab_ids": self._read("mcly_vocab_ids", row).astype(np.int64),
            "liquid_mask": self._read("liquid_mask_256", row).astype(np.float32) / 255.0,
            "liquid_type": self._read("liquid_type_256", row).astype(np.int64),
            "liquid_height": self._read("liquid_height_256", row).astype(np.float32),
            "mcnk_flags": self._read("mcnk_flags_16", row).astype(np.int64),
            "normal_xyz": self._read("normal_xyz_257", row).astype(np.float32) / 127.0,
            "shadow_mask": self._read("shadow_mask_256", row).astype(np.float32) / 255.0,
            "object_visibility": self._read("object_visibility_256", row).astype(np.float32) / 255.0,
            "ground_intent_height": self._read("ground_intent_height_257", row).astype(np.float32),
            "object_instance_mask": self._read("object_instance_mask", row).astype(np.int64),
            "placements": self.placements_by_row.get(row, []),
        }
        # True MCNK hole bitmasks are a post-build attachment; -1 = unknown.
        if HOLES_ARRAY in self.root:
            record["holes_bits"] = np.asarray(self.root[HOLES_ARRAY][row]).astype(np.int64)
        else:
            record["holes_bits"] = np.full((16, 16), -1, dtype=np.int64)
        return record

    def rows_for_buckets(self, buckets: list[str] | None) -> list[int]:
        """Rows whose baked-in curation ``difficulty_bucket`` is in *buckets*.

        ``None`` returns every row.  Rows without curation metadata (store
        built without a manifest) only pass when no filter is requested.
        """
        if buckets is None:
            return list(range(self.index.num_rows))
        if "difficulty_bucket" not in self.index.column_names:
            raise ValueError(
                "store has no baked-in curation metadata (difficulty_bucket); "
                "rebuild with --curation-manifest to use bucket filtering"
            )
        wanted = {b.lower() for b in buckets}
        col = self.index["difficulty_bucket"].to_pylist()
        return [i for i, b in enumerate(col) if b is not None and str(b).lower() in wanted]


# ---------------------------------------------------------------------------
# Tileset texture images (C# extract-tilesets export; attached post-build)
# ---------------------------------------------------------------------------

TILESETS_GROUP = "tilesets"
TILESET_RGB_SIZE = 256


def attach_tileset_images(
    store_path: Path, manifests: list[Path], builds: list[str] | None = None
) -> dict:
    """Attach per-era tileset texture images to an existing V25 store.

    ``manifests`` are JSON files written by ``WowViewer.Tool.Harvest
    extract-tilesets`` (BLP decoded from the era's own MPQs — the same path
    can carry different pixels per era, which is exactly why the vocabulary is
    era-scoped).  Writes a ``tilesets`` group aligned to vocab ids:

    * ``tileset_rgb_256`` (V, 256, 256, 3) uint8 — the texture image
    * ``tileset_present`` (V,) uint8 — 1 where an image was resolved
    """
    from PIL import Image

    store_path = Path(store_path)
    root = zarr.open_group(str(store_path), mode="r+")
    vocab_table = pq.read_table(store_path / "tileset_vocab.parquet")
    vocab_rows = vocab_table.to_pylist()
    vocab_size = int(root.attrs.get("vocab_size", max(r["vocab_id"] for r in vocab_rows) + 1))

    # (build, normalized path) -> PNG file
    images: dict[tuple[str, str], Path] = {}
    for m_i, manifest_path in enumerate(manifests):
        with open(manifest_path, encoding="utf-8") as f:
            data = json.load(f)
        build_id = builds[m_i] if builds is not None else None
        if build_id is None:
            client_root = str(data.get("client_root", ""))
            for part in reversed(Path(client_root).parts):
                if part and part[0].isdigit() and "_" in part:
                    build_id = part
                    break
        if build_id is None:
            raise ValueError(f"{manifest_path}: cannot derive build id; pass builds=[...]")
        base = Path(manifest_path).parent
        for entry in data["tilesets"]:
            norm = str(entry["path"]).strip().lower().replace("\\", "/")
            images[(build_id, norm)] = base / entry["file"]

    if TILESETS_GROUP in root:
        del root[TILESETS_GROUP]
    group = root.create_group(TILESETS_GROUP)
    group.create_array(
        "tileset_rgb_256",
        shape=(vocab_size, TILESET_RGB_SIZE, TILESET_RGB_SIZE, 3),
        chunks=(1, TILESET_RGB_SIZE, TILESET_RGB_SIZE, 3),
        dtype=np.uint8,
        compressors=DEFAULT_CODEC,
    )
    group.create_array(
        "tileset_present",
        shape=(vocab_size,),
        chunks=(vocab_size,),
        dtype=np.uint8,
        compressors=DEFAULT_CODEC,
    )

    present = np.zeros(vocab_size, dtype=np.uint8)
    matched = 0
    for row in vocab_rows:
        vid = int(row["vocab_id"])
        key = (str(row.get("build") or ""), str(row["tileset_path"]))
        png = images.get(key)
        if png is None or not png.exists():
            continue
        img = Image.open(png).convert("RGB")
        if img.size != (TILESET_RGB_SIZE, TILESET_RGB_SIZE):
            img = img.resize((TILESET_RGB_SIZE, TILESET_RGB_SIZE), Image.LANCZOS)
        group["tileset_rgb_256"][vid] = np.asarray(img, dtype=np.uint8)
        present[vid] = 1
        matched += 1
    group["tileset_present"][:] = present

    root.attrs.update(
        {
            "tileset_images_attached": True,
            "tileset_images_sources": [str(p) for p in manifests],
            "tileset_images_matched": matched,
        }
    )
    return {"vocab_size": vocab_size, "matched": matched}


# ---------------------------------------------------------------------------
# True MCNK hole bitmasks (C# extract-holes export; attached post-build)
# ---------------------------------------------------------------------------

HOLES_ARRAY = "holes_bits_16"


def attach_holes_bits(
    store_path: Path, exports: list[Path], builds: list[str] | None = None
) -> dict:
    """Attach raw per-chunk MCNK hole bitmasks to an existing V25 store.

    ``exports`` are JSON files written by ``WowViewer.Tool.Harvest
    extract-holes`` (era-aware C# reader — the same hole field the terrain
    renderer consumes).  Joins on (build, map, tile_x, tile_y) and writes
    ``holes_bits_16`` (N, 16, 16) int32: the uint16 hole-group bitmask per
    chunk, ``-1`` where the export carries no data for a tile.

    This replaces the excluded ``holes_16`` V18 signal, which was derived from
    the wrong MCNK header field (Spec 094 audit) and is unusable.
    """
    store_path = Path(store_path)
    root = zarr.open_group(str(store_path), mode="r+")
    index = pq.read_table(store_path / "index.parquet")
    n = index.num_rows

    holes_by_key: dict[tuple[str, str, int, int], np.ndarray] = {}
    for export_i, export_path in enumerate(exports):
        with open(export_path, encoding="utf-8") as f:
            data = json.load(f)
        # The store joins on canonical underscore build ids; take an explicit
        # override when given, else derive from the export's staged client
        # directory name (output/tmp/wowarchive-clients/<build_id>/...).
        build_id = builds[export_i] if builds is not None else None
        if build_id is None:
            client_root = str(data.get("client_root", ""))
            for part in reversed(Path(client_root).parts):
                if part and part[0].isdigit() and "_" in part:
                    build_id = part
                    break
        if build_id is None:
            raise ValueError(
                f"{export_path}: cannot derive build id from client_root; pass builds=[...]"
            )
        for map_name, tiles in data["maps"].items():
            for tile in tiles:
                mask = np.asarray(tile["holes"], dtype=np.int32).reshape(16, 16)
                holes_by_key[(build_id, map_name, int(tile["x"]), int(tile["y"]))] = mask

    builds = index["build"].to_pylist()
    maps = index["map"].to_pylist()
    xs = index["tile_x"].to_pylist()
    ys = index["tile_y"].to_pylist()

    out = np.full((n, 16, 16), -1, dtype=np.int32)
    matched = 0
    holed = 0
    for i in range(n):
        mask = holes_by_key.get((str(builds[i]), str(maps[i]), int(xs[i]), int(ys[i])))
        if mask is None:
            continue
        out[i] = mask
        matched += 1
        if mask.max() > 0:
            holed += 1

    if HOLES_ARRAY in root:
        del root[HOLES_ARRAY]
    root.create_array(
        HOLES_ARRAY,
        shape=(n, 16, 16),
        chunks=(1, 16, 16),
        dtype=np.int32,
        compressors=DEFAULT_CODEC,
    )
    root[HOLES_ARRAY][:] = out
    root.attrs.update(
        {
            "holes_bits_attached": True,
            "holes_bits_sources": [str(p) for p in exports],
            "holes_bits_matched": matched,
            "holes_bits_holed_tiles": holed,
        }
    )
    return {"rows": n, "matched": matched, "holed": holed}


# ---------------------------------------------------------------------------
# PM4 segment records (pre-parsed; attached post-build)
# ---------------------------------------------------------------------------

PM4_SEGMENTS_TABLE = "pm4_segments.parquet"


def _segment_to_row(rec: Pm4SegmentSignalRecord) -> dict:
    b = rec.bounds
    return {
        "segment_id": rec.segment_id,
        "has_bounds": b is not None,
        "bmin_x": float(b.min[0]) if b else 0.0,
        "bmin_y": float(b.min[1]) if b else 0.0,
        "bmin_z": float(b.min[2]) if b else 0.0,
        "bmax_x": float(b.max[0]) if b else 0.0,
        "bmax_y": float(b.max[1]) if b else 0.0,
        "bmax_z": float(b.max[2]) if b else 0.0,
        "footprint_hull_json": json.dumps(rec.footprint_hull),
        "hs_min": float(rec.height_stats.minimum_plane_distance),
        "hs_max": float(rec.height_stats.maximum_plane_distance),
        "hs_avg": float(rec.height_stats.average_plane_distance),
        "surface_family_histogram_json": json.dumps(rec.surface_family_histogram),
        "ts_surface_count": int(rec.topology_stats.surface_count),
        "ts_total_index_count": int(rec.topology_stats.total_index_count),
        "ts_anchor_point_count": int(rec.topology_stats.anchor_point_count),
        "ts_anchor_normal_count": int(rec.topology_stats.anchor_normal_count),
        "as_linked_position_ref_count": int(rec.anchor_signals.linked_position_ref_count),
        "as_normal_heading_count": int(rec.anchor_signals.normal_heading_count),
        "as_terminator_count": int(rec.anchor_signals.terminator_count),
        "as_floor_minimum": int(rec.anchor_signals.floor_minimum),
        "as_floor_maximum": int(rec.anchor_signals.floor_maximum),
        "as_heading_min": rec.anchor_signals.heading_minimum_degrees,
        "as_heading_max": rec.anchor_signals.heading_maximum_degrees,
        "as_heading_mean": rec.anchor_signals.heading_mean_degrees,
        "signal_version": rec.signal_version,
        "signal_store_row": rec.signal_store_row,
        "tile_coordinates": ",".join(rec.tile_coordinates),
    }


def _row_to_segment(row: dict) -> Pm4SegmentSignalRecord:
    bounds = None
    if row["has_bounds"]:
        bounds = Pm4Bounds3(
            min=(row["bmin_x"], row["bmin_y"], row["bmin_z"]),
            max=(row["bmax_x"], row["bmax_y"], row["bmax_z"]),
        )
    return Pm4SegmentSignalRecord(
        segment_id=row["segment_id"],
        bounds=bounds,
        footprint_hull=[tuple(p) for p in json.loads(row["footprint_hull_json"])],
        height_stats=Pm4SegmentHeightStats(
            minimum_plane_distance=row["hs_min"],
            maximum_plane_distance=row["hs_max"],
            average_plane_distance=row["hs_avg"],
        ),
        surface_family_histogram=json.loads(row["surface_family_histogram_json"]),
        topology_stats=Pm4SegmentTopologyStats(
            surface_count=row["ts_surface_count"],
            total_index_count=row["ts_total_index_count"],
            anchor_point_count=row["ts_anchor_point_count"],
            anchor_normal_count=row["ts_anchor_normal_count"],
        ),
        anchor_signals=Pm4SegmentAnchorSignals(
            linked_position_ref_count=row["as_linked_position_ref_count"],
            normal_heading_count=row["as_normal_heading_count"],
            terminator_count=row["as_terminator_count"],
            floor_minimum=row["as_floor_minimum"],
            floor_maximum=row["as_floor_maximum"],
            heading_minimum_degrees=row["as_heading_min"],
            heading_maximum_degrees=row["as_heading_max"],
            heading_mean_degrees=row["as_heading_mean"],
        ),
        signal_version=row["signal_version"] or "",
        signal_store_row=row["signal_store_row"],
        tile_coordinates=[t for t in row["tile_coordinates"].split(",") if t],
    )


def attach_pm4_segments(store_path: Path, segment_export_json: Path) -> int:
    """Attach pre-parsed PM4 segment records (C# export JSON) to a V25 store.

    Uses ``harvester.pm4_asset_matching.json_import`` — no PM4 binary parsing
    happens in Python (FR-102-402).  Returns the number of records written.
    """
    from harvester.pm4_asset_matching.json_import import import_segment_export

    records = import_segment_export(segment_export_json)
    rows = [_segment_to_row(r) for r in records]
    pq.write_table(pa.Table.from_pylist(rows), Path(store_path) / PM4_SEGMENTS_TABLE)
    return len(rows)


def load_pm4_segment_records(
    store_path: Path, tile_coordinate: str | None = None
) -> list[Pm4SegmentSignalRecord]:
    """Load pre-parsed PM4 segment records from a V25 store (or any directory).

    ``tile_coordinate`` (e.g. ``"27_29"``) filters to segments touching that tile.
    """
    table_path = Path(store_path) / PM4_SEGMENTS_TABLE
    if not table_path.exists():
        return []
    table = pq.read_table(table_path)
    rows = table.to_pylist()
    records = [_row_to_segment(r) for r in rows]
    if tile_coordinate is not None:
        records = [r for r in records if tile_coordinate in r.tile_coordinates]
    return records


# ---------------------------------------------------------------------------
# Structured prediction output store (inference CLI)
# ---------------------------------------------------------------------------


def write_prediction_store(
    output: Path,
    predictions: dict[str, np.ndarray],
    placements: list[dict] | None = None,
    attrs: dict | None = None,
    overwrite: bool = True,
) -> Path:
    """Write model predictions into a structured Zarr group (Blosc LZ4 level 1).

    ``predictions`` maps array names (e.g. ``height_257``, ``wdl_height_33``,
    ``object_mask_256``, ``clean_minimap_256``, ``alpha_256``, ``mcly_vocab_ids``,
    ``mtex_probs``) to per-tile numpy arrays (leading batch/tile dimension).
    ``placements`` (optional) is written to ``placements.parquet``.
    """
    output = Path(output)
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"{output} exists")
        import shutil

        shutil.rmtree(output)

    root = zarr.open_group(str(output), mode="w")
    for name, arr in predictions.items():
        arr = np.asarray(arr)
        chunks = (1, *arr.shape[1:]) if arr.ndim > 1 else arr.shape
        root.create_array(
            name,
            shape=arr.shape,
            chunks=chunks,
            dtype=arr.dtype,
            compressors=DEFAULT_CODEC,
        )
        root[name][:] = arr

    if placements:
        pq.write_table(pa.Table.from_pylist(placements), output / "placements.parquet")

    root.attrs.update(
        {
            "v25_prediction_store": True,
            "v25_dataset_version": V25_DATASET_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            **(attrs or {}),
        }
    )
    return output
