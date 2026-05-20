from __future__ import annotations

import argparse
import json
import random
import struct
import subprocess
import threading
import warnings
from collections import deque
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image as _PILImage
from PIL import ImageDraw as _PILImageDraw
import pyarrow.parquet as pq
try:
    import zarr
    import zarr.storage
except ImportError:  # pragma: no cover - optional for raw-only inspection runs
    zarr = None  # type: ignore[assignment]

from build_v16_dataset import (
    ENDS_MAGIC,
    NPZB_MAGIC,
    REQUIRED_KEYS,
    _decode_metadata_json,
    _derive_liquid_supervision,
    _extract_tile_coords_from_metadata,
    _find_client_root,
    _find_harvest_tool,
    _normalize_map_name,
)


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DATASET_ROOT = _PROJECT_ROOT / "output" / "datasets" / "v16"
_DEFAULT_OUTPUT = _DATASET_ROOT / "harvest_signal_inspection"
_KIND_CHOICES = ("mh2o", "mclq", "wl", "unified", "mcnk_liquid", "object", "placement")


def _pump_stderr(stderr_pipe, map_name: str, tail: deque[str]) -> None:
    try:
        for raw in iter(stderr_pipe.readline, b""):
            if not raw:
                break
            line = raw.decode("utf-8", errors="replace").rstrip()
            if not line:
                continue
            tail.append(line)
            print(f"    [harvest:{map_name}] {line}", flush=True)
    finally:
        try:
            stderr_pipe.close()
        except Exception:
            pass


def _open_store(zarr_path: Path):
    if zarr is None:
        raise RuntimeError("zarr is not installed; finalized-store comparison is unavailable in this interpreter")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Object at .* is not recognized as a component of a Zarr hierarchy\.",
            category=UserWarning,
        )
        store = zarr.storage.LocalStore(str(zarr_path), read_only=True)
        root = zarr.open_group(store=store, mode="r")
    return store, root


def _load_index_lookup(zarr_path: Path) -> tuple[dict[tuple[str, int, int], int], object | None, dict[int, dict[str, object]]]:
    index_path = zarr_path / "index.parquet"
    if zarr is None or not index_path.exists():
        return {}, None, {}

    store, root = _open_store(zarr_path)
    table = pq.read_table(str(index_path))
    rows_by_tile_id: dict[int, dict[str, object]] = {}
    lookup: dict[tuple[str, int, int], int] = {}
    try:
        for i in range(table.num_rows):
            row = {col: table.column(col)[i].as_py() for col in table.column_names}
            tile_id = int(row.get("tile_id", i))
            map_name = str(row.get("map", ""))
            tile_x = int(row.get("tile_x", -1) or -1)
            tile_y = int(row.get("tile_y", -1) or -1)
            rows_by_tile_id[tile_id] = row
            lookup[(map_name, tile_x, tile_y)] = tile_id
    except Exception:
        store.close()
        raise
    return lookup, (store, root), rows_by_tile_id


def _to_2d(arr: np.ndarray | None) -> np.ndarray | None:
    if arr is None:
        return None
    out = np.asarray(arr)
    if out.ndim < 2:
        return None
    if out.ndim > 2:
        out = np.squeeze(out)
        if out.ndim != 2:
            return None
    return out


def _presence_from_raw(arr: np.ndarray | None) -> np.ndarray | None:
    src = _to_2d(arr)
    if src is None:
        return None
    if src.dtype == np.bool_:
        return src.astype(np.float32)
    out = src.astype(np.float32)
    if np.issubdtype(src.dtype, np.integer):
        return (out > 0.0).astype(np.float32)
    return (np.abs(out) > 1e-6).astype(np.float32)


def _extract_mcnk_flag_grid(data: dict[str, np.ndarray], meta: dict[str, object]) -> np.ndarray | None:
    raw_chunks = meta.get("raw_chunks")
    if not isinstance(raw_chunks, list):
        return None

    flags = np.zeros((16, 16), dtype=np.uint32)
    any_chunk = False
    for raw_chunk in raw_chunks:
        if not isinstance(raw_chunk, dict):
            continue
        if str(raw_chunk.get("scope", "")).lower() != "mcnk":
            continue
        if str(raw_chunk.get("chunk_id", "")).upper() != "MCNK":
            continue

        chunk_x = raw_chunk.get("chunk_x")
        chunk_y = raw_chunk.get("chunk_y")
        entry_name = raw_chunk.get("entry_name")
        if not isinstance(chunk_x, int) or not isinstance(chunk_y, int) or not isinstance(entry_name, str):
            continue
        if not (0 <= chunk_x < 16 and 0 <= chunk_y < 16):
            continue

        payload = data.get(entry_name)
        if payload is None:
            continue
        raw = np.asarray(payload)
        if raw.ndim != 1 or raw.size < 4:
            continue

        flags[chunk_y, chunk_x] = struct.unpack_from("<I", raw.astype(np.uint8, copy=False).tobytes(), 0)[0]
        any_chunk = True

    return flags if any_chunk else None


def _classify_mcnk_liquid_type(flags: int) -> int:
    has_water = (flags & 0x0C) != 0
    has_magma = (flags & 0x10) != 0
    has_slime = (flags & 0x20) != 0
    kind_count = int(has_water) + int(has_magma) + int(has_slime)
    if kind_count <= 0:
        return 0
    if kind_count > 1:
        return 4
    if has_slime:
        return 3
    if has_magma:
        return 2
    return 1


def _mcnk_liquid_presence_from_flags(flags: np.ndarray | None) -> np.ndarray | None:
    if flags is None:
        return None
    return ((np.asarray(flags, dtype=np.uint32) & 0x3C) != 0).astype(np.float32)


def _mcnk_liquid_types_from_flags(flags: np.ndarray | None) -> np.ndarray | None:
    if flags is None:
        return None
    src = np.asarray(flags, dtype=np.uint32)
    out = np.zeros(src.shape, dtype=np.uint8)
    for y in range(src.shape[0]):
        for x in range(src.shape[1]):
            out[y, x] = _classify_mcnk_liquid_type(int(src[y, x]))
    return out


def _normalize_u8_mask(arr: np.ndarray | None, target: int = 256) -> np.ndarray:
    if arr is None:
        return np.zeros((target, target), dtype=np.uint8)
    src = np.asarray(arr).astype(np.float32)
    if src.ndim != 2:
        return np.zeros((target, target), dtype=np.uint8)
    if src.shape[0] != target or src.shape[1] != target:
        src = src[:target, :target]
        if src.shape[0] < target or src.shape[1] < target:
            out = np.zeros((target, target), dtype=np.float32)
            out[: src.shape[0], : src.shape[1]] = src
            src = out
    src = np.clip(src, 0.0, 1.0)
    return (src * 255.0).astype(np.uint8)


def _normalize_instance_u8(arr: np.ndarray | None, target: int = 256) -> np.ndarray:
    if arr is None:
        return np.zeros((target, target), dtype=np.uint8)
    src = np.asarray(arr)
    if src.ndim != 2:
        return np.zeros((target, target), dtype=np.uint8)
    src = src[:target, :target]
    if src.shape[0] < target or src.shape[1] < target:
        out = np.zeros((target, target), dtype=src.dtype)
        out[: src.shape[0], : src.shape[1]] = src
        src = out
    src = src.astype(np.int32, copy=False)
    nonzero = src > 0
    out = np.zeros_like(src, dtype=np.uint8)
    if np.any(nonzero):
        max_val = int(src[nonzero].max())
        out[nonzero] = np.clip((src[nonzero] / max(max_val, 1)) * 255.0, 0.0, 255.0).astype(np.uint8)
    return out


def _to_gray_u8(arr: np.ndarray | None, *, absolute: bool = False) -> np.ndarray:
    if arr is None:
        return np.zeros((256, 256), dtype=np.uint8)
    src = np.asarray(arr).astype(np.float32)
    if src.ndim == 3:
        src = np.squeeze(src)
    if src.ndim != 2:
        return np.zeros((256, 256), dtype=np.uint8)
    src = src[:256, :256]
    if src.shape[0] < 256 or src.shape[1] < 256:
        out = np.zeros((256, 256), dtype=np.float32)
        out[: src.shape[0], : src.shape[1]] = src
        src = out
    if absolute:
        src = np.abs(src)
    lo = float(np.min(src))
    hi = float(np.max(src))
    if hi - lo < 1e-8:
        return np.zeros_like(src, dtype=np.uint8)
    return (np.clip((src - lo) / (hi - lo), 0.0, 1.0) * 255.0).astype(np.uint8)


def _resize_rgb_or_gray(arr: np.ndarray, size: int = 256) -> np.ndarray:
    img = _PILImage.fromarray(arr)
    img = img.resize((size, size), _PILImage.Resampling.NEAREST)
    return np.asarray(img)


def _draw_label(img_u8: np.ndarray, text: str) -> np.ndarray:
    if img_u8.ndim == 2:
        rgb = np.repeat(img_u8[:, :, None], 3, axis=2)
    else:
        rgb = img_u8
    img = _PILImage.fromarray(rgb.astype(np.uint8), "RGB")
    drw = _PILImageDraw.Draw(img)
    drw.rectangle([(0, 0), (img.width, 18)], fill=(0, 0, 0))
    drw.text((4, 3), text, fill=(255, 255, 255))
    return np.asarray(img)


def _render_mcnk_liquid_types_u8(types: np.ndarray | None, target: int = 256) -> np.ndarray:
    if types is None:
        return np.zeros((target, target, 3), dtype=np.uint8)

    src = np.asarray(types, dtype=np.uint8)
    if src.ndim != 2:
        return np.zeros((target, target, 3), dtype=np.uint8)

    palette = np.array(
        [
            [0, 0, 0],        # 0 none
            [32, 140, 255],   # 1 water/ocean
            [255, 120, 24],   # 2 magma
            [80, 220, 120],   # 3 slime
            [255, 64, 200],   # 4 mixed/ambiguous
        ],
        dtype=np.uint8,
    )
    clipped = np.clip(src.astype(np.int32, copy=False), 0, len(palette) - 1)
    rgb = palette[clipped]
    return _resize_rgb_or_gray(rgb, target)


def _count_rows(arr: np.ndarray | None) -> int:
    if arr is None:
        return 0
    src = np.asarray(arr)
    return int(src.shape[0]) if src.ndim >= 2 else 0


def _sample_stats(data: dict[str, np.ndarray], meta: dict[str, object]) -> dict[str, object]:
    mh2o_presence = _presence_from_raw(data.get("mh2o_presence_mask"))
    mh2o_height_presence = _presence_from_raw(data.get("mh2o_surface_height"))
    mclq_presence = _presence_from_raw(data.get("mclq_presence_mask"))
    mclq_height_presence = _presence_from_raw(data.get("mclq_surface_height"))
    wl_presence = _presence_from_raw(data.get("wl_liquid_mask"))
    unified_presence = _presence_from_raw(data.get("unified_liquid_mask"))
    object_presence = _presence_from_raw(data.get("object_mask_257"))
    instance_mask = _to_2d(data.get("object_instance_mask_257"))
    mcnk_flags = _extract_mcnk_flag_grid(data, meta)
    mcnk_liquid_presence = _mcnk_liquid_presence_from_flags(mcnk_flags)
    mcnk_liquid_types = _mcnk_liquid_types_from_flags(mcnk_flags)

    mddf_rows = _count_rows(data.get("placement_mddf_data"))
    modf_rows = _count_rows(data.get("placement_modf_data"))
    placement_meta_mddf = int(meta.get("placement_mddf_count", 0) or 0)
    placement_meta_modf = int(meta.get("placement_modf_count", 0) or 0)

    mcnk_water_chunks = 0
    mcnk_magma_chunks = 0
    mcnk_slime_chunks = 0
    mcnk_mixed_chunks = 0
    if mcnk_liquid_types is not None:
        mcnk_water_chunks = int(np.count_nonzero(mcnk_liquid_types == 1))
        mcnk_magma_chunks = int(np.count_nonzero(mcnk_liquid_types == 2))
        mcnk_slime_chunks = int(np.count_nonzero(mcnk_liquid_types == 3))
        mcnk_mixed_chunks = int(np.count_nonzero(mcnk_liquid_types == 4))

    return {
        "mh2o_presence_sum": float(mh2o_presence.sum()) if mh2o_presence is not None else 0.0,
        "mh2o_height_presence_sum": float(mh2o_height_presence.sum()) if mh2o_height_presence is not None else 0.0,
        "mclq_presence_sum": float(mclq_presence.sum()) if mclq_presence is not None else 0.0,
        "mclq_height_presence_sum": float(mclq_height_presence.sum()) if mclq_height_presence is not None else 0.0,
        "wl_presence_sum": float(wl_presence.sum()) if wl_presence is not None else 0.0,
        "unified_presence_sum": float(unified_presence.sum()) if unified_presence is not None else 0.0,
        "mcnk_liquid_chunk_count": int(np.count_nonzero(mcnk_liquid_presence)) if mcnk_liquid_presence is not None else 0,
        "mcnk_water_chunks": mcnk_water_chunks,
        "mcnk_magma_chunks": mcnk_magma_chunks,
        "mcnk_slime_chunks": mcnk_slime_chunks,
        "mcnk_mixed_chunks": mcnk_mixed_chunks,
        "object_mask_sum": float(object_presence.sum()) if object_presence is not None else 0.0,
        "object_instance_nonzero": int(np.count_nonzero(instance_mask)) if instance_mask is not None else 0,
        "object_instance_max": int(instance_mask.max()) if instance_mask is not None and instance_mask.size > 0 else 0,
        "placement_mddf_rows": mddf_rows,
        "placement_modf_rows": modf_rows,
        "placement_meta_mddf": placement_meta_mddf,
        "placement_meta_modf": placement_meta_modf,
    }


def _matches_kind(kind: str, stats: dict[str, object]) -> bool:
    if kind == "mh2o":
        return float(stats["mh2o_presence_sum"]) > 0.0 or float(stats["mh2o_height_presence_sum"]) > 0.0
    if kind == "mclq":
        return float(stats["mclq_presence_sum"]) > 0.0 or float(stats["mclq_height_presence_sum"]) > 0.0
    if kind == "wl":
        return float(stats["wl_presence_sum"]) > 0.0
    if kind == "unified":
        return float(stats["unified_presence_sum"]) > 0.0
    if kind == "mcnk_liquid":
        return int(stats["mcnk_liquid_chunk_count"]) > 0
    if kind == "object":
        return float(stats["object_mask_sum"]) > 0.0 or int(stats["object_instance_nonzero"]) > 0
    if kind == "placement":
        return (
            int(stats["placement_mddf_rows"]) > 0
            or int(stats["placement_modf_rows"]) > 0
            or int(stats["placement_meta_mddf"]) > 0
            or int(stats["placement_meta_modf"]) > 0
        )
    return False


def _reservoir_add(pool: list[dict[str, object]], item: dict[str, object], *, seen: int, cap: int, rng: random.Random) -> None:
    if len(pool) < cap:
        pool.append(item)
        return
    slot = rng.randint(0, seen - 1)
    if slot < cap:
        pool[slot] = item


def _stream_samples(
    *,
    build: str,
    maps: list[str],
    kinds: list[str],
    sample_count: int,
    sample_seed: int,
) -> list[dict[str, object]]:
    harvest_tool = _find_harvest_tool()
    client_root = _find_client_root(build)
    if client_root is None:
        raise RuntimeError(f"Staged client root not found for build {build}")

    rng = random.Random(int(sample_seed))
    pools: dict[str, list[dict[str, object]]] = {kind: [] for kind in kinds}
    seen_counts: dict[str, int] = {kind: 0 for kind in kinds}

    for map_name in maps:
        cmd = [
            str(harvest_tool),
            "harvest-stream",
            "--client-root",
            str(client_root),
            "--map",
            map_name,
            "--build",
            build,
        ]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        if proc.stdout is None or proc.stderr is None:
            proc.terminate()
            raise RuntimeError(f"Failed to open harvest-stream pipes for {map_name}")

        stderr_tail: deque[str] = deque(maxlen=40)
        stderr_thread = threading.Thread(target=_pump_stderr, args=(proc.stderr, map_name, stderr_tail), daemon=True)
        stderr_thread.start()
        saw_end = False

        while True:
            header = proc.stdout.read(8)
            if not header:
                break
            if len(header) < 8:
                raise RuntimeError(f"Truncated stream header for {map_name}")

            magic = header[:4]
            if magic == ENDS_MAGIC:
                saw_end = True
                break
            if magic != NPZB_MAGIC:
                raise RuntimeError(f"Unexpected stream magic {magic!r} for {map_name}")

            length = struct.unpack("<I", header[4:8])[0]
            blob = proc.stdout.read(length)
            if not blob or len(blob) < length:
                raise RuntimeError(f"Truncated NPZ blob for {map_name}")

            data = dict(np.load(BytesIO(blob), allow_pickle=False))
            if not REQUIRED_KEYS.issubset(data.keys()):
                continue

            meta = _decode_metadata_json(data)
            tile_x, tile_y = _extract_tile_coords_from_metadata(meta)
            actual_map = _normalize_map_name(meta.get("map_name", map_name), map_name)
            stats = _sample_stats(data, meta)
            derived_liquid_mask, _, derived_has_liquid, derived_source = _derive_liquid_supervision(data)

            base_item: dict[str, object] = {
                "build": build,
                "map": actual_map,
                "tile_x": int(tile_x),
                "tile_y": int(tile_y),
                "tile_name": str(meta.get("tile_name", "")),
                "source_adt_path": str(meta.get("source_adt_path", "")),
                "metadata": meta,
                "stats": stats,
                "derived_has_liquid": bool(derived_has_liquid),
                "derived_source": derived_source,
                "npz_arrays": {key: np.asarray(value) for key, value in data.items()},
                "derived_liquid_mask": derived_liquid_mask,
            }

            for kind in kinds:
                if not _matches_kind(kind, stats):
                    continue
                seen_counts[kind] += 1
                item = dict(base_item)
                item["sample_kind"] = kind
                _reservoir_add(pools[kind], item, seen=seen_counts[kind], cap=sample_count, rng=rng)

        if proc.poll() is None:
            proc.terminate()
        proc.wait()
        stderr_thread.join(timeout=2.0)
        if not saw_end:
            raise RuntimeError(f"Harvest stream ended without ENDS sentinel for {map_name}")

    combined: list[dict[str, object]] = []
    for kind in kinds:
        combined.extend(sorted(pools[kind], key=lambda row: (str(row["map"]), int(row["tile_x"]), int(row["tile_y"]))))
    return combined


def _visualize_sample(
    sample: dict[str, object],
    zarr_match: dict[str, object] | None,
    *,
    panel_size: int = 256,
) -> np.ndarray:
    arrays: dict[str, np.ndarray] = sample["npz_arrays"]  # type: ignore[assignment]
    minimap = np.asarray(arrays["minimap_rgb_256"]).astype(np.uint8)

    raw_mh2o = _presence_from_raw(arrays.get("mh2o_presence_mask"))
    raw_mclq = _presence_from_raw(arrays.get("mclq_presence_mask"))
    raw_wl = _presence_from_raw(arrays.get("wl_liquid_mask"))
    raw_unified = _presence_from_raw(arrays.get("unified_liquid_mask"))
    raw_object = _presence_from_raw(arrays.get("object_mask_257"))
    raw_instance = _to_2d(arrays.get("object_instance_mask_257"))
    derived_liquid = np.asarray(sample["derived_liquid_mask"]).astype(np.float32)
    mcnk_flags = _extract_mcnk_flag_grid(arrays, sample["metadata"])  # type: ignore[arg-type]
    mcnk_types = _mcnk_liquid_types_from_flags(mcnk_flags)

    panels = [
        _draw_label(_resize_rgb_or_gray(minimap, panel_size), "minimap"),
        _draw_label(_resize_rgb_or_gray(_normalize_u8_mask(raw_mh2o), panel_size), "raw mh2o"),
        _draw_label(_resize_rgb_or_gray(_normalize_u8_mask(raw_mclq), panel_size), "raw mclq"),
        _draw_label(_resize_rgb_or_gray(_normalize_u8_mask(raw_wl), panel_size), "raw wl"),
        _draw_label(_resize_rgb_or_gray(_normalize_u8_mask(raw_unified), panel_size), "raw unified"),
        _draw_label(_render_mcnk_liquid_types_u8(mcnk_types, panel_size), "mcnk liquid flags"),
        _draw_label(_resize_rgb_or_gray(_normalize_u8_mask(derived_liquid), panel_size), "python derived"),
        _draw_label(_resize_rgb_or_gray(_normalize_u8_mask(raw_object), panel_size), "raw object"),
        _draw_label(_resize_rgb_or_gray(_normalize_instance_u8(raw_instance), panel_size), "raw instance"),
    ]

    if zarr_match is not None:
        panels.extend(
            [
                _draw_label(_resize_rgb_or_gray(_normalize_u8_mask(zarr_match.get("liquid_mask")), panel_size), "zarr liquid"),
                _draw_label(_resize_rgb_or_gray(_normalize_u8_mask(zarr_match.get("object_mask")), panel_size), "zarr object"),
                _draw_label(_resize_rgb_or_gray(_normalize_instance_u8(zarr_match.get("object_instance_mask")), panel_size), "zarr instance"),
            ]
        )

    return np.concatenate(panels, axis=1)


def _write_outputs(
    *,
    build: str,
    output_dir: Path,
    samples: list[dict[str, object]],
    zarr_lookup: dict[tuple[str, int, int], int],
    zarr_store_root: object | None,
    zarr_rows_by_tile_id: dict[int, dict[str, object]],
) -> None:
    build_dir = output_dir / build
    build_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    strips: list[np.ndarray] = []

    for i, sample in enumerate(samples):
        map_name = str(sample["map"])
        tile_x = int(sample["tile_x"])
        tile_y = int(sample["tile_y"])
        kind = str(sample["sample_kind"])

        zarr_match = None
        zarr_row = None
        if zarr_store_root is not None:
            tile_id = zarr_lookup.get((map_name, tile_x, tile_y))
            if tile_id is not None:
                store, root = zarr_store_root
                zarr_row = zarr_rows_by_tile_id.get(tile_id)
                zarr_match = {
                    "tile_id": tile_id,
                    "liquid_mask": root["liquid_mask"][tile_id].astype(np.float32),
                    "object_mask": root["object_mask"][tile_id].astype(np.float32),
                    "object_instance_mask": root["object_instance_mask"][tile_id].astype(np.int32),
                }

        npz_path = build_dir / f"sample_{i:02d}_{kind}_{map_name}_{tile_x}_{tile_y}.npz"
        arrays: dict[str, np.ndarray] = sample["npz_arrays"]  # type: ignore[assignment]
        save_payload = {k: np.asarray(v) for k, v in arrays.items() if k != "metadata.json"}
        save_payload["derived_liquid_mask"] = np.asarray(sample["derived_liquid_mask"])
        mcnk_flags = _extract_mcnk_flag_grid(arrays, sample["metadata"])  # type: ignore[arg-type]
        if mcnk_flags is not None:
            save_payload["derived_mcnk_flags_16"] = mcnk_flags.astype(np.uint32, copy=False)
            save_payload["derived_mcnk_liquid_presence_16"] = _mcnk_liquid_presence_from_flags(mcnk_flags).astype(np.float32, copy=False)
            save_payload["derived_mcnk_liquid_types_16"] = _mcnk_liquid_types_from_flags(mcnk_flags).astype(np.uint8, copy=False)
        save_payload["metadata_json_utf8"] = np.frombuffer(json.dumps(sample["metadata"], indent=2).encode("utf-8"), dtype=np.uint8)
        np.savez_compressed(npz_path, **save_payload)

        strip = _visualize_sample(sample, zarr_match)
        title = (
            f"{build} sample={i:02d} kind={kind} map={map_name} xy=({tile_x},{tile_y}) "
            f"derived={sample['derived_source']} mddf={sample['stats']['placement_mddf_rows']} "
            f"modf={sample['stats']['placement_modf_rows']} obj_sum={sample['stats']['object_mask_sum']:.1f}"
        )
        img = _PILImage.fromarray(strip, "RGB")
        drw = _PILImageDraw.Draw(img)
        drw.rectangle([(0, 0), (img.width, 18)], fill=(24, 24, 24))
        drw.text((4, 3), title, fill=(240, 240, 240))
        labeled = np.asarray(img)
        strips.append(labeled)

        image_path = build_dir / f"sample_{i:02d}_{kind}_{map_name}_{tile_x}_{tile_y}.png"
        img.save(image_path)

        summary_rows.append(
            {
                "sample_index": i,
                "sample_kind": kind,
                "build": build,
                "map": map_name,
                "tile_x": tile_x,
                "tile_y": tile_y,
                "tile_name": sample["tile_name"],
                "source_adt_path": sample["source_adt_path"],
                "derived_has_liquid": sample["derived_has_liquid"],
                "derived_source": sample["derived_source"],
                "stats": sample["stats"],
                "zarr_row": zarr_row,
                "npz_path": str(npz_path),
                "image_path": str(image_path),
            }
        )

    if strips:
        cols = 1
        width = max(strip.shape[1] for strip in strips)
        height = sum(strip.shape[0] for strip in strips)
        canvas = _PILImage.new("RGB", (width * cols, height), (12, 12, 12))
        y = 0
        for strip in strips:
            img = _PILImage.fromarray(strip, "RGB")
            canvas.paste(img, (0, y))
            y += strip.shape[0]
        canvas.save(build_dir / f"{build}.overview.png")

    with (build_dir / f"{build}.samples.json").open("w", encoding="utf-8") as fh:
        json.dump(summary_rows, fh, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect raw V16 harvest NPZ samples and compare them to finalized Zarr outputs")
    parser.add_argument("--build", required=True, help="Build key, e.g. 3_3_5_12340")
    parser.add_argument("--maps", nargs="+", required=True, help="Maps to stream from the staged client")
    parser.add_argument("--kinds", nargs="+", choices=_KIND_CHOICES, default=["mh2o", "mclq", "object", "placement"], help="Sample categories to collect")
    parser.add_argument("--sample-count", type=int, default=4, help="Reservoir sample size per category")
    parser.add_argument("--sample-seed", type=int, default=1234, help="Reservoir sampling seed")
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT, help="Output directory for summaries, NPZs, and PNGs")
    args = parser.parse_args()

    zarr_path = _DATASET_ROOT / f"{args.build}.zarr"
    zarr_lookup, zarr_store_root, zarr_rows_by_tile_id = _load_index_lookup(zarr_path)
    try:
        samples = _stream_samples(
            build=args.build,
            maps=[str(v) for v in args.maps],
            kinds=[str(v) for v in args.kinds],
            sample_count=int(args.sample_count),
            sample_seed=int(args.sample_seed),
        )
        _write_outputs(
            build=args.build,
            output_dir=args.output_dir,
            samples=samples,
            zarr_lookup=zarr_lookup,
            zarr_store_root=zarr_store_root,
            zarr_rows_by_tile_id=zarr_rows_by_tile_id,
        )
        print(f"Wrote {len(samples)} samples to {args.output_dir / args.build}")
    finally:
        if zarr_store_root is not None:
            store, _root = zarr_store_root
            store.close()


if __name__ == "__main__":
    main()
