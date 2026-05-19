"""Build V16 consolidated Zarr dataset directly from game client archives.

Single-pass pipeline: C# harvester streams NPZ blobs -> Python reads from pipe -> Zarr.
NO intermediate files on disk. The Zarr store IS the dataset.

Now carries ALL available NPZ signals including per-instance object mask and placement data.

Usage:
    cd wow-viewer/data-harvester

    # Build one build (auto-discovered terrain maps):
    uv run python scripts/build_v16_dataset.py build --build 3_3_5_12340

    # Build multiple builds:
    uv run python scripts/build_v16_dataset.py build --builds 3_3_5_12340 4_0_0_11927

    # Limit tiles (for testing):
    uv run python scripts/build_v16_dataset.py build --build 3_3_5_12340 --limit 100

    # Only specific maps:
    uv run python scripts/build_v16_dataset.py build --build 3_3_5_12340 --maps Azeroth Northrend

    # Check stats:
    uv run python scripts/build_v16_dataset.py stats --build 3_3_5_12340
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import subprocess
import sys
import threading
import time
import warnings
from collections import deque
from io import BytesIO
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import zarr
import zarr.codecs
import zarr.storage

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_HARVEST_TOOL_DIR = _PROJECT_ROOT / "tools" / "harvest" / "WowViewer.Tool.Harvest" / "bin" / "Debug" / "net10.0"
_DATASET_ROOT = _PROJECT_ROOT / "output" / "datasets" / "v16"
_CLIENT_ROOTS = _PROJECT_ROOT.parent / "output" / "tmp" / "wowarchive-clients"

# ── NPZ key → Zarr array name mapping ──────────────────────────────────
# All signals from the C# harvester that map to fixed-shape Zarr arrays.
OUTPUT_ARRAY_NAMES = {
    "height_257": "height_257",
    "mcnr_normal_xyz": "normal_xyz",
    "mcal_alpha_pack_256": "alpha_256",
    "hole_mask_16": "holes_16",
    "unified_liquid_mask": "liquid_mask",
    "unified_liquid_height": "liquid_height",
    "object_mask_257": "object_mask",
    "object_precise_mask_257": "object_precise_mask",
    "object_instance_mask_257": "object_instance_mask",
    "minimap_rgb_256": "minimap_rgb",
    "mcsh_shadow_mask_256": "shadow_mask",
    "mcly_texture_ids": "mcly_texture_ids",
    "mcly_layer_mask": "mcly_layer_mask",
}

DTYPES = {
    "height_257": np.float32, "normal_xyz": np.float32, "normal_mask": np.bool_,
    "alpha_256": np.float32, "holes_16": np.bool_, "liquid_mask": np.float32,
    "liquid_height": np.float32, "object_mask": np.bool_, "object_precise_mask": np.float32,
    "object_instance_mask": np.int32, "minimap_rgb": np.uint8,
    "shadow_mask": np.float32, "mcly_texture_ids": np.int32, "mcly_layer_mask": np.float32,
}

FILL_VALUES = {
    "height_257": 0.0, "normal_xyz": 0.0, "normal_mask": False,
    "alpha_256": 0.0, "holes_16": False, "liquid_mask": 0.0,
    "liquid_height": 0.0, "object_mask": False, "object_precise_mask": 0.0,
    "object_instance_mask": 0, "minimap_rgb": 0,
    "shadow_mask": 0.0, "mcly_texture_ids": -1, "mcly_layer_mask": 0.0,
}

SHAPES = {
    "height_257": (257, 257), "normal_xyz": (257, 257, 3), "normal_mask": (257, 257),
    "alpha_256": (256, 256, 4), "holes_16": (16, 16), "liquid_mask": (256, 256),
    "liquid_height": (256, 256), "object_mask": (257, 257), "object_precise_mask": (257, 257),
    "object_instance_mask": (257, 257), "minimap_rgb": (256, 256, 3),
    "shadow_mask": (256, 256), "mcly_texture_ids": (16, 16, 4), "mcly_layer_mask": (16, 16, 4),
}

CHUNK_SIZES = {
    "height_257": (64, 257, 257), "normal_xyz": (64, 257, 257, 3),
    "normal_mask": (256, 257, 257), "alpha_256": (64, 256, 256, 4),
    "holes_16": (1024, 16, 16), "liquid_mask": (64, 256, 256),
    "liquid_height": (64, 256, 256), "object_mask": (256, 257, 257),
    "object_precise_mask": (256, 257, 257), "object_instance_mask": (256, 257, 257),
    "minimap_rgb": (64, 256, 256, 3), "shadow_mask": (64, 256, 256),
    "mcly_texture_ids": (1024, 16, 16, 4), "mcly_layer_mask": (256, 16, 16, 4),
}

ALL_ARRAY_KEYS = [
    "height_257", "normal_xyz", "normal_mask", "alpha_256", "holes_16",
    "liquid_mask", "liquid_height", "object_mask", "object_precise_mask",
    "object_instance_mask", "minimap_rgb", "shadow_mask",
    "mcly_texture_ids", "mcly_layer_mask",
]

LIQUID_SOURCE_KEYS = ("mh2o", "mclq", "unified", "wl")

# Integration keys: derive has_* flags for these signals in the Parquet index.
# Include all fixed-shape arrays plus explicit liquid-source provenance flags.
SIGNAL_FLAG_KEYS = [
    *ALL_ARRAY_KEYS,
    *(f"liquid_source_{name}" for name in LIQUID_SOURCE_KEYS),
]

REQUIRED_KEYS = {"minimap_rgb_256", "height_257"}
DEFAULT_CODEC = "lz4"
DEFAULT_CLEVEL = 1
DEFAULT_SHUFFLE = "shuffle"
WRITE_RETRY_ATTEMPTS = 8
WRITE_RETRY_BASE_DELAY_SECONDS = 0.15
WRITE_BATCH_SIZE = 16


def _decode_metadata_json(tile_blob: dict[str, np.ndarray]) -> dict[str, object]:
    payload = tile_blob.get("metadata.json")
    if payload is None:
        return {}
    try:
        if hasattr(payload, "tobytes"):
            raw = payload.tobytes()
        elif isinstance(payload, bytes):
            raw = payload
        else:
            raw = bytes(payload)
        decoded = json.loads(raw.decode("utf-8"))
    except Exception:
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _tile_rejection_report_path(output_path: Path, build_name: str) -> Path:
    return output_path.parent / f"{build_name}.rejected_tiles.jsonl"


def _resume_state_path(output_path: Path) -> Path:
    return output_path / "_resume_state.json"


def _load_resume_state(output_path: Path) -> dict[str, object] | None:
    path = _resume_state_path(output_path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _open_zarr_group_readonly(zarr_path: Path):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Object at .* is not recognized as a component of a Zarr hierarchy\.",
            category=UserWarning,
        )
        store = zarr.storage.LocalStore(str(zarr_path), read_only=True)
        root = zarr.open_group(store=store, mode="r")
    return store, root


def _load_completed_final_store_state(output_path: Path) -> dict[str, object] | None:
    if not output_path.exists():
        return None

    idx_path = output_path / "index.parquet"
    if not idx_path.exists():
        return None

    table = pq.read_table(str(idx_path), columns=["map"])
    store, root = _open_zarr_group_readonly(output_path)
    try:
        array_length = int(root["height_257"].shape[0])
    finally:
        store.close()

    if table.num_rows != array_length:
        return None

    state = _load_resume_state(output_path) or {}
    if not state:
        maps: list[str] = []
        seen: set[str] = set()
        for value in table.column("map"):
            map_name = str(value.as_py())
            if map_name in seen:
                continue
            seen.add(map_name)
            maps.append(map_name)
        state = {
            "build": output_path.stem.replace(".zarr", ""),
            "requested_maps": maps,
            "completed_maps": maps,
            "valid_tiles": table.num_rows,
            "skipped_zero_usable_maps": 0,
            "rejected_tile_count": 0,
            "codec": "unknown-final-store",
            "clevel": -1,
            "shuffle": "unknown-final-store",
            "capacity": table.num_rows,
            "finalized": True,
            "inferred_from_final_store": True,
        }
    else:
        state = dict(state)
        state["finalized"] = True

    return state


def _write_resume_state(
    output_path: Path,
    *,
    build: str,
    requested_maps: list[str],
    completed_maps: list[str],
    valid: int,
    skipped_zero_usable_maps: int,
    rejected_tile_count: int,
    codec_name: str,
    codec_level: int,
    codec_shuffle: str,
    capacity: int,
    finalized: bool = False,
) -> None:
    state = {
        "build": build,
        "requested_maps": requested_maps,
        "completed_maps": completed_maps,
        "valid_tiles": valid,
        "skipped_zero_usable_maps": skipped_zero_usable_maps,
        "rejected_tile_count": rejected_tile_count,
        "codec": codec_name,
        "clevel": codec_level,
        "shuffle": codec_shuffle,
        "capacity": capacity,
        "finalized": finalized,
    }
    _resume_state_path(output_path).write_text(json.dumps(state, indent=2), encoding="utf-8")

# ── Placement data columns for the companion Parquet table ────────────
PLACEMENT_COLUMNS_MDDF = [
    "nameId", "uniqueId", "posX", "posY", "posZ", "rotX", "rotY", "rotZ", "scale",
]
PLACEMENT_COLUMNS_MODF = [
    "nameId", "uniqueId", "posX", "posY", "posZ", "rotX", "rotY", "rotZ",
    "bbMinX", "bbMinY", "bbMinZ", "bbMaxX", "bbMaxY", "bbMaxZ",
]

NPZB_MAGIC = b"NPZB"
ENDS_MAGIC = b"ENDS"


def _dir_size_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _is_retryable_windows_file_error(ex: BaseException) -> bool:
    if isinstance(ex, PermissionError):
        return True
    if isinstance(ex, OSError) and getattr(ex, "winerror", None) in {5, 32}:
        return True
    return False


def _flush_tile_batch_with_retry(
    arrays: dict[str, zarr.Array],
    start_index: int,
    pending_arrays: dict[str, list[np.ndarray]],
    pending_count: int,
    *,
    map_name: str,
) -> int:
    if pending_count <= 0:
        return start_index

    for key in ALL_ARRAY_KEYS:
        batch_value = np.stack(pending_arrays[key], axis=0)
        for attempt in range(1, WRITE_RETRY_ATTEMPTS + 1):
            try:
                arrays[key][start_index:start_index + pending_count] = batch_value
                break
            except Exception as ex:
                if not _is_retryable_windows_file_error(ex) or attempt == WRITE_RETRY_ATTEMPTS:
                    raise RuntimeError(
                        f"Failed writing tile batch start={start_index} count={pending_count} "
                        f"for map {map_name} array={key} after {attempt} attempts: {ex}"
                    ) from ex
                delay = WRITE_RETRY_BASE_DELAY_SECONDS * attempt
                print(
                    f"    Warning: retrying Zarr batch write for map {map_name} "
                    f"array={key} start={start_index} count={pending_count} after filesystem error: {ex}",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(delay)

    for key in ALL_ARRAY_KEYS:
        pending_arrays[key].clear()

    return start_index + pending_count


def _tail_text(lines: deque[str]) -> str:
    if not lines:
        return "(no stderr output)"
    return "\n".join(lines)


def _pump_stderr(stderr_pipe, map_name: str, tail: deque[str]) -> None:
    try:
        for raw in iter(stderr_pipe.readline, b""):
            if not raw:
                break
            line = raw.decode("utf-8", errors="replace").rstrip()
            if not line:
                continue
            tail.append(line)
            print(f"    [harvest:{map_name}] {line}", file=sys.stderr, flush=True)
    finally:
        try:
            stderr_pipe.close()
        except Exception:
            pass


def _find_harvest_tool() -> Path:
    exe = _HARVEST_TOOL_DIR / "WowViewer.Tool.Harvest.exe"
    if exe.exists():
        return exe
    for p in sorted((_PROJECT_ROOT / "tools" / "harvest" / "WowViewer.Tool.Harvest" / "bin" / "Debug").glob("*/WowViewer.Tool.Harvest.exe")):
        if p.exists():
            return p
    raise FileNotFoundError("Harvest tool not found. Build it first.")


def _find_client_root(build: str) -> Path | None:
    parent = _CLIENT_ROOTS / build
    if not parent.exists():
        return None
    for child in parent.iterdir():
        if child.is_dir() and ((child / "WoW.exe").exists() or (child / "Data").exists()):
            return child
    return None


def _process_tile_data(data: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, bool]] | None:
    if "minimap_rgb_256" not in data or "height_257" not in data:
        return None

    tile_arrays: dict[str, np.ndarray] = {}
    has_signals: dict[str, bool] = {}

    for src_key, dst_key in OUTPUT_ARRAY_NAMES.items():
        if src_key in data:
            tile_arrays[dst_key] = _normalize_array(data[src_key], dst_key)
            has_signals[dst_key] = True
        else:
            shape = SHAPES[dst_key]
            dtype = DTYPES[dst_key]
            fill = FILL_VALUES[dst_key]
            tile_arrays[dst_key] = np.full(shape, fill, dtype=dtype)
            has_signals[dst_key] = False

    # Rebuild liquid supervision from raw liquid signals with explicit priority.
    # This avoids over-trusting legacy unified_liquid_* payloads when they only carry WL* fallback.
    liquid_mask, liquid_height, liquid_has_signal, liquid_source = _derive_liquid_supervision(data)
    tile_arrays["liquid_mask"] = liquid_mask
    tile_arrays["liquid_height"] = liquid_height
    has_signals["liquid_mask"] = liquid_has_signal
    has_signals["liquid_height"] = liquid_has_signal
    for source_name in LIQUID_SOURCE_KEYS:
        has_signals[f"liquid_source_{source_name}"] = liquid_source == source_name

    if "mcnr_normal_xyz" in data:
        nrm = data["mcnr_normal_xyz"].astype(np.float32)
        normal_mask = (np.abs(nrm).sum(axis=-1) > 1e-6)
        zero_mask = ~normal_mask
        nrm[zero_mask] = [0.0, 0.0, 1.0]
        norms = np.linalg.norm(nrm, axis=-1, keepdims=True)
        norms = np.where(norms < 1e-6, 1.0, norms)
        nrm = nrm / norms
        tile_arrays["normal_xyz"] = nrm.astype(np.float32)
        has_signals["normal_xyz"] = True
    else:
        tile_arrays["normal_xyz"] = np.zeros((257, 257, 3), dtype=np.float32)
        normal_mask = np.zeros((257, 257), dtype=np.bool_)
        has_signals["normal_xyz"] = False

    tile_arrays["normal_mask"] = normal_mask.astype(np.bool_)
    has_signals["normal_mask"] = True

    return tile_arrays, has_signals


def _derive_liquid_supervision(
    data: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, bool, str | None]:
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

    def _coerce_liq(arr: np.ndarray, key: str) -> np.ndarray:
        return _normalize_array(arr, key)

    # 1) MH2O (preferred)
    mh2o_type = _to_2d(data.get("mh2o_type_mask"))
    if mh2o_type is not None:
        mh2o_mask = (mh2o_type.astype(np.float32) > 0.0).astype(np.float32)
        if float(mh2o_mask.max()) > 0.0:
            mh2o_height = _to_2d(data.get("mh2o_surface_height"))
            if mh2o_height is None:
                mh2o_height = np.zeros_like(mh2o_mask, dtype=np.float32)
            return (
                _coerce_liq(mh2o_mask, "liquid_mask"),
                _coerce_liq(mh2o_height.astype(np.float32), "liquid_height"),
                True,
                "mh2o",
            )

    # 2) MCLQ (next priority)
    mclq_type = _to_2d(data.get("mclq_type_mask"))
    if mclq_type is not None:
        mclq_mask = (mclq_type.astype(np.float32) > 0.0).astype(np.float32)
        if float(mclq_mask.max()) > 0.0:
            mclq_height = _to_2d(data.get("mclq_surface_height"))
            if mclq_height is None:
                mclq_height = np.zeros_like(mclq_mask, dtype=np.float32)
            return (
                _coerce_liq(mclq_mask, "liquid_mask"),
                _coerce_liq(mclq_height.astype(np.float32), "liquid_height"),
                True,
                "mclq",
            )

    # 3) Existing unified signal
    unified_mask = _to_2d(data.get("unified_liquid_mask"))
    if unified_mask is not None:
        unified_mask = unified_mask.astype(np.float32)
        if unified_mask.max() > 1.5:
            unified_mask = unified_mask / 255.0
        unified_mask = np.clip(unified_mask, 0.0, 1.0)
        if float(unified_mask.max()) > 0.0:
            unified_height = _to_2d(data.get("unified_liquid_height"))
            if unified_height is None:
                unified_height = np.zeros_like(unified_mask, dtype=np.float32)
            return (
                _coerce_liq(unified_mask, "liquid_mask"),
                _coerce_liq(unified_height.astype(np.float32), "liquid_height"),
                True,
                "unified",
            )

    # 4) WL* last-resort fallback
    wl_mask = _to_2d(data.get("wl_liquid_mask"))
    if wl_mask is not None:
        wl_mask = wl_mask.astype(np.float32)
        if wl_mask.max() > 1.5:
            wl_mask = wl_mask / 255.0
        wl_mask = np.clip(wl_mask, 0.0, 1.0)
        if float(wl_mask.max()) > 0.0:
            wl_height = _to_2d(data.get("wl_liquid_height"))
            if wl_height is None:
                wl_height = np.zeros_like(wl_mask, dtype=np.float32)
            return (
                _coerce_liq(wl_mask, "liquid_mask"),
                _coerce_liq(wl_height.astype(np.float32), "liquid_height"),
                True,
                "wl",
            )

    # No usable liquid signal
    return (
        np.zeros(SHAPES["liquid_mask"], dtype=np.float32),
        np.zeros(SHAPES["liquid_height"], dtype=np.float32),
        False,
        None,
    )


def _extract_metadata(data: dict[str, np.ndarray]) -> dict:
    meta_raw = data.get("metadata.json")
    if meta_raw is None:
        return {}
    try:
        if isinstance(meta_raw, str):
            return json.loads(meta_raw)
        return json.loads(meta_raw.tobytes().decode())
    except Exception:
        return {}


def _try_parse_tile_coords_from_stem(stem: str) -> tuple[int | None, int | None]:
    parts = stem.rsplit("_", 2)
    if len(parts) < 3:
        return None, None
    try:
        return int(parts[-2]), int(parts[-1])
    except (TypeError, ValueError):
        return None, None


def _try_parse_alpha_tile_coords(value: str) -> tuple[int | None, int | None]:
    marker = "alpha-tile("
    idx = value.lower().find(marker)
    if idx < 0:
        return None, None
    inside = value[idx + len(marker):]
    end = inside.find(")")
    if end < 0:
        return None, None
    pair = inside[:end].split(",", 1)
    if len(pair) != 2:
        return None, None
    try:
        tx = int(pair[0].strip())
        ty = int(pair[1].strip())
    except (TypeError, ValueError):
        return None, None
    if tx < 0 or tx > 63 or ty < 0 or ty > 63:
        return None, None
    return tx, ty


def _extract_tile_coords_from_metadata(meta: dict[str, object]) -> tuple[int, int]:
    tx = meta.get("tile_x")
    ty = meta.get("tile_y")
    if tx is not None and ty is not None:
        try:
            return int(tx), int(ty)
        except (TypeError, ValueError):
            pass

    tile_name = str(meta.get("tile_name", "") or "")
    if tile_name:
        alpha_tx, alpha_ty = _try_parse_alpha_tile_coords(tile_name)
        if alpha_tx is not None and alpha_ty is not None:
            return alpha_tx, alpha_ty
        parsed_tx, parsed_ty = _try_parse_tile_coords_from_stem(tile_name)
        if parsed_tx is not None and parsed_ty is not None:
            return parsed_tx, parsed_ty

    source = str(meta.get("source_adt_path", "") or "")
    if source:
        alpha_tx, alpha_ty = _try_parse_alpha_tile_coords(source)
        if alpha_tx is not None and alpha_ty is not None:
            return alpha_tx, alpha_ty
        parsed_tx, parsed_ty = _try_parse_tile_coords_from_stem(Path(source).stem)
        if parsed_tx is not None and parsed_ty is not None:
            return parsed_tx, parsed_ty

    return 0, 0


def _extract_placements(data: dict[str, np.ndarray], meta: dict) -> tuple[list[dict], list[dict], list[str], list[str]]:
    mddf_rows = []
    modf_rows = []
    mddf_names = meta.get("placement_mddf_names", [])
    modf_names = meta.get("placement_modf_names", [])

    mddf_data = data.get("placement_mddf_data")
    if mddf_data is not None and mddf_data.ndim == 2 and mddf_data.shape[0] > 0:
        for i in range(mddf_data.shape[0]):
            row = {col: float(mddf_data[i, j]) for j, col in enumerate(PLACEMENT_COLUMNS_MDDF) if j < mddf_data.shape[1]}
            row["instance_type"] = "mddf"
            row["instance_idx"] = i
            name_id = int(row.get("nameId", -1))
            row["asset_path"] = mddf_names[name_id] if 0 <= name_id < len(mddf_names) else ""
            mddf_rows.append(row)

    modf_data = data.get("placement_modf_data")
    if modf_data is not None and modf_data.ndim == 2 and modf_data.shape[0] > 0:
        for i in range(modf_data.shape[0]):
            row = {col: float(modf_data[i, j]) for j, col in enumerate(PLACEMENT_COLUMNS_MODF) if j < modf_data.shape[1]}
            row["instance_type"] = "modf"
            row["instance_idx"] = i
            name_id = int(row.get("nameId", -1))
            row["asset_path"] = modf_names[name_id] if 0 <= name_id < len(modf_names) else ""
            modf_rows.append(row)

    return mddf_rows, modf_rows, mddf_names, modf_names


def _normalize_array(arr: np.ndarray, dst_key: str) -> np.ndarray:
    arr = arr.astype(DTYPES.get(dst_key, np.float32))
    if dst_key == "alpha_256":
        if arr.max() > 1.5:
            arr = arr / 255.0
        arr = np.clip(arr, 0.0, 1.0)
    elif dst_key == "liquid_mask":
        if arr.max() > 1.5:
            arr = arr / 255.0
        arr = np.clip(arr, 0.0, 1.0)
    elif dst_key in ("holes_16", "object_mask"):
        arr = arr.astype(np.bool_)
    elif dst_key == "object_instance_mask":
        arr = arr.astype(np.int32)
    return _coerce_array_shape(arr, dst_key)


def _coerce_array_shape(arr: np.ndarray, dst_key: str) -> np.ndarray:
    target_shape = SHAPES[dst_key]
    if arr.shape == target_shape:
        return arr

    # Common case for terrain signals with variable layer counts:
    # squeeze accidental singleton axes, then restore missing trailing axes.
    arr = np.squeeze(arr)
    while arr.ndim < len(target_shape):
        arr = np.expand_dims(arr, axis=-1)

    fill = FILL_VALUES.get(dst_key, 0)
    coerced = np.full(target_shape, fill, dtype=DTYPES[dst_key])

    copy_rank = min(arr.ndim, len(target_shape))
    src_slices = []
    dst_slices = []
    for axis in range(copy_rank):
        extent = min(arr.shape[axis], target_shape[axis])
        src_slices.append(slice(0, extent))
        dst_slices.append(slice(0, extent))

    if src_slices:
        coerced[tuple(dst_slices)] = arr[tuple(src_slices)].astype(DTYPES[dst_key], copy=False)

    return coerced


def _discover_maps_for_build(harvest_tool: Path, client_root: Path) -> list[str]:
    def getv(row: dict, key: str, default=None):
        return row.get(key, row.get(key[:1].upper() + key[1:], default))

    cmd = [
        str(harvest_tool),
        "discover-maps",
        "--client-root",
        str(client_root),
    ]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr, flush=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"discover-maps failed for {client_root} with exit code {proc.returncode}"
        )

    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as ex:
        raise RuntimeError(
            f"discover-maps returned invalid JSON for {client_root}: {ex}"
        ) from ex

    if not isinstance(payload, list):
        raise RuntimeError(f"discover-maps returned unexpected payload type: {type(payload)!r}")

    included = [getv(row, "map") for row in payload if getv(row, "include")]
    print(f"Discovered {len(included)} trainable maps from WDT summaries")
    for row in payload:
        map_name = getv(row, "map", "<unknown>")
        reason = getv(row, "reason", "unknown")
        tiles = getv(row, "tilesWithData", 0)
        has_wmo = getv(row, "hasWorldModelAsset", False)
        has_tile = getv(row, "hasReadableTile", False)
        status = "include" if getv(row, "include") else "skip"
        print(
            f"  [{status}] {map_name}: reason={reason}, "
            f"tiles={tiles}, wmo={has_wmo}, readable_tile={has_tile}"
        )

    if not included:
        raise RuntimeError(f"No trainable maps discovered for {client_root}")

    return included


def _write_index(rows: list[dict], output_path: Path) -> None:
    schema_fields = [
        pa.field("tile_id", pa.int64()),
        pa.field("build", pa.string()),
        pa.field("map", pa.string()),
        pa.field("tile_x", pa.int32()),
        pa.field("tile_y", pa.int32()),
        pa.field("height_mean", pa.float32()),
        pa.field("height_std", pa.float32()),
    ]
    bool_fields = sorted({
        k
        for row in rows
        for k in row.keys()
        if k.startswith("has_")
    }) if rows else []
    for bf in bool_fields:
        schema_fields.append(pa.field(bf, pa.bool_()))

    schema = pa.schema(schema_fields)
    col_data = {k: [] for k in schema.names}
    for row in rows:
        for k in schema.names:
            col_data[k].append(row.get(k, False if k.startswith("has_") else 0))

    table = pa.table(col_data, schema=schema)
    pq.write_table(table, str(output_path / "index.parquet"))


def _write_harvest_metrics(
    *,
    build: str,
    output_path: Path,
    index_rows: list[dict],
    placements_total: int,
    skipped_zero_usable_maps: int,
    rejected_tile_count: int,
    elapsed_seconds: float,
) -> Path:
    tile_count = len(index_rows)
    signal_counts: dict[str, int] = {}
    for row in index_rows:
        for key, value in row.items():
            if not key.startswith("has_"):
                continue
            signal_counts[key] = signal_counts.get(key, 0) + (1 if bool(value) else 0)

    coverage = {
        key: {
            "count": int(count),
            "fraction": float(count / tile_count) if tile_count > 0 else 0.0,
        }
        for key, count in sorted(signal_counts.items())
    }

    map_counts: dict[str, int] = {}
    for row in index_rows:
        map_name = str(row.get("map", ""))
        map_counts[map_name] = map_counts.get(map_name, 0) + 1

    payload = {
        "build": build,
        "tile_count": tile_count,
        "placements_total": int(placements_total),
        "skipped_zero_usable_maps": int(skipped_zero_usable_maps),
        "rejected_missing_required_tiles": int(rejected_tile_count),
        "elapsed_seconds": float(elapsed_seconds),
        "tiles_per_second": float(tile_count / max(elapsed_seconds, 0.01)),
        "signal_coverage": coverage,
        "map_tile_counts": dict(sorted(map_counts.items())),
    }
    out_path = output_path / "harvest_metrics.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def _read_index_rows(output_path: Path) -> list[dict]:
    idx_path = output_path / "index.parquet"
    table = pq.read_table(str(idx_path))
    return [
        {col: table.column(col)[i].as_py() for col in table.column_names}
        for i in range(table.num_rows)
    ]


def _read_parquet_rows(parquet_path: Path) -> list[dict]:
    table = pq.read_table(str(parquet_path))
    return [
        {col: table.column(col)[i].as_py() for col in table.column_names}
        for i in range(table.num_rows)
    ]


def _ordered_maps_from_index_rows(index_rows: list[dict]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for row in index_rows:
        map_name = str(row.get("map", ""))
        if not map_name or map_name in seen:
            continue
        seen.add(map_name)
        ordered.append(map_name)
    return ordered


def _normalize_map_name(meta_map: object, requested_map: str) -> str:
    raw = str(meta_map or "").strip()
    if not raw:
        return requested_map
    low = raw.lower()
    if low in {"memory", "<memory>", "unknown", "<unknown>"}:
        return requested_map
    return raw


def _is_placeholder_map_name(name: str) -> bool:
    low = str(name or "").strip().lower()
    return low in {"", "memory", "<memory>", "unknown", "<unknown>"}


def _stream_valid_tile_metadata(
    harvest_tool: Path,
    client_root: Path,
    map_name: str,
    build_version: str | None,
) -> list[dict[str, object]]:
    cmd = [
        str(harvest_tool),
        "harvest-stream",
        "--client-root",
        str(client_root),
        "--map",
        map_name,
    ]
    if build_version:
        cmd.extend(["--build", build_version])

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    if proc.stdout is None or proc.stderr is None:
        proc.terminate()
        raise RuntimeError(f"Failed to open harvest-stream pipes for map {map_name}.")

    stderr_tail: deque[str] = deque(maxlen=40)
    stderr_thread = threading.Thread(
        target=_pump_stderr,
        args=(proc.stderr, map_name, stderr_tail),
        daemon=True,
    )
    stderr_thread.start()

    rows: list[dict[str, object]] = []
    saw_end_marker = False
    stream_error: str | None = None

    while True:
        header = proc.stdout.read(8)
        if not header:
            stream_error = "stdout closed before ENDS sentinel"
            break
        if len(header) < 8:
            stream_error = f"truncated stream header ({len(header)}/8 bytes)"
            break

        magic = header[:4]
        if magic == ENDS_MAGIC:
            saw_end_marker = True
            break
        if magic != NPZB_MAGIC:
            stream_error = f"unexpected stream magic {magic!r}"
            break

        length = struct.unpack("<I", header[4:8])[0]
        if length == 0 or length > 50_000_000:
            stream_error = f"invalid NPZ blob length {length}"
            break

        blob = proc.stdout.read(length)
        if not blob or len(blob) < length:
            stream_error = f"truncated NPZ blob ({len(blob) if blob else 0}/{length} bytes)"
            break

        try:
            data = dict(np.load(BytesIO(blob), allow_pickle=False))
        except Exception as ex:
            stream_error = f"failed to decode streamed NPZ blob: {ex}"
            break

        if not REQUIRED_KEYS.issubset(data.keys()):
            continue

        meta = _decode_metadata_json(data)
        tx, ty = _extract_tile_coords_from_metadata(meta)
        actual_map = _normalize_map_name(meta.get("map_name", map_name), map_name)
        rows.append({"map": actual_map, "tile_x": tx, "tile_y": ty})

    if proc.poll() is None and (stream_error is not None or not saw_end_marker):
        proc.terminate()

    return_code = proc.wait()
    stderr_thread.join(timeout=2.0)

    if stream_error is not None:
        raise RuntimeError(
            f"Harvest stream failed for map {map_name}: {stream_error}\n"
            f"stderr tail:\n{_tail_text(stderr_tail)}"
        )
    if not saw_end_marker:
        raise RuntimeError(
            f"Harvest stream ended without ENDS sentinel for map {map_name}.\n"
            f"stderr tail:\n{_tail_text(stderr_tail)}"
        )
    if return_code != 0:
        raise RuntimeError(
            f"Harvest stream exited with code {return_code} for map {map_name}.\n"
            f"stderr tail:\n{_tail_text(stderr_tail)}"
        )

    return rows


def _write_placements(all_placements: list[dict], output_path: Path) -> None:
    if not all_placements:
        return
    fields = [
        pa.field("tile_id", pa.int64()),
        pa.field("instance_type", pa.string()),
        pa.field("instance_idx", pa.int32()),
        pa.field("asset_path", pa.string()),
    ]
    for col in PLACEMENT_COLUMNS_MDDF:
        fields.append(pa.field(col, pa.float32()))
    for col in PLACEMENT_COLUMNS_MODF:
        if col not in [f.name for f in fields]:
            fields.append(pa.field(col, pa.float32()))

    schema = pa.schema(fields)
    col_data = {f.name: [] for f in fields}
    for row in all_placements:
        for f in fields:
            val = row.get(f.name, 0.0 if f.type == pa.float32() else "")
            col_data[f.name].append(val)

    table = pa.table(col_data, schema=schema)
    pq.write_table(table, str(output_path / "placements.parquet"))


def _format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def _format_ratio(raw_bytes: int, stored_bytes: int) -> str:
    if stored_bytes <= 0:
        return "n/a"
    return f"{raw_bytes / stored_bytes:.2f}x"


def cmd_build(args: argparse.Namespace) -> None:
    builds = args.builds or [args.build]
    harvest_tool = _find_harvest_tool()
    print(f"Harvest tool: {harvest_tool}")

    maps_override = getattr(args, "maps", None)
    limit = args.limit
    resume = args.resume
    codec_name = args.codec
    codec_level = args.clevel
    codec_shuffle = args.shuffle
    rebuild_existing = args.rebuild_existing

    for build in builds:
        client_root = _find_client_root(build)
        if client_root is None:
            print(f"SKIP build {build}: no client root found at {_CLIENT_ROOTS / build}")
            continue

        output_path = _DATASET_ROOT / f"{build}.zarr"
        staging_path = _DATASET_ROOT / f"{build}.zarr.partial"
        completed_state = None if rebuild_existing else _load_completed_final_store_state(output_path)
        if completed_state is not None and not staging_path.exists():
            completed_maps = completed_state.get("completed_maps", [])
            print(
                f"SKIP build {build}: final store already complete at {output_path} "
                f"({len(completed_maps)} maps, {completed_state.get('valid_tiles', 'unknown')} tiles)"
            )
            continue
        if staging_path.exists() and not resume:
            shutil.rmtree(staging_path)
        if not staging_path.exists():
            staging_path.mkdir(parents=True, exist_ok=True)

        build_version = build.replace("_", ".")
        map_names = maps_override or _discover_maps_for_build(harvest_tool, client_root)

        print(f"\n{'='*60}")
        print(f"Building V16 dataset for {build}")
        print(f"Client: {client_root}")
        print(f"Maps: {map_names}")
        print(f"Output: {output_path}")
        print(f"Staging: {staging_path}")
        print(f"Rejected tiles report: {_tile_rejection_report_path(output_path, build)}")
        print(f"Resume: {resume}")
        print(f"Codec: {codec_name} clevel={codec_level} shuffle={codec_shuffle}")

        try:
            _build_zarr_streaming(
                harvest_tool=harvest_tool,
                client_root=client_root,
                build=build,
                build_version=build_version,
                map_names=map_names,
                output_path=staging_path,
                limit=limit,
                rejected_tiles_report_path=_tile_rejection_report_path(output_path, build),
                resume=resume,
                codec_name=codec_name,
                codec_level=codec_level,
                codec_shuffle=codec_shuffle,
            )
            if output_path.exists():
                shutil.rmtree(output_path)
            staging_path.replace(output_path)
            print(f"Promoted staged dataset -> {output_path}")
        except Exception:
            print(
                f"Build failed for {build}. Partial output preserved at {staging_path}",
                file=sys.stderr,
                flush=True,
            )
            raise


def _build_zarr_streaming(
    harvest_tool: Path,
    client_root: Path,
    build: str,
    build_version: str,
    map_names: list[str],
    output_path: Path,
    limit: int | None,
    rejected_tiles_report_path: Path | None = None,
    resume: bool = False,
    codec_name: str = DEFAULT_CODEC,
    codec_level: int = DEFAULT_CLEVEL,
    codec_shuffle: str = DEFAULT_SHUFFLE,
) -> None:
    codec = zarr.codecs.BloscCodec(cname=codec_name, clevel=codec_level, shuffle=codec_shuffle)
    resume_state = _load_resume_state(output_path) if resume else None

    if resume and resume_state is None:
        has_meaningful_partial = output_path.exists() and any(output_path.iterdir())
        if has_meaningful_partial:
            substantive_entries = [
                path.name
                for path in output_path.iterdir()
                if path.name not in {"zarr.json", ".zgroup", ".zattrs"}
            ]
            if substantive_entries:
                raise RuntimeError(
                    f"Resume requested for {output_path}, but no {_resume_state_path(output_path).name} was found."
                )
        print(
            f"  Resume requested for {build}, but no {_resume_state_path(output_path).name} exists yet. "
            f"Starting a fresh staged build at {output_path}.",
            flush=True,
        )

    store = zarr.storage.LocalStore(str(output_path), read_only=False)
    root = zarr.open_group(store=store, mode="a" if resume_state is not None else "w")

    arrays: dict[str, zarr.Array] = {}
    index_rows: list[dict] = []
    all_placements: list[dict] = []
    valid = 0
    skipped_zero_usable_maps = 0
    rejected_tile_count = 0
    t0 = time.perf_counter()
    capacity = 50000
    completed_maps: list[str] = []
    pending_arrays: dict[str, list[np.ndarray]] = {key: [] for key in ALL_ARRAY_KEYS}
    pending_count = 0

    if resume_state is not None:
        expected_maps = resume_state.get("requested_maps", [])
        if expected_maps != map_names:
            raise RuntimeError(
                f"Resume map list mismatch for {build}. Existing partial requested_maps={expected_maps} "
                f"but current maps={map_names}."
            )
        if resume_state.get("codec") != codec_name or int(resume_state.get("clevel", -1)) != codec_level or resume_state.get("shuffle") != codec_shuffle:
            raise RuntimeError(
                f"Resume codec mismatch for {build}. Existing partial uses "
                f"{resume_state.get('codec')} clevel={resume_state.get('clevel')} shuffle={resume_state.get('shuffle')}, "
                f"current request is {codec_name} clevel={codec_level} shuffle={codec_shuffle}."
            )
        capacity = int(resume_state.get("capacity", capacity))
        completed_maps = [str(name) for name in resume_state.get("completed_maps", [])]
        skipped_zero_usable_maps = int(resume_state.get("skipped_zero_usable_maps", 0))
        rejected_tile_count = int(resume_state.get("rejected_tile_count", 0))
        idx_path = output_path / "index.parquet"
        if idx_path.exists():
            table = pq.read_table(str(idx_path))
            index_rows = [
                {col: table.column(col)[i].as_py() for col in table.column_names}
                for i in range(table.num_rows)
            ]
            valid = len(index_rows)
        else:
            valid = int(resume_state.get("valid_tiles", 0))
        pl_path = output_path / "placements.parquet"
        if pl_path.exists():
            pl_table = pq.read_table(str(pl_path))
            all_placements = [
                {col: pl_table.column(col)[i].as_py() for col in pl_table.column_names}
                for i in range(pl_table.num_rows)
            ]

    rejected_tiles_report = None
    if rejected_tiles_report_path is not None:
        rejected_tiles_report_path.parent.mkdir(parents=True, exist_ok=True)
        if rejected_tiles_report_path.exists() and resume_state is None:
            rejected_tiles_report_path.unlink()
        rejected_tiles_report = rejected_tiles_report_path.open("a" if resume_state is not None else "w", encoding="utf-8")

    for key in ALL_ARRAY_KEYS:
        if key in root:
            arrays[key] = root[key]
        else:
            shape = (capacity,) + SHAPES[key]
            chunks = CHUNK_SIZES.get(key, (64,) + SHAPES[key])
            arrays[key] = root.create_array(
                key, shape=shape, chunks=chunks, dtype=DTYPES[key],
                compressors=[codec], fill_value=FILL_VALUES.get(key, 0),
            )

    for map_name in map_names:
        if map_name in completed_maps:
            print(f"\n  Skipping completed map: {map_name}")
            continue

        print(f"\n  Streaming map: {map_name}")

        cmd = [
            str(harvest_tool), "harvest-stream",
            "--client-root", str(client_root),
            "--map", map_name,
        ]
        if build_version:
            cmd.extend(["--build", build_version])

        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
        )
        if proc.stdout is None or proc.stderr is None:
            proc.terminate()
            raise RuntimeError(f"Failed to open harvest-stream pipes for map {map_name}.")

        stderr_tail: deque[str] = deque(maxlen=40)
        stderr_thread = threading.Thread(
            target=_pump_stderr,
            args=(proc.stderr, map_name, stderr_tail),
            daemon=True,
        )
        stderr_thread.start()

        tile_count = 0
        dropped_missing_required = 0
        map_placements = 0
        map_blob_bytes = 0
        saw_end_marker = False
        stream_error: str | None = None
        while True:
            header = proc.stdout.read(8)
            if not header:
                stream_error = "stdout closed before ENDS sentinel"
                break
            if len(header) < 8:
                stream_error = f"truncated stream header ({len(header)}/8 bytes)"
                break

            magic = header[:4]
            if magic == ENDS_MAGIC:
                saw_end_marker = True
                break
            if magic != NPZB_MAGIC:
                stream_error = f"unexpected stream magic {magic!r}"
                break

            length = struct.unpack("<I", header[4:8])[0]
            if length == 0 or length > 50_000_000:
                stream_error = f"invalid NPZ blob length {length}"
                break

            blob = proc.stdout.read(length)
            if not blob or len(blob) < length:
                stream_error = f"truncated NPZ blob ({len(blob) if blob else 0}/{length} bytes)"
                break

            try:
                data = dict(np.load(BytesIO(blob), allow_pickle=False))
            except Exception as ex:
                stream_error = f"failed to decode streamed NPZ blob: {ex}"
                break

            result = _process_tile_data(data)
            if result is None:
                dropped_missing_required += 1
                rejected_tile_count += 1
                missing = sorted(REQUIRED_KEYS - set(data.keys()))
                meta = _decode_metadata_json(data)
                source_adt_path = str(meta.get("source_adt_path", ""))
                tx, ty = _extract_tile_coords_from_metadata(meta)
                if rejected_tiles_report is not None:
                    rejected_map = _normalize_map_name(meta.get("map_name", map_name), map_name)
                    rejected_tiles_report.write(
                        json.dumps(
                            {
                                "build": build,
                                "map_name": rejected_map,
                                "source_adt_path": source_adt_path,
                                "tile_x": tx,
                                "tile_y": ty,
                                "missing_required_keys": missing,
                                "available_keys": sorted(data.keys()),
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    rejected_tiles_report.flush()
                if dropped_missing_required <= 5:
                    print(
                        f"    Warning: dropped tile blob missing required keys {missing}; "
                        f"available keys: {sorted(data.keys())}",
                        file=sys.stderr,
                        flush=True,
                    )
                continue

            tile_arrays, has_signals = result

            h_mean = float(np.mean(tile_arrays["height_257"]))
            h_std = float(np.std(tile_arrays["height_257"])) + 1e-8

            meta = _extract_metadata(data)
            tx, ty = 0, 0
            actual_map = map_name
            if meta:
                tx, ty = _extract_tile_coords_from_metadata(meta)
                actual_map = _normalize_map_name(meta.get("map_name", map_name), map_name)

            # Extract placement data
            tile_id = valid + pending_count
            mddf_rows, modf_rows, mddf_names, modf_names = _extract_placements(data, meta)
            for row in mddf_rows:
                row["tile_id"] = tile_id
                all_placements.append(row)
            for row in modf_rows:
                row["tile_id"] = tile_id
                all_placements.append(row)
            map_placements += len(mddf_rows) + len(modf_rows)

            row = {
                "tile_id": tile_id, "build": build, "map": actual_map,
                "tile_x": tx, "tile_y": ty,
                "height_mean": h_mean, "height_std": h_std,
                "n_mddf": len(mddf_rows), "n_modf": len(modf_rows),
            }
            for key in SIGNAL_FLAG_KEYS:
                row[f"has_{key}"] = bool(has_signals.get(key, False))
            index_rows.append(row)

            needed = valid + pending_count + 1
            while needed >= capacity:
                capacity += 50000
                for key in ALL_ARRAY_KEYS:
                    arrays[key].resize((capacity,) + SHAPES[key])

            for key in ALL_ARRAY_KEYS:
                pending_arrays[key].append(tile_arrays[key])
            pending_count += 1

            if pending_count >= WRITE_BATCH_SIZE:
                valid = _flush_tile_batch_with_retry(
                    arrays,
                    valid,
                    pending_arrays,
                    pending_count,
                    map_name=actual_map,
                )
                pending_count = 0

            tile_count += 1
            map_blob_bytes += length
            if tile_count == 1 or tile_count % 10 == 0:
                elapsed = time.perf_counter() - t0
                total_written = valid + pending_count
                rate = total_written / max(elapsed, 0.01)
                store_mb = _dir_size_bytes(output_path) / 1024 / 1024
                print(
                    f"    Progress {map_name}: map_tiles={tile_count} total_tiles={total_written} "
                    f"placements={map_placements} raw_npz={map_blob_bytes / 1024 / 1024:.1f} MB "
                    f"store={store_mb:.1f} MB rate={rate:.1f} tiles/s",
                    flush=True,
                )

            if limit is not None and valid + pending_count >= limit:
                proc.terminate()
                break

        if proc.poll() is None and (stream_error is not None or not saw_end_marker):
            proc.terminate()

        return_code = proc.wait()
        stderr_thread.join(timeout=2.0)

        if stream_error is not None:
            raise RuntimeError(
                f"Harvest stream failed for map {map_name}: {stream_error}\n"
                f"stderr tail:\n{_tail_text(stderr_tail)}"
            )
        if not saw_end_marker:
            raise RuntimeError(
                f"Harvest stream ended without ENDS sentinel for map {map_name}.\n"
                f"stderr tail:\n{_tail_text(stderr_tail)}"
            )
        if return_code != 0:
            raise RuntimeError(
                f"Harvest stream exited with code {return_code} for map {map_name}.\n"
                f"stderr tail:\n{_tail_text(stderr_tail)}"
            )
        if pending_count > 0:
            valid = _flush_tile_batch_with_retry(
                arrays,
                valid,
                pending_arrays,
                pending_count,
                map_name=map_name,
            )
            pending_count = 0
        if tile_count == 0:
            skipped_zero_usable_maps += 1
            completed_maps.append(map_name)
            _write_resume_state(
                output_path,
                build=build,
                requested_maps=map_names,
                completed_maps=completed_maps,
                valid=valid,
                skipped_zero_usable_maps=skipped_zero_usable_maps,
                rejected_tile_count=rejected_tile_count,
                codec_name=codec_name,
                codec_level=codec_level,
                codec_shuffle=codec_shuffle,
                capacity=capacity,
            )
            print(
                f"    Warning: skipping map {map_name} because harvest produced zero usable V16 tiles. "
                f"Dropped missing-required blobs: {dropped_missing_required}. "
                f"Report: {rejected_tiles_report_path}",
                file=sys.stderr,
                flush=True,
            )
            continue

        if dropped_missing_required > 0:
            print(
                f"    Warning: dropped {dropped_missing_required} blobs for map {map_name} "
                f"because required dataset keys were missing. "
                f"Report: {rejected_tiles_report_path}",
                file=sys.stderr,
                flush=True,
            )

        print(
            f"    Map {map_name}: {tile_count} tiles streamed, placements={map_placements}, "
            f"raw_npz={map_blob_bytes / 1024 / 1024:.1f} MB, "
            f"dropped_missing_required={dropped_missing_required}",
            flush=True,
        )

        completed_maps.append(map_name)
        if index_rows:
            _write_index(index_rows, output_path)
        if all_placements:
            _write_placements(all_placements, output_path)
        _write_resume_state(
            output_path,
            build=build,
            requested_maps=map_names,
            completed_maps=completed_maps,
            valid=valid,
            skipped_zero_usable_maps=skipped_zero_usable_maps,
            rejected_tile_count=rejected_tile_count,
            codec_name=codec_name,
            codec_level=codec_level,
            codec_shuffle=codec_shuffle,
            capacity=capacity,
        )

        if limit is not None and valid >= limit:
            break

    for key in ALL_ARRAY_KEYS:
        arrays[key].resize((valid,) + SHAPES[key])

    if valid == 0:
        raise RuntimeError(
            "Harvest stream produced zero usable tiles across all requested maps."
        )

    if index_rows:
        _write_index(index_rows, output_path)

    if all_placements:
        _write_placements(all_placements, output_path)

    store.close()
    _write_resume_state(
        output_path,
        build=build,
        requested_maps=map_names,
        completed_maps=completed_maps,
        valid=valid,
        skipped_zero_usable_maps=skipped_zero_usable_maps,
        rejected_tile_count=rejected_tile_count,
        codec_name=codec_name,
        codec_level=codec_level,
        codec_shuffle=codec_shuffle,
        capacity=valid,
        finalized=True,
    )

    total_bytes = _dir_size_bytes(output_path)
    liq_count = sum(1 for r in index_rows if r.get("has_liquid_mask", False))
    inst_count = sum(1 for r in index_rows if r.get("has_object_instance_mask", False))
    elapsed = time.perf_counter() - t0
    if rejected_tiles_report is not None:
        rejected_tiles_report.close()
    print(f"\nDone. {valid} tiles -> {output_path}")
    print(f"Size: {total_bytes / 1024 / 1024:.1f} MB")
    print(f"Liquid: {liq_count}/{valid}, Instance mask: {inst_count}/{valid}")
    print(f"Placements: {len(all_placements)} total")
    print(f"Skipped zero-usable maps: {skipped_zero_usable_maps}")
    print(f"Rejected missing-required tiles: {rejected_tile_count}")
    if rejected_tiles_report_path is not None:
        print(f"Rejected tiles report: {rejected_tiles_report_path}")
    metrics_path = _write_harvest_metrics(
        build=build,
        output_path=output_path,
        index_rows=index_rows,
        placements_total=len(all_placements),
        skipped_zero_usable_maps=skipped_zero_usable_maps,
        rejected_tile_count=rejected_tile_count,
        elapsed_seconds=elapsed,
    )
    print(f"Harvest metrics: {metrics_path}")
    print(f"Time: {elapsed:.0f}s ({valid / max(elapsed, 0.01):.1f} tiles/s)")


def cmd_stats(args: argparse.Namespace) -> None:
    builds = args.builds or [args.build]
    aggregate_total_rows = 0
    aggregate_unique_tiles: set[tuple[str, int, int]] = set()
    aggregate_per_build_unique: dict[str, int] = {}
    for build in builds:
        zarr_path = _DATASET_ROOT / f"{build}.zarr"
        if not zarr_path.exists():
            print(f"SKIP {build}: no Zarr store at {zarr_path}")
            continue
        store_bytes = _dir_size_bytes(zarr_path)
        total_raw_array_bytes = 0
        total_array_disk_bytes = 0
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Object at .* is not recognized as a component of a Zarr hierarchy\.",
                category=UserWarning,
            )
            store = zarr.storage.LocalStore(str(zarr_path), read_only=True)
            root = zarr.open_group(store=store, mode="r")
            n = root["height_257"].shape[0]
            print(f"\n{build}: {n} tiles")
            for k in sorted(root.array_keys()):
                a = root[k]
                raw_bytes = int(np.prod(a.shape, dtype=np.int64) * np.dtype(a.dtype).itemsize)
                disk_bytes = _dir_size_bytes(zarr_path / k) if (zarr_path / k).exists() else 0
                total_raw_array_bytes += raw_bytes
                total_array_disk_bytes += disk_bytes
                print(
                    f"  {k}: shape={a.shape} dtype={a.dtype} "
                    f"raw={_format_bytes(raw_bytes)} disk={_format_bytes(disk_bytes)} "
                    f"ratio={_format_ratio(raw_bytes, disk_bytes)}"
                )
            store.close()

        print(
            f"  arrays total: raw={_format_bytes(total_raw_array_bytes)} "
            f"disk={_format_bytes(total_array_disk_bytes)} "
            f"ratio={_format_ratio(total_raw_array_bytes, total_array_disk_bytes)}"
        )
        print(
            f"  full store: disk={_format_bytes(store_bytes)} "
            f"savings_vs_raw_arrays={_format_bytes(max(total_raw_array_bytes - store_bytes, 0))}"
        )

        idx_path = zarr_path / "index.parquet"
        if idx_path.exists():
            table = pq.read_table(str(idx_path))
            idx_bytes = idx_path.stat().st_size
            print(
                f"  index.parquet: {table.num_rows} rows, {table.num_columns} cols, "
                f"disk={_format_bytes(idx_bytes)}"
            )
            for col in table.column_names:
                if col.startswith("has_"):
                    count_scalar = pc.sum(table.column(col))
                    count = 0 if count_scalar is None else int(count_scalar.as_py() or 0)
                    print(f"    {col}: {count}/{table.num_rows}")
            local_unique: set[tuple[str, int, int]] = set()
            if {"map", "tile_x", "tile_y"}.issubset(set(table.column_names)):
                for row in table.select(["map", "tile_x", "tile_y"]).to_pylist():
                    map_name = str(row.get("map") or "")
                    tx = int(row.get("tile_x") or 0)
                    ty = int(row.get("tile_y") or 0)
                    key = (map_name, tx, ty)
                    local_unique.add(key)
                    aggregate_unique_tiles.add(key)
                aggregate_total_rows += int(table.num_rows)
                aggregate_per_build_unique[build] = len(local_unique)
            if table.num_rows != n:
                print(
                    f"  WARNING: array length ({n}) does not match index rows ({table.num_rows}). "
                    f"Build may be incomplete or corrupted."
                )
        else:
            print("  WARNING: index.parquet missing. This store looks incomplete or failed before finalization.")

        metrics_path = zarr_path / "harvest_metrics.json"
        if metrics_path.exists():
            print(f"  harvest_metrics.json: {metrics_path}")

        pl_path = zarr_path / "placements.parquet"
        if pl_path.exists():
            pl_table = pq.read_table(str(pl_path))
            pl_bytes = pl_path.stat().st_size
            print(f"  placements.parquet: {pl_table.num_rows} placements, disk={_format_bytes(pl_bytes)}")
        partial_path = _DATASET_ROOT / f"{build}.zarr.partial"
        if partial_path.exists():
            print(f"  WARNING: staged partial output still exists at {partial_path}")

    if len(builds) > 1 and aggregate_total_rows > 0:
        unique_count = len(aggregate_unique_tiles)
        duplication_factor = aggregate_total_rows / max(unique_count, 1)
        print("\nCross-build overlap summary:")
        print(f"  total rows across builds: {aggregate_total_rows}")
        print(f"  unique (map,tile_x,tile_y): {unique_count}")
        print(f"  duplication factor: {duplication_factor:.2f}x")
        for build in sorted(aggregate_per_build_unique):
            print(f"  {build}: unique tile coords={aggregate_per_build_unique[build]}")


def cmd_merge_builds(args: argparse.Namespace) -> None:
    builds = args.builds or sorted(
        d.stem.replace(".zarr", "")
        for d in _DATASET_ROOT.glob("*.zarr")
        if d.is_dir()
    )
    if not builds:
        raise RuntimeError("No source build stores found under output/datasets/v16.")

    output_name = args.output_name or "merged_all"
    output_path = _DATASET_ROOT / f"{output_name}.zarr"
    output_partial = _DATASET_ROOT / f"{output_name}.zarr.partial"
    dedupe_mode = str(args.dedupe_mode or "coords_height")
    if dedupe_mode not in {"none", "coords", "coords_height"}:
        raise RuntimeError("--dedupe-mode must be one of: none, coords, coords_height")

    if output_path.exists() and not args.rebuild_existing:
        raise RuntimeError(
            f"Output merged store already exists at {output_path}. Use --rebuild-existing to overwrite."
        )

    if output_partial.exists():
        shutil.rmtree(output_partial)
    if output_path.exists() and args.rebuild_existing:
        shutil.rmtree(output_path)

    source_handles: dict[str, tuple[zarr.storage.LocalStore, zarr.Group]] = {}
    source_index_rows: dict[str, list[dict]] = {}
    source_placements_rows: dict[str, list[dict]] = {}
    t0 = time.perf_counter()

    for build in builds:
        zarr_path = _DATASET_ROOT / f"{build}.zarr"
        if not zarr_path.exists():
            print(f"SKIP {build}: no store at {zarr_path}")
            continue
        store, root = _open_zarr_group_readonly(zarr_path)
        source_handles[build] = (store, root)
        idx_path = zarr_path / "index.parquet"
        if not idx_path.exists():
            store.close()
            raise RuntimeError(f"Source build {build} has no index.parquet.")
        source_index_rows[build] = _read_parquet_rows(idx_path)
        pl_path = zarr_path / "placements.parquet"
        source_placements_rows[build] = _read_parquet_rows(pl_path) if pl_path.exists() else []

    try:
        records: list[tuple[str, int, dict]] = []
        if dedupe_mode == "none":
            for build in builds:
                rows = source_index_rows.get(build, [])
                for row in rows:
                    records.append((build, int(row["tile_id"]), row))
        else:
            selected: dict[tuple, tuple[str, int, dict]] = {}
            ordered_keys: list[tuple] = []
            for build in builds:
                rows = source_index_rows.get(build, [])
                for row in rows:
                    map_name = str(row.get("map", ""))
                    tx = int(row.get("tile_x") or 0)
                    ty = int(row.get("tile_y") or 0)
                    if dedupe_mode == "coords":
                        key = (map_name, tx, ty)
                    else:
                        key = (
                            map_name,
                            tx,
                            ty,
                            round(float(row.get("height_mean", 0.0)), 4),
                            round(float(row.get("height_std", 0.0)), 4),
                        )
                    if key not in selected:
                        ordered_keys.append(key)
                    # Keep latest source in requested build order.
                    selected[key] = (build, int(row["tile_id"]), row)
            for key in ordered_keys:
                records.append(selected[key])

        if not records:
            raise RuntimeError("No records selected for merge.")

        capacity = len(records)
        codec = zarr.codecs.BloscCodec(cname=DEFAULT_CODEC, clevel=DEFAULT_CLEVEL, shuffle=DEFAULT_SHUFFLE)
        out_store = zarr.storage.LocalStore(str(output_partial), read_only=False)
        out_root = zarr.open_group(store=out_store, mode="w")
        out_arrays: dict[str, zarr.Array] = {}
        for key in ALL_ARRAY_KEYS:
            chunks = CHUNK_SIZES.get(key, (64,) + SHAPES[key])
            out_arrays[key] = out_root.create_array(
                key,
                shape=(capacity,) + SHAPES[key],
                chunks=chunks,
                dtype=DTYPES[key],
                compressors=[codec],
                fill_value=FILL_VALUES.get(key, 0),
            )

        merged_index_rows: list[dict] = []
        source_to_merged_tile: dict[tuple[str, int], int] = {}
        batch_size = int(args.batch_size or 64)
        pending_arrays: dict[str, list[np.ndarray]] = {key: [] for key in ALL_ARRAY_KEYS}
        write_pos = 0

        for build, src_tile_id, row in records:
            _, src_root = source_handles[build]
            for key in ALL_ARRAY_KEYS:
                pending_arrays[key].append(src_root[key][src_tile_id].astype(DTYPES[key], copy=False))

            merged_row = dict(row)
            merged_row["tile_id"] = write_pos
            merged_row["build"] = str(row.get("build") or build)
            merged_row["source_build"] = build
            merged_row["source_tile_id"] = src_tile_id
            merged_index_rows.append(merged_row)
            source_to_merged_tile[(build, src_tile_id)] = write_pos
            write_pos += 1

            if len(pending_arrays["height_257"]) >= batch_size:
                start = write_pos - len(pending_arrays["height_257"])
                for key in ALL_ARRAY_KEYS:
                    out_arrays[key][start:write_pos] = np.stack(pending_arrays[key], axis=0)
                    pending_arrays[key].clear()

        remaining = len(pending_arrays["height_257"])
        if remaining > 0:
            start = write_pos - remaining
            for key in ALL_ARRAY_KEYS:
                out_arrays[key][start:write_pos] = np.stack(pending_arrays[key], axis=0)
                pending_arrays[key].clear()

        _write_index(merged_index_rows, output_partial)

        merged_placements: list[dict] = []
        for build in builds:
            for row in source_placements_rows.get(build, []):
                old_tile_id = int(row.get("tile_id", -1))
                new_tile_id = source_to_merged_tile.get((build, old_tile_id))
                if new_tile_id is None:
                    continue
                new_row = dict(row)
                new_row["tile_id"] = int(new_tile_id)
                new_row["source_build"] = build
                new_row["source_tile_id"] = old_tile_id
                merged_placements.append(new_row)
        if merged_placements:
            _write_placements(merged_placements, output_partial)

        elapsed = time.perf_counter() - t0
        metrics_path = _write_harvest_metrics(
            build=output_name,
            output_path=output_partial,
            index_rows=merged_index_rows,
            placements_total=len(merged_placements),
            skipped_zero_usable_maps=0,
            rejected_tile_count=0,
            elapsed_seconds=elapsed,
        )
        merge_manifest = {
            "output_name": output_name,
            "output_path": str(output_path),
            "source_builds": builds,
            "dedupe_mode": dedupe_mode,
            "selected_tiles": len(records),
            "source_tiles_total": int(sum(len(source_index_rows.get(b, [])) for b in builds)),
            "batch_size": batch_size,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "harvest_metrics_path": str(metrics_path),
        }
        (output_partial / "merge_manifest.json").write_text(json.dumps(merge_manifest, indent=2), encoding="utf-8")
        _write_resume_state(
            output_partial,
            build=output_name,
            requested_maps=sorted({str(r.get("map", "")) for r in merged_index_rows}),
            completed_maps=sorted({str(r.get("map", "")) for r in merged_index_rows}),
            valid=len(merged_index_rows),
            skipped_zero_usable_maps=0,
            rejected_tile_count=0,
            codec_name=DEFAULT_CODEC,
            codec_level=DEFAULT_CLEVEL,
            codec_shuffle=DEFAULT_SHUFFLE,
            capacity=len(merged_index_rows),
            finalized=True,
        )
        out_store.close()

        # Safety gate: never publish an incomplete merged store.
        missing_paths: list[str] = []
        for key in ALL_ARRAY_KEYS:
            if not (output_partial / key).exists():
                missing_paths.append(key)
        for required_file in ("index.parquet", "harvest_metrics.json", "merge_manifest.json", "_resume_state.json"):
            if not (output_partial / required_file).exists():
                missing_paths.append(required_file)
        if missing_paths:
            raise RuntimeError(
                "Refusing to publish incomplete merged store. Missing: "
                + ", ".join(missing_paths)
            )

        os_replace_target = output_path
        output_partial.replace(os_replace_target)
        print(f"Merged store: {output_path}")
        print(f"Tiles: {len(merged_index_rows)} (from {sum(len(source_index_rows.get(b, [])) for b in builds)} source rows)")
        print(f"Placements: {len(merged_placements)}")
        print(f"Dedupe mode: {dedupe_mode}")
        print(f"Metrics: {output_path / 'harvest_metrics.json'}")
    finally:
        for store, _root in source_handles.values():
            store.close()


def cmd_repair_index(args: argparse.Namespace) -> None:
    builds = args.builds or [args.build]
    harvest_tool = _find_harvest_tool()

    for build in builds:
        output_path = _DATASET_ROOT / f"{build}.zarr"
        if not output_path.exists():
            print(f"SKIP {build}: no final store at {output_path}")
            continue

        client_root = _find_client_root(build)
        if client_root is None:
            raise RuntimeError(f"Could not find staged client root for build {build}.")

        idx_path = output_path / "index.parquet"
        if not idx_path.exists():
            raise RuntimeError(f"Build {build} has no index.parquet to repair.")

        if not args.no_backup:
            backup_path = output_path / "index.parquet.bak"
            if not backup_path.exists():
                shutil.copy2(idx_path, backup_path)
                print(f"Backed up {idx_path} -> {backup_path}")

        index_rows = _read_index_rows(output_path)
        ordered_maps = _ordered_maps_from_index_rows(index_rows)
        build_version = build.replace("_", ".")

        print(f"Repairing index coordinates for {build}")
        print(f"Client: {client_root}")
        print(f"Maps: {ordered_maps}")

        if ordered_maps and all(_is_placeholder_map_name(m) for m in ordered_maps):
            discovered_maps = _discover_maps_for_build(harvest_tool, client_root)
            print("Index map labels are placeholder-only; attempting full map relabel from fresh stream order.")
            print(f"Discovered maps: {discovered_maps}")
            streamed_all: list[dict[str, object]] = []
            for map_name in discovered_maps:
                streamed_all.extend(_stream_valid_tile_metadata(harvest_tool, client_root, map_name, build_version))

            if len(streamed_all) != len(index_rows):
                raise RuntimeError(
                    f"Repair relabel count mismatch for {build}: "
                    f"index has {len(index_rows)} rows, discovered stream produced {len(streamed_all)} valid tiles."
                )

            for idx, streamed in enumerate(streamed_all):
                index_rows[idx]["map"] = str(streamed["map"])
                index_rows[idx]["tile_x"] = int(streamed["tile_x"])
                index_rows[idx]["tile_y"] = int(streamed["tile_y"])

            _write_index(index_rows, output_path)
            print(f"Wrote repaired index: {idx_path}")
            continue

        for map_name in ordered_maps:
            row_indices = [i for i, row in enumerate(index_rows) if str(row.get("map")) == map_name]
            streamed_rows = _stream_valid_tile_metadata(harvest_tool, client_root, map_name, build_version)

            if len(streamed_rows) != len(row_indices):
                raise RuntimeError(
                    f"Coordinate repair count mismatch for {build}/{map_name}: "
                    f"index has {len(row_indices)} rows, stream produced {len(streamed_rows)} valid tiles."
                )

            for row_idx, streamed in zip(row_indices, streamed_rows):
                index_rows[row_idx]["map"] = str(streamed["map"])
                index_rows[row_idx]["tile_x"] = int(streamed["tile_x"])
                index_rows[row_idx]["tile_y"] = int(streamed["tile_y"])

            print(f"  repaired {map_name}: {len(row_indices)} rows")

        _write_index(index_rows, output_path)
        print(f"Wrote repaired index: {idx_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build V16 consolidated Zarr dataset")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--build", type=str, help="Single build key (e.g. 3_3_5_12340)")
    common.add_argument("--builds", nargs="+", help="Multiple build keys")

    build_p = sub.add_parser("build", parents=[common])
    build_p.add_argument("--limit", type=int, default=None, help="Max tiles to extract")
    build_p.add_argument("--maps", nargs="+", default=None, help="Specific maps to extract")
    build_p.add_argument("--resume", action="store_true", help="Resume from <build>.zarr.partial if a compatible resume state exists")
    build_p.add_argument("--rebuild-existing", action="store_true", help="Rebuild even if a final <build>.zarr already looks complete")
    build_p.add_argument("--codec", choices=["lz4", "zstd"], default=DEFAULT_CODEC, help="Blosc codec for future writes")
    build_p.add_argument("--clevel", type=int, default=DEFAULT_CLEVEL, help="Blosc compression level")
    build_p.add_argument("--shuffle", choices=["noshuffle", "shuffle", "bitshuffle"], default=DEFAULT_SHUFFLE, help="Blosc shuffle mode")

    stats_p = sub.add_parser("stats", parents=[common])
    repair_p = sub.add_parser("repair-index", parents=[common])
    repair_p.add_argument("--no-backup", action="store_true", help="Skip creating index.parquet.bak before rewriting index.parquet")
    merge_p = sub.add_parser("merge-builds", parents=[common], help="Merge per-build stores into one combined Zarr store")
    merge_p.add_argument("--output-name", type=str, default="merged_all", help="Output store name (without .zarr)")
    merge_p.add_argument(
        "--dedupe-mode",
        choices=["none", "coords", "coords_height"],
        default="coords_height",
        help="Deduplication key: none, map+coords, or map+coords+height stats",
    )
    merge_p.add_argument("--batch-size", type=int, default=64, help="Array copy batch size during merge")
    merge_p.add_argument("--rebuild-existing", action="store_true", help="Overwrite existing merged output store")

    args = parser.parse_args()

    if args.command == "build":
        cmd_build(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "repair-index":
        cmd_repair_index(args)
    elif args.command == "merge-builds":
        cmd_merge_builds(args)


if __name__ == "__main__":
    main()
