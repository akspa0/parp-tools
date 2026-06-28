"""Spec 077 Phase 5 (US4) ADT-free processed prior generation.

The teacher-prior pipeline (spec 077 Phase 3) produces a processed
minimap prior using ADT-backed placement arrays for supervision. The
runtime path (this module) produces the same 5-channel prior shape from
a *predicted* object mask instead — no ADT data required at inference
time.

The contract mirrors ``teacher_prior.PRIOR_CHANNELS``:

* ``[..., 0:3]`` = object-suppressed RGB (per-tile median fill on
  predicted object pixels)
* ``[..., 3]``   = predicted object mask (uint8)
* ``[..., 4]``   = mask confidence band (uint8)

The first pass uses a *provided* predicted mask (e.g. produced by an
object-mask lane trained later, or by a simple minimap heuristic for
the first proof). The object-mask training lane itself is documented
as T034 and lands separately.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr
import zarr.codecs
import zarr.storage

_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from harvester.teacher_prior import PRIOR_CHANNELS  # noqa: E402
from harvester.teacher_prior import suppress_object_pixels  # noqa: E402

DEFAULT_CODEC = zarr.codecs.BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle")


def build_adt_free_prior_tensor(
    minimap_rgb: np.ndarray,
    predicted_mask: np.ndarray,
    confidence_band: np.ndarray | None = None,
) -> np.ndarray:
    """Build the 5-channel ADT-free processed prior for one tile.

    Parameters
    ----------
    minimap_rgb:
        (256, 256, 3) uint8 raw minimap.
    predicted_mask:
        (256, 256) uint8 predicted object mask (0..255). Values
        ``>= 128`` are treated as object pixels; below that, as
        terrain.
    confidence_band:
        Optional (256, 256) uint8 confidence map. When ``None`` the
        confidence band defaults to ``255`` everywhere (the mask
        itself is the confidence).
    """
    if minimap_rgb.ndim != 3 or minimap_rgb.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 minimap; got {minimap_rgb.shape}")
    if minimap_rgb.dtype != np.uint8:
        minimap_rgb = minimap_rgb.astype(np.uint8)
    if predicted_mask.shape != minimap_rgb.shape[:2]:
        raise ValueError(
            f"Predicted mask shape {predicted_mask.shape} does not match "
            f"minimap {minimap_rgb.shape[:2]}"
        )
    bin_mask = (predicted_mask >= 128).astype(np.uint8)
    suppressed = suppress_object_pixels(minimap_rgb, bin_mask)
    if confidence_band is None:
        confidence_band = np.full(minimap_rgb.shape[:2], 255, dtype=np.uint8)
    return np.concatenate(
        [suppressed, bin_mask[:, :, None], confidence_band[:, :, None]],
        axis=2,
    ).astype(np.uint8, copy=False)


def _read_minimap(v18_path: Path) -> np.ndarray:
    store = zarr.storage.LocalStore(str(v18_path), read_only=True)
    root = zarr.open_group(store, mode="r")
    return np.asarray(root["minimap_rgb"][:])


def _read_predicted_masks(path: Path, expected_n: int) -> np.ndarray:
    """Read a predicted-mask array from NPZ or Zarr."""
    if path.suffix == ".npz":
        data = np.load(str(path), allow_pickle=False)
        return np.asarray(data["predicted_mask"]).astype(np.uint8, copy=False)
    if path.is_dir() and (path / "predicted_object_mask_256").exists():
        store = zarr.storage.LocalStore(str(path), read_only=True)
        root = zarr.open_group(store, mode="r")
        return np.asarray(root["predicted_object_mask_256"][:]).astype(np.uint8, copy=False)
    if path.suffix == ".zarr" and (path / "predicted_object_mask_256").exists():
        store = zarr.storage.LocalStore(str(path), read_only=True)
        root = zarr.open_group(store, mode="r")
        return np.asarray(root["predicted_object_mask_256"][:]).astype(np.uint8, copy=False)
    raise FileNotFoundError(
        f"Could not find a predicted-mask array at {path}. "
        f"Expected an NPZ with key 'predicted_mask' or a Zarr group with "
        f"'predicted_object_mask_256'."
    )


def _write_zarr(
    output_path: Path,
    raw_minimap: np.ndarray,
    predicted_mask: np.ndarray,
    processed_prior: np.ndarray,
    metadata: dict,
) -> Path:
    if output_path.exists():
        shutil.rmtree(output_path)
    store = zarr.storage.LocalStore(str(output_path), read_only=False)
    root = zarr.group(store=store)
    root.create_array(
        "raw_minimap_rgb_256",
        data=raw_minimap,
        chunks=(min(8, raw_minimap.shape[0]), 256, 256, 3),
        compressors=DEFAULT_CODEC,
    )
    root.create_array(
        "predicted_object_mask_256",
        data=predicted_mask,
        chunks=(min(8, predicted_mask.shape[0]), 256, 256),
        compressors=DEFAULT_CODEC,
    )
    root.create_array(
        "processed_minimap_prior_256",
        data=processed_prior,
        chunks=(min(8, processed_prior.shape[0]), 256, 256, processed_prior.shape[3]),
        compressors=DEFAULT_CODEC,
    )
    root.attrs.update(dict(metadata.items()))
    return output_path


def _write_tiles_parquet(records: list[dict], path: Path) -> None:
    if not records:
        path.write_text("", encoding="utf-8")
        return
    table = pa.table(
        {
            "build": [r["build"] for r in records],
            "map": [r["map_name"] for r in records],
            "tile_id": [r["tile_id"] for r in records],
            "raw_minimap_key": [r["raw_minimap_key"] for r in records],
            "predicted_mask_key": [r["predicted_mask_key"] for r in records],
            "processed_prior_key": [r["processed_prior_key"] for r in records],
        }
    )
    pq.write_table(table, str(path))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build spec 077 ADT-free processed prior from a predicted object mask."
    )
    parser.add_argument("--v18-path", type=Path, required=True,
                        help="Source V18 Zarr store (provides raw minimap).")
    parser.add_argument("--predicted-mask", type=Path, required=True,
                        help="Path to a predicted-mask NPZ or Zarr store.")
    parser.add_argument("--output-root", type=Path, required=True,
                        help="Directory under which <build>.zarr is written.")
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument("--start-tile-id", type=int, default=0)
    return parser.parse_args(argv)


def main_with_args(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.v18_path.exists():
        print(f"V18 store not found: {args.v18_path}", file=sys.stderr)
        return 2
    if not args.predicted_mask.exists():
        print(f"Predicted mask path not found: {args.predicted_mask}", file=sys.stderr)
        return 2

    build = args.v18_path.stem.replace(".zarr", "")
    raw = _read_minimap(args.v18_path)
    predicted = _read_predicted_masks(args.predicted_mask, expected_n=raw.shape[0])
    if predicted.shape[0] != raw.shape[0]:
        print(
            f"Predicted-mask tile count {predicted.shape[0]} does not match "
            f"minimap {raw.shape[0]}; truncating to the smaller count.",
            file=sys.stderr,
        )
    n_tiles = min(raw.shape[0], predicted.shape[0])

    raw_list: list[np.ndarray] = []
    mask_list: list[np.ndarray] = []
    prior_list: list[np.ndarray] = []
    records: list[dict] = []
    for i in range(args.start_tile_id, n_tiles):
        if args.max_tiles is not None and len(records) >= args.max_tiles:
            break
        minimap = raw[i]
        mask = predicted[i]
        tensor = build_adt_free_prior_tensor(minimap, mask)
        raw_list.append(minimap)
        mask_list.append(mask)
        prior_list.append(tensor)
        records.append(
            {
                "build": build,
                "map_name": "",
                "tile_id": i,
                "raw_minimap_key": f"raw_minimap_rgb_256/{len(records)}",
                "predicted_mask_key": f"predicted_object_mask_256/{len(records)}",
                "processed_prior_key": f"processed_minimap_prior_256/{len(records)}",
            }
        )

    raw_arr = np.stack(raw_list, axis=0).astype(np.uint8, copy=False)
    mask_arr = np.stack(mask_list, axis=0).astype(np.uint8, copy=False)
    prior_arr = np.stack(prior_list, axis=0).astype(np.uint8, copy=False)
    args.output_root.mkdir(parents=True, exist_ok=True)
    output_path = args.output_root / f"{build}.zarr"
    metadata = {
        "schema": "spec-077-adt-free-prior",
        "schema_version": "1",
        "build": build,
        "source_v18_path": str(args.v18_path),
        "predicted_mask_path": str(args.predicted_mask),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tile_count": len(records),
        "phase5_prior_channels": list(PRIOR_CHANNELS),
    }
    _write_zarr(output_path, raw_arr, mask_arr, prior_arr, metadata)
    _write_tiles_parquet(records, output_path / "tiles.parquet")
    print(f"Wrote {len(records)} ADT-free prior tiles to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main_with_args())
