"""Build precomputed Spec 077 albedo guidance stores.

Reads a V18 tensor-pack Zarr store containing ``alpha_256`` and writes a
sidecar Zarr store containing ``albedo_rgb_256``. The albedo is the same
texture-identity guidance signal used by ``HeightOnlyPriorDataset`` when
``include_albedo=True``: MCAL alpha weights composited with stable per-texture
colours from ``mcly_texture_ids``.
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

from harvester.compositor import composite_texture_identity_albedo  # noqa: E402

DEFAULT_CODEC = zarr.codecs.BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle")


def _nearest_resize_hwc(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Nearest-neighbor resize for HWC arrays."""
    if arr.ndim != 3:
        raise ValueError(f"Expected HWC array; got shape {arr.shape}")
    h, w = arr.shape[0], arr.shape[1]
    if h == target_h and w == target_w:
        return arr
    ys = np.linspace(0, h - 1, target_h).astype(np.int64)
    xs = np.linspace(0, w - 1, target_w).astype(np.int64)
    return arr[ys[:, None], xs[None, :], :]


def _read_index_rows(v18_path: Path, tile_count: int) -> list[dict]:
    index_path = v18_path / "index.parquet"
    if not index_path.exists():
        return [
            {
                "build": v18_path.stem.replace(".zarr", ""),
                "map": "",
                "map_name": "",
                "tile_id": idx,
                "tile_x": idx % 64,
                "tile_y": idx // 64,
            }
            for idx in range(tile_count)
        ]
    table = pq.read_table(str(index_path))
    rows: list[dict] = []
    for idx in range(min(table.num_rows, tile_count)):
        rows.append({col: table.column(col)[idx].as_py() for col in table.column_names})
    return rows


def _write_tiles_parquet(rows: list[dict], output_path: Path) -> None:
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    normalized: dict[str, list] = {
        "build": [],
        "map": [],
        "map_name": [],
        "tile_id": [],
        "tile_x": [],
        "tile_y": [],
        "albedo_key": [],
    }
    for idx, row in enumerate(rows):
        tile_id = int(row.get("tile_id", idx))
        map_name = str(row.get("map_name", row.get("map", "")))
        normalized["build"].append(str(row.get("build", "")))
        normalized["map"].append(str(row.get("map", map_name)))
        normalized["map_name"].append(map_name)
        normalized["tile_id"].append(tile_id)
        normalized["tile_x"].append(int(row.get("tile_x", tile_id % 64)))
        normalized["tile_y"].append(int(row.get("tile_y", tile_id // 64)))
        normalized["albedo_key"].append(f"albedo_rgb_256/{tile_id}")
    pq.write_table(pa.table(normalized), str(output_path))


def build_albedo_store(
    *,
    v18_path: Path,
    output_root: Path,
    max_tiles: int = 0,
    overwrite: bool = False,
) -> Path:
    if not v18_path.exists():
        raise FileNotFoundError(f"V18 store not found: {v18_path}")
    store = zarr.storage.LocalStore(str(v18_path), read_only=True)
    root = zarr.open_group(store, mode="r")
    if "alpha_256" not in root:
        raise KeyError(f"No alpha_256 array under {v18_path}")

    alpha = root["alpha_256"]
    mcly_texture_ids = root["mcly_texture_ids"] if "mcly_texture_ids" in root else None
    mcly_layer_mask = root["mcly_layer_mask"] if "mcly_layer_mask" in root else None
    tile_count = int(alpha.shape[0])
    if max_tiles > 0:
        tile_count = min(tile_count, int(max_tiles))
    if tile_count <= 0:
        raise ValueError(f"No alpha tiles available under {v18_path}")

    build = str(dict(root.attrs).get("build", v18_path.stem.replace(".zarr", "")))
    output_path = output_root / f"{build}.zarr"
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"Albedo store already exists: {output_path}; pass --overwrite to replace it")
        shutil.rmtree(output_path)

    output_root.mkdir(parents=True, exist_ok=True)
    out_store = zarr.storage.LocalStore(str(output_path), read_only=False)
    out_root = zarr.group(store=out_store)
    out = out_root.create_array(
        "albedo_rgb_256",
        shape=(tile_count, 256, 256, 3),
        chunks=(min(8, tile_count), 256, 256, 3),
        dtype=np.uint8,
        compressors=DEFAULT_CODEC,
    )

    for idx in range(tile_count):
        alpha_tile = np.asarray(alpha[idx], dtype=np.float32)
        if float(alpha_tile.max(initial=0.0)) > 1.5:
            alpha_tile = alpha_tile / 255.0
        alpha_tile = np.clip(alpha_tile, 0.0, 1.0)
        if alpha_tile.shape[0] != 256 or alpha_tile.shape[1] != 256:
            alpha_tile = _nearest_resize_hwc(alpha_tile, 256, 256)
        tex_tile = np.asarray(mcly_texture_ids[idx], dtype=np.int32) if mcly_texture_ids is not None else None
        layer_mask_tile = np.asarray(mcly_layer_mask[idx], dtype=np.float32) if mcly_layer_mask is not None else None
        albedo = composite_texture_identity_albedo(alpha_tile, tex_tile, layer_mask_tile)
        out[idx] = np.clip(np.rint(albedo * 255.0), 0, 255).astype(np.uint8)

    rows = _read_index_rows(v18_path, tile_count)
    _write_tiles_parquet(rows, output_path / "tiles.parquet")
    metadata = {
        "schema": "spec-077-albedo-guidance",
        "build": build,
        "source_v18_path": str(v18_path),
        "array": "albedo_rgb_256",
        "dtype": "uint8",
        "range": "0..255",
        "tile_count": tile_count,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "compositor": "harvester.compositor.composite_texture_identity_albedo",
        "colour_source": "stable hash of mcly_texture_ids with placeholder fallback",
    }
    out_root.attrs.update(metadata)
    (output_path / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build precomputed Spec 077 albedo guidance stores.")
    parser.add_argument("--v18-path", type=Path, required=True, help="Input V18 <build>.zarr store containing alpha_256.")
    parser.add_argument("--output-root", type=Path, required=True, help="Output root for <build>.zarr albedo stores.")
    parser.add_argument("--max-tiles", type=int, default=0, help="Optional smoke cap. 0 = all alpha tiles.")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Replace an existing output albedo store.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        output_path = build_albedo_store(
            v18_path=args.v18_path,
            output_root=args.output_root,
            max_tiles=int(args.max_tiles),
            overwrite=bool(args.overwrite),
        )
    except Exception as exc:  # pragma: no cover - CLI diagnostics
        print(f"Failed to build albedo store: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote albedo store: {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
