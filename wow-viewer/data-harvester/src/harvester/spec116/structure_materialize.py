"""Spec 116 US5: materialize a frozen structure classifier's predicted output into a store.

The geometry model must train on the structure classifier's **predicted** structure (errors and
all), never on ground-truth labels — exactly the FR-006 rule the terrain-feature materializer
already enforces. This module runs one frozen ``StructureSlotNet`` checkpoint over the selected
curriculum rows and writes a derived store:

- ``structure_family``: int8 (N, 16, 16) predicted family per chunk.
- ``structure_confidence``: float16 (N, 16, 16) max softmax probability per chunk.
- ``structure_legal``: bool (N, 16, 16) whether a legal same-family reference was resolved.
- ``index.parquet``: row-aligned with the source curriculum (source_row_index, map, tile_x,
  tile_y, split, source).
- attrs: schema, checkpoint path + sha256, taxonomy revision, slot, source store identity.

Source stores are NEVER mutated; the derived store is immutable once written. Swapping the
checkpoint means writing a new store, which is what keeps the structure model independently
replaceable (constitution IV).
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from harvester.spec116.structure_contract import sha256_file
from harvester.spec116.structure_infer import resolve_legal_reference
from harvester.spec116.structure_model import CHUNK_GRID, build_structure_model
from harvester.v50.terrain_feature_labels import (
    CLASS_COUNT,
    TAXONOMY_REVISION,
    load_texture_name_dump,
)

STRUCTURE_STORE_SCHEMA = "v116-structure-store-v1"


class StructureMaterializeError(ValueError):
    """Raised when a structure store cannot be materialized as declared."""


def materialize_structure(
    *,
    store: Path,
    checkpoint: Path,
    output: Path,
    dumps: list[Path],
    slot: int,
    base: int = 0,
    device: str = "cpu",
    write: bool = False,
) -> dict:
    """Run a frozen structure checkpoint over the source store and optionally persist.

    Returns the materialization plan (or the written store summary if ``write=True``). The
    source store is never mutated; the output is a new Zarr group + index.parquet.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    import zarr

    # --- Load checkpoint. --- #
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    ckpt_slot = int(ckpt["slot"])
    ckpt_base = int(ckpt["base"])
    ckpt_taxonomy = str(ckpt["taxonomy_revision"])
    if ckpt_taxonomy != TAXONOMY_REVISION:
        raise StructureMaterializeError(
            f"checkpoint taxonomy {ckpt_taxonomy!r} != code {TAXONOMY_REVISION!r}"
        )
    if ckpt_slot != slot:
        raise StructureMaterializeError(
            f"checkpoint is for slot {ckpt_slot}, but materialization requested slot {slot}"
        )
    checkpoint_sha = sha256_file(checkpoint)

    # --- Open source store. --- #
    source_group = zarr.open_group(str(store), mode="r")
    if "minimap_rgb" not in source_group:
        raise StructureMaterializeError(f"store is missing minimap_rgb: {store}")
    index_path = store / "index.parquet"
    if not index_path.exists():
        raise StructureMaterializeError(f"store has no index.parquet: {store}")
    index_rows = pq.read_table(index_path).to_pylist()
    row_count = len(index_rows)

    names_by_tile = load_texture_name_dump(dumps) if dumps else {}

    # --- Build model. --- #
    model, _ = build_structure_model(slot=slot, base=base if base else ckpt_base)
    model.load_state_dict(ckpt["model"])
    model.eval()
    dev = torch.device(device)
    model = model.to(dev)

    # --- Plan. --- #
    plan = {
        "schema": "v116-structure-materialize-plan-v1",
        "source_store": str(store),
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_sha,
            "slot": slot,
            "base": ckpt_base,
            "taxonomy_revision": TAXONOMY_REVISION,
        },
        "selected_rows": row_count,
        "output_arrays": {
            "structure_family": {"shape": [row_count, CHUNK_GRID, CHUNK_GRID], "dtype": "int8"},
            "structure_confidence": {"shape": [row_count, CHUNK_GRID, CHUNK_GRID], "dtype": "float16"},
            "structure_legal": {"shape": [row_count, CHUNK_GRID, CHUNK_GRID], "dtype": "bool"},
        },
    }

    if not write:
        return plan

    # --- Write derived store. --- #
    if output.exists() and any(output.iterdir()):
        raise StructureMaterializeError(
            f"refusing to overwrite non-empty structure store {output}; choose a new path"
        )

    output.mkdir(parents=True, exist_ok=False)
    group = zarr.open_group(str(output), mode="w")

    family_array = group.create_array(
        "structure_family",
        shape=(row_count, CHUNK_GRID, CHUNK_GRID),
        chunks=(1, CHUNK_GRID, CHUNK_GRID),
        dtype=np.int8,
    )
    confidence_array = group.create_array(
        "structure_confidence",
        shape=(row_count, CHUNK_GRID, CHUNK_GRID),
        chunks=(1, CHUNK_GRID, CHUNK_GRID),
        dtype=np.float16,
    )
    legal_array = group.create_array(
        "structure_legal",
        shape=(row_count, CHUNK_GRID, CHUNK_GRID),
        chunks=(1, CHUNK_GRID, CHUNK_GRID),
        dtype=np.bool_,
    )

    for row in range(row_count):
        meta = index_rows[row]
        key = (str(meta.get("map")), int(meta.get("tile_x", -1)), int(meta.get("tile_y", -1)))
        texture_names = names_by_tile.get(key)
        has_table = texture_names is not None and len(texture_names) > 0

        rgb = np.asarray(source_group["minimap_rgb"][row], dtype=np.float32) / 255.0
        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(dev)

        with torch.no_grad():
            logits = model(rgb_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        predicted = probs.argmax(axis=0).astype(np.int8)
        confidence = probs.max(axis=0).astype(np.float16)
        legal = np.zeros((CHUNK_GRID, CHUNK_GRID), dtype=np.bool_)

        if has_table:
            for cy in range(CHUNK_GRID):
                for cx in range(CHUNK_GRID):
                    ref = resolve_legal_reference(int(predicted[cy, cx]), texture_names)
                    legal[cy, cx] = ref is not None

        family_array[row] = predicted
        confidence_array[row] = confidence
        legal_array[row] = legal

    # --- Row-aligned index. --- #
    derived_index = [
        {
            "source_row_index": row,
            "map": str(index_rows[row].get("map")),
            "tile_x": int(index_rows[row].get("tile_x", -1)),
            "tile_y": int(index_rows[row].get("tile_y", -1)),
            "split": str(index_rows[row].get("split", "train")),
            "source": str(index_rows[row].get("source", "authored")),
        }
        for row in range(row_count)
    ]
    pq.write_table(pa.Table.from_pylist(derived_index), output / "index.parquet")

    # --- Store attrs. --- #
    created_utc = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    group.attrs.update(
        {
            "schema": STRUCTURE_STORE_SCHEMA,
            "created_utc": created_utc,
            "source_store": str(store),
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "slot": slot,
            "base": ckpt_base,
            "taxonomy_revision": TAXONOMY_REVISION,
            "class_count": CLASS_COUNT,
            "chunk_grid": CHUNK_GRID,
        }
    )

    summary = {
        **plan,
        "schema": STRUCTURE_STORE_SCHEMA,
        "created_utc": created_utc,
        "output": str(output),
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Spec 116 US5: materialize predicted structure into a derived store (dry-run by default)"
    )
    ap.add_argument("--store", required=True, type=Path, help="source v50 curriculum Zarr store")
    ap.add_argument("--checkpoint", required=True, type=Path, help="frozen StructureSlotNet checkpoint")
    ap.add_argument("--output", required=True, type=Path, help="derived structure store output directory")
    ap.add_argument("--dumps", nargs="*", type=Path, default=[], help="texture-name dump JSON files")
    ap.add_argument("--slot", required=True, type=int, choices=[1, 2, 3], help="detail slot")
    ap.add_argument("--base", type=int, default=0, help="override base width (0 = use checkpoint's)")
    ap.add_argument("--device", default="cpu", help="cpu (default) or cuda")
    ap.add_argument("--write", action="store_true", help="write the derived store (default: print plan only)")
    args = ap.parse_args(argv)

    result = materialize_structure(
        store=args.store,
        checkpoint=args.checkpoint,
        output=args.output,
        dumps=args.dumps,
        slot=args.slot,
        base=args.base,
        device=args.device,
        write=args.write,
    )
    print(json.dumps(result, indent=2, default=str), flush=True)

    if not args.write:
        print("DRY RUN ONLY -- pass --write to emit the derived structure store.", flush=True)
        return 0

    print(f"wrote structure store: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
