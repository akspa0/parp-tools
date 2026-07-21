"""Spec 116 US3: structure inference — predict layer structure from the minimap alone.

From a raw minimap image, a frozen ``StructureSlotNet`` predicts per-chunk surface family
probabilities. When a tile MTEX table is supplied, the **legality resolver** picks a legal
same-family local texture id from that tile's own table (SC-004 = 100% legal references).
When no table is available (OOD), the resolver never fabricates a reference (D-05) and the
audit record flags ``legal_table_available=false``.

The inference contract is ``v50-structure-infer-v1``: a per-tile audit record with the
checkpoint identity, input store refs, the legality flag, and per-tile predictions.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from harvester.spec116.structure_contract import sha256_file, validate_structure_infer
from harvester.spec116.structure_model import CHUNK_GRID, build_structure_model
from harvester.v50.terrain_feature_labels import (
    TAXONOMY_REVISION,
    classify_texture_name,
)

STRUCTURE_INFER_SCHEMA = "v50-structure-infer-v1"


class StructureInferError(ValueError):
    """Raised when the structure inference contract is violated."""


def resolve_legal_reference(
    predicted_family: int,
    texture_names: list[str],
) -> int | None:
    """Pick a legal same-family local texture id from the tile's own MTEX table.

    Returns the local index (0-based) of the first texture in ``texture_names`` whose
    classified family matches ``predicted_family``, or ``None`` if no match exists. The
    resolver never fabricates a reference: if the tile's table has no texture of the
    predicted family, the chunk is marked low-confidence and gets no reference (D-05).
    """
    for local_id, name in enumerate(texture_names):
        if classify_texture_name(name) == predicted_family:
            return local_id
    return None


def infer_structure_images(
    *,
    checkpoint: Path,
    inputs: list[Path],
    tile_table: Path | None,
    slot: int,
    base: int = 0,
    device: str = "cpu",
) -> dict:
    """Run structure inference over loose 256x256 minimap tile(s) -- no v50 store required.

    Mirrors ``harvester.v50.terrain_feature_infer``'s deployment contract: RGB in, prediction
    out, no client-derived ground truth anywhere, so it runs unchanged on an arbitrary image --
    including a hand-painted OOD tile that was never part of any harvested store (spec US3
    acceptance 4). When ``tile_table`` is supplied, the SAME table resolves legal references for
    every input image (this is a manual ad hoc run over a handful of tiles, not a store-joined
    batch with one table per tile). Omitting it exercises the OOD path (D-05): no reference is
    fabricated and the audit flags ``legal_table_available=false``.

    ``tile_table`` is a JSON file: either a bare list of texture name strings, or an object with a
    ``texture_names`` key holding that list (the tile's own MTEX table, in order).
    """
    import torch

    from harvester.v50.direct_geometry_infer import discover_tiles, load_tile_rgb

    tiles = discover_tiles(inputs)

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    ckpt_slot = int(ckpt["slot"])
    ckpt_base = int(ckpt["base"])
    ckpt_taxonomy = str(ckpt["taxonomy_revision"])
    if ckpt_taxonomy != TAXONOMY_REVISION:
        raise StructureInferError(
            f"checkpoint taxonomy {ckpt_taxonomy!r} != code {TAXONOMY_REVISION!r}"
        )
    if ckpt_slot != slot:
        raise StructureInferError(
            f"checkpoint is for slot {ckpt_slot}, but inference requested slot {slot}"
        )

    model, _ = build_structure_model(slot=slot, base=base if base else ckpt_base)
    model.load_state_dict(ckpt["model"])
    model.eval()
    dev = torch.device(device)
    model = model.to(dev)

    texture_names: list[str] | None = None
    if tile_table is not None:
        payload = json.loads(tile_table.read_text(encoding="utf-8"))
        texture_names = list(payload["texture_names"]) if isinstance(payload, dict) else list(payload)
        if not texture_names:
            raise StructureInferError(f"--tile-table is empty: {tile_table}")
    has_table = texture_names is not None

    per_tile: list[dict] = []
    all_legal = True
    total_chunks = CHUNK_GRID * CHUNK_GRID

    for tile_path in tiles:
        rgb = load_tile_rgb(tile_path).astype(np.float32) / 255.0
        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(dev)
        with torch.no_grad():
            logits = model(rgb_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        predicted_families = probs.argmax(axis=0)
        confidences = probs.max(axis=0)

        references: list[list[int | None]] = []
        legal_count = 0
        low_confidence_count = 0

        if has_table:
            for cy in range(CHUNK_GRID):
                row_refs: list[int | None] = []
                for cx in range(CHUNK_GRID):
                    ref = resolve_legal_reference(int(predicted_families[cy, cx]), texture_names)
                    row_refs.append(ref)
                    if ref is not None:
                        legal_count += 1
                    else:
                        low_confidence_count += 1
                references.append(row_refs)
        else:
            # OOD: no table, no references (D-05).
            references = [[None] * CHUNK_GRID for _ in range(CHUNK_GRID)]
            low_confidence_count = total_chunks

        if legal_count < total_chunks and has_table:
            all_legal = False

        per_tile.append({
            "tile_row": len(per_tile),
            "input": str(tile_path),
            "slot": slot,
            "legal_table_available": has_table,
            "legal_reference_count": legal_count,
            "low_confidence_count": low_confidence_count,
            "total_chunks": total_chunks,
            "predicted_family_grid": predicted_families.tolist(),
            "confidence_grid": confidences.tolist(),
            "references": references if has_table else None,
        })

    sc004 = all_legal if has_table else False

    record = {
        "schema": STRUCTURE_INFER_SCHEMA,
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
            "taxonomy_revision": TAXONOMY_REVISION,
        },
        "inputs": [{"path": str(t), "sha256": sha256_file(t)} for t in tiles],
        "legal_table_available": has_table,
        "sc004_all_references_legal": sc004,
        "per_tile": per_tile,
    }
    validate_structure_infer(record)
    return record


def infer_structure(
    *,
    checkpoint: Path,
    store: Path,
    dumps: list[Path],
    slot: int,
    base: int = 32,
    device: str = "cpu",
) -> dict:
    """Run structure inference over every tile in a v50 store and build the audit record.

    For each tile:
    1. Load the minimap RGB and run the frozen ``StructureSlotNet`` → per-chunk family
       probabilities (16x16, CLASS_COUNT).
    2. If the tile has a texture-name dump, resolve each chunk's predicted family to a legal
       local texture id from the tile's own MTEX table (SC-004).
    3. If no dump is available (OOD), set ``legal_table_available=false`` and emit no
       references (D-05).

    Returns the ``v50-structure-infer-v1`` audit record.
    """
    import pyarrow.parquet as pq
    import torch
    import zarr

    from harvester.v50.terrain_feature_labels import load_texture_name_dump

    # --- Load checkpoint. --- #
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    ckpt_slot = int(ckpt["slot"])
    ckpt_base = int(ckpt["base"])
    ckpt_taxonomy = str(ckpt["taxonomy_revision"])
    if ckpt_taxonomy != TAXONOMY_REVISION:
        raise StructureInferError(
            f"checkpoint taxonomy {ckpt_taxonomy!r} != code {TAXONOMY_REVISION!r}"
        )
    if ckpt_slot != slot:
        raise StructureInferError(
            f"checkpoint is for slot {ckpt_slot}, but inference requested slot {slot}"
        )

    model, _ = build_structure_model(slot=slot, base=base if base else ckpt_base)
    model.load_state_dict(ckpt["model"])
    model.eval()
    dev = torch.device(device)
    model = model.to(dev)

    # --- Open store. --- #
    group = zarr.open_group(str(store), mode="r")
    if "minimap_rgb" not in group:
        raise StructureInferError(f"store is missing minimap_rgb: {store}")
    index_path = store / "index.parquet"
    if not index_path.exists():
        raise StructureInferError(f"store has no index.parquet: {store}")
    index_rows = pq.read_table(index_path).to_pylist()
    row_count = len(index_rows)

    names_by_tile = load_texture_name_dump(dumps) if dumps else {}

    # --- Inference. --- #
    per_tile: list[dict] = []
    all_legal = True
    any_table = False

    for row in range(row_count):
        meta = index_rows[row]
        key = (str(meta.get("map")), int(meta.get("tile_x", -1)), int(meta.get("tile_y", -1)))
        texture_names = names_by_tile.get(key)
        has_table = texture_names is not None and len(texture_names) > 0
        if has_table:
            any_table = True

        rgb = np.asarray(group["minimap_rgb"][row], dtype=np.float32) / 255.0
        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(dev)

        with torch.no_grad():
            logits = model(rgb_tensor)  # (1, CLASS_COUNT, 16, 16)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()  # (CLASS_COUNT, 16, 16)

        predicted_families = probs.argmax(axis=0)  # (16, 16)
        confidences = probs.max(axis=0)  # (16, 16)

        # Legality resolution.
        references: list[list[int | None]] = []  # (16, 16) of local_id or None
        legal_count = 0
        total_chunks = CHUNK_GRID * CHUNK_GRID
        low_confidence_count = 0

        if has_table:
            for cy in range(CHUNK_GRID):
                row_refs: list[int | None] = []
                for cx in range(CHUNK_GRID):
                    family = int(predicted_families[cy, cx])
                    ref = resolve_legal_reference(family, texture_names)
                    row_refs.append(ref)
                    if ref is not None:
                        legal_count += 1
                    else:
                        low_confidence_count += 1
                references.append(row_refs)
        else:
            # OOD: no table, no references (D-05).
            references = [[None] * CHUNK_GRID for _ in range(CHUNK_GRID)]
            low_confidence_count = total_chunks

        if legal_count < total_chunks and has_table:
            all_legal = False

        per_tile.append({
            "tile_row": row,
            "map": str(meta.get("map")),
            "tile_x": int(meta.get("tile_x", -1)),
            "tile_y": int(meta.get("tile_y", -1)),
            "slot": slot,
            "legal_table_available": has_table,
            "legal_reference_count": legal_count,
            "low_confidence_count": low_confidence_count,
            "total_chunks": total_chunks,
            "predicted_family_grid": predicted_families.tolist(),
            "confidence_grid": confidences.tolist(),
            "references": references if has_table else None,
        })

    # --- Build audit record. --- #
    legal_table_available = any_table
    sc004 = all_legal if legal_table_available else False

    record = {
        "schema": STRUCTURE_INFER_SCHEMA,
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
            "taxonomy_revision": TAXONOMY_REVISION,
        },
        "inputs": [
            {
                "path": str(store.resolve()),
                "sha256": sha256_file(store / "index.parquet"),
            }
        ],
        "legal_table_available": legal_table_available,
        "sc004_all_references_legal": sc004,
        "per_tile": per_tile,
    }
    validate_structure_infer(record)
    return record


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Spec 116 US3: structure inference from minimap (legality resolver + OOD audit)"
    )
    ap.add_argument("--checkpoint", required=True, type=Path, help="frozen StructureSlotNet checkpoint")
    ap.add_argument("--store", type=Path, default=None,
                    help="v50 curriculum Zarr store (batch mode; mutually exclusive with --inputs)")
    ap.add_argument("--dumps", nargs="*", type=Path, default=[],
                    help="store mode: texture-name dump JSON files (OOD if omitted)")
    ap.add_argument("--inputs", nargs="+", type=Path, default=None,
                    help="loose 256x256 minimap tile(s) or folder(s) (mutually exclusive with --store); "
                         "runs unchanged on a hand-painted OOD image with no store backing")
    ap.add_argument("--tile-table", type=Path, default=None,
                    help="--inputs mode: one MTEX table JSON (list of texture names, or "
                         "{'texture_names': [...]}) applied to every input image; omit for OOD")
    ap.add_argument("--slot", required=True, type=int, choices=[1, 2, 3], help="detail slot")
    ap.add_argument("--base", type=int, default=0, help="override base width (0 = use checkpoint's)")
    ap.add_argument("--device", default="cpu", help="cpu (default) or cuda")
    ap.add_argument("--output", type=Path, default=None, help="output audit JSON (required with --write)")
    ap.add_argument("--write", action="store_true", help="write the audit record (default: print summary)")
    args = ap.parse_args(argv)

    if args.write and args.output is None:
        ap.error("--write requires --output")
    if (args.store is None) == (args.inputs is None):
        ap.error("exactly one of --store or --inputs is required")
    if args.tile_table is not None and args.inputs is None:
        ap.error("--tile-table only applies to --inputs mode (store mode uses --dumps)")

    if args.inputs is not None:
        record = infer_structure_images(
            checkpoint=args.checkpoint,
            inputs=args.inputs,
            tile_table=args.tile_table,
            slot=args.slot,
            base=args.base,
            device=args.device,
        )
    else:
        record = infer_structure(
            checkpoint=args.checkpoint,
            store=args.store,
            dumps=args.dumps,
            slot=args.slot,
            base=args.base,
            device=args.device,
        )

    print("=== Spec 116 US3: structure inference ===")
    print(f"checkpoint: {record['checkpoint']['path']}")
    print(f"legal_table_available: {record['legal_table_available']}")
    print(f"sc004_all_references_legal: {record['sc004_all_references_legal']}")
    print(f"tiles: {len(record['per_tile'])}")
    for tile in record["per_tile"][:5]:
        label = tile["input"] if "input" in tile else f"({tile['tile_x']},{tile['tile_y']})"
        print(
            f"  tile {label}: "
            f"legal={tile['legal_reference_count']}/{tile['total_chunks']} "
            f"low_conf={tile['low_confidence_count']}"
        )
    if len(record["per_tile"]) > 5:
        print(f"  ... ({len(record['per_tile']) - 5} more)")

    if not args.write:
        print("\nDRY RUN ONLY -- pass --write to emit the audit record.")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"\nwrote audit: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
