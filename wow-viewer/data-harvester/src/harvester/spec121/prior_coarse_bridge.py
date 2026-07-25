"""Spec 121 T014: bridge a frozen Stage A checkpoint's lattice predictions into the detailer's
``--coarse-store`` contract (``v114-coarse-relief-v1``).

The detailer trainer consumes a materialized coarse store via ``--coarse-store`` and validates it
with ``validate_coarse_store`` (schema, source_filter, row alignment). This bridge produces that
exact schema from a Stage A checkpoint (MitB0LatticeNet or LatticeNet, auto-detected from the
checkpoint's ``backbone_config``), so the detailer runs with zero trainer changes — the predicted
WDL prior IS the coarse field.

The lattice's native 545 samples are two regular grids (17x17 outer + 16x16 inner). Each is
independently bilinear-upsampled to 257x257 and averaged, matching the ``_dense_lattice_field``
preview logic in the trainer and the Spec 117 bridge rule.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from harvester.spec117.lattice_contract import INNER_DIM, OUTER_DIM
from harvester.spec121.lattice_backbone_model import (
    LATTICE_NET_ID,
    MIT_B0_LATTICE_ID,
    build_stage_a_model,
    config_from_payload,
)
from harvester.v50.direct_geometry_materialize import COARSE_ARRAY, COARSE_STORE_SCHEMA
from harvester.v50.height_relative_train import (
    TrainerContractError,
    curriculum_identity,
    select_training_rows,
    validate_curriculum_contract,
    validate_source_selection,
)
from harvester.v50.model_stage_contract import sha256_file


class PriorBridgeError(ValueError):
    """Raised when a Stage A checkpoint cannot be bridged into a coarse store."""


def _upsample_to_257(outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
    """Bilinear-upsample both grids to 257x257 and average (Spec 117 bridge rule)."""
    import torch
    import torch.nn.functional as functional

    def _up(g: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(np.asarray(g, dtype=np.float32))[None, None]
        return functional.interpolate(t, size=(257, 257), mode="bilinear", align_corners=True)[0, 0].numpy()

    return (_up(outer) + _up(inner)) / 2.0


def _rebuild_stage_a(checkpoint: dict, device: str):
    """Rebuild the Stage A model from checkpoint's backbone_config (no Hub access needed)."""
    import torch

    backbone = checkpoint.get("backbone_config", {})
    arch = str(backbone.get("architecture", LATTICE_NET_ID))
    if arch == LATTICE_NET_ID:
        model, _ = build_stage_a_model(LATTICE_NET_ID, base=int(backbone.get("base", 64)))
    elif arch == MIT_B0_LATTICE_ID:
        segformer_config = backbone.get("segformer_config", {})
        model, _ = build_stage_a_model(
            MIT_B0_LATTICE_ID,
            mit_config=config_from_payload(segformer_config),
        )
    else:
        raise PriorBridgeError(f"unknown architecture {arch!r} in checkpoint backbone_config")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model.to(torch.device(device))


def bridge_prior_to_coarse(
    *,
    store: Path,
    checkpoint_path: Path,
    output: Path,
    source: str,
    release: str,
    device: str = "cpu",
    write: bool = False,
) -> dict:
    """Run a frozen Stage A checkpoint over selected rows and optionally persist a coarse store.

    Returns the plan (dry-run) or the written store summary (write=True). The source store is
    never mutated; the derived coarse store is immutable once written.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    import zarr

    from harvester.v50.contracts import require_store_release

    source_group = zarr.open_group(str(store), mode="r")
    try:
        require_store_release(source_group, release, store=store)
    except ValueError as exc:
        raise PriorBridgeError(str(exc)) from exc
    index_path = store / "index.parquet"
    if not index_path.exists():
        raise PriorBridgeError(f"store has no index.parquet: {store}")
    index = pq.read_table(index_path).to_pylist()
    try:
        array_lengths = {name: int(source_group[name].shape[0]) for name in source_group.array_keys()}
        validate_curriculum_contract(attrs=dict(source_group.attrs), array_lengths=array_lengths, index_rows=index)
        validate_source_selection(attrs=dict(source_group.attrs), source=source)
        selected = select_training_rows(index, source)
    except TrainerContractError as exc:
        raise PriorBridgeError(str(exc)) from exc
    if not selected:
        raise PriorBridgeError(f"source filter {source!r} selected zero rows")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_sha = sha256_file(checkpoint_path)
    model = _rebuild_stage_a(ckpt, device)

    plan = {
        "schema": "v121-prior-coarse-bridge-plan-v1",
        "source_store": str(store),
        "source_filter": source,
        "selected_rows": len(selected),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha,
            "architecture": str(ckpt.get("backbone_config", {}).get("architecture", "?")),
            "epoch": int(ckpt.get("epoch", 0)),
            "val_mae": float(ckpt.get("val_mae", float("nan"))),
        },
        "output_array": {
            "name": COARSE_ARRAY,
            "shape": [len(selected), 257, 257],
            "dtype": "float16",
        },
    }
    if not write:
        return plan

    if output.exists() and any(output.iterdir()):
        raise PriorBridgeError(f"refusing to overwrite non-empty coarse store {output}; choose a new path")

    output.mkdir(parents=True, exist_ok=False)
    group = zarr.open_group(str(output), mode="w")
    array = group.create_array(
        COARSE_ARRAY,
        shape=(len(selected), 257, 257),
        chunks=(1, 257, 257),
        dtype=np.float16,
    )

    outer_count = OUTER_DIM * OUTER_DIM
    with torch.no_grad():
        for position, row_index in enumerate(selected):
            rgb = np.asarray(source_group["minimap_rgb"][row_index], dtype=np.uint8)
            rgb_t = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
            pred = model(rgb_t).squeeze(0).cpu().numpy()
            outer = pred[:outer_count].reshape(OUTER_DIM, OUTER_DIM)
            inner = pred[outer_count:].reshape(INNER_DIM, INNER_DIM)
            dense = _upsample_to_257(outer, inner)
            array[position] = dense.astype(np.float16)

    # Write index.parquet matching the coarse store schema (source_group_id, split, minimap_source).
    coarse_index = []
    for position, row_index in enumerate(selected):
        src = index[row_index]
        coarse_index.append({
            "source_group_id": int(position),
            "split": str(src.get("split", "train")),
            "minimap_source": str(src.get("minimap_source", "authored")),
            "tile_row": int(src.get("tile_row", -1)),
            "map": str(src.get("map", "?")),
            "tile_x": int(src.get("tile_x", -1)),
            "tile_y": int(src.get("tile_y", -1)),
        })
    pq.write_table(pa.Table.from_pylist(coarse_index), output / "index.parquet")

    curriculum_id = curriculum_identity(store)
    attrs = {
        "schema": COARSE_STORE_SCHEMA,
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_filter": source,
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": checkpoint_sha,
            "architecture": str(ckpt.get("backbone_config", {}).get("architecture", "?")),
            "epoch": int(ckpt.get("epoch", 0)),
        },
        "curriculum_identity": curriculum_id,
    }
    group.attrs.update(attrs)

    return {
        "schema": COARSE_STORE_SCHEMA,
        "path": str(output.resolve()),
        "rows": len(selected),
        "checkpoint_sha": checkpoint_sha,
        "source_filter": source,
    }


__all__ = [
    "PriorBridgeError",
    "bridge_prior_to_coarse",
]
