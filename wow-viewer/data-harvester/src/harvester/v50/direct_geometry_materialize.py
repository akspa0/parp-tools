"""Spec 114 T058: materialize a frozen coarse checkpoint's outputs into a derived Zarr store.

The residual detailer must train on GENERATED upstream outputs (including their errors), never on
teacher-forced truth. This module runs one frozen coarse geometry checkpoint over the selected
rows of the dual-source curriculum and writes a derived store:

- ``coarse_relief``: float16 (N, 257, 257) predictions in the exact row order of ``index.parquet``;
- ``index.parquet``: source row index, ``source_group_id``, ``split``, ``minimap_source`` — so the
  detailer trainer reuses the identical frozen split with zero re-derivation;
- attrs: schema ``v114-coarse-relief-v1``, checkpoint path + sha256, model variant, source filter,
  and the source curriculum's content identity.

Source stores are NEVER mutated. The derived store is immutable once written (non-empty outputs
are refused); swapping the coarse checkpoint means writing a new derived store, which is what
makes the detailer independently replaceable.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from harvester.v50.contracts import require_store_release, validate_release
from harvester.v50.direct_geometry_infer import (
    InferenceContractError,
    load_geometry_checkpoint,
    predict_relief,
    predict_relief_with_feature_map,
)
from harvester.v50.height_relative_train import (
    SOURCE_CHOICES,
    TrainerContractError,
    curriculum_identity,
    select_training_rows,
    validate_curriculum_contract,
    validate_source_selection,
)
from harvester.v50.model_stage_contract import sha256_file

COARSE_STORE_SCHEMA = "v114-coarse-relief-v1"
COARSE_ARRAY = "coarse_relief"


class MaterializationError(ValueError):
    """Raised when a coarse-output store cannot be materialized as declared."""


def load_selected_rows(store: Path, *, source: str, release: str) -> tuple[dict, list[dict], list[int]]:
    """Validate the source curriculum and return (attrs, index_rows, selected_indices)."""
    import pyarrow.parquet as pq
    import zarr

    group = zarr.open_group(str(store), mode="r")
    try:
        require_store_release(group, release, store=store)
    except ValueError as exc:
        raise MaterializationError(str(exc)) from exc
    index = pq.read_table(store / "index.parquet").to_pylist()
    try:
        array_lengths = {name: int(group[name].shape[0]) for name in group.array_keys()}
        validate_curriculum_contract(
            attrs=dict(group.attrs), array_lengths=array_lengths, index_rows=index
        )
        validate_source_selection(attrs=dict(group.attrs), source=source)
        selected = select_training_rows(index, source)
    except TrainerContractError as exc:
        raise MaterializationError(str(exc)) from exc
    if not selected:
        raise MaterializationError(f"source filter {source!r} selected zero rows")
    return dict(group.attrs), index, selected


def load_feature_store(feature_store: Path, *, selected: list[int]) -> tuple[object, int, dict[int, int]]:
    """Open and validate a Spec 115/117 ``v115-feature-map-v1`` store for materialization.

    Mirrors ``direct_geometry_train.py``'s own ``--feature-store`` validation exactly (same schema
    const, same ``class_count``/``feature_map`` checks, same full-row-coverage requirement) so a
    checkpoint trained with a feature store can also be materialized with the SAME feature store —
    this was a real gap: the trainer accepted ``--feature-store`` (Spec 115/117) but this
    materializer had no way to reproduce the extra input channels, so a feature-augmented
    checkpoint failed at ``load_state_dict`` with a channel-count mismatch.
    """
    import pyarrow.parquet as pq
    import zarr

    feature_group = zarr.open_group(str(feature_store), mode="r")
    feature_attrs = dict(feature_group.attrs)
    if feature_attrs.get("schema") != "v115-feature-map-v1":
        raise MaterializationError(f"--feature-store is not a v115-feature-map-v1 store: {feature_store}")
    class_count = int(feature_attrs.get("class_count", 0))
    if class_count < 1 or "feature_map" not in feature_group:
        raise MaterializationError(f"--feature-store has no usable feature_map array: {feature_store}")
    feature_index = pq.read_table(feature_store / "index.parquet").to_pylist()
    row_to_position = {int(r["source_row_index"]): pos for pos, r in enumerate(feature_index)}
    missing = [i for i in selected if i not in row_to_position]
    if missing:
        raise MaterializationError(
            f"--feature-store is missing {len(missing)} selected curriculum rows "
            f"(e.g. {missing[:5]}); materialize it with the same --source"
        )
    return feature_group, class_count, row_to_position


def build_materialization_plan(
    *,
    store: Path,
    checkpoint_path: Path,
    source: str,
    index: list[dict],
    selected: list[int],
    checkpoint: dict,
    checkpoint_sha: str,
    feature_store: Path | None = None,
    feature_class_count: int = 0,
) -> dict:
    split_counts = {
        split: sum(str(index[i].get("split")) == split for i in selected)
        for split in ("train", "val", "test")
    }
    split_counts = {key: value for key, value in split_counts.items() if value}
    plan = {
        "schema": "v114-coarse-materialize-plan-v1",
        "source_store": str(store),
        "source_filter": source,
        "selected_rows": len(selected),
        "split_counts": split_counts,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha,
            "model_variant": checkpoint.get("model_variant"),
            "epoch": int(checkpoint.get("epoch", 0)),
            "val_mae": float(checkpoint.get("val_mae", float("nan"))),
        },
        "output_array": {
            "name": COARSE_ARRAY,
            "shape": [len(selected), 257, 257],
            "dtype": "float16",
            "estimated_bytes": len(selected) * 257 * 257 * 2,
        },
    }
    if feature_store is not None:
        plan["feature_store"] = {
            "path": str(feature_store),
            "class_count": feature_class_count,
            "input_channels": 3 + feature_class_count,
        }
    return plan


def materialize_coarse_relief(
    *,
    store: Path,
    checkpoint_path: Path,
    output: Path,
    source: str,
    release: str,
    device: str,
    write: bool,
    feature_store: Path | None = None,
) -> dict:
    """Shared CLI/test path: validate, predict, optionally persist the derived store.

    ``feature_store`` is optional and MUST be the same Spec 115/117 generated feature-map store the
    checkpoint was trained with (``--feature-store`` on ``direct_geometry_train.py``) — required
    whenever the checkpoint's architecture has ``in_channels > 3``, since the model literally cannot
    run without those extra channels concatenated onto RGB.
    """
    attrs, index, selected = load_selected_rows(store, source=source, release=release)
    feature_group = None
    feature_class_count = 0
    feature_row_to_position: dict[int, int] = {}
    if feature_store is not None:
        feature_group, feature_class_count, feature_row_to_position = load_feature_store(
            feature_store, selected=selected
        )
    try:
        model, checkpoint, _identity = load_geometry_checkpoint(
            checkpoint_path, device=device, in_channels=3 + feature_class_count
        )
    except InferenceContractError as exc:
        raise MaterializationError(str(exc)) from exc
    checkpoint_sha = sha256_file(checkpoint_path)
    plan = build_materialization_plan(
        store=store,
        checkpoint_path=checkpoint_path,
        source=source,
        index=index,
        selected=selected,
        checkpoint=checkpoint,
        checkpoint_sha=checkpoint_sha,
        feature_store=feature_store,
        feature_class_count=feature_class_count,
    )
    if not write:
        return plan
    if output.exists() and any(output.iterdir()):
        raise MaterializationError(
            f"refusing to overwrite non-empty coarse store {output}; choose a new path"
        )

    import pyarrow as pa
    import pyarrow.parquet as pq
    import zarr

    output.mkdir(parents=True, exist_ok=False)
    group = zarr.open_group(str(output), mode="w")
    array = group.create_array(
        COARSE_ARRAY,
        shape=(len(selected), 257, 257),
        chunks=(1, 257, 257),
        dtype=np.float16,
    )

    source_group = zarr.open_group(str(store), mode="r")
    for position, row_index in enumerate(selected):
        rgb = np.asarray(source_group["minimap_rgb"][row_index], dtype=np.uint8)
        if feature_group is not None:
            feats = np.asarray(
                feature_group["feature_map"][feature_row_to_position[row_index]], dtype=np.float32
            )
            relief = predict_relief_with_feature_map(model, rgb, feats, device=device)
        else:
            relief = predict_relief(model, rgb, device=device)
        array[position] = relief.astype(np.float16)

    index_rows = [
        {
            "source_row_index": int(row_index),
            "source_group_id": str(index[row_index]["source_group_id"]),
            "split": str(index[row_index]["split"]),
            "minimap_source": str(index[row_index]["minimap_source"]),
        }
        for row_index in selected
    ]
    pq.write_table(pa.Table.from_pylist(index_rows), output / "index.parquet")

    summary = {
        **plan,
        "schema": COARSE_STORE_SCHEMA,
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "curriculum_identity": curriculum_identity(store),
    }
    group.attrs.update(
        {
            "schema": COARSE_STORE_SCHEMA,
            "created_utc": summary["created_utc"],
            "source_store": str(store),
            "source_filter": source,
            "curriculum_identity": summary["curriculum_identity"],
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
            "model_variant": checkpoint.get("model_variant"),
            "checkpoint_epoch": int(checkpoint.get("epoch", 0)),
            "row_count": len(selected),
            "feature_store": plan.get("feature_store"),
        }
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Spec 114 coarse-relief materializer (dry run by default)"
    )
    ap.add_argument("--store", required=True, type=Path, help="dual-source curriculum store")
    ap.add_argument("--checkpoint", required=True, type=Path, help="frozen coarse checkpoint_best.pt")
    ap.add_argument("--output", required=True, type=Path, help="new derived coarse store path")
    ap.add_argument("--source", required=True, choices=sorted(SOURCE_CHOICES))
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--release", default="v50.1", type=validate_release)
    ap.add_argument("--feature-store", type=Path, default=None,
                    help="REQUIRED if the checkpoint was trained with --feature-store (Spec "
                         "115/117): the exact same v115-feature-map-v1 store, so materialization "
                         "can reconstruct the checkpoint's real in_channels and feed it the same "
                         "generated channels it trained on. Omit for an RGB-only checkpoint.")
    ap.add_argument("--write", action="store_true",
                    help="persist the derived store; default prints the validated plan only")
    args = ap.parse_args(argv)

    if args.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but unavailable; use --device cpu.")
    try:
        summary = materialize_coarse_relief(
            store=args.store,
            checkpoint_path=args.checkpoint,
            feature_store=args.feature_store,
            output=args.output,
            source=args.source,
            release=args.release,
            device=args.device,
            write=args.write,
        )
    except MaterializationError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(summary, indent=2), flush=True)
    if not args.write:
        print("DRY RUN ONLY: add --write to persist the derived coarse store.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
