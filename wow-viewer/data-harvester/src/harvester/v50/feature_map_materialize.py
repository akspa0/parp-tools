"""Spec 115 US2: materialize a frozen terrain-feature classifier's GENERATED output into a store.

The deconfounded geometry model must train on the classifier's *generated* feature map (errors and
all), never on ground-truth labels -- exactly the FR-006 rule the coarse-relief materializer already
enforces for the detailer. This module runs one frozen classifier checkpoint over the selected
curriculum rows and writes a derived store:

- ``feature_map``: float16 (N, K, 256, 256) class probabilities in the exact row order of
  ``index.parquet``. Stored at the classifier's native 256 resolution, matching the 256x256 RGB the
  geometry model concatenates it onto at the input -- no resampling in the training loop.
- ``index.parquet``: source row index, ``source_group_id``, ``split``, ``minimap_source`` -- so the
  geometry trainer reuses the identical frozen split with zero re-derivation.
- attrs: schema, classifier checkpoint path + sha256, taxonomy revision, source curriculum identity.

Source stores are NEVER mutated; the derived store is immutable once written. Swapping the classifier
checkpoint means writing a new store, which is what keeps the classifier independently replaceable.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from harvester.v50.contracts import require_store_release, validate_release
from harvester.v50.direct_geometry_infer import InferenceContractError
from harvester.v50.height_relative_train import (
    SOURCE_CHOICES,
    TrainerContractError,
    curriculum_identity,
    select_training_rows,
    validate_curriculum_contract,
    validate_source_selection,
)
from harvester.v50.model_stage_contract import sha256_file
from harvester.v50.terrain_feature_infer import load_terrain_feature_checkpoint, predict_feature_map
from harvester.v50.terrain_feature_labels import CLASS_COUNT, FAMILY_NAMES, TAXONOMY_REVISION

FEATURE_STORE_SCHEMA = "v115-feature-map-v1"
FEATURE_ARRAY = "feature_map"
FEATURE_SIZE = 256


class FeatureMaterializationError(ValueError):
    """Raised when a feature-map store cannot be materialized as declared."""


def load_selected_rows(store: Path, *, source: str, release: str) -> tuple[dict, list[dict], list[int]]:
    """Validate the source curriculum and return (attrs, index_rows, selected_indices)."""
    import pyarrow.parquet as pq
    import zarr

    group = zarr.open_group(str(store), mode="r")
    try:
        require_store_release(group, release, store=store)
    except ValueError as exc:
        raise FeatureMaterializationError(str(exc)) from exc
    index = pq.read_table(store / "index.parquet").to_pylist()
    try:
        array_lengths = {name: int(group[name].shape[0]) for name in group.array_keys()}
        validate_curriculum_contract(
            attrs=dict(group.attrs), array_lengths=array_lengths, index_rows=index
        )
        validate_source_selection(attrs=dict(group.attrs), source=source)
        selected = select_training_rows(index, source)
    except TrainerContractError as exc:
        raise FeatureMaterializationError(str(exc)) from exc
    if not selected:
        raise FeatureMaterializationError(f"source filter {source!r} selected zero rows")
    return dict(group.attrs), index, selected


def build_plan(
    *,
    store: Path,
    checkpoint_path: Path,
    source: str,
    index: list[dict],
    selected: list[int],
    checkpoint: dict,
    checkpoint_sha: str,
) -> dict:
    split_counts = {
        split: sum(str(index[i].get("split")) == split for i in selected)
        for split in ("train", "val", "test")
    }
    split_counts = {key: value for key, value in split_counts.items() if value}
    return {
        "schema": "v115-feature-materialize-plan-v1",
        "source_store": str(store),
        "source_filter": source,
        "selected_rows": len(selected),
        "split_counts": split_counts,
        "families": list(FAMILY_NAMES),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha,
            "model_variant": checkpoint.get("model_variant"),
            "taxonomy_revision": checkpoint.get("taxonomy_revision"),
            "epoch": int(checkpoint.get("epoch", 0)),
        },
        "output_array": {
            "name": FEATURE_ARRAY,
            "shape": [len(selected), CLASS_COUNT, FEATURE_SIZE, FEATURE_SIZE],
            "dtype": "float16",
            "estimated_bytes": len(selected) * CLASS_COUNT * FEATURE_SIZE * FEATURE_SIZE * 2,
        },
    }


def materialize_feature_maps(
    *,
    store: Path,
    checkpoint_path: Path,
    output: Path,
    source: str,
    release: str,
    device: str,
    write: bool,
) -> dict:
    """Shared CLI/test path: validate, predict, optionally persist the derived feature store."""
    attrs, index, selected = load_selected_rows(store, source=source, release=release)
    try:
        model, checkpoint, _identity = load_terrain_feature_checkpoint(checkpoint_path, device=device)
    except InferenceContractError as exc:
        raise FeatureMaterializationError(str(exc)) from exc
    if checkpoint.get("taxonomy_revision") != TAXONOMY_REVISION:
        raise FeatureMaterializationError(
            f"classifier taxonomy {checkpoint.get('taxonomy_revision')!r} != code {TAXONOMY_REVISION!r}"
        )
    checkpoint_sha = sha256_file(checkpoint_path)
    plan = build_plan(
        store=store,
        checkpoint_path=checkpoint_path,
        source=source,
        index=index,
        selected=selected,
        checkpoint=checkpoint,
        checkpoint_sha=checkpoint_sha,
    )
    if not write:
        return plan
    if output.exists() and any(output.iterdir()):
        raise FeatureMaterializationError(
            f"refusing to overwrite non-empty feature store {output}; choose a new path"
        )

    import pyarrow as pa
    import pyarrow.parquet as pq
    import zarr

    output.mkdir(parents=True, exist_ok=False)
    group = zarr.open_group(str(output), mode="w")
    array = group.create_array(
        FEATURE_ARRAY,
        shape=(len(selected), CLASS_COUNT, FEATURE_SIZE, FEATURE_SIZE),
        chunks=(1, CLASS_COUNT, FEATURE_SIZE, FEATURE_SIZE),
        dtype=np.float16,
    )

    source_group = zarr.open_group(str(store), mode="r")
    for position, row_index in enumerate(selected):
        rgb = np.asarray(source_group["minimap_rgb"][row_index], dtype=np.uint8)
        # Native (K, 256, 256), the classifier's own output resolution — stored as-is so the
        # geometry trainer concatenates it onto the 256x256 RGB with no resampling per epoch.
        array[position] = predict_feature_map(model, rgb, device=device).astype(np.float16)

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
        "schema": FEATURE_STORE_SCHEMA,
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "curriculum_identity": curriculum_identity(store),
    }
    group.attrs.update(
        {
            "schema": FEATURE_STORE_SCHEMA,
            "created_utc": summary["created_utc"],
            "source_store": str(store),
            "source_filter": source,
            "curriculum_identity": summary["curriculum_identity"],
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
            "model_variant": checkpoint.get("model_variant"),
            "taxonomy_revision": TAXONOMY_REVISION,
            "class_count": CLASS_COUNT,
            "family_names": list(FAMILY_NAMES),
            "row_count": len(selected),
        }
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Spec 115 terrain-feature-map materializer (dry run by default)"
    )
    ap.add_argument("--store", required=True, type=Path, help="dual-source curriculum store")
    ap.add_argument("--checkpoint", required=True, type=Path, help="frozen classifier checkpoint_best.pt")
    ap.add_argument("--output", required=True, type=Path, help="new derived feature store path")
    ap.add_argument("--source", required=True, choices=sorted(SOURCE_CHOICES))
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--release", default="v50.1", type=validate_release)
    ap.add_argument("--write", action="store_true",
                    help="persist the derived store; default prints the validated plan only")
    args = ap.parse_args(argv)

    if args.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but unavailable; use --device cpu.")
    try:
        summary = materialize_feature_maps(
            store=args.store,
            checkpoint_path=args.checkpoint,
            output=args.output,
            source=args.source,
            release=args.release,
            device=args.device,
            write=args.write,
        )
    except FeatureMaterializationError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(summary, indent=2), flush=True)
    if not args.write:
        print("DRY RUN ONLY: add --write to persist the derived feature store.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
