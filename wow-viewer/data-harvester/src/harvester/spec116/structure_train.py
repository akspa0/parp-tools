"""Spec 116 US3: per-slot structure classifier trainer (dry run by default; USER runs the CUDA pass).

Each detail slot (1-3) is an independently trained, independently promoted model (constitution
IV / D-04). This trainer builds one ``StructureSlotNet`` for a single slot, trains it on the
US4 held-out split, and gates on **per-class IoU/recall** (D-08) — never aggregate accuracy.

The training problem mirrors Spec 115's terrain-feature classifier: the label distribution is
heavily imbalanced (terrain dominates, road/water/structure are rare), so:

- the loss is class-weighted (capped inverse-frequency, computed in-run from the real labels);
- the promotion metric is **per-class IoU/recall**, with the rarest class as the key signal;
- the in-run baseline is the majority class, making the degenerate "predict terrain everywhere"
  solution explicit and impossible to mistake for success.

**Dry-run-first (FR-015)**: without ``--confirm-run`` the trainer prints a plan (architecture,
split counts, class weights, time/memory estimate) and exits. The user owns every CUDA run
(FR-018). A ``--device cpu`` path exists for tiny fixture validation in tests only.

**Preconditions**: the held-out split must have ``verified_violation_count == 0`` (FR-010) and
the US1 vocabulary decision must be available (FR-002). Missing either is a hard refusal.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from harvester.spec116.held_out_split import load_split
from harvester.spec116.relational_extract import CHUNKS_PER_AXIS
from harvester.spec116.relief_stratification import (
    RELIEF_STD_THRESHOLD,
    chunk_strata,
    stratified_mae,
    tile_mean_baseline_mae,
)
from harvester.spec116.structure_contract import sha256_file, validate_structure_run
from harvester.spec116.structure_model import build_structure_model
from harvester.v50.terrain_feature_labels import (
    CLASS_COUNT,
    FAMILY_NAMES,
    TAXONOMY_REVISION,
    classify_texture_name,
    load_texture_name_dump,
    rule_set_sha256,
)

STRUCTURE_RUN_SCHEMA = "v50-structure-run-v1"
MAX_CLASS_WEIGHT = 15.0
GEOMETRY_RESCORE_SCHEMA = "v116-geometry-rescore-v1"


class StructureTrainError(ValueError):
    """Raised when the structure trainer contract is violated."""


# --------------------------------------------------------------------------- #
# Class weights (capped inverse-frequency, same pattern as Spec 115).          #
# --------------------------------------------------------------------------- #
def compute_class_weights(
    label_counts: dict[str, int],
    *,
    max_weight: float = MAX_CLASS_WEIGHT,
) -> list[float]:
    """Capped inverse-frequency weights, in family-ordinal order.

    The ``unknown`` class (ordinal 0) is given weight 0 and masked out of the loss: it means
    "no rule matched", not a terrain family. An absent class gets neutral weight (1.0), never
    infinite.
    """
    total = sum(label_counts.get(name, 0) for name in FAMILY_NAMES)
    if total <= 0:
        raise StructureTrainError("label store reports zero labelled chunks")
    weights: list[float] = []
    for index, name in enumerate(FAMILY_NAMES):
        if index == 0:  # UNKNOWN
            weights.append(0.0)
            continue
        count = label_counts.get(name, 0)
        if count <= 0:
            weights.append(1.0)
            continue
        weights.append(min(total / (CLASS_COUNT * count), max_weight))
    return weights


# --------------------------------------------------------------------------- #
# Confusion metrics (per-class IoU/recall; aggregate accuracy reported only).  #
# --------------------------------------------------------------------------- #
def confusion_metrics(confusion: np.ndarray) -> dict:
    """Per-class IoU/recall plus macro IoU and aggregate accuracy (reported only, never gated)."""
    per_class: dict[str, dict[str, float]] = {}
    ious: list[float] = []
    recalls: list[float] = []
    for family in range(CLASS_COUNT):
        true_positive = float(confusion[family, family])
        predicted = float(confusion[:, family].sum())
        actual = float(confusion[family, :].sum())
        union = predicted + actual - true_positive
        iou = true_positive / union if union > 0 else 0.0
        recall = true_positive / actual if actual > 0 else 0.0
        per_class[FAMILY_NAMES[family]] = {"iou": iou, "recall": recall}
        if actual > 0:
            ious.append(iou)
            recalls.append(recall)
    total = float(confusion.sum())
    return {
        "per_class": per_class,
        "macro_iou": float(np.mean(ious)) if ious else 0.0,
        "aggregate_accuracy_reported_only": float(np.trace(confusion) / total) if total > 0 else 0.0,
    }


def _rarest_class_name(label_counts: dict[str, int]) -> str:
    """The rarest non-unknown class by chunk count (the key gate metric, D-08)."""
    candidates = {
        name: label_counts.get(name, 0)
        for index, name in enumerate(FAMILY_NAMES)
        if index > 0 and label_counts.get(name, 0) > 0
    }
    if not candidates:
        raise StructureTrainError("no non-unknown classes have any labelled chunks")
    return min(candidates, key=candidates.get)


# --------------------------------------------------------------------------- #
# Per-slot label construction from the v50 store.                              #
# --------------------------------------------------------------------------- #
def build_slot_labels(
    *,
    store: Path,
    dumps: list[Path],
    slot: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Build per-tile (16, 16) family label arrays for one detail slot.

    Returns ``(labels, tile_mask, label_counts)`` where:
    - ``labels`` is (N, 16, 16) int64; -1 means masked (absent slot or no dump).
    - ``tile_mask`` is (N,) bool; False means the tile has no dump and is excluded.
    - ``label_counts`` is family-name -> chunk count (excluding masked and unknown).

    The label for chunk (cy, cx) at slot S is the family of ``mcly_texture_ids[cy, cx, S]``
    resolved through the tile's own texture-name dump. Absent slots (``< 0``) are masked, never
    coerced into a class.
    """
    import pyarrow.parquet as pq
    import zarr

    group = zarr.open_group(str(store), mode="r")
    for required in ("mcly_texture_ids", "minimap_rgb"):
        if required not in group:
            raise StructureTrainError(f"store is missing {required!r}: {store}")

    index_path = store / "index.parquet"
    if not index_path.exists():
        raise StructureTrainError(f"store has no index.parquet: {store}")
    index_rows = pq.read_table(index_path).to_pylist()

    ids_array = group["mcly_texture_ids"]
    row_count = int(ids_array.shape[0])
    if row_count != len(index_rows):
        raise StructureTrainError(
            f"index rows ({len(index_rows)}) != mcly_texture_ids rows ({row_count})"
        )

    names_by_tile = load_texture_name_dump(dumps)

    labels = np.full((row_count, CHUNKS_PER_AXIS, CHUNKS_PER_AXIS), -1, dtype=np.int64)
    tile_mask = np.zeros(row_count, dtype=bool)
    label_counts: dict[str, int] = dict.fromkeys(FAMILY_NAMES, 0)

    for row in range(row_count):
        meta = index_rows[row]
        key = (str(meta.get("map")), int(meta.get("tile_x", -1)), int(meta.get("tile_y", -1)))
        texture_names = names_by_tile.get(key)
        if not texture_names:
            continue  # excluded: no dump
        tile_mask[row] = True
        tile_ids = np.asarray(ids_array[row], dtype=np.int32)

        for cy in range(CHUNKS_PER_AXIS):
            for cx in range(CHUNKS_PER_AXIS):
                local_id = int(tile_ids[cy, cx, slot])
                if local_id < 0:
                    continue  # absent slot: masked
                if 0 <= local_id < len(texture_names):
                    family = classify_texture_name(texture_names[local_id])
                else:
                    family = 0  # UNKNOWN: out-of-range local index
                labels[row, cy, cx] = family
                if family > 0:
                    label_counts[FAMILY_NAMES[family]] += 1

    return labels, tile_mask, label_counts


# --------------------------------------------------------------------------- #
# Training plan (dry-run output).                                             #
# --------------------------------------------------------------------------- #
def build_training_plan(
    *,
    slot: int,
    architecture: dict,
    vocabulary_decision: str,
    split_counts: dict,
    label_counts: dict[str, int],
    class_weights: list[float],
    batch_size: int,
    epochs: int,
    lr: float,
    max_class_weight: float,
    device: str,
) -> dict:
    train_steps = math.ceil(max(split_counts.get("train", 0), 1) / batch_size)
    # Rough time/memory estimate for the dry-run plan.
    param_count = architecture.get("param_count", 0)
    mem_mb = param_count * 4 * 4 / 1e6  # params + grads + Adam states + AMP buffer
    time_min = train_steps * epochs * 0.05  # ~3s/step on CUDA for base=32
    return {
        "schema": "v116-structure-plan-v1",
        "slot": slot,
        "vocabulary_decision": vocabulary_decision,
        "architecture": architecture,
        "split_counts": split_counts,
        "label_chunk_counts": label_counts,
        "class_weights": {FAMILY_NAMES[i]: w for i, w in enumerate(class_weights)},
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": lr,
        "max_class_weight": max_class_weight,
        "device": device,
        "train_steps_per_epoch": train_steps,
        "estimated_time_minutes": round(time_min, 1),
        "estimated_memory_mb": round(mem_mb, 1),
        "deployment_inputs": ["minimap_rgb"],
        "promotion_metric": "per_class_iou_recall (NOT aggregate accuracy: D-08)",
        "rarest_class": _rarest_class_name(label_counts),
    }


# --------------------------------------------------------------------------- #
# Core training loop (only runs with --confirm-run).                          #
# --------------------------------------------------------------------------- #
def train_structure(
    *,
    store: Path,
    split_dir: Path,
    dumps: list[Path],
    vocabulary_report: Path,
    output: Path,
    slot: int,
    base: int = 32,
    epochs: int = 60,
    batch_size: int = 16,
    lr: float = 3e-4,
    max_class_weight: float = MAX_CLASS_WEIGHT,
    device: str = "cuda",
    seed: int = 116,
    patience: int = 12,
) -> dict:
    """Run the per-slot structure training and write the ``v50-structure-run-v1`` record.

    This is the user-run CUDA path. Tests call it with ``device="cpu"`` on a tiny fixture.
    """
    import torch
    import zarr
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    # --- Preconditions: held-out split must be clean, vocabulary must exist. --- #
    split_manifest, split_rows = load_split(split_dir)
    if split_manifest["verified_violation_count"] != 0:
        raise StructureTrainError(
            f"held-out split has {split_manifest['verified_violation_count']} violations; "
            "refusing to train on a leaky split (FR-010)"
        )

    vocab_report = json.loads(vocabulary_report.read_text(encoding="utf-8"))
    from harvester.spec116.family_slot_consistency import recommendation_from_report
    vocabulary_decision = recommendation_from_report(vocab_report)

    # --- Build labels for this slot. --- #
    labels, tile_mask, label_counts = build_slot_labels(
        store=store, dumps=dumps, slot=slot
    )
    if slot < 1 or slot > 3:
        raise StructureTrainError(f"slot must be 1-3, got {slot}")

    # --- Partition by held-out split. --- #
    # Map (map, tile_x, tile_y) -> split label from the split parquet.
    split_map: dict[tuple[str, int, int], str] = {}
    for row in split_rows:
        key = (str(row["map"]), int(row["tile_x"]), int(row["tile_y"]))
        split_map[key] = str(row["split"])

    import pyarrow.parquet as pq
    index_rows = pq.read_table(store / "index.parquet").to_pylist()

    train_rows: list[int] = []
    held_out_rows: list[int] = []
    for row_idx, meta in enumerate(index_rows):
        if not tile_mask[row_idx]:
            continue  # excluded: no dump
        key = (str(meta.get("map")), int(meta.get("tile_x", -1)), int(meta.get("tile_y", -1)))
        split_label = split_map.get(key, "excluded")
        if split_label == "train":
            train_rows.append(row_idx)
        elif split_label == "held_out":
            held_out_rows.append(row_idx)

    if len(train_rows) < 2 or len(held_out_rows) < 1:
        raise StructureTrainError(
            f"insufficient rows after split: train={len(train_rows)} held_out={len(held_out_rows)}"
        )

    # --- Class weights. --- #
    class_weights = compute_class_weights(label_counts, max_weight=max_class_weight)
    rarest_class = _rarest_class_name(label_counts)

    # --- Model. --- #
    model, model_identity = build_structure_model(slot=slot, base=base)
    architecture = model_identity["architecture"]

    build_training_plan(
        slot=slot,
        architecture=architecture,
        vocabulary_decision=vocabulary_decision,
        split_counts={"train": len(train_rows), "held_out": len(held_out_rows)},
        label_counts=label_counts,
        class_weights=class_weights,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        max_class_weight=max_class_weight,
        device=device,
    )

    # --- Training. --- #
    group = zarr.open_group(str(store), mode="r")
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    class SlotDataset(Dataset):
        def __init__(self, rows: list[int]) -> None:
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, i: int):
            row = self.rows[i]
            rgb = np.asarray(group["minimap_rgb"][row], dtype=np.float32) / 255.0
            label = labels[row].copy()
            # Mask: -1 (absent slot) and unknown (0) are both ignored.
            label = np.where(label >= 0, label, -1)
            label = np.where(label > 0, label, -1)  # mask unknown
            return (
                torch.from_numpy(rgb).permute(2, 0, 1),
                torch.from_numpy(label),
            )

    dev = torch.device(device)
    model = model.to(dev)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=dev)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor, ignore_index=-1)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    train_loader = DataLoader(
        SlotDataset(train_rows), batch_size=batch_size, shuffle=True, drop_last=False,
    )
    val_loader = DataLoader(
        SlotDataset(held_out_rows), batch_size=batch_size, shuffle=False,
    )

    # Majority-class baseline: predict the most common non-unknown class everywhere.
    majority_family = max(
        (f for f in range(1, CLASS_COUNT) if label_counts.get(FAMILY_NAMES[f], 0) > 0),
        key=lambda f: label_counts.get(FAMILY_NAMES[f], 0),
    )
    majority_confusion = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64)
    for row in held_out_rows:
        label = labels[row]
        keep = (label >= 0) & (label > 0)
        counts = np.bincount(label[keep], minlength=CLASS_COUNT)
        majority_confusion[:, majority_family] += counts
    baseline_metrics = confusion_metrics(majority_confusion)

    output.mkdir(parents=True, exist_ok=True)
    best_macro_iou = -1.0
    best_epoch = 0
    stale = 0
    best_metrics: dict | None = None

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        steps = 0
        for rgb, label in train_loader:
            rgb = rgb.to(dev, non_blocking=True)
            label = label.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(rgb)
            loss = criterion(logits, label)
            loss.backward()
            opt.step()
            running += float(loss.detach())
            steps += 1

        model.eval()
        confusion = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64)
        with torch.no_grad():
            for rgb, label in val_loader:
                rgb = rgb.to(dev, non_blocking=True)
                predicted = model(rgb).argmax(dim=1).cpu().numpy().ravel()
                actual = label.numpy().ravel()
                keep = actual >= 0
                np.add.at(confusion, (actual[keep], predicted[keep]), 1)

        metrics = confusion_metrics(confusion)
        macro_iou = metrics["macro_iou"]
        print(
            f"epoch {epoch:3d} loss={running / max(steps, 1):.4f} "
            f"macro_iou={macro_iou:.4f} acc={metrics['aggregate_accuracy_reported_only']:.4f}",
            flush=True,
        )

        if macro_iou > best_macro_iou:
            best_macro_iou = macro_iou
            best_epoch = epoch
            stale = 0
            best_metrics = metrics
            torch.save(
                {
                    "model": model.state_dict(),
                    "slot": slot,
                    "base": base,
                    "taxonomy_revision": TAXONOMY_REVISION,
                    "num_classes": CLASS_COUNT,
                    "epoch": epoch,
                    "macro_iou": macro_iou,
                    "metrics": metrics,
                },
                output / "checkpoint_best.pt",
            )
        else:
            stale += 1
            if stale >= patience:
                print(f"early stop at epoch {epoch} (patience {patience})", flush=True)
                break

    assert best_metrics is not None

    # --- Build the v50-structure-run-v1 record. --- #
    checkpoint_path = output / "checkpoint_best.pt"
    run_record = _build_run_record(
        slot=slot,
        vocabulary_decision=vocabulary_decision,
        architecture=architecture,
        store=store,
        split_dir=split_dir,
        split_manifest=split_manifest,
        dumps=dumps,
        config={
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "max_class_weight": max_class_weight,
            "device": device,
        },
        split_counts={"train": len(train_rows), "held_out": len(held_out_rows)},
        majority_family=majority_family,
        baseline_metrics=baseline_metrics,
        best_epoch=best_epoch,
        best_metrics=best_metrics,
        checkpoint_path=checkpoint_path,
        rarest_class=rarest_class,
    )
    validate_structure_run(run_record)
    (output / "structure_run.json").write_text(
        json.dumps(run_record, indent=2), encoding="utf-8"
    )
    return run_record


def _build_run_record(
    *,
    slot: int,
    vocabulary_decision: str,
    architecture: dict,
    store: Path,
    split_dir: Path,
    split_manifest: dict,
    dumps: list[Path],
    config: dict,
    split_counts: dict,
    majority_family: int,
    baseline_metrics: dict,
    best_epoch: int,
    best_metrics: dict,
    checkpoint_path: Path,
    rarest_class: str,
) -> dict:
    """Assemble the ``v50-structure-run-v1`` record (validated by structure_contract)."""
    per_class = best_metrics["per_class"]
    rarest_iou = per_class[rarest_class]["iou"]
    rarest_recall = per_class[rarest_class]["recall"]

    # SC-003: the model beats the majority-class baseline on per-class IoU/recall.
    baseline_rarest_iou = baseline_metrics["per_class"][rarest_class]["iou"]
    sc003 = bool(rarest_iou > baseline_rarest_iou)

    return {
        "schema": STRUCTURE_RUN_SCHEMA,
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "feature": "116-relational-terrain-layers",
        "slot": slot,
        "vocabulary_decision": vocabulary_decision,
        "identity": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
        },
        "inputs": {
            "store": {
                "path": str(store.resolve()),
                "sha256": sha256_file(store / "index.parquet"),
            },
            "held_out_split": {
                "path": str(split_dir / "split.json"),
                "sha256": sha256_file(split_dir / "split.json"),
                "verified_violation_count": int(split_manifest["verified_violation_count"]),
            },
            "texture_name_dumps": [
                {"path": str(d.resolve()), "sha256": sha256_file(d)} for d in dumps
            ],
            "taxonomy_revision": TAXONOMY_REVISION,
            "rule_set_sha256": rule_set_sha256(),
        },
        "architecture": {
            "class": architecture["class"],
            "base": architecture["base"],
            "slot": architecture["slot"],
            "num_classes": architecture["num_classes"],
            "param_count": architecture["param_count"],
        },
        "config": config,
        "split_counts": split_counts,
        "baselines": {
            "majority_class": {
                "family": FAMILY_NAMES[majority_family],
                "per_class_iou": {
                    name: baseline_metrics["per_class"][name]["iou"]
                    for name in FAMILY_NAMES
                },
                "per_class_recall": {
                    name: baseline_metrics["per_class"][name]["recall"]
                    for name in FAMILY_NAMES
                },
            }
        },
        "best_epoch": best_epoch,
        "metrics": {
            "per_class": per_class,
            "macro_iou": best_metrics["macro_iou"],
            "rarest_class_iou": rarest_iou,
            "rarest_class_recall": rarest_recall,
            "aggregate_accuracy_reported_only": best_metrics["aggregate_accuracy_reported_only"],
        },
        "promotion_verdict": "pending",
        "gate": {
            "rule": "per_class_iou_recall",
            "rarest_class": rarest_class,
            "sc003": sc003,
        },
    }


# --------------------------------------------------------------------------- #
# US4 T019: relief-stratified re-score of an EXISTING Spec 114 geometry        #
# checkpoint. Read-only evaluation -- no training, no gradient step.           #
# --------------------------------------------------------------------------- #
def rescore_geometry_checkpoint(
    *,
    checkpoint_path: Path,
    store: Path,
    split_dir: Path,
    relief_threshold: float = RELIEF_STD_THRESHOLD,
    in_channels: int | None = None,
    feature_store: Path | None = None,
    source: str = "all",
    device: str = "cpu",
) -> dict:
    """Re-score an existing Spec 114/115 geometry checkpoint on the US4 held-out split.

    Reports flat vs relief-bearing MAE plus the trivial (tile-mean) baseline per stratum
    (FR-011/SC-006), and whether the checkpoint beats the trivial baseline on relief-bearing
    regions (SC-007-shaped honesty bar). Never trains; never writes to the checkpoint or store.

    Checkpoints trained with a Spec 115 ``--feature-store`` (deconfounded, 8 input channels: RGB
    + 5-class terrain-feature probabilities) need that same generated feature map reconstructed
    at rescore time -- pass ``feature_store`` pointing at the same ``v115-feature-map-v1`` store
    used for training. ``in_channels`` is then auto-derived (``3 + class_count``) rather than
    left for the caller to compute by hand; an explicit ``in_channels`` overrides that.

    ``source`` mirrors ``direct_geometry_train.py``'s own ``--source`` ({"all", "authored",
    "synthetic"}): a feature-map store only covers the row domain it was materialized from (e.g.
    Spec 115's authored-only feature map does not cover the dual curriculum's synthetic rows), so
    the held-out rows must be filtered to the same domain the checkpoint was actually trained and
    the feature store was actually built on -- rather than failing deep in the per-row loop with a
    "missing rows" error whose cause isn't obvious from the message alone.
    """
    import pyarrow.parquet as pq
    import torch
    import zarr

    from harvester.v50.direct_geometry_model import build_geometry_model
    from harvester.v50.height_relative_model import decode_relative_height, encode_relative_height
    from harvester.v50.height_relative_train import SOURCE_CHOICES

    if source not in SOURCE_CHOICES:
        raise StructureTrainError(f"source must be one of {sorted(SOURCE_CHOICES)}, got {source!r}")

    split_manifest, split_rows = load_split(split_dir)
    if split_manifest["verified_violation_count"] != 0:
        raise StructureTrainError(
            f"held-out split has {split_manifest['verified_violation_count']} violations; "
            "refusing to rescore on a leaky split (FR-010)"
        )

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    architecture = ckpt.get("model_variant")
    if not architecture:
        raise StructureTrainError(f"checkpoint has no 'model_variant': {checkpoint_path}")

    feature_group = None
    feature_row_to_position: dict[int, int] = {}
    feature_class_count = 0
    if feature_store is not None:
        feature_group = zarr.open_group(str(feature_store), mode="r")
        feature_attrs = dict(feature_group.attrs)
        if feature_attrs.get("schema") != "v115-feature-map-v1":
            raise StructureTrainError(
                f"--feature-store is not a v115-feature-map-v1 store: {feature_store}"
            )
        feature_class_count = int(feature_attrs.get("class_count", 0))
        if feature_class_count < 1 or "feature_map" not in feature_group:
            raise StructureTrainError(f"--feature-store has no usable feature_map array: {feature_store}")
        feature_index = pq.read_table(feature_store / "index.parquet").to_pylist()
        feature_row_to_position = {
            int(r["source_row_index"]): pos for pos, r in enumerate(feature_index)
        }

    resolved_in_channels = in_channels if in_channels is not None else (3 + feature_class_count)

    model, _ = build_geometry_model(architecture, in_channels=resolved_in_channels)
    model.load_state_dict(ckpt["model"])
    dev = torch.device(device)
    model = model.to(dev)
    model.eval()

    group = zarr.open_group(str(store), mode="r")
    for required in ("minimap_rgb", "height_257"):
        if required not in group:
            raise StructureTrainError(f"store is missing {required!r}: {store}")
    index_rows = pq.read_table(store / "index.parquet").to_pylist()

    split_map: dict[tuple[str, int, int], str] = {}
    for row in split_rows:
        key = (str(row["map"]), int(row["tile_x"]), int(row["tile_y"]))
        split_map[key] = str(row["split"])

    held_out_rows: list[int] = []
    for row_idx, meta in enumerate(index_rows):
        key = (str(meta.get("map")), int(meta.get("tile_x", -1)), int(meta.get("tile_y", -1)))
        if split_map.get(key, "excluded") != "held_out":
            continue
        if source != "all" and str(meta.get("minimap_source")) != source:
            continue
        held_out_rows.append(row_idx)
    if not held_out_rows:
        raise StructureTrainError(
            f"held-out split has zero rows present in this store for source={source!r}"
        )

    if feature_group is not None:
        missing = [row for row in held_out_rows if row not in feature_row_to_position]
        if missing:
            raise StructureTrainError(
                f"--feature-store is missing {len(missing)} of {len(held_out_rows)} held-out rows "
                f"(first missing source_row_index={missing[0]}) -- if this checkpoint was trained "
                f"with --source authored, pass --source authored here too so held-out rows are "
                f"restricted to the same domain the feature store actually covers"
            )

    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    strata: list[np.ndarray] = []

    with torch.no_grad():
        for row in held_out_rows:
            rgb = np.asarray(group["minimap_rgb"][row], dtype=np.float32) / 255.0
            channels = torch.from_numpy(rgb).permute(2, 0, 1)
            if feature_group is not None:
                position = feature_row_to_position[row]
                feats = np.asarray(feature_group["feature_map"][position], dtype=np.float32)
                channels = torch.cat([channels, torch.from_numpy(feats)], dim=0)
            x = channels.unsqueeze(0).to(dev)
            height = np.asarray(group["height_257"][row], dtype=np.float32)
            _, tile_min, tile_max = encode_relative_height(height)
            predicted_norm = model(x).squeeze(0).cpu().numpy()
            predicted_height = decode_relative_height(predicted_norm, tile_min, tile_max)
            predictions.append(predicted_height)
            targets.append(height)
            strata.append(chunk_strata(height, threshold=relief_threshold))

    stratified = stratified_mae(predictions, targets, strata)
    trivial_baseline = tile_mean_baseline_mae(targets)
    relief_mae = stratified["relief"]["mae"]
    sc007_beats_trivial_on_relief = relief_mae is not None and relief_mae < trivial_baseline

    return {
        "schema": GEOMETRY_RESCORE_SCHEMA,
        "source": source,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "architecture": architecture,
            "in_channels": resolved_in_channels,
        },
        "feature_store": (
            {"path": str(feature_store.resolve()), "class_count": feature_class_count}
            if feature_store is not None else None
        ),
        "store": {"path": str(store.resolve()), "sha256": sha256_file(store / "index.parquet")},
        "held_out_split": {
            "path": str(split_dir / "split.json"),
            "sha256": sha256_file(split_dir / "split.json"),
            "verified_violation_count": int(split_manifest["verified_violation_count"]),
        },
        "relief_threshold": relief_threshold,
        "held_out_row_count": len(held_out_rows),
        "stratified_mae": stratified,
        "trivial_baseline_mae": trivial_baseline,
        "sc007_beats_trivial_on_relief": sc007_beats_trivial_on_relief,
        "absolute_comparison_to_prior_runs_invalid": True,
    }


DETAILER_RESCORE_SCHEMA = "v116-detailer-rescore-v1"


def rescore_detailer_checkpoint(
    *,
    checkpoint_path: Path,
    store: Path,
    split_dir: Path,
    coarse_store: Path | None = None,
    relief_threshold: float = RELIEF_STD_THRESHOLD,
    feature_store: Path | None = None,
    source: str = "all",
    device: str = "cpu",
) -> dict:
    """Re-score an existing detailer (coarse + residual) checkpoint on the US4 held-out split.

    ``rescore_geometry_checkpoint`` above only knows single-stage geometry checkpoints
    (``direct_cnn_v112``/``mit_b0_regression``) -- it cannot reconstruct a detailer's coarse+
    residual composition (``geometry_detailer_model.GeometryDetailerNet``, which needs a
    per-row materialized coarse relief as a SECOND input alongside RGB, then adds the model's
    residual output to it). This is that same relief-stratified honesty gate, extended to the
    detailer stage: reuses ``chunk_strata``/``stratified_mae``/``tile_mean_baseline_mae`` verbatim,
    only the per-row prediction step differs (coarse+residual composition instead of one forward
    pass).

    ``coarse_store`` defaults to the ``v114-coarse-relief-v1`` store path the checkpoint itself
    recorded at train time (``run_identity["coarse_store"]``) -- pass an explicit override only if
    that path has moved. ``feature_store`` must be the SAME store the checkpoint was trained with
    if its ``run_identity["feature_store"]`` is not null (the checkpoint records the path and
    class_count it used, but not the array data itself, matching the coarse rescorer's identical
    requirement). Never trains; never writes to the checkpoint, coarse store, or curriculum store.
    """
    import pyarrow.parquet as pq
    import torch
    import zarr

    from harvester.v50.direct_geometry_materialize import COARSE_ARRAY, COARSE_STORE_SCHEMA
    from harvester.v50.geometry_detailer_model import (
        DETAILER_ARCHITECTURE_ID,
        GeometryDetailerNet,
        compose_final,
    )
    from harvester.v50.height_relative_model import decode_relative_height, encode_relative_height
    from harvester.v50.height_relative_train import SOURCE_CHOICES

    if source not in SOURCE_CHOICES:
        raise StructureTrainError(f"source must be one of {sorted(SOURCE_CHOICES)}, got {source!r}")

    split_manifest, split_rows = load_split(split_dir)
    if split_manifest["verified_violation_count"] != 0:
        raise StructureTrainError(
            f"held-out split has {split_manifest['verified_violation_count']} violations; "
            "refusing to rescore on a leaky split (FR-010)"
        )

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    variant = ckpt.get("model_variant")
    if variant != DETAILER_ARCHITECTURE_ID:
        raise StructureTrainError(
            f"checkpoint model_variant {variant!r} is not a detailer checkpoint "
            f"({DETAILER_ARCHITECTURE_ID!r}); use --rescore-checkpoint for a single-stage "
            "direct_cnn_v112/mit_b0_regression geometry checkpoint instead"
        )

    resolved_coarse_store = coarse_store or Path(str(ckpt.get("coarse_store", "")))
    if not str(resolved_coarse_store) or not resolved_coarse_store.exists():
        raise StructureTrainError(
            f"coarse store not found: {resolved_coarse_store!r} (pass --coarse-store to override "
            "the path the checkpoint recorded at train time)"
        )
    coarse_group = zarr.open_group(str(resolved_coarse_store), mode="r")
    if dict(coarse_group.attrs).get("schema") != COARSE_STORE_SCHEMA:
        raise StructureTrainError(
            f"--coarse-store is not a {COARSE_STORE_SCHEMA!r} store: {resolved_coarse_store}"
        )
    coarse_index = pq.read_table(resolved_coarse_store / "index.parquet").to_pylist()
    coarse_row_to_position = {int(r["source_row_index"]): pos for pos, r in enumerate(coarse_index)}

    checkpoint_feature_info = ckpt.get("feature_store")
    feature_group = None
    feature_row_to_position: dict[int, int] = {}
    feature_class_count = 0
    if feature_store is not None:
        feature_group = zarr.open_group(str(feature_store), mode="r")
        feature_attrs = dict(feature_group.attrs)
        if feature_attrs.get("schema") != "v115-feature-map-v1":
            raise StructureTrainError(
                f"--feature-store is not a v115-feature-map-v1 store: {feature_store}"
            )
        feature_class_count = int(feature_attrs.get("class_count", 0))
        if feature_class_count < 1 or "feature_map" not in feature_group:
            raise StructureTrainError(f"--feature-store has no usable feature_map array: {feature_store}")
        feature_index = pq.read_table(feature_store / "index.parquet").to_pylist()
        feature_row_to_position = {
            int(r["source_row_index"]): pos for pos, r in enumerate(feature_index)
        }
    elif checkpoint_feature_info:
        raise StructureTrainError(
            "checkpoint was trained with a --feature-store "
            f"(class_count={checkpoint_feature_info.get('class_count')}, "
            f"recorded path={checkpoint_feature_info.get('path')!r}) but none was given here; "
            "pass --feature-store pointing at the same v115-feature-map-v1 store"
        )

    in_channels = 3 + feature_class_count
    model = GeometryDetailerNet(in_channels=in_channels)
    try:
        model.load_state_dict(ckpt["model"])
    except (KeyError, RuntimeError) as exc:
        raise StructureTrainError(
            f"checkpoint weights do not match detailer architecture with in_channels={in_channels}: {exc}"
        ) from exc
    dev = torch.device(device)
    model = model.to(dev)
    model.eval()

    group = zarr.open_group(str(store), mode="r")
    for required in ("minimap_rgb", "height_257"):
        if required not in group:
            raise StructureTrainError(f"store is missing {required!r}: {store}")
    index_rows = pq.read_table(store / "index.parquet").to_pylist()

    split_map: dict[tuple[str, int, int], str] = {}
    for row in split_rows:
        key = (str(row["map"]), int(row["tile_x"]), int(row["tile_y"]))
        split_map[key] = str(row["split"])

    held_out_rows: list[int] = []
    for row_idx, meta in enumerate(index_rows):
        key = (str(meta.get("map")), int(meta.get("tile_x", -1)), int(meta.get("tile_y", -1)))
        if split_map.get(key, "excluded") != "held_out":
            continue
        if source != "all" and str(meta.get("minimap_source")) != source:
            continue
        held_out_rows.append(row_idx)
    if not held_out_rows:
        raise StructureTrainError(
            f"held-out split has zero rows present in this store for source={source!r}"
        )
    missing_coarse = [row for row in held_out_rows if row not in coarse_row_to_position]
    if missing_coarse:
        raise StructureTrainError(
            f"--coarse-store is missing {len(missing_coarse)} of {len(held_out_rows)} held-out "
            f"rows (first missing source_row_index={missing_coarse[0]}) -- materialize it with "
            f"the same --source used here"
        )
    if feature_group is not None:
        missing_feature = [row for row in held_out_rows if row not in feature_row_to_position]
        if missing_feature:
            raise StructureTrainError(
                f"--feature-store is missing {len(missing_feature)} of {len(held_out_rows)} "
                f"held-out rows (first missing source_row_index={missing_feature[0]})"
            )

    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    strata: list[np.ndarray] = []

    with torch.no_grad():
        for row in held_out_rows:
            rgb = np.asarray(group["minimap_rgb"][row], dtype=np.float32) / 255.0
            channels = torch.from_numpy(rgb).permute(2, 0, 1)
            if feature_group is not None:
                position = feature_row_to_position[row]
                feats = np.asarray(feature_group["feature_map"][position], dtype=np.float32)
                channels = torch.cat([channels, torch.from_numpy(feats)], dim=0)
            rgb_t = channels.unsqueeze(0).to(dev)
            coarse = np.asarray(
                coarse_group[COARSE_ARRAY][coarse_row_to_position[row]], dtype=np.float32
            )
            coarse_t = torch.from_numpy(coarse).unsqueeze(0).to(dev)
            height = np.asarray(group["height_257"][row], dtype=np.float32)
            _, tile_min, tile_max = encode_relative_height(height)
            residual = model(rgb_t, coarse_t)
            final_norm = compose_final(coarse_t, residual, clamp=True).squeeze(0).cpu().numpy()
            predicted_height = decode_relative_height(final_norm, tile_min, tile_max)
            predictions.append(predicted_height)
            targets.append(height)
            strata.append(chunk_strata(height, threshold=relief_threshold))

    stratified = stratified_mae(predictions, targets, strata)
    trivial_baseline = tile_mean_baseline_mae(targets)
    relief_mae = stratified["relief"]["mae"]
    sc007_beats_trivial_on_relief = relief_mae is not None and relief_mae < trivial_baseline

    return {
        "schema": DETAILER_RESCORE_SCHEMA,
        "source": source,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "model_variant": variant,
            "in_channels": in_channels,
        },
        "coarse_store": {
            "path": str(resolved_coarse_store.resolve()),
            "sha256": sha256_file(resolved_coarse_store / "index.parquet"),
        },
        "feature_store": (
            {"path": str(feature_store.resolve()), "class_count": feature_class_count}
            if feature_store is not None else None
        ),
        "store": {"path": str(store.resolve()), "sha256": sha256_file(store / "index.parquet")},
        "held_out_split": {
            "path": str(split_dir / "split.json"),
            "sha256": sha256_file(split_dir / "split.json"),
            "verified_violation_count": int(split_manifest["verified_violation_count"]),
        },
        "relief_threshold": relief_threshold,
        "held_out_row_count": len(held_out_rows),
        "stratified_mae": stratified,
        "trivial_baseline_mae": trivial_baseline,
        "sc007_beats_trivial_on_relief": sc007_beats_trivial_on_relief,
        "absolute_comparison_to_prior_runs_invalid": True,
    }


# --------------------------------------------------------------------------- #
# CLI (dry-run-first; user owns every CUDA run).                               #
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Spec 116 US3 per-slot structure classifier training (dry run by default)"
    )
    ap.add_argument("--store", required=True, type=Path, help="curriculum Zarr store")
    ap.add_argument("--split", required=True, type=Path, help="held-out split directory (US4)")
    ap.add_argument("--dumps", nargs="+", type=Path, default=None,
                    help="texture-name dump JSON files (required unless --rescore-checkpoint)")
    ap.add_argument("--vocabulary", type=Path, default=None,
                    help="US1 family_slot_consistency report JSON (required unless --rescore-checkpoint)")
    ap.add_argument("--output", type=Path, default=None,
                    help="run output directory (required unless --rescore-checkpoint)")
    ap.add_argument("--slot", type=int, choices=[1, 2, 3], default=None,
                    help="detail slot to train (required unless --rescore-checkpoint)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--seed", type=int, default=116)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--max-class-weight", type=float, default=MAX_CLASS_WEIGHT)
    ap.add_argument("--device", default="cuda", help="cuda (default) or cpu (tests only)")
    ap.add_argument("--confirm-run", action="store_true", help="required to launch training")
    ap.add_argument("--rescore-checkpoint", type=Path, default=None,
                    help="US4 T019: re-score an EXISTING Spec 114/115 geometry checkpoint on "
                         "--split, stratified by relief. Switches the CLI to read-only "
                         "evaluation mode; --dumps/--vocabulary/--output/--slot are not used.")
    ap.add_argument("--rescore-detailer-checkpoint", type=Path, default=None,
                    help="Rescore an EXISTING detailer (coarse + residual) checkpoint instead of "
                         "a single-stage geometry checkpoint -- use this, not --rescore-checkpoint, "
                         "for anything trained by v50_train_geometry_detailer.py (model_variant "
                         "detailer_unet_v1). Mutually exclusive with --rescore-checkpoint.")
    ap.add_argument("--coarse-store", type=Path, default=None,
                    help="detailer rescore mode: the v114-coarse-relief-v1 store to use. Defaults "
                         "to the path the checkpoint itself recorded at train time; pass this only "
                         "if that path has moved.")
    ap.add_argument("--relief-threshold", type=float, default=RELIEF_STD_THRESHOLD,
                    help="per-chunk height std at/above which a chunk counts as relief-bearing")
    ap.add_argument("--feature-store", type=Path, default=None,
                    help="rescore mode: a v115-feature-map-v1 store (from "
                         "v50_materialize_feature_maps.py or spec116_structure_to_feature_map.py) "
                         "-- reconstructs the same generated feature channels a Spec 115 "
                         "deconfounded checkpoint was trained with. Required for any checkpoint "
                         "whose model_variant is mit_b0_regression with in_channels > 3.")
    ap.add_argument("--rescore-in-channels", type=int, default=None,
                    help="override the rescored checkpoint's input channel count. Auto-derived "
                         "as 3 (RGB) or 3 + --feature-store's class_count when omitted -- you "
                         "should not normally need to set this by hand.")
    ap.add_argument("--rescore-source", default="all", choices=["all", "authored", "synthetic"],
                    help="rescore mode: restrict held-out rows to this minimap_source domain "
                         "(matches direct_geometry_train.py's --source). Required to be "
                         "'authored' when --feature-store only covers the authored subset.")
    ap.add_argument("--rescore-output", type=Path, default=None,
                    help="optional path to write the rescore report JSON")
    ap.add_argument("--print-only", action="store_true",
                    help="rescore mode: print the report only, never write --rescore-output")
    args = ap.parse_args(argv)

    if args.rescore_checkpoint is not None and args.rescore_detailer_checkpoint is not None:
        ap.error("--rescore-checkpoint and --rescore-detailer-checkpoint are mutually exclusive")

    if args.rescore_checkpoint is not None:
        device = args.device
        if device == "cuda":
            import torch
            if not torch.cuda.is_available():
                device = "cpu"
        report = rescore_geometry_checkpoint(
            checkpoint_path=args.rescore_checkpoint,
            store=args.store,
            split_dir=args.split,
            relief_threshold=args.relief_threshold,
            in_channels=args.rescore_in_channels,
            feature_store=args.feature_store,
            source=args.rescore_source,
            device=device,
        )
        print(json.dumps(report, indent=2), flush=True)
        if not args.print_only and args.rescore_output is not None:
            args.rescore_output.parent.mkdir(parents=True, exist_ok=True)
            args.rescore_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"\nwrote report: {args.rescore_output}", flush=True)
        return 0

    if args.rescore_detailer_checkpoint is not None:
        device = args.device
        if device == "cuda":
            import torch
            if not torch.cuda.is_available():
                device = "cpu"
        report = rescore_detailer_checkpoint(
            checkpoint_path=args.rescore_detailer_checkpoint,
            store=args.store,
            split_dir=args.split,
            coarse_store=args.coarse_store,
            relief_threshold=args.relief_threshold,
            feature_store=args.feature_store,
            source=args.rescore_source,
            device=device,
        )
        print(json.dumps(report, indent=2), flush=True)
        if not args.print_only and args.rescore_output is not None:
            args.rescore_output.parent.mkdir(parents=True, exist_ok=True)
            args.rescore_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"\nwrote report: {args.rescore_output}", flush=True)
        return 0

    missing = [
        name for name, value in (
            ("--dumps", args.dumps), ("--vocabulary", args.vocabulary),
            ("--output", args.output), ("--slot", args.slot),
        ) if value is None
    ]
    if missing:
        ap.error(f"{', '.join(missing)} required unless --rescore-checkpoint is given")

    # --- Load preconditions for the dry-run plan. --- #
    split_manifest, split_rows = load_split(args.split)
    if split_manifest["verified_violation_count"] != 0:
        raise SystemExit(
            f"held-out split has {split_manifest['verified_violation_count']} violations; "
            "refusing to train on a leaky split (FR-010)"
        )

    vocab_report = json.loads(args.vocabulary.read_text(encoding="utf-8"))
    from harvester.spec116.family_slot_consistency import recommendation_from_report
    vocabulary_decision = recommendation_from_report(vocab_report)

    labels, tile_mask, label_counts = build_slot_labels(
        store=args.store, dumps=args.dumps, slot=args.slot
    )

    # Partition by split for the plan counts.
    import pyarrow.parquet as pq
    index_rows = pq.read_table(args.store / "index.parquet").to_pylist()
    split_map: dict[tuple[str, int, int], str] = {}
    for row in split_rows:
        key = (str(row["map"]), int(row["tile_x"]), int(row["tile_y"]))
        split_map[key] = str(row["split"])
    train_count = 0
    held_out_count = 0
    for row_idx, meta in enumerate(index_rows):
        if not tile_mask[row_idx]:
            continue
        key = (str(meta.get("map")), int(meta.get("tile_x", -1)), int(meta.get("tile_y", -1)))
        split_label = split_map.get(key, "excluded")
        if split_label == "train":
            train_count += 1
        elif split_label == "held_out":
            held_out_count += 1

    class_weights = compute_class_weights(label_counts, max_weight=args.max_class_weight)
    model, model_identity = build_structure_model(slot=args.slot, base=args.base)

    plan = build_training_plan(
        slot=args.slot,
        architecture=model_identity["architecture"],
        vocabulary_decision=vocabulary_decision,
        split_counts={"train": train_count, "held_out": held_out_count},
        label_counts=label_counts,
        class_weights=class_weights,
        batch_size=args.batch,
        epochs=args.epochs,
        lr=args.lr,
        max_class_weight=args.max_class_weight,
        device=args.device,
    )
    print(json.dumps(plan, indent=2), flush=True)

    if not args.confirm_run:
        print("DRY RUN ONLY: add --confirm-run to launch user-owned training.", flush=True)
        return 0

    # --- User-owned training run. --- #
    if args.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise SystemExit("CUDA is not available; user-run training refuses CPU.")

    run_record = train_structure(
        store=args.store,
        split_dir=args.split,
        dumps=args.dumps,
        vocabulary_report=args.vocabulary,
        output=args.output,
        slot=args.slot,
        base=args.base,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        max_class_weight=args.max_class_weight,
        device=args.device,
        seed=args.seed,
        patience=args.patience,
    )
    print(json.dumps(run_record["metrics"], indent=2), flush=True)
    print(
        f"best epoch {run_record['best_epoch']} "
        f"macro_iou={run_record['metrics']['macro_iou']:.4f} "
        f"rarest_class_iou={run_record['metrics']['rarest_class_iou']:.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
