"""Train the v60 real object-mask detector from an existing v50 Zarr store.

This lane predicts real ``object_precise_mask`` and/or ``object_mask`` from ``minimap_rgb``.
It does not claim a clean-minimap target and it never uses the empty geometry-visible signal as a
replacement. GPU training requires ``--confirm-run`` and remains user-owned.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
import zarr
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from harvester.v60.real_object_mask_model import (  # noqa: E402
    REAL_OBJECT_TARGETS,
    RealObjectMaskNet,
    project_mask_257_to_256,
    real_object_mask_loss,
)

THRESHOLDS = np.round(np.arange(0.20, 0.81, 0.05), 2)
STORE_SCHEMA = "v60-real-object-mask-experiment-v1"
TARGETS_BY_MODE = {
    "precise": ("object_precise_mask",),
    "footprint": ("object_mask",),
    "both": REAL_OBJECT_TARGETS,
}


@dataclass(frozen=True)
class RowRef:
    row_index: int
    map_name: str
    tile_x: int
    tile_y: int
    source_group_id: str
    minimap_source: str
    split: str


def _d4(array: np.ndarray, index: int) -> np.ndarray:
    result = np.rot90(array, index % 4)
    if index >= 4:
        result = np.fliplr(result)
    return np.ascontiguousarray(result)


def _edge_channel(rgb: np.ndarray) -> np.ndarray:
    gray = rgb.astype(np.float32).mean(axis=2)
    gy, gx = np.gradient(gray)
    return np.clip(np.hypot(gx, gy) / 128.0, 0.0, 1.0).astype(np.float32)


def _target_mode(mode: str) -> tuple[str, ...]:
    try:
        return tuple(TARGETS_BY_MODE[mode])
    except KeyError as exc:
        raise ValueError(f"unknown target mode {mode!r}") from exc


def _load_rows(
    store: Path,
    source: str,
    split_policy: str,
    val_map: str,
    input_kind: str = "rgb",
    validation_limit: int = 0,
) -> list[RowRef]:
    rows = pq.read_table(store / "index.parquet").to_pylist()
    if validation_limit < 0:
        raise ValueError("validation_limit must be non-negative")
    selected: list[RowRef] = []
    for index, row in enumerate(rows):
        minimap_source = str(row.get("minimap_source", ""))
        if source != "all" and minimap_source != source:
            continue
        map_name = str(row.get("map", ""))
        if split_policy == "map_holdout":
            split = "val" if map_name == val_map else "train"
        else:
            split = str(row.get("split", ""))
            if split not in {"train", "val"}:
                continue
        source_store = str(row.get("source_store", ""))
        if "0_5_3_3368" not in source_store:
            raise ValueError(f"row {index} is not from the expected 0_5_3_3368 source: {source_store}")
        selected.append(
            RowRef(
                row_index=index,
                map_name=map_name,
                tile_x=int(row.get("tile_x", -1)),
                tile_y=int(row.get("tile_y", -1)),
                source_group_id=str(row.get("source_group_id", "")),
                minimap_source=minimap_source,
                split=split,
            )
        )
    if not selected:
        raise ValueError("no rows remain after source/split filtering")
    groups: dict[str, set[str]] = {}
    for row in selected:
        if row.source_group_id:
            groups.setdefault(row.source_group_id, set()).add(row.split)
    conflicts = {group: sorted(splits) for group, splits in groups.items() if len(splits) > 1}
    if conflicts:
        sample = list(conflicts.items())[:5]
        raise ValueError(f"source groups cross train/validation split: {sample}")
    if not any(row.split == "train" for row in selected) or not any(row.split == "val" for row in selected):
        raise ValueError("split must contain both train and validation rows")
    if validation_limit:
        train_rows = [row for row in selected if row.split == "train"]
        val_rows = [row for row in selected if row.split == "val"][:validation_limit]
        if not val_rows:
            raise ValueError("validation_limit removed every validation row")
        selected = train_rows + val_rows
    return selected


def _check_store(group: zarr.Group, store: Path) -> dict[str, Any]:
    required = ("minimap_rgb", "object_mask", "object_precise_mask", "index.parquet")
    missing = [name for name in required if name != "index.parquet" and name not in group]
    if not (store / "index.parquet").is_file():
        missing.append("index.parquet")
    if missing:
        raise ValueError(f"v50 store is missing required real-mask inputs: {missing}")
    release = str(group.attrs.get("release", ""))
    if release != "v50.1":
        raise ValueError(f"expected v50.1 store release, got {release!r}")
    if tuple(group["minimap_rgb"].shape[1:]) != (256, 256, 3):
        raise ValueError(f"unexpected minimap_rgb shape {group['minimap_rgb'].shape}")
    for name in ("object_mask", "object_precise_mask"):
        if tuple(group[name].shape[1:]) != (257, 257):
            raise ValueError(f"unexpected {name} shape {group[name].shape}")
    geometry_present = "object_geometry_visible_mask_257" in group
    return {
        "release": release,
        "model_family": str(group.attrs.get("model_family", "")),
        "row_count": int(group["minimap_rgb"].shape[0]),
        "geometry_visible_present": geometry_present,
    }


def _geometry_audit(group: zarr.Group, rows: list[RowRef]) -> dict[str, Any]:
    name = "object_geometry_visible_mask_257"
    if name not in group:
        return {"present": False, "rows_audited": 0, "nonzero_rows": 0, "used_as_target": False}
    nonzero = 0
    for row in rows:
        values = np.asarray(group[name][row.row_index])
        if np.any(values > 0.5):
            nonzero += 1
    return {
        "present": True,
        "rows_audited": len(rows),
        "nonzero_rows": nonzero,
        "used_as_target": False,
    }


class RealObjectMaskDataset(Dataset):
    def __init__(self, group: zarr.Group, rows: list[RowRef], target_names: tuple[str, ...], *, input_kind: str, augment: bool) -> None:
        self.group = group
        self.rows = rows
        self.target_names = target_names
        self.input_kind = input_kind
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        rgb = np.asarray(self.group["minimap_rgb"][row.row_index], dtype=np.uint8)
        targets = np.stack(
            [project_mask_257_to_256(np.asarray(self.group[name][row.row_index])) for name in self.target_names],
            axis=0,
        )
        if self.augment:
            transform = int(np.random.randint(0, 8))
            rgb = _d4(rgb, transform)
            targets = np.stack([_d4(target, transform) for target in targets], axis=0)
        inputs = rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
        if self.input_kind == "rgb_edge":
            inputs = np.concatenate((inputs, _edge_channel(rgb)[None]), axis=0)
        return (
            torch.from_numpy(np.ascontiguousarray(inputs)),
            torch.from_numpy(np.ascontiguousarray(targets)),
        )


def _filter_blank(group: zarr.Group, rows: list[RowRef], minimum_std: float) -> tuple[list[RowRef], int]:
    if minimum_std <= 0.0:
        return rows, 0
    kept: list[RowRef] = []
    dropped = 0
    for row in rows:
        rgb = np.asarray(group["minimap_rgb"][row.row_index], dtype=np.float32)
        if float(rgb.std()) >= minimum_std:
            kept.append(row)
        else:
            dropped += 1
    return kept, dropped


def _metric_state(target_count: int) -> dict[str, np.ndarray]:
    return {
        "intersection": np.zeros((target_count, len(THRESHOLDS)), dtype=np.float64),
        "union": np.zeros((target_count, len(THRESHOLDS)), dtype=np.float64),
        "predicted": np.zeros((target_count, len(THRESHOLDS)), dtype=np.float64),
        "truth": np.zeros(target_count, dtype=np.float64),
        "positive_pixels": np.zeros(target_count, dtype=np.float64),
        "tiles_with_positive": np.zeros(target_count, dtype=np.float64),
    }


@torch.no_grad()
def evaluate(model: RealObjectMaskNet, loader: DataLoader, device: torch.device, use_amp: bool, positive_weight: float = 4.0) -> dict[str, Any]:
    model.eval()
    names = model.target_names
    state = _metric_state(len(names))
    loss_sum = 0.0
    count = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(inputs)
            loss, _ = real_object_mask_loss(logits, targets, names, positive_weight=positive_weight)
        probability = logits.float().sigmoid()
        target_bool = targets >= 0.5
        loss_sum += float(loss.detach()) * inputs.shape[0]
        count += int(inputs.shape[0])
        for target_index in range(len(names)):
            truth = target_bool[:, target_index]
            state["truth"][target_index] += float(truth.sum())
            state["positive_pixels"][target_index] += float(targets[:, target_index].sum())
            state["tiles_with_positive"][target_index] += float(truth.flatten(1).any(dim=1).sum())
            for threshold_index, threshold in enumerate(THRESHOLDS):
                prediction = probability[:, target_index] >= float(threshold)
                state["intersection"][target_index, threshold_index] += float((prediction & truth).sum())
                state["union"][target_index, threshold_index] += float((prediction | truth).sum())
                state["predicted"][target_index, threshold_index] += float(prediction.sum())
    metrics: dict[str, Any] = {"loss": loss_sum / max(count, 1), "rows": count}
    selection_scores: list[float] = []
    for target_index, name in enumerate(names):
        intersection = state["intersection"][target_index]
        union = state["union"][target_index]
        predicted = state["predicted"][target_index]
        truth = state["truth"][target_index]
        iou = intersection / np.maximum(union, 1.0)
        dice = 2.0 * intersection / np.maximum(predicted + truth, 1.0)
        best_index = int(np.argmax(iou))
        threshold = float(THRESHOLDS[best_index])
        metrics[name] = {
            "positive_fraction": float(state["positive_pixels"][target_index] / max(count * 256 * 256, 1)),
            "tiles_with_positive": int(state["tiles_with_positive"][target_index]),
            "best_threshold": threshold,
            "iou": float(iou[best_index]),
            "dice": float(dice[best_index]),
            "precision": float(intersection[best_index] / max(predicted[best_index], 1.0)),
            "recall": float(intersection[best_index] / max(truth, 1.0)),
            "iou_at_0.5": float(iou[int(np.argmin(np.abs(THRESHOLDS - 0.5)))]),
            "zero_mask_baseline_iou": 0.0,
        }
        selection_scores.append(metrics[name]["iou"])
    metrics["selection_score"] = float(min(selection_scores)) if selection_scores else 0.0
    return metrics


def _preview(model: RealObjectMaskNet, group: zarr.Group, rows: list[RowRef], device: torch.device, input_kind: str, output: Path, metrics: dict[str, Any]) -> None:
    model.eval()
    panels: list[Image.Image] = []
    font = ImageFont.load_default()
    for row in rows[:4]:
        rgb = np.asarray(group["minimap_rgb"][row.row_index], dtype=np.uint8)
        targets = np.stack([project_mask_257_to_256(np.asarray(group[name][row.row_index])) for name in model.target_names])
        inputs = rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
        if input_kind == "rgb_edge":
            inputs = np.concatenate((inputs, _edge_channel(rgb)[None]), axis=0)
        with torch.no_grad():
            prediction = torch.sigmoid(model(torch.from_numpy(inputs[None]).to(device)).float())[0].cpu().numpy()
        for index, name in enumerate(model.target_names):
            truth = targets[index] >= 0.5
            probability = prediction[index]
            threshold = float(metrics[name]["best_threshold"])
            predicted = probability >= threshold
            truth_image = np.repeat((truth.astype(np.uint8) * 255)[..., None], 3, axis=2)
            probability_image = np.repeat((np.clip(probability, 0.0, 1.0) * 255).astype(np.uint8)[..., None], 3, axis=2)
            agreement = np.zeros((256, 256, 3), dtype=np.uint8)
            agreement[predicted & truth] = (0, 200, 0)
            agreement[predicted & ~truth] = (220, 0, 0)
            agreement[~predicted & truth] = (0, 90, 255)
            row_image = Image.fromarray(np.concatenate((rgb, truth_image, probability_image, agreement), axis=1))
            draw = ImageDraw.Draw(row_image)
            draw.rectangle((0, 0, 1024, 18), fill=(0, 0, 0))
            draw.text((4, 3), f"{row.map_name} {row.tile_x}_{row.tile_y} | {name} | IoU={metrics[name]['iou']:.3f}", fill=(255, 255, 0), font=font)
            panels.append(row_image)
    if panels:
        canvas = Image.new("RGB", (1024, len(panels) * 256), (0, 0, 0))
        for index, panel in enumerate(panels):
            canvas.paste(panel, (0, index * 256))
        output.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output)


def _input_contract(input_kind: str) -> str:
    return {"rgb": "minimap_rgb", "rgb_edge": "minimap_rgb_edge"}[input_kind]


def _input_channels(input_kind: str) -> int:
    return {"rgb": 3, "rgb_edge": 4}[input_kind]


def _plan(
    store: Path,
    source: str,
    split_policy: str,
    val_map: str,
    target_mode: str,
    input_kind: str,
    validation_limit: int,
) -> dict[str, Any]:
    group = zarr.open_group(str(store), mode="r")
    store_info = _check_store(group, store)
    rows = _load_rows(store, source, split_policy, val_map, input_kind, validation_limit)
    train_rows = [row for row in rows if row.split == "train"]
    val_rows = [row for row in rows if row.split == "val"]
    return {
        "schema": STORE_SCHEMA,
        "store": str(store.resolve()),
        "dataset": {
            **store_info,
            "source_build": "0_5_3_3368",
            "source_filter": source,
            "split_policy": split_policy,
            "train_rows": len(train_rows),
            "validation_rows": len(val_rows),
        },
        "input_contract": _input_contract(input_kind),
        "targets": list(_target_mode(target_mode)),
        "val_map": val_map,
        "validation_limit": validation_limit,
        "source_counts": {source_name: sum(row.minimap_source == source_name for row in rows) for source_name in {row.minimap_source for row in rows}},
        "source_group_count": len({row.source_group_id for row in rows if row.source_group_id}),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source", choices=("authored", "synthetic", "all"), default="authored")
    parser.add_argument("--split", dest="split_policy", choices=("manifest", "map_holdout"), default="map_holdout")
    parser.add_argument("--val-map", default="Azeroth")
    parser.add_argument("--targets", choices=tuple(TARGETS_BY_MODE), default="both")
    parser.add_argument("--input", dest="input_kind", choices=("rgb", "rgb_edge"), default="rgb")
    parser.add_argument("--validation-rows", type=int, default=0, help="cap validation rows after deterministic split; 0 keeps all")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-rgb-std", type=float, default=1.0)
    parser.add_argument("--positive-weight", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-aug", action="store_true")
    parser.add_argument("--confirm-run", action="store_true", help="required to launch CUDA training")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.store.is_dir():
        raise SystemExit(f"store does not exist: {args.store}")
    plan = _plan(args.store, args.source, args.split_policy, args.val_map, args.targets, args.input_kind, args.validation_rows)
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    if args.plan_only:
        return 0
    if not args.confirm_run:
        raise SystemExit("refusing to train without --confirm-run; use --plan-only for an offline check")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; refusing to train on CPU")
    if args.epochs < 1 or args.batch < 1:
        raise SystemExit("--epochs and --batch must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"output is non-empty; choose a fresh path: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    group = zarr.open_group(str(args.store), mode="r")
    rows = _load_rows(args.store, args.source, args.split_policy, args.val_map, args.input_kind, args.validation_rows)
    train_rows, val_rows = [row for row in rows if row.split == "train"], [row for row in rows if row.split == "val"]
    train_rows, dropped_train = _filter_blank(group, train_rows, args.min_rgb_std)
    val_rows, dropped_val = _filter_blank(group, val_rows, args.min_rgb_std)
    target_names = _target_mode(args.targets)
    train_loader = DataLoader(RealObjectMaskDataset(group, train_rows, target_names, input_kind=args.input_kind, augment=not args.no_aug), batch_size=args.batch, shuffle=True, num_workers=0, pin_memory=True, drop_last=False)
    val_loader = DataLoader(RealObjectMaskDataset(group, val_rows, target_names, input_kind=args.input_kind, augment=False), batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=True)
    model = RealObjectMaskNet(in_channels=_input_channels(args.input_kind), target_names=target_names).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=not args.no_amp)
    best_score = -math.inf
    best_epoch = 0
    no_improve = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=not args.no_amp):
                logits = model(inputs)
                loss, _ = real_object_mask_loss(logits, targets, target_names, positive_weight=args.positive_weight)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss.detach()) * inputs.shape[0]
            train_count += int(inputs.shape[0])
        scheduler.step()
        validation = evaluate(model, val_loader, device, not args.no_amp, args.positive_weight)
        record = {"epoch": epoch, "train_loss": train_loss / max(train_count, 1), "validation": validation}
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        score = float(validation["selection_score"])
        if score > best_score:
            best_score, best_epoch, no_improve = score, epoch, 0
            torch.save({"model": model.state_dict(), "model_config": {"in_channels": model.in_channels, "target_names": list(target_names)}, "epoch": epoch, "validation": validation, "history": history}, args.output / "checkpoint_best.pt")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                break
    checkpoint = torch.load(args.output / "checkpoint_best.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    final_metrics = evaluate(model, val_loader, device, not args.no_amp, args.positive_weight)
    preview = args.output / "validation-preview.png"
    _preview(model, group, val_rows, device, args.input_kind, preview, final_metrics)
    report = {
        **plan,
        "dataset": {**plan["dataset"], "train_rows_after_blank_filter": len(train_rows), "validation_rows_after_blank_filter": len(val_rows), "dropped_blank_train": dropped_train, "dropped_blank_validation": dropped_val},
        "input_contract": plan["input_contract"],
        "targets": list(target_names),
        "target_metrics": final_metrics,
        "selection_score": float(final_metrics["selection_score"]),
        "best_epoch": best_epoch,
        "history": history,
        "geometry_visible_mask_audit": _geometry_audit(group, rows),
        "split_audit": {"source_group_conflicts": 0, "map_counts": {name: sum(row.map_name == name for row in rows) for name in sorted({row.map_name for row in rows})}, "source_counts": plan["source_counts"]},
        "preview_artifacts": [str(preview.resolve())],
        "training_command": " ".join(sys.argv),
    }
    (args.output / "experiment_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.output / 'experiment_report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
