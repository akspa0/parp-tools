"""Train Spec 102 M0: RGB minimap -> one strict terrain-visible geometry mask."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import zarr
from torch import nn
from torch.utils.data import DataLoader, Dataset

from harvester.spec102.m0 import (
    STRICT_OBJECT_TARGET_KEY,
    M0ObjectMask,
    segmentation_loss,
    strict_object_target_256,
)
from harvester.spec102.m0_coverage import validate_m0_coverage_audit
from harvester.spec102.m0_scope import validate_m0_build_local_scope
from harvester.spec102.m0_validation import M0ValidationSample, render_m0_validation_panel
from harvester.spec102.signal_audit import validate_m0_training_audit


class MaskDataset(Dataset):
    def __init__(self, rgb: np.ndarray, mask: np.ndarray, rows: list[int]) -> None:
        self.rgb = rgb
        self.mask = mask
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        rgb = np.ascontiguousarray(self.rgb[row].transpose(2, 0, 1), dtype=np.float32) / 255.0
        mask = np.ascontiguousarray(self.mask[row] > 0.5, dtype=np.float32)[None]
        return torch.from_numpy(rgb), torch.from_numpy(mask)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def repository_revision() -> str:
    """Record the exact committed trainer revision instead of a vague run label."""
    repository = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("M0 cannot record its git revision; refusing an untraceable GPU run") from error
    revision = result.stdout.strip()
    if len(revision) != 40:
        raise RuntimeError("M0 received an invalid git revision; refusing an untraceable GPU run")
    return revision


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    intersection = union = predicted = target = loss_sum = samples = 0.0
    with torch.inference_mode():
        for rgb, mask in loader:
            rgb = rgb.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(rgb)
                loss, _ = segmentation_loss(logits, mask)
            binary = torch.sigmoid(logits) >= 0.5
            truth = mask >= 0.5
            intersection += float((binary & truth).sum().item())
            union += float((binary | truth).sum().item())
            predicted += float(binary.sum().item())
            target += float(truth.sum().item())
            loss_sum += float(loss.item()) * rgb.shape[0]
            samples += rgb.shape[0]
    return {
        "loss": loss_sum / max(samples, 1.0),
        "iou": intersection / max(union, 1.0),
        "dice": (2.0 * intersection) / max(predicted + target, 1.0),
    }


def write_validation_grid(
    path: Path,
    model: nn.Module,
    dataset: MaskDataset,
    device: torch.device,
    metadata_by_row: dict[int, dict],
    epoch: int,
    split: str = "validation_map",
    threshold: float = 0.5,
    checkpoint_label: str = "current epoch",
    count: int = 4,
) -> None:
    rows = min(count, len(dataset))
    samples: list[M0ValidationSample] = []
    model.eval()
    for index in range(rows):
        rgb, target = dataset[index]
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            probability = torch.sigmoid(model(rgb[None].to(device)))[0, 0].float().cpu().numpy()
        source = np.clip(rgb.permute(1, 2, 0).numpy() * 255.0, 0, 255).astype(np.uint8)
        row = dataset.rows[index]
        metadata = metadata_by_row[row]
        samples.append(M0ValidationSample(
            row=row,
            build=str(metadata["build"]),
            map_name=str(metadata["map"]),
            tile_x=int(metadata["tile_x"]),
            tile_y=int(metadata["tile_y"]),
            source_rgb=source,
            probability=probability,
            target=target[0].numpy(),
        ))
    canvas = render_m0_validation_panel(
        samples, split=split, epoch=epoch, threshold=threshold, checkpoint_label=checkpoint_label,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Spec 102 M0 object-mask model")
    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--signal-audit-report", required=True, type=Path)
    parser.add_argument("--coverage-report", required=True, type=Path)
    parser.add_argument("--raw-v18-store", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--prior-gate-report", type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=102)
    parser.add_argument("--positive-weight", type=float, default=4.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    args = parser.parse_args()

    if args.epochs < 1 or args.epochs > 12:
        raise ValueError("Spec 102 M0 permits 1-12 epochs; extension requires a passed prior gate")
    manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    store_index = pq.read_table(args.store / "index.parquet").to_pylist()
    scope = validate_m0_build_local_scope(manifest, source_index=store_index)
    audit_report = validate_m0_training_audit(
        args.signal_audit_report,
        store=args.store,
        split_manifest=args.split_manifest,
        expected_scope=scope.audit_binding,
        scoped_rows=scope.scoped_rows,
    )
    validate_m0_coverage_audit(
        args.coverage_report,
        raw_v18_store=args.raw_v18_store,
        store=args.store,
        split_manifest=args.split_manifest,
        expected_scope=scope.audit_binding,
    )
    artifact_binding = scope.artifact_binding(
        store=args.store,
        split_manifest=args.split_manifest,
        coverage_report=args.coverage_report,
    )
    code_revision = repository_revision()
    if args.epochs > 3:
        if args.prior_gate_report is None or not args.prior_gate_report.is_file():
            raise RuntimeError("M0 runs beyond 3 epochs require a passed build-local --prior-gate-report")
        if args.resume_checkpoint is None:
            raise RuntimeError("M0 runs beyond 3 epochs must resume the bound three-epoch checkpoint")
        try:
            prior_gate = json.loads(args.prior_gate_report.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"cannot read M0 prior gate report: {error}") from error
        if prior_gate.get("schema") != "spec102-m0-training-report-v3":
            raise RuntimeError("M0 extension rejects legacy or unknown prior gate reports")
        if prior_gate.get("gate_scope") != "build_local_3_3_5_only" or prior_gate.get("cross_era_evaluated") is not False:
            raise RuntimeError("M0 extension requires a 3.3.5-only prior decision report")
        if prior_gate.get("m0_artifact_binding") != artifact_binding:
            raise RuntimeError("M0 extension prior report is bound to a different store or split")
        if prior_gate.get("signal_audit_fingerprint") != audit_report.get("scoped_signal_fingerprint"):
            raise RuntimeError("M0 extension prior report is bound to a different audited signal snapshot")
        if prior_gate.get("epochs") != 3 or prior_gate.get("extension_authorized") is not True:
            raise RuntimeError("M0 extension requires an improving three-epoch build-local decision")
        prior_checkpoint = prior_gate.get("best_checkpoint")
        if not isinstance(prior_checkpoint, str) or Path(prior_checkpoint).resolve() != args.resume_checkpoint.resolve():
            raise RuntimeError("M0 extension must resume the checkpoint named by its prior decision report")
    rows_by_split = scope.rows_by_split
    metadata_by_row = scope.metadata_by_row

    if not torch.cuda.is_available():
        raise RuntimeError("M0 is CUDA-only; CPU fallback is prohibited")
    seed_everything(args.seed)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()

    group = zarr.open_group(str(args.store), mode="r")
    target_name = STRICT_OBJECT_TARGET_KEY
    if target_name not in group:
        raise RuntimeError(
            f"M0 strict geometry target '{target_name}' is missing; refusing all target fallbacks"
        )
    rgb = np.asarray(group["minimap_rgb"][:])
    strict_masks = np.asarray(group[target_name][:], dtype=np.float32)
    if strict_masks.shape[1:] != (257, 257):
        raise RuntimeError(f"M0 strict geometry target has invalid shape {strict_masks.shape}")
    masks = np.stack([strict_object_target_256(mask) for mask in strict_masks], axis=0)
    if rgb.shape[0] != len(store_index) or masks.shape[0] != len(store_index):
        raise RuntimeError("Store row count does not match the hash-bound M0 scope")
    eligible_rows = np.asarray(
        [row for split_rows in rows_by_split.values() for row in split_rows], dtype=np.int64
    )
    positive_prevalence = float((masks[eligible_rows] > 0.5).mean())
    if positive_prevalence <= 0.0:
        raise RuntimeError(f"M0 target '{target_name}' is empty; refusing a meaningless GPU run")

    datasets = {split: MaskDataset(rgb, masks, rows) for split, rows in rows_by_split.items()}
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=0, pin_memory=True),
        "validation_map": DataLoader(datasets["validation_map"], batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True),
        "test_build_local": DataLoader(datasets["test_build_local"], batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True),
    }

    model = M0ObjectMask().to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if not 3_000_000 <= parameter_count <= 12_000_000:
        raise RuntimeError(f"M0 parameter count {parameter_count} violates the 3-12M contract")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_manifest = {
        "schema": "spec102-m0-input-v3",
        "forward_inputs": [{"name": "minimap_rgb", "shape": ["batch", 3, 256, 256], "deployment_source": "raw minimap RGB"}],
        "target_only": [{
            "name": target_name,
            "shape": ["batch", 257, 257],
            "projection": "four_corner_max_to_256",
        }],
        "output": {"name": "object_geometry_visible_mask_logits", "shape": ["batch", 1, 256, 256]},
        "prohibited_inputs": [
            "height_257", "normal_xyz_257", "wdl_height_33", target_name,
            "object_mask_256", "object_visibility_256", "object_precise_mask_257",
        ],
        "m0_training_scope": scope.audit_binding,
        "m0_artifact_binding": artifact_binding,
        "cross_era_evaluated": False,
        "signal_audit_report": str(args.signal_audit_report.resolve()),
        "signal_audit_fingerprint": audit_report["scoped_signal_fingerprint"],
        "coverage_report": str(args.coverage_report.resolve()),
        "coverage_report_sha256": artifact_binding["coverage_report_sha256"],
        "coverage_raw_v18_store": str(args.raw_v18_store.resolve()),
        "command": list(sys.argv),
        "code_revision": code_revision,
    }
    (args.output_dir / "input_manifest.json").write_text(json.dumps(input_manifest, indent=2), encoding="utf-8")

    history: list[dict] = []
    best_iou = -1.0
    start_epoch = 0
    if args.resume_checkpoint is not None:
        resume = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
        if resume.get("schema") != "spec102-m0-checkpoint-v2":
            raise RuntimeError("--resume-checkpoint is not a Spec 102 M0 checkpoint")
        if resume.get("m0_artifact_binding") != artifact_binding:
            raise RuntimeError("--resume-checkpoint was not trained on this hash-bound 3.3.5-only scope")
        model.load_state_dict(resume["model"], strict=True)
        start_epoch = int(resume["epoch"])
        best_iou = float(resume["validation"]["iou"])
        torch.save(resume, args.output_dir / "checkpoint_best.pt")
    if start_epoch >= args.epochs:
        raise RuntimeError(f"resume checkpoint epoch {start_epoch} already reaches requested epoch {args.epochs}")
    started = time.perf_counter()
    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        train_loss = samples = 0.0
        for batch_rgb, batch_mask in loaders["train"]:
            batch_rgb = batch_rgb.to(device, non_blocking=True)
            batch_mask = batch_mask.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(batch_rgb)
                loss, _ = segmentation_loss(
                    logits,
                    batch_mask,
                    positive_weight=args.positive_weight,
                    dice_weight=args.dice_weight,
                )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite M0 loss at epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss.item()) * batch_rgb.shape[0]
            samples += batch_rgb.shape[0]
        validation = evaluate(model, loaders["validation_map"], device)
        record = {"epoch": epoch, "train_loss": train_loss / max(samples, 1), "validation": validation}
        history.append(record)
        print(json.dumps(record), flush=True)
        write_validation_grid(
            args.output_dir / "validation" / f"epoch_{epoch:02d}.png",
            model,
            datasets["validation_map"],
            device,
            metadata_by_row=metadata_by_row,
            epoch=epoch,
        )
        if validation["iou"] > best_iou:
            best_iou = validation["iou"]
            torch.save(
                {
                    "schema": "spec102-m0-checkpoint-v2",
                    "model": model.state_dict(),
                    "config": {
                        "base_channels": 40,
                        "parameter_count": parameter_count,
                        "positive_weight": args.positive_weight,
                        "dice_weight": args.dice_weight,
                        "target": target_name,
                        "single_output": True,
                    },
                    "input_manifest": input_manifest,
                    "m0_training_scope": scope.audit_binding,
                    "m0_artifact_binding": artifact_binding,
                    "code_revision": code_revision,
                    "epoch": epoch,
                    "validation": validation,
                },
                args.output_dir / "checkpoint_best.pt",
            )

    checkpoint = torch.load(args.output_dir / "checkpoint_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_metrics = evaluate(model, loaders["test_build_local"], device)
    write_validation_grid(
        args.output_dir / "validation" / "best_test_build_local.png",
        model,
        datasets["test_build_local"],
        device,
        metadata_by_row=metadata_by_row,
        epoch=int(checkpoint["epoch"]),
        split="test_build_local",
        checkpoint_label="best_checkpoint",
    )
    gate_passed = best_iou >= 0.25 and test_metrics["iou"] >= 0.10
    extension_authorized = (
        args.epochs == 3
        and not gate_passed
        and len(history) == 3
        and all(np.isfinite(record["train_loss"]) and np.isfinite(record["validation"]["iou"]) for record in history)
        and history[-1]["validation"]["iou"] > history[0]["validation"]["iou"]
        and test_metrics["iou"] > 0.0
    )
    report = {
        "schema": "spec102-m0-training-report-v3",
        "gate_scope": "build_local_3_3_5_only",
        "cross_era_evaluated": False,
        "m0_training_scope": scope.audit_binding,
        "m0_artifact_binding": artifact_binding,
        "signal_audit_report": str(args.signal_audit_report.resolve()),
        "signal_audit_fingerprint": audit_report["scoped_signal_fingerprint"],
        "coverage_report": str(args.coverage_report.resolve()),
        "coverage_report_sha256": artifact_binding["coverage_report_sha256"],
        "coverage_raw_v18_store": str(args.raw_v18_store.resolve()),
        "command": list(sys.argv),
        "code_revision": code_revision,
        "parameter_count": parameter_count,
        "target": target_name,
        "target_positive_prevalence": positive_prevalence,
        "epochs": args.epochs,
        "resumed_from_epoch": start_epoch,
        "resume_checkpoint": str(args.resume_checkpoint.resolve()) if args.resume_checkpoint else None,
        "history": history,
        "best_validation_iou": best_iou,
        "test_build_local": test_metrics,
        "zero_mask_baseline_iou": 0.0,
        "gate_requirements": {"validation_iou_min": 0.25, "test_build_local_iou_min": 0.10},
        "gate_passed": gate_passed,
        "extension_authorized": extension_authorized,
        "best_checkpoint": str((args.output_dir / "checkpoint_best.pt").resolve()),
        "peak_vram_gb": torch.cuda.max_memory_allocated() / (1024**3),
        "wall_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
