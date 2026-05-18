from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

_src_dir = Path(__file__).resolve().parent.parent / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader

from harvester.v15_model import V15Model
from harvester.v16_dataset import V16Dataset

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DATASET_ROOT = _PROJECT_ROOT / "output" / "datasets" / "v16"
_REPORT_ROOT = _DATASET_ROOT / "validation"

_TENSOR_SPECS: dict[str, dict[str, Any]] = {
    "input": {"shape": (3, 256, 256), "dtype": torch.float32, "min": 0.0, "max": 1.0},
    "height": {"shape": (1, 257, 257), "dtype": torch.float32},
    "normals": {"shape": (3, 257, 257), "dtype": torch.float32, "min": -1.0, "max": 1.0},
    "normal_mask": {"shape": (1, 257, 257), "dtype": torch.float32, "min": 0.0, "max": 1.0},
    "alpha": {"shape": (4, 256, 256), "dtype": torch.float32, "min": 0.0, "max": 1.0},
    "holes": {"shape": (1, 16, 16), "dtype": torch.float32, "min": 0.0, "max": 1.0},
    "liquid": {"shape": (1, 256, 256), "dtype": torch.float32, "min": 0.0, "max": 1.0},
    "liquid_height": {"shape": (1, 256, 256), "dtype": torch.float32},
    "weight": {"shape": (1, 257, 257), "dtype": torch.float32, "min": 0.0, "max": 1.0},
    "instance_mask": {"shape": (1, 257, 257), "dtype": torch.int64, "min": 0},
    "mcly_ids": {"shape": (16, 16, 4), "dtype": torch.int64, "min": 0, "max": 15},
    "mcly_mask": {"shape": (16, 16, 4), "dtype": torch.float32, "min": 0.0},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate that V16 Zarr datasets are readable by the current training stack"
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=_DATASET_ROOT,
        help="Root directory containing V16 .zarr stores",
    )
    parser.add_argument("--build", type=str, help="Single build key")
    parser.add_argument("--builds", nargs="+", help="Multiple build keys")
    parser.add_argument(
        "--train-samples",
        type=int,
        default=64,
        help="How many train samples to sanity-check through V16Dataset",
    )
    parser.add_argument(
        "--val-samples",
        type=int,
        default=32,
        help="How many val samples to sanity-check through V16Dataset",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for DataLoader/model checks")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Validation split fraction")
    parser.add_argument("--seed", type=int, default=42, help="Dataset split seed")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "auto"],
        help="Device for model forward-pass validation",
    )
    parser.add_argument(
        "--skip-model-forward",
        action="store_true",
        help="Skip the V15Model forward-pass check and validate dataset loading only",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPORT_ROOT,
        help="Directory where the validation JSON report will be written",
    )
    return parser.parse_args()


def _resolve_builds(args: argparse.Namespace, dataset_dir: Path) -> list[str]:
    builds = args.builds or ([args.build] if args.build else [])
    if builds:
        return builds
    return sorted(path.stem for path in dataset_dir.glob("*.zarr"))


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _open_index_summary(dataset_dir: Path, build: str) -> dict[str, Any]:
    zarr_path = dataset_dir / f"{build}.zarr"
    index_path = zarr_path / "index.parquet"
    if not index_path.exists():
        raise RuntimeError(f"Missing index.parquet for build {build}: {index_path}")

    table = pq.read_table(str(index_path))
    signal_counts: dict[str, int] = {}
    for col in table.column_names:
        if not col.startswith("has_"):
            continue
        count_scalar = pc.sum(table.column(col))
        signal_counts[col] = 0 if count_scalar is None else int(count_scalar.as_py() or 0)

    placements_path = zarr_path / "placements.parquet"
    rejected_path = dataset_dir / f"{build}.rejected_tiles.jsonl"

    return {
        "build": build,
        "zarr_path": str(zarr_path),
        "tile_count": int(table.num_rows),
        "column_count": len(table.column_names),
        "placement_rows": int(pq.read_metadata(str(placements_path)).num_rows) if placements_path.exists() else 0,
        "rejected_tiles_report": str(rejected_path) if rejected_path.exists() else None,
        "signal_counts": signal_counts,
    }


def _entry_for_sample(dataset: V16Dataset, idx: int) -> dict[str, Any]:
    entry = dataset._index_entries[dataset._indices[idx]]
    return {
        "build": entry.get("_build"),
        "tile_id": entry.get("tile_id"),
        "map": entry.get("map"),
        "tile_x": entry.get("tile_x"),
        "tile_y": entry.get("tile_y"),
    }


def _record_issue(issues: list[dict[str, Any]], split: str, sample_meta: dict[str, Any], message: str) -> None:
    issue = {"split": split, **sample_meta, "message": message}
    issues.append(issue)


def _check_tensor(
    name: str,
    tensor: torch.Tensor,
    sample_meta: dict[str, Any],
    split: str,
    issues: list[dict[str, Any]],
) -> None:
    spec = _TENSOR_SPECS[name]
    if tuple(tensor.shape) != spec["shape"]:
        _record_issue(
            issues,
            split,
            sample_meta,
            f"{name} shape mismatch: expected {spec['shape']} got {tuple(tensor.shape)}",
        )
    if tensor.dtype != spec["dtype"]:
        _record_issue(
            issues,
            split,
            sample_meta,
            f"{name} dtype mismatch: expected {spec['dtype']} got {tensor.dtype}",
        )

    if tensor.is_floating_point():
        if not torch.isfinite(tensor).all():
            _record_issue(issues, split, sample_meta, f"{name} contains non-finite values")
        value_min = float(tensor.min().item())
        value_max = float(tensor.max().item())
    else:
        value_min = int(tensor.min().item())
        value_max = int(tensor.max().item())

    if "min" in spec and value_min < spec["min"] - 1e-5:
        _record_issue(
            issues,
            split,
            sample_meta,
            f"{name} min out of range: expected >= {spec['min']} got {value_min}",
        )
    if "max" in spec and value_max > spec["max"] + 1e-5:
        _record_issue(
            issues,
            split,
            sample_meta,
            f"{name} max out of range: expected <= {spec['max']} got {value_max}",
        )


def _validate_sample_semantics(
    sample: dict[str, Any],
    split: str,
    sample_meta: dict[str, Any],
    issues: list[dict[str, Any]],
) -> None:
    has_normals = bool(sample["has_normals"])
    has_alpha = bool(sample["has_alpha"])
    has_holes = bool(sample["has_holes"])
    has_liquid = bool(sample["has_liquid"])
    has_instance = bool(sample["has_instance"])
    has_mcly = bool(sample["has_mcly"])

    normal_mask_sum = float(sample["normal_mask"].sum().item())
    alpha_sum = float(sample["alpha"].sum().item())
    holes_sum = float(sample["holes"].sum().item())
    liquid_sum = float(sample["liquid"].sum().item())
    instance_max = int(sample["instance_mask"].max().item())
    mcly_mask_sum = float(sample["mcly_mask"].sum().item())

    if not has_normals and normal_mask_sum != 0.0:
        _record_issue(issues, split, sample_meta, "has_normals is false but normal_mask is nonzero")
    if not has_alpha and alpha_sum != 0.0:
        _record_issue(issues, split, sample_meta, "has_alpha is false but alpha tensor is nonzero")
    if not has_holes and holes_sum != 0.0:
        _record_issue(issues, split, sample_meta, "has_holes is false but holes tensor is nonzero")
    if not has_liquid and liquid_sum != 0.0:
        _record_issue(issues, split, sample_meta, "has_liquid is false but liquid tensor is nonzero")
    if not has_instance and instance_max != 0:
        _record_issue(issues, split, sample_meta, "has_instance is false but instance_mask has nonzero ids")
    if not has_mcly and mcly_mask_sum != 0.0:
        _record_issue(issues, split, sample_meta, "has_mcly is false but mcly_mask is nonzero")


def _validate_split(dataset: V16Dataset, split: str, sample_limit: int) -> dict[str, Any]:
    sample_count = min(len(dataset), max(sample_limit, 0))
    issues: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []

    coverage = {
        "has_normals": 0,
        "has_alpha": 0,
        "has_holes": 0,
        "has_liquid": 0,
        "has_instance": 0,
        "has_mcly": 0,
    }

    for idx in range(sample_count):
        sample_meta = _entry_for_sample(dataset, idx)
        try:
            sample = dataset[idx]
        except Exception as ex:
            _record_issue(issues, split, sample_meta, f"dataset sample load failed: {ex}")
            continue

        for key in _TENSOR_SPECS:
            _check_tensor(key, sample[key], sample_meta, split, issues)

        _validate_sample_semantics(sample, split, sample_meta, issues)

        for key in coverage:
            if bool(sample[key]):
                coverage[key] += 1

        if len(examples) < 8:
            examples.append(
                {
                    **sample_meta,
                    "has_normals": bool(sample["has_normals"]),
                    "has_alpha": bool(sample["has_alpha"]),
                    "has_holes": bool(sample["has_holes"]),
                    "has_liquid": bool(sample["has_liquid"]),
                    "has_instance": bool(sample["has_instance"]),
                    "has_mcly": bool(sample["has_mcly"]),
                    "height_min": float(sample["height"].min().item()),
                    "height_max": float(sample["height"].max().item()),
                    "weight_mean": float(sample["weight"].mean().item()),
                }
            )

    return {
        "split": split,
        "dataset_len": len(dataset),
        "validated_samples": sample_count,
        "coverage": coverage,
        "issues": issues,
        "example_samples": examples,
    }


def _validate_dataloader_batch(
    dataset: V16Dataset,
    split: str,
    batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    if len(dataset) == 0:
        return {"split": split, "ok": True, "reason": "empty_split"}

    try:
        loader = DataLoader(
            dataset,
            batch_size=min(batch_size, len(dataset)),
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=False,
        )
        batch = next(iter(loader))

        return {
            "split": split,
            "ok": True,
            "batch_size": int(batch["input"].shape[0]),
            "tensor_shapes": {key: list(value.shape) for key, value in batch.items() if isinstance(value, torch.Tensor)},
        }
    except Exception as ex:
        return {
            "split": split,
            "ok": False,
            "error": str(ex),
        }


def _validate_model_forward(
    dataset: V16Dataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> dict[str, Any]:
    if len(dataset) == 0:
        return {"ok": True, "reason": "empty_split"}

    try:
        loader = DataLoader(
            dataset,
            batch_size=min(batch_size, len(dataset)),
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=False,
        )
        batch = next(iter(loader))
        model = V15Model().to(device)
        model.eval()

        with torch.no_grad():
            outputs = model(batch["input"].to(device))

        height, normals, alpha, holes, liquid, liquid_height, mcly = outputs
        return {
            "ok": True,
            "device": str(device),
            "output_shapes": {
                "height": list(height.shape),
                "normals": list(normals.shape),
                "alpha": list(alpha.shape),
                "holes": list(holes.shape),
                "liquid": list(liquid.shape),
                "liquid_height": list(liquid_height.shape),
                "mcly_logits": list(mcly.shape),
            },
        }
    except Exception as ex:
        return {
            "ok": False,
            "device": str(device),
            "error": str(ex),
            "note": "This includes model initialization failures such as missing pretrained ConvNeXt weights.",
        }


def main() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r"Object at .* is not recognized as a component of a Zarr hierarchy\.",
        category=UserWarning,
    )

    args = parse_args()
    builds = _resolve_builds(args, args.dataset_dir)
    if not builds:
        raise SystemExit("No V16 builds found to validate")

    device = _resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    build_summaries = [_open_index_summary(args.dataset_dir, build) for build in builds]

    train_ds = V16Dataset(
        dataset_dir=args.dataset_dir,
        builds=builds,
        split="train",
        val_fraction=args.val_fraction,
        seed=args.seed,
        augment=False,
    )
    val_ds = V16Dataset(
        dataset_dir=args.dataset_dir,
        builds=builds,
        split="val",
        val_fraction=args.val_fraction,
        seed=args.seed,
        augment=False,
    )

    train_validation = _validate_split(train_ds, "train", args.train_samples)
    val_validation = _validate_split(val_ds, "val", args.val_samples)

    batch_checks = {
        "train": _validate_dataloader_batch(train_ds, "train", args.batch_size, args.num_workers),
        "val": _validate_dataloader_batch(val_ds, "val", args.batch_size, args.num_workers),
    }

    model_forward: dict[str, Any]
    if args.skip_model_forward:
        model_forward = {"ok": True, "skipped": True}
    else:
        model_forward = _validate_model_forward(train_ds, args.batch_size, args.num_workers, device)

    issues = train_validation["issues"] + val_validation["issues"]
    overall_ok = bool(
        train_validation["validated_samples"] > 0
        and batch_checks["train"]["ok"]
        and batch_checks["val"]["ok"]
        and model_forward["ok"]
        and not issues
    )

    report = {
        "generated_at": datetime.now().isoformat(),
        "dataset_dir": str(args.dataset_dir),
        "builds": builds,
        "overall_ok": overall_ok,
        "trainer_contract": {
            "consumed_targets": ["height", "normals", "alpha", "holes", "liquid", "liquid_height", "mcly"],
            "dataset_exposes_instance_mask": True,
            "trainer_uses_instance_mask": False,
            "dataset_exposes_mcly_arrays": True,
            "trainer_uses_mcly_targets": True,
            "dataset_exposes_liquid_height": True,
            "trainer_uses_liquid_height": True,
            "note": "Current train_v16.py supervises MCLY and liquid-height targets from the V16 Zarr contract.",
        },
        "build_summaries": build_summaries,
        "split_validation": {
            "train": {k: v for k, v in train_validation.items() if k != "issues"},
            "val": {k: v for k, v in val_validation.items() if k != "issues"},
        },
        "batch_checks": batch_checks,
        "model_forward": model_forward,
        "issues": issues,
    }

    report_name = "all-builds" if len(builds) > 1 else builds[0]
    report_path = args.output_dir / f"{report_name}.training_readiness.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote {report_path}")
    print(f"overall_ok={overall_ok}")
    print(f"builds={builds}")
    print(f"train_samples={train_validation['validated_samples']} val_samples={val_validation['validated_samples']}")
    print(f"issues={len(issues)}")


if __name__ == "__main__":
    main()
