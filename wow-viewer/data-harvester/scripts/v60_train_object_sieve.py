#!/usr/bin/env python3
"""User-run v60 object-sieve trainer for procedural or real-library-derived corpora."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from harvester.v60.object_library_sieve import (  # noqa: E402
    CLEAN_SIGNAL,
    INPUT_SIGNAL,
    MASK_SIGNAL,
    load_object_library_sieve_manifest,
    validate_object_library_sieve_corpus,
)
from harvester.v60.object_sieve_model import ObjectSieveNet, object_sieve_loss  # noqa: E402


def _metric(
    pred_clean: np.ndarray,
    pred_mask: np.ndarray,
    objectified: np.ndarray,
    clean: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    clean_mae = float(np.abs(pred_clean - clean).mean())
    predicted = pred_mask >= 0.5
    target = mask >= 0.5
    intersection = float(np.logical_and(predicted, target).sum())
    union = float(np.logical_or(predicted, target).sum())
    target_count = float(target.sum())
    predicted_count = float(predicted.sum())
    iou = 1.0 if union == 0.0 else intersection / union
    nonempty_iou = None if target_count == 0.0 else intersection / max(union, 1.0)
    precision = 1.0 if predicted_count == 0.0 and target_count == 0.0 else intersection / max(predicted_count, 1.0)
    recall = 1.0 if target_count == 0.0 and predicted_count == 0.0 else intersection / max(target_count, 1.0)
    total_pixels = float(target.size)
    return {
        "clean_mae": clean_mae,
        "contaminated_input_mae": float(np.abs(objectified - clean).mean()),
        "mask_iou": float(iou),
        "nonempty_mask_iou": float(nonempty_iou) if nonempty_iou is not None else float("nan"),
        "mask_precision": float(precision),
        "mask_recall": float(recall),
        "target_coverage": float(target.mean()),
        "zero_mask_baseline_iou": 1.0 if target_count == 0.0 else 0.0,
        "all_foreground_baseline_iou": target_count / total_pixels,
    }


def _aggregate(records: list[dict[str, float]]) -> dict[str, float]:
    if not records:
        return {}
    output: dict[str, float] = {}
    for key in records[0]:
        values = np.asarray([record[key] for record in records], dtype=np.float64)
        values = values[np.isfinite(values)]
        if len(values):
            output[key] = float(values.mean())
    return output


def _evaluate(model, rows: list[dict], root: Path, device) -> dict:
    import torch

    model.eval()
    by_regime: dict[str, list[dict[str, float]]] = {}
    all_records: list[dict[str, float]] = []
    with torch.no_grad():
        for row in rows:
            with np.load(root / str(row["npz"]), allow_pickle=False) as payload:
                objectified = np.asarray(payload[INPUT_SIGNAL], dtype=np.float32)
                clean = np.asarray(payload[CLEAN_SIGNAL], dtype=np.float32)
                mask = np.asarray(payload[MASK_SIGNAL], dtype=np.float32)
            x = torch.from_numpy(objectified[None, None]).to(device)
            prediction = model(x)
            pred_clean = prediction.clean_terrain.squeeze().cpu().numpy()
            pred_mask = torch.sigmoid(prediction.contamination_logits).squeeze().cpu().numpy()
            record = _metric(pred_clean, pred_mask, objectified, clean, mask)
            all_records.append(record)
            by_regime.setdefault(str(row["placement_regime"]), []).append(record)
    return {
        "overall": _aggregate(all_records),
        "by_regime": {regime: _aggregate(records) for regime, records in sorted(by_regime.items())},
        "row_count": len(rows),
    }


def _plan(*, manifest: dict, validation: dict, variant: str, epochs: int, batch: int, mask_weight: float) -> dict:
    model = ObjectSieveNet(variant=variant)  # type: ignore[arg-type]
    rows = manifest["rows"]
    train_rows = [row for row in rows if row.get("split") == "train"]
    validation_rows = [row for row in rows if row.get("split") == "validation"]
    return {
        "schema": "v60-object-sieve-training-plan-v1",
        "corpus_schema": manifest["schema"],
        "corpus_root": str(Path(validation["corpus_root"]).resolve()),
        "variant": variant,
        "architecture": {"class": "ObjectSieveNet", "input": "1x256x256", "parameters": sum(p.numel() for p in model.parameters())},
        "clean_output": "input_plus_zero_initialized_residual",
        "split_counts": {"train": len(train_rows), "validation": len(validation_rows)},
        "regimes": sorted({str(row["placement_regime"]) for row in rows}),
        "library_object_count": validation.get("library_object_count"),
        "epochs": epochs,
        "batch": batch,
        "mask_weight": mask_weight,
        "input_signal": INPUT_SIGNAL,
        "targets": [CLEAN_SIGNAL, MASK_SIGNAL],
        "ground_truth_mask_as_input": False,
        "validation": validation,
        "cuda_required_for_confirmed_run": True,
        "dry_run": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a v60 object sieve; dry-run unless --confirm-run")
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--variant", choices=["clean_only", "auxiliary_mask_loss", "predicted_mask_guided"], default="auxiliary_mask_loss")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mask-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=6001)
    parser.add_argument("--confirm-run", action="store_true", help="launch user-owned CUDA training")
    args = parser.parse_args()
    manifest = load_object_library_sieve_manifest(args.corpus)
    validation = validate_object_library_sieve_corpus(args.corpus)
    if not validation["valid"]:
        raise SystemExit("refusing to train invalid corpus: " + "; ".join(validation["failures"][:8]))
    train_rows = [row for row in manifest["rows"] if row.get("split") == "train"]
    validation_rows = [row for row in manifest["rows"] if row.get("split") == "validation"]
    if not train_rows or not validation_rows:
        raise SystemExit(f"corpus needs train and validation rows, got train={len(train_rows)} validation={len(validation_rows)}")
    plan = _plan(manifest=manifest, validation=validation, variant=args.variant, epochs=args.epochs, batch=args.batch, mask_weight=args.mask_weight)
    print(json.dumps(plan, indent=2), flush=True)
    if not args.confirm_run:
        print("DRY RUN ONLY: add --confirm-run to launch user-owned CUDA training.", flush=True)
        return 0

    import torch
    from torch.utils.data import DataLoader, Dataset

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; confirmed v60 training refuses CPU")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    class CorpusDataset(Dataset):
        def __init__(self, selected_rows: list[dict]) -> None:
            self.rows = selected_rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int):
            row = self.rows[index]
            with np.load(args.corpus / str(row["npz"]), allow_pickle=False) as payload:
                objectified = np.asarray(payload[INPUT_SIGNAL], dtype=np.float32)
                clean = np.asarray(payload[CLEAN_SIGNAL], dtype=np.float32)
                mask = np.asarray(payload[MASK_SIGNAL], dtype=np.float32)
            return (
                torch.from_numpy(objectified[None]),
                torch.from_numpy(clean[None]),
                torch.from_numpy(mask[None]),
            )

    device = torch.device("cuda")
    model = ObjectSieveNet(variant=args.variant).to(device)  # type: ignore[arg-type]
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        CorpusDataset(train_rows),
        batch_size=args.batch,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        generator=generator,
    )
    args.output.mkdir(parents=True)
    (args.output / "training_plan.json").write_text(json.dumps({**plan, "dry_run": False}, indent=2), encoding="utf-8")
    best_clean = float("inf")
    history: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for objectified, clean, mask in loader:
            objectified, clean, mask = objectified.to(device), clean.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(objectified)
            losses_dict = object_sieve_loss(prediction, clean, mask, args.variant, mask_weight=args.mask_weight)  # type: ignore[arg-type]
            losses_dict["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(losses_dict["total_loss"].detach().item()))
        metrics = _evaluate(model, validation_rows, args.corpus, device)
        record = {"epoch": epoch, "train_total_loss": float(np.mean(losses)), **metrics}
        history.append(record)
        torch.save({"model": model.state_dict(), "variant": args.variant, "epoch": epoch, "metrics": metrics}, args.output / "checkpoint_last.pt")
        clean_score = float(metrics["overall"]["clean_mae"])
        if clean_score < best_clean:
            best_clean = clean_score
            torch.save({"model": model.state_dict(), "variant": args.variant, "epoch": epoch, "metrics": metrics}, args.output / "checkpoint_best.pt")
        print(f"[epoch {epoch:03d}] train_loss={np.mean(losses):.6f} val_clean_mae={clean_score:.6f} val_nonempty_mask_iou={metrics['overall'].get('nonempty_mask_iou', float('nan')):.4f}", flush=True)
    report = {
        "schema": "v60-object-sieve-experiment-report-v1",
        "plan": plan,
        "best_clean_mae": best_clean,
        "final": history[-1] if history else {},
        "history": history,
        "gate": {
            "clean_beats_contaminated_input_identity": bool(
                history and best_clean < float(history[-1]["overall"]["contaminated_input_mae"])
            ),
            "best_clean_mae": best_clean,
            "identity_baseline_mae": history[-1]["overall"].get("contaminated_input_mae") if history else None,
        },
        "per_signal": {"clean_terrain": "reported as clean_mae", "contamination_mask": "reported as mask_iou/nonempty_mask_iou/precision/recall"},
    }
    (args.output / "experiment_report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
