#!/usr/bin/env python3
"""User-run trainer for the v60 footprint-guided object marker."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from harvester.v60.object_marker import (  # noqa: E402
    EMBEDDING_DIM,
    FOOTPRINT_SIGNAL,
    IMAGE_SIGNAL,
    KNOWN_SIGNAL,
    ObjectMarkerError,
    ObjectMarkerNet,
    build_library_gallery_inputs,
    load_object_marker_manifest,
    marker_input_tensor,
    marker_loss,
    retrieve_library_identity,
    validate_object_marker_corpus,
)


def _plan(manifest: dict, validation: dict, args: argparse.Namespace) -> dict:
    model = ObjectMarkerNet(base=args.base, embedding_dim=EMBEDDING_DIM)
    train = [row for row in manifest["rows"] if row.get("split") == "train"]
    held_out = [row for row in manifest["rows"] if row.get("split") == "validation"]
    return {
        "schema": "v60-object-marker-plan-v1",
        "stage": "object_marker",
        "architecture": {
            "id": "object_marker_net",
            "input": "minimap_rgb_256(3)+object_candidate_mask_256(1)",
            "embedding_dim": EMBEDDING_DIM,
            "base": args.base,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        },
        "corpus_schema": manifest["schema"],
        "corpus_validation": validation,
        "split_counts": {"train": len(train), "validation": len(held_out)},
        "positive_counts": {
            "train": sum(int(row["known_object"]) for row in train),
            "validation": sum(int(row["known_object"]) for row in held_out),
        },
        "signals": [IMAGE_SIGNAL, FOOTPRINT_SIGNAL, KNOWN_SIGNAL],
        "identity_resolution": "frozen_v50_gallery_nearest_embedding",
        "epochs": args.epochs,
        "batch": args.batch,
        "seed": args.seed,
        "known_threshold": args.known_threshold,
        "metric_weight": args.metric_weight,
        "ground_truth_identity_as_input": False,
        "dry_run": True,
    }


class _MarkerDataset:
    def __init__(self, root: Path, rows: list[dict], identity_index: dict[str, int]) -> None:
        self.root = root
        self.rows = rows
        self.identity_index = identity_index

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with np.load(self.root / str(row["npz"]), allow_pickle=False) as payload:
            image = np.asarray(payload[IMAGE_SIGNAL], dtype=np.float32)
            footprint = np.asarray(payload[FOOTPRINT_SIGNAL], dtype=np.float32)
        tensor = marker_input_tensor(image, footprint).squeeze(0)
        known = int(row["known_object"])
        identity = self.identity_index.get(str(row.get("library_id")), -1) if known else -1
        return tensor, known, identity


def _metrics(model, rows, root: Path, gallery_images, gallery_masks, gallery_ids, device, threshold: float) -> dict:
    model.eval()
    gallery_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(gallery_images), 64):
            gallery_inputs = [
                marker_input_tensor(image, mask).squeeze(0)
                for image, mask in zip(
                    gallery_images[start : start + 64],
                    gallery_masks[start : start + 64],
                    strict=True,
                )
            ]
            gallery_tensor = torch.stack(gallery_inputs).to(device)
            gallery_chunks.append(model(gallery_tensor)["embedding"].detach().cpu().numpy())
    gallery_embeddings = np.concatenate(gallery_chunks, axis=0)
    known_targets: list[int] = []
    known_predictions: list[int] = []
    retrieval_top1 = 0
    retrieval_top5 = 0
    retrieval_count = 0
    by_kind: dict[str, dict[str, int]] = {}
    with torch.no_grad():
        for row in rows:
            with np.load(root / str(row["npz"]), allow_pickle=False) as payload:
                inputs = marker_input_tensor(payload[IMAGE_SIGNAL], payload[FOOTPRINT_SIGNAL]).to(device)
            outputs = model(inputs)
            confidence = float(torch.sigmoid(outputs["known_logit"])[0].item())
            known = int(row["known_object"])
            predicted = int(confidence >= 0.5)
            known_targets.append(known)
            known_predictions.append(predicted)
            kind = str(row.get("candidate_kind", "unknown"))
            bucket = by_kind.setdefault(kind, {"count": 0, "predicted_known": 0})
            bucket["count"] += 1
            bucket["predicted_known"] += predicted
            if known:
                retrieval_count += 1
                result = retrieve_library_identity(
                    outputs["embedding"][0].detach().cpu().numpy(),
                    gallery_embeddings,
                    gallery_ids,
                    known_confidence=confidence,
                    known_threshold=threshold,
                    top_k=5,
                )
                top_matches = [match["library_id"] for match in result["top_matches"]]
                target = str(row["library_id"])
                retrieval_top1 += int(top_matches and top_matches[0] == target)
                retrieval_top5 += int(target in top_matches)
    true_positive = sum(p == 1 and t == 1 for p, t in zip(known_predictions, known_targets, strict=True))
    predicted_positive = sum(known_predictions)
    actual_positive = sum(known_targets)
    return {
        "known_precision": true_positive / predicted_positive if predicted_positive else 0.0,
        "known_recall": true_positive / actual_positive if actual_positive else 0.0,
        "known_coverage": predicted_positive / max(len(rows), 1),
        "retrieval_top1": retrieval_top1 / retrieval_count if retrieval_count else 0.0,
        "retrieval_top5": retrieval_top5 / retrieval_count if retrieval_count else 0.0,
        "retrieval_positive_count": retrieval_count,
        "by_candidate_kind": by_kind,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train v60 object marker; dry-run unless --confirm-run")
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--object-library", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--metric-weight", type=float, default=1.0)
    parser.add_argument("--known-threshold", type=float, default=0.55)
    parser.add_argument("--seed", type=int, default=6001)
    parser.add_argument("--confirm-run", action="store_true", help="launch user-owned CUDA training")
    args = parser.parse_args()
    try:
        manifest = load_object_marker_manifest(args.corpus)
        validation = validate_object_marker_corpus(args.corpus)
    except (FileNotFoundError, ObjectMarkerError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    if not validation["valid"]:
        raise SystemExit("refusing to train invalid marker corpus: " + "; ".join(validation["failures"][:8]))
    train_rows = [row for row in manifest["rows"] if row.get("split") == "train"]
    validation_rows = [row for row in manifest["rows"] if row.get("split") == "validation"]
    if not train_rows or not validation_rows:
        raise SystemExit(f"marker corpus needs train and validation rows, got {len(train_rows)}/{len(validation_rows)}")
    plan = _plan(manifest, validation, args)
    print(json.dumps(plan, indent=2), flush=True)
    if not args.confirm_run:
        print("DRY RUN ONLY: add --confirm-run to launch user-owned CUDA training.", flush=True)
        return 0
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    from torch.utils.data import DataLoader

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; confirmed v60 marker training refuses CPU")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    positive_ids = sorted({str(row["library_id"]) for row in train_rows if int(row["known_object"])})
    identity_index = {value: index for index, value in enumerate(positive_ids)}
    dataset = _MarkerDataset(args.corpus, train_rows, identity_index)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=0, pin_memory=True)
    model = ObjectMarkerNet(base=args.base, embedding_dim=EMBEDDING_DIM).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    gallery_images, gallery_masks, gallery_ids = build_library_gallery_inputs(args.object_library)
    args.output.mkdir(parents=True)
    (args.output / "training_plan.json").write_text(json.dumps({**plan, "dry_run": False}, indent=2), encoding="utf-8")
    history: list[dict] = []
    best_top1 = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for inputs, known, identity in loader:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs.to(device, non_blocking=True))
            loss_values = marker_loss(
                outputs,
                known.to(device),
                identity.to(device),
                metric_weight=args.metric_weight,
            )
            loss_values["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss_values["total_loss"].detach().item()))
        metrics = _metrics(
            model, validation_rows, args.corpus, gallery_images, gallery_masks, gallery_ids, device, args.known_threshold
        )
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), **metrics}
        history.append(record)
        checkpoint = {
            "model": model.state_dict(),
            "architecture": {"base": args.base, "embedding_dim": EMBEDDING_DIM, "input_channels": 4},
            "epoch": epoch,
            "metrics": metrics,
            "gallery": {"source_library": str(args.object_library.resolve()), "count": len(gallery_ids)},
        }
        torch.save(checkpoint, args.output / "checkpoint_last.pt")
        if float(metrics["retrieval_top1"]) >= best_top1:
            best_top1 = float(metrics["retrieval_top1"])
            torch.save(checkpoint, args.output / "checkpoint_best.pt")
        print(
            f"[epoch {epoch:03d}] train_loss={np.mean(losses):.6f} "
            f"val_known_precision={metrics['known_precision']:.4f} "
            f"val_known_recall={metrics['known_recall']:.4f} "
            f"val_retrieval_top1={metrics['retrieval_top1']:.4f}",
            flush=True,
        )
    report = {
        "schema": "v60-object-marker-experiment-report-v1",
        "plan": plan,
        "best_retrieval_top1": best_top1,
        "final": history[-1] if history else {},
        "history": history,
        "per_signal": {"knownness": "known_precision/known_recall/known_coverage", "identity": "retrieval_top1/retrieval_top5"},
        "gate": {"promotion": "pending_user_review", "ground_truth_identity_as_input": False},
    }
    (args.output / "experiment_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
