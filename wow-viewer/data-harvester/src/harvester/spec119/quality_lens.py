"""Spec 119 US3 library quality lens (T025/T026, FR-008/FR-009).

Runs the FROZEN classifier over the full library and emits (a) per-asset penultimate-layer
embeddings, (b) a disagreement report of candidate mislabels sorted by wrong-class confidence,
(c) near-duplicate pairs by embedding cosine similarity, and (d) low-coverage flags. Embeddings
are deterministic from a frozen checkpoint (``eval()`` + ``no_grad()`` + no stochastic ops).
The retrieval integration itself is explicitly out of scope (FR-012) — this delivers the
embedding + the quality report only.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from harvester.spec119.infer import load_classifier_checkpoint
from harvester.spec119.library_data import (
    captured_rows,
    load_asset_rows,
    open_library,
    read_image,
    row_coverages,
)
from harvester.spec119.object_library_contract import ObjectLibraryContractError


def compute_embeddings(model, group, rows: Sequence[dict[str, Any]]) -> np.ndarray:
    """Penultimate-layer vectors, one per row, in row order (deterministic, FR-009)."""
    import torch

    model.eval()
    vectors = []
    with torch.no_grad():
        for row in rows:
            image = read_image(group, row["_row_index"])
            x = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
            vectors.append(model.embedding(x).squeeze(0).numpy().astype(np.float32))
    return np.stack(vectors) if vectors else np.zeros((0, 0), dtype=np.float32)


def predict_all(model, class_index: dict[str, int], group, rows: Sequence[dict[str, Any]]):
    """Per-row (predicted_index, confidence, probabilities) in row order."""
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        for row in rows:
            image = read_image(group, row["_row_index"])
            x = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
            probs = torch.softmax(model(x).squeeze(0), dim=0).numpy()
            out.append((int(probs.argmax()), float(probs.max()), probs))
    return out


def find_mislabels(
    rows: Sequence[dict[str, Any]],
    predictions: Sequence[tuple[int, float, Any]],
    labeled_indices: Sequence[int],
    class_index: dict[str, int],
) -> list[dict[str, Any]]:
    """Candidate mislabels sorted by confidence in the wrong class, descending (FR-008)."""
    index_class = {index: name for name, index in class_index.items()}
    report = []
    for row, (predicted, confidence, _probs), labeled in zip(
        rows, predictions, labeled_indices, strict=True
    ):
        if predicted != labeled:
            report.append(
                {
                    "library_id": row["library_id"],
                    "asset_path": row["normalized_asset_path"],
                    "labeled_class": index_class[labeled],
                    "predicted_class": index_class[predicted],
                    "confidence": confidence,
                }
            )
    report.sort(key=lambda entry: entry["confidence"], reverse=True)
    return report


def find_near_duplicates(
    embeddings: np.ndarray,
    library_ids: Sequence[str],
    threshold: float = 0.95,
    top_k: int = 200,
) -> list[dict[str, Any]]:
    """Top-K embedding pairs above a cosine-similarity threshold (FR-008)."""
    if embeddings.shape[0] < 2:
        return []
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    similarity = (embeddings / norms) @ (embeddings / norms).T
    pairs = []
    n = embeddings.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            score = float(similarity[i, j])
            if score >= threshold:
                pairs.append(
                    {
                        "library_id_a": library_ids[i],
                        "library_id_b": library_ids[j],
                        "cosine_similarity": score,
                    }
                )
    pairs.sort(key=lambda entry: entry["cosine_similarity"], reverse=True)
    return pairs[:top_k]


def flag_low_coverage(
    rows: Sequence[dict[str, Any]], coverages: Sequence[float], threshold: float
) -> list[dict[str, Any]]:
    """Captures below the blank threshold, flagged separately from mislabels (US3)."""
    return [
        {
            "library_id": row["library_id"],
            "asset_path": row["normalized_asset_path"],
            "coverage": float(coverage),
        }
        for row, coverage in zip(rows, coverages, strict=True)
        if coverage < threshold
    ]


def main() -> int:
    """CLI per contracts/cli-contract.md §5 (dry-run-first; --write emits the artifacts)."""
    ap = argparse.ArgumentParser(
        description="Spec 119 library quality lens (frozen classifier -> embeddings + report)"
    )
    ap.add_argument("--store", required=True, type=Path, help="object-library zarr (read-only)")
    ap.add_argument("--checkpoint", required=True, type=Path, help="frozen classifier checkpoint")
    ap.add_argument("--output-root", required=True, type=Path)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--near-duplicate-threshold", type=float, default=0.95)
    ap.add_argument("--near-duplicate-top-k", type=int, default=200)
    ap.add_argument("--blank-threshold", type=float, default=0.01)
    ap.add_argument("--write", action="store_true",
                    help="write embeddings.parquet + quality_report.json; without this flag only "
                         "print summary counts")
    args = ap.parse_args()

    try:
        group = open_library(args.store)
        rows = captured_rows(load_asset_rows(args.store))
        model, class_index = load_classifier_checkpoint(args.checkpoint)
    except ObjectLibraryContractError as exc:
        raise SystemExit(str(exc)) from exc

    coverages = row_coverages(group, rows)
    index_class = {index: name for name, index in class_index.items()}
    # Labeled class per row: blank rows are 'empty' (D-04), else the checkpoint's coarse label.
    from harvester.spec119.object_library_contract import coarse_label_for_row

    labeled = [
        class_index.get(coarse_label_for_row(str(row["asset_type"]), coverage, args.blank_threshold), 0)
        for row, coverage in zip(rows, coverages, strict=True)
    ]
    predictions = predict_all(model, class_index, group, rows)
    embeddings = compute_embeddings(model, group, rows)
    library_ids = [row["library_id"] for row in rows]

    mislabels = find_mislabels(rows, predictions, labeled, class_index)
    near_duplicates = find_near_duplicates(
        embeddings, library_ids,
        threshold=args.near_duplicate_threshold, top_k=args.near_duplicate_top_k,
    )
    low_coverage = flag_low_coverage(rows, coverages, args.blank_threshold)
    summary = {
        "total": len(rows),
        "mislabel_count": len(mislabels),
        "near_duplicate_pair_count": len(near_duplicates),
        "low_coverage_count": len(low_coverage),
    }
    print(json.dumps({"schema": "v119-quality-plan-v1", "summary": summary}, indent=2), flush=True)
    if not args.write:
        print("DRY RUN ONLY: add --write to emit embeddings.parquet + quality_report.json.", flush=True)
        return 0

    import pyarrow as pa
    import pyarrow.parquet as pq

    output = Path(args.output_root) / args.run_name
    output.mkdir(parents=True, exist_ok=True)
    embedding_rows = [
        {
            "library_id": row["library_id"],
            "embedding": embeddings[i].tolist(),
            "predicted_class": index_class[predictions[i][0]],
            "labeled_class": index_class[labeled[i]],
            "agreement": bool(predictions[i][0] == labeled[i]),
        }
        for i, row in enumerate(rows)
    ]
    pq.write_table(pa.Table.from_pylist(embedding_rows), output / "embeddings.parquet")
    report = {
        "schema": "v119-quality-report-v1",
        "store": str(args.store.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "near_duplicate_threshold": args.near_duplicate_threshold,
        "blank_threshold": args.blank_threshold,
        "summary": summary,
        "mislabels": mislabels,
        "near_duplicates": near_duplicates,
        "low_coverage": low_coverage,
    }
    (output / "quality_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {output / 'embeddings.parquet'}", flush=True)
    print(f"wrote {output / 'quality_report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "compute_embeddings",
    "find_mislabels",
    "find_near_duplicates",
    "flag_low_coverage",
    "main",
    "predict_all",
]
