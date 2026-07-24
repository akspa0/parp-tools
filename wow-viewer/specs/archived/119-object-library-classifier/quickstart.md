# Quickstart: Object-Library Segmentation & Classifier (Spec 119)

**Date**: 2026-07-23 | **Spec**: [spec.md](spec.md) | **CLI contract**: [contracts/cli-contract.md](contracts/cli-contract.md)

All commands are PowerShell-ready and run from the data-harvester dir. **Training is user-run**
(FR-010): every training command prints its plan and exits unless you pass `--confirm-run`.

## Prerequisites

1. A built object-library zarr (from the Spec 118 `capture-objects` pipeline, now WMO-inclusive).
   If you only have the smoke store, build the full one first (user-run harvest):
   ```powershell
   cd i:\parp\parp-tools\wow-viewer\data-harvester
   uv run python scripts/build_object_library.py `
     --from-harvest-stream `
     --harvest-project i:\parp\parp-tools\wow-viewer\tools\harvest\WowViewer.Tool.Harvest\WowViewer.Tool.Harvest.csproj `
     --client-root H:\CLIENTS\0_5_3_3368 `
     --output-root i:\parp\parp-tools\wow-viewer\output\object-library `
     --run-name objlib_0_5_3_3368 `
     --run
   ```
   This guide assumes the store is at
   `i:\parp\parp-tools\wow-viewer\output\object-library\objlib_0_5_3_3368.zarr`.

## §1 — Build the family-isolated held-out split (US1 prerequisite)

```powershell
uv run python scripts/spec119_build_split.py `
  --store i:\parp\parp-tools\wow-viewer\output\object-library\objlib_0_5_3_3368.zarr `
  --output i:\parp\parp-tools\wow-viewer\output\object-library\objlib_split.json `
  --held-out-fraction 0.2 `
  --seed 0 `
  --write
```
Dry-run (omit `--write`) first to see family/row counts and confirm `verified_violation_count=0`
(the leakage check — FR-004). The script **refuses to write** a split with violations.

## §2 — Train the classifier (US1, P1)

Dry-run first (prints param count, majority-class baseline, class weights, train/held-out counts):
```powershell
uv run python scripts/spec119_train_classifier.py `
  --store i:\parp\parp-tools\wow-viewer\output\object-library\objlib_0_5_3_3368.zarr `
  --split i:\parp\parp-tools\wow-viewer\output\object-library\objlib_split.json `
  --output-root i:\parp\parp-tools\wow-viewer\output\object-library\runs `
  --run-name classifier_v1 `
  --base 16 --epochs 60 --lr 1e-3 --pct-start 0.1
```
Then actually train (USER runs CUDA):
```powershell
uv run python scripts/spec119_train_classifier.py `
  --store i:\parp\parp-tools\wow-viewer\output\object-library\objlib_0_5_3_3368.zarr `
  --split i:\parp\parp-tools\wow-viewer\output\object-library\objlib_split.json `
  --output-root i:\parp\parp-tools\wow-viewer\output\object-library\runs `
  --run-name classifier_v1 `
  --base 16 --epochs 60 --lr 1e-3 --pct-start 0.1 `
  --confirm-run
```
**Gate (SC-001)**: held-out top-1 accuracy must be ≥15pp above the majority-class baseline
(reported in the run record's `baselines`). If it isn't, do not proceed to US2 — diagnose the data
(class imbalance, label noise, leakage) first.

## §3 — Train the segmenter (US2, P2)

Only after §2 passes SC-001. Dry-run, then train:
```powershell
uv run python scripts/spec119_train_segmenter.py `
  --store i:\parp\parp-tools\wow-viewer\output\object-library\objlib_0_5_3_3368.zarr `
  --split i:\parp\parp-tools\wow-viewer\output\object-library\objlib_split.json `
  --output-root i:\parp\parp-tools\wow-viewer\output\object-library\runs `
  --run-name segmenter_v1 `
  --base 16 --epochs 60 --lr 1e-3 --pct-start 0.1 `
  --confirm-run
```
**Gate (SC-002)**: held-out per-pixel IoU must be ≥0.20 above the better of the
all-foreground/all-background trivial baselines (reported in `baselines`).

## §4 — Loose-image inference (FR-013)

Run a frozen checkpoint on a loose PNG with no store/ground truth:
```powershell
# Classifier
uv run python scripts/spec119_infer.py `
  --checkpoint i:\parp\parp-tools\wow-viewer\output\object-library\runs\classifier_v1\classifier.pt `
  --inputs i:\parp\parp-tools\wow-viewer\output\object-library-smoke\img_0.png `
  --output i:\parp\parp-tools\wow-viewer\output\object-library\runs\classifier_v1\pred_0.json

# Segmenter
uv run python scripts/spec119_infer.py `
  --checkpoint i:\parp\parp-tools\wow-viewer\output\object-library\runs\segmenter_v1\segmenter.pt `
  --inputs i:\parp\parp-tools\wow-viewer\output\object-library-smoke\img_0.png `
  --output i:\parp\parp-tools\wow-viewer\output\object-library\runs\segmenter_v1\
```

## §5 — Quality lens (US3, P3)

Only after §2 (needs a frozen classifier). Dry-run, then write:
```powershell
uv run python scripts/spec119_quality_lens.py `
  --store i:\parp\parp-tools\wow-viewer\output\object-library\objlib_0_5_3_3368.zarr `
  --checkpoint i:\parp\parp-tools\wow-viewer\output\object-library\runs\classifier_v1\classifier.pt `
  --output-root i:\parp\parp-tools\wow-viewer\output\object-library\runs `
  --run-name classifier_v1 `
  --write
```
Writes `embeddings.parquet` + `quality_report.json` (mislabels, near-duplicate clusters,
low-coverage flags). **Gate (SC-004)**: manually inspect the top-flagged mislabels — ≥50% should be
genuinely mislabeled or genuinely confusable.

## Out of scope (do not pull in)

- Minimap-crop-to-library-asset retrieval (FR-012 — that is the Spec 118 minimap chain's job).
- Multi-variant captures (rotated/scaled) — the library currently has one variant per asset; the
  split isolates by asset so this stays valid if multi-variant is added later.
- A finer-than-coarse taxonomy as the primary metric (SC-001 is on the coarse split; `--fine-labels`
  is a secondary, heuristic-labeled cut).
