# CLI Contract: Object-Library Segmentation & Classifier (Spec 119)

**Date**: 2026-07-23 | **Spec**: [spec.md](spec.md) | **Data model**: [data-model.md](data-model.md)

All CLIs live in `wow-viewer/data-harvester/scripts/` and are run via
`cd wow-viewer/data-harvester && uv run python scripts/<script>.py ...`. All are **dry-run-first**
(FR-010): without an explicit `--confirm-run` (training) or `--write` (derivation), they print the
plan and exit without writing or launching heavy work.

## 1. Split builder — `spec119_build_split.py`

Builds the family-isolated held-out split (D-01) from an object-library zarr.

```
uv run python scripts/spec119_build_split.py \
  --store <path-to-objlib.zarr> \
  --output <path-to-split.json> \
  --held-out-fraction 0.2 \
  --seed 0 \
  [--write]
```

- **Without `--write`**: prints the family count, train/held-out row counts, and the
  `verified_violation_count` (leakage check, must be 0) and exits.
- **With `--write`**: writes the `HeldOutSplit` JSON (data-model).
- **Refuses** if `verified_violation_count > 0` (a leaky split is an error, not a warning — FR-004).

## 2. Classifier trainer — `spec119_train_classifier.py`

```
uv run python scripts/spec119_train_classifier.py \
  --store <path-to-objlib.zarr> \
  --split <path-to-split.json> \
  --output-root <dir> \
  --run-name <name> \
  --base 16 \
  --epochs 60 \
  --lr 1e-3 \
  --pct-start 0.1 \
  --blank-threshold 0.01 \
  [--fine-labels] \
  [--confirm-run]
```

- **Without `--confirm-run`**: prints the plan (param count, train/held-out counts, majority-class
  baseline, class weights) and exits (FR-010).
- **With `--confirm-run`**: trains (USER runs CUDA), writes `ClassifierCheckpoint` +
  `ModelStageRun` (`stage=object_library_classifier`, `promotion_verdict=pending`).
- `--fine-labels`: switches the target to the heuristic `FineFamilyLabel` (D-03); the run record
  marks it heuristic; SC-001 is still reported on the coarse split for comparability.
- Records the majority-class baseline in `baselines` (FR-005) and per-class precision/recall in
  `metrics` (FR-007).

## 3. Segmenter trainer — `spec119_train_segmenter.py`

```
uv run python scripts/spec119_train_segmenter.py \
  --store <path-to-objlib.zarr> \
  --split <path-to-split.json> \
  --output-root <dir> \
  --run-name <name> \
  --base 16 \
  --epochs 60 \
  --lr 1e-3 \
  --pct-start 0.1 \
  --blank-threshold 0.01 \
  [--confirm-run]
```

- Same dry-run-first gate. Writes `SegmenterCheckpoint` + `ModelStageRun`
  (`stage=object_library_segmenter`).
- Records the all-foreground and all-background trivial IoU baselines in `baselines` (SC-002) and
  per-coverage-bucket IoU in `metrics`.
- Excludes blank captures from training (D-04); the exclusion count is in the run record.

## 4. Loose-image inference — `spec119_infer.py`

Runs a frozen checkpoint on one or more loose PNGs with no store/ground truth (FR-013).

```
uv run python scripts/spec119_infer.py \
  --checkpoint <path-to-classifier.pt|segmenter.pt> \
  --inputs <png1> [<png2> ...] \
  --output <path-to-predictions.json|mask-png-dir>
```

- Classifier → writes a JSON of `{input, predicted_class, confidence, per_class_probs}`.
- Segmenter → writes a `<input-stem>_mask.png` (255 foreground / 0 background) per input.
- Reconstructs the architecture from the checkpoint's `base` (D-02); refuses a checkpoint whose
  `architecture` block is missing `base`.

## 5. Quality lens — `spec119_quality_lens.py`

Runs the frozen classifier over the full library → embeddings + disagreement report (US3, FR-008).

```
uv run python scripts/spec119_quality_lens.py \
  --store <path-to-objlib.zarr> \
  --checkpoint <path-to-classifier.pt> \
  --output-root <dir> \
  --run-name <name> \
  --near-duplicate-threshold 0.95 \
  --near-duplicate-top-k 200 \
  --blank-threshold 0.01 \
  [--write]
```

- **Without `--write`**: prints summary counts (mislabel count, near-duplicate pair count,
  low-coverage count) and exits.
- **With `--write`**: writes `embeddings.parquet` + `quality_report.json` (data-model).
- Deterministic from a frozen checkpoint (FR-009): `eval()`, `no_grad()`, no stochastic ops.

## Shared conventions

- All paths are explicit CLI args (no hardcoded client/store paths — Rule 4/9).
- `--store` is read-only; no script mutates the source zarr (FR-011).
- Run records use `v50-model-stage-run-v1` (D-05); the two new stages
  (`object_library_classifier`, `object_library_segmenter`) are added to
  `model_stage_contract.STAGES`.
- Training scripts reuse `harvester.v50.lr_schedule.make_onecycle_scheduler` with the
  warmup-aware stale counter (D-05, the Spec 117 fix).
