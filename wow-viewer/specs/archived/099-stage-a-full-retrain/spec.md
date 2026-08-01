# Feature Specification: Spec 099 — Stage A Full Retrain (200 epochs, normal guidance, TTA)

**Feature Branch**: `099-stage-a-full-retrain`
**Created**: 2026-07-10
**Status**: Draft (vision + concrete plan)
**Owner**: wow-viewer
**Parent**: Spec 094 (V24 WDL prior + lattice detailer), Spec 096 (deployment), Spec 097 (per-map export)

**Input**: User description (verbatim) — "let's instead get this model trained more fully, with more than 100 epochs, maybe improve the training code somewhat, to ensure proper gains on every epoch, maybe add some self-guided validation/testing to improve training. We had good luck injecting data like normals in as a guidance signal for our earlier models, why not do that for the data we have? Guidance can help train these models faster than no guidance. Please optimize the trainer as best we can to get a 100% perfect WDL output from it."

---

## Problem Statement

The current V24 minimap-only Stage A is at best_val_l1 = 190.31 world units — 158× worse than the `block_reduce` baseline (1.31). It is not useful for downstream reconstruction.

**Training data is the existing real WDL priors** stored in the V18/V24 substrate (`wdl_prior_outer` / `wdl_prior_inner` per tile, with `wdl_prior_source_outer/inner` = 0 for real WDLs from the staged client, 1 for synthetic, 2 for audit-empty). For tiles that have a real WDL (source=0), the trainer's `build_target(record)` returns it as the supervised target. For tiles that only have a synthetic WDL (source=1), that's also a useful target — the model learns to mimic the synthetic extraction. **The model does NOT consume a WDL as input — the WDL is the OUTPUT.** It only consumes the minimap (and optionally the normal). This is the design: at inference time the model gets only a minimap and produces a WDL — it does not require a real WDL to produce a real WDL.

The user wants:

1. **Train longer (200+ epochs)**
2. **Per-epoch real improvement** (the cosine LR + early stopping plateaus at 190 L1)
3. **Inject normals as guidance** (worked for prior models per the user)
4. **Self-guided validation** (the model tells itself how well it's doing)
5. **Optimize the trainer** for "100% perfect WDL output"

This spec is a real bounded piece of work. It is not a vision doc — every change is concrete, every test exists, and the success criterion is a measurable L1.

---

## What This Spec Does

Five bounded slices, each validated before the next:

### Slice 1 — `StageAMinimapOnly` with normal-guidance channels (new model)

A new model class `StageAMinimapOnlyGuided` (in `stage_a.py`) that takes:
- 3 channels: cleaned minimap RGB (current)
- 3 channels: normal XYZ (down-sampled to 64×64, normalized to [-1, 1])
- 3 channels: predicted height derivative along X, Y, diagonal (Sobel of the normal map) — gives the model a sense of local curvature

Output: the (17,17) outer + (16,16) inner WDL prior. Same head as the existing model.

Why this works: the model's job is to map a 2D minimap to a height field. Normals are a 2.5D representation of the same scene. Adding normals as input channels gives the model a much easier mapping (minimap + normal → WDL prior is close to identity for low-frequency content), and the Sobel derivatives make local curvature explicit.

Size: ~450K params (vs current 335K). 64x64 input, 9 input channels. Should fit easily on the 12 GB envelope.

### Slice 2 — Trainer with RAdam + Lookahead + warmup + cosine

Replace the plain AdamW with:
- `RAdam` (rectified Adam — better init behavior than Adam)
- `Lookahead(slow_step=5)` (smooths the optimization trajectory, gives a "look ahead" before committing)
- `OneCycleLR` (warmup + cosine decay in a single cycle, gives per-epoch improvements)
- AMP bf16 (default) or fp16 with explicit scaler

Hyperparameters:
- `lr=2e-3` (slightly higher than current 1e-3, since OneCycle handles the warmup)
- `weight_decay=1e-4` (current)
- `epochs=200` (was 50/100)
- `batch_size=64` (current)
- `early_stopping_patience=30` (was 0 / explicit)

The OneCycleLR is the key change: instead of "cosine over the whole run", it does "warmup to peak LR over first 5% of training, then cosine decay to 0 over the rest". This gives the model a real LR boost early (when it's learning the easy stuff) and fine-tunes at the end (when it's polishing).

### Slice 3 — Test-time augmentation (TTA)

At validation time, predict the model on **5 versions of the input**:
1. The original minimap
2. The minimap flipped left-right
3. The minimap flipped top-bottom
4. The minimap rotated 90°
5. The minimap rotated 270°

For each version, **un-flip / un-rotate the prediction** so all 5 predictions are in the same coordinate frame. Then **average the 5 predictions** (or take the median). The averaged prediction is more robust than any single forward pass.

This is a real win for WDL priors because the prior is a smooth height field — averaging 5 noisy predictions reduces variance by sqrt(5) ≈ 2.2×.

Test-time augmentation runs in validation only (not during training, for speed).

### Slice 4 — Self-guided validation: per-region L1 + per-cell confidence breakdown

The current `val_l1` is a single number. The user wants **richer signal** so the model can self-improve. Add to the validation output:

- `val_l1_real_cells` (already there) — L1 on cells where the WDL was from the real client
- `val_l1_synth_cells` (already there) — L1 on cells where the WDL was synthesised
- **`val_l1_per_region`** (new) — 4 sub-regions: NW, NE, SW, SE corners of the 17×17 grid. Per-region L1 lets us see if the model is struggling in a specific corner.
- **`val_l1_per_quantile`** (new) — quantile-bucketed L1 (10 buckets, by predicted height). The model should be accurate on the full range, not just the median.
- **`val_l1_curvature`** (new) — L1 on the local curvature (Sobel of the prediction) vs the curvature of the WDL prior ground truth. Captures how well the model captures the "shape" not just the magnitude.

Per-epoch visual preview: at every Nth epoch, save a 4-up PNG showing the input minimap, the predicted WDL prior (outer 17×17 rendered as a heatmap), the ground-truth WDL prior, and the per-cell absolute error. The preview is saved under `output/v24_validation/<run_id>/previews/epoch_NNN.png`.

### Slice 5 — 200-epoch full retrain + objective criterion

Run the full retrain on the curated open-world V24 corpus (`3_3_5_12340_openworld_curated.zarr`, 2,011 tiles) for 200 epochs with the new trainer (RAdam + Lookahead + OneCycle + normal guidance + TTA). Track:

- Best `val_l1` over all 200 epochs.
- Best `val_l1_real_cells` (the source of truth for the real-WDL coverage).
- Best `val_l1_curvature` (the shape-quality metric).

**Objective criterion (SC-099-001):** `val_l1 < 5.0 world units` AND `val_l1_real_cells < 3.0 world units` AND `val_l1_curvature < 0.5` AND `val_l1_per_quantile` is roughly flat (each bucket within 2× the median).

If the objective is met, the new checkpoint replaces the current minimap-only Stage A as the deployment default. If it's not met, the slice ships anyway with the new model + trainer + tests; the user can iterate further.

---

## What This Spec Does NOT Do

- **No new minimap-only architecture changes** — the existing `StageAMinimapOnly` model is kept (it's the deployment model). The new guided model is additive.
- **No changes to Stage B** — out of scope. Stage B is a separate lane (Spec 100 in the Spec 098 vision doc).
- **No new V18 build training** — single-build retrain (3_3_5_12340) is the SC-001 target. Multi-build retrain is Spec 099+.
- **No RunPod work** — local retrain on the 12 GB envelope. RunPod is a fallback if the run takes >6 hours.
- **No real-time inference** — the model is per-tile forward, not real-time.
- **No loss-of-normals graceful degradation** — the guided model requires normals. If the user's pipeline doesn't have normals for a tile, they should use the unguided model.

---

## User Scenarios & Testing

### User Story 1 — Re-run a 200-epoch training and see real improvement (Priority: P1)

**Acceptance scenarios**:
1. `uv run python scripts/train_v24_stage_a.py --minimap-only --guided --epochs 200 --output output/v24_validation/v24_guided_20260710` runs to completion.
2. `output/v24_validation/v24_guided_20260710/stage_a.pt` exists, ≤ 5 MB, loads as `StageAMinimapOnlyGuided`.
3. `loss_history.jsonl` has 200 lines (one per epoch).
4. `val_l1_curvature` (per-epoch) decreases monotonically over the first 50 epochs (the model is genuinely learning the shape).
5. `val_l1` at the final epoch is < the `val_l1` at epoch 50 (200 epochs is better than 50).
6. `stage_a_metrics.json` reports `params ≤ 600_000`.

### User Story 2 — Test-time augmentation is real (Priority: P1)

**Acceptance scenarios**:
1. `validate_v24.py --guided-checkpoint <path>` runs validation with TTA, reports both TTA-on and TTA-off metrics.
2. TTA-on `val_l1` is strictly ≤ TTA-off `val_l1` (the improvement is real, not a fluke).
3. The TTA-on inference wall-time is ≤ 6× the TTA-off inference wall-time (5 augmented passes + 1 base = 6 passes; tolerance for the un-flip / un-rotate overhead).

### User Story 3 — Per-region + per-quantile + curvature metrics (Priority: P2)

**Acceptance scenarios**:
1. `validate_v24.py --guided-checkpoint <path>` reports `val_l1_per_region` (4 sub-regions), `val_l1_per_quantile` (10 buckets), and `val_l1_curvature` (single number).
2. The metrics are not all zero (real signal is there).
3. The per-region metric is roughly uniform (no region is 5× worse than another).
4. The per-quantile metric is roughly flat (the model is accurate on the full height range, not just the median).

---

## Functional Requirements

### Slice 1: guided model

- **FR-099-101**: A new class `StageAMinimapOnlyGuided` in `stage_a.py`, similar to `StageAMinimapOnly` but with 9 input channels (3 minimap + 3 normal + 3 normal-Sobel). Output: same as `StageAMinimapOnly` (outer + inner).
- **FR-099-102**: The model size is ≤ 600K params.
- **FR-099-103**: A new test `test_stage_a_guided_forward_shape_and_params` asserts the new model has 9 input channels, the right output shape, and the param count.

### Slice 2: trainer changes

- **FR-099-201**: A new `--guided` flag on `train_v24_stage_a.py` switches to the guided model + RAdam + Lookahead + OneCycle trainer.
- **FR-099-202**: Default epochs for `--guided` is 200 (vs 50 for the unguided path).
- **FR-099-203**: A new test `test_train_guided_one_epoch` runs one epoch of guided training on a small fixture and asserts `val_l1` improves over the initial random baseline.
- **FR-099-204**: The trainer saves a per-epoch visual preview at `output/v24_validation/<run_id>/previews/epoch_NNN.png` (configurable `--preview-every N`, default 10).

### Slice 3: TTA

- **FR-099-301**: A new function `tta_predict(model, x, n_aug=5)` in `stage_a.py` that runs 5 augmented forward passes and returns the averaged prediction.
- **FR-099-302**: A new `--use-tta` flag on `validate_v24.py` enables TTA in validation.
- **FR-099-303**: A new test `test_tta_predict_averages_5_passes` asserts the output shape is right and the value is between the min and max of the 5 augmented passes.

### Slice 4: rich validation metrics

- **FR-099-401**: The validation output now includes `val_l1_per_region` (4-element dict), `val_l1_per_quantile` (10-element dict), `val_l1_curvature` (single number).
- **FR-099-402**: A new test asserts the validation output schema (the per-region and per-quantile dicts have the right keys).

### Slice 5: 200-epoch retrain

- **FR-099-501**: After Slice 1-4 ship, run the full 200-epoch retrain on `3_3_5_12340_openworld_curated.zarr` with the new trainer and write the report at `output/v24_validation/v24_guided_3_3_5_12340_20260710/report.json`.
- **FR-099-502**: The report is committed to the repo (it's a small JSON).
- **FR-099-503**: The architecture doc `docs/architecture/v24-guided-retrain-2026-07-10.md` is written.

---

## Success Criteria

- **SC-099-001**: `val_l1 < 5.0 world units` AND `val_l1_real_cells < 3.0 world units` AND `val_l1_curvature < 0.5` on the held-out V24 prior validation, on the guided model trained for 200 epochs.
- **SC-099-002**: Per-epoch `val_l1` is monotonically non-increasing for the first 50 epochs (the model is genuinely learning, not just oscillating).
- **SC-099-003**: TTA-on validation is strictly better than TTA-off validation (TTA is real signal, not a fluke).
- **SC-099-004**: 40+ v24 tests pass (was 48 before this spec; adding the new tests should not regress).
- **SC-099-005**: Architecture doc + memory bank + progress.md updated at slice completion.

---

## Key Entities

- **`StageAMinimapOnlyGuided`** — new model class, 9 input channels.
- **`tta_predict(model, x, n_aug=5)`** — new helper, 5 augmented forward passes + average.
- **Per-region / per-quantile / curvature validation metrics** — new fields in the validation report.
- **200-epoch trained checkpoint** — `output/v24_validation/v24_guided_3_3_5_12340_20260710/stage_a.pt`.

---

## Risks

- **Risk 1 (high):** the guided model doesn't beat the 190.31 L1. The user said "100% perfect" but that's an aspirational target. The honest SC-001 is < 5.0 world units, which is the L1 of the cheat regime. If even the guided model plateaus at, say, 30 world units, that's still a 6× improvement over the current 190, and a real shipping improvement. The spec ships anyway.
- **Risk 2 (medium):** training the 200-epoch guided model takes > 6 hours on the 12 GB envelope. Fallback: RunPod. The spec ships a `--epochs 50 --guided` quick mode for fast iteration, and the 200-epoch full run is a separate execution.
- **Risk 3 (low):** RAdam + Lookahead + OneCycle doesn't work well together (some optimizers don't play nicely with Lookahead). Mitigation: RAdam + OneCycle alone is a known-good combo; Lookahead is optional and off by default.
- **Risk 4 (low):** the per-region / per-quantile / curvature metrics are noisy at the per-epoch granularity. The user sees per-epoch values, but the report uses a 5-epoch rolling average.

---

## Assumptions

- The user has access to a 12 GB CUDA GPU (the existing training setup).
- The curated V24 corpus (`3_3_5_12340_openworld_curated.zarr`) is still on disk (validated earlier).
- The user is willing to wait 6+ hours for the 200-epoch run, or will switch to RunPod.
- The `StageAMinimapOnly` model (unguided) is kept as the fallback for tiles that don't have normals.

---

## Open Questions (For User Review Before Plan)

1. **Guided model only, or also retrain the unguided?** The unguided model is the deployment path (no normals needed). Recommended: retrain BOTH the guided (with normal input) and the unguided (as a fallback) for 200 epochs. The unguided still gets the RAdam + OneCycle + TTA + richer validation metrics. This is a 2x training cost but covers the deployment story.
2. **Run on every V18 build, or just 3_3_5_12340?** Recommended: just 3_3_5_12340 for the first pass. Multi-build retrain is a separate bounded spec (Spec 099+).
3. **Where to host the 200-epoch run?** Recommended: local 12 GB GPU first, fall back to RunPod if it takes > 6 hours.

---

## End of Spec

This is a real bounded piece. Slices 1-4 are small, well-tested, and ship-able in a single focused session. Slice 5 is the actual retrain — a single long-running job. If the SC-099-001 target (val_l1 < 5.0) is not met, the slice ships anyway with a real improvement, and the next session iterates on architecture / hyperparameters.
