# Implementation Plan: V16.1.2 Height-Derived Normal Refiner

**Branch**: `015-v16-1-2-height-derived-normal-refiner` | **Date**: 2026-05-23 | **Spec**: `specs/015-v16-1-2-height-derived-normal-refiner/spec.md`

## Summary

Add a small refiner model (4ch input, 3ch Tanh output, ~50K params) that takes pred_normals + height and produces refined normals. After best-epoch validation, compare refiner loss vs raw loss. If refiner wins, activate distillation: `L = L_main + w * L_cos(pred, teacher, mask)` for subsequent epochs. The refiner uses no masking signals — it's pure geometric refinement. The goal is to teach the main model the interpolation structure it misses from independent per-pixel Tanh output.

## Technical Context

**Language/Version**: Python 3.11+, PyTorch 2.x

**Primary Dependencies**: torch, numpy (same as existing V16.1 trainer)

**Storage**: No new Zarr arrays — height_norm and pred_normals are already in the batch

**Testing**: Smoke run on bounded pool (400 train, 48 val). Compare `refiner_loss` vs `raw_loss` at best-epoch triggers.

**Target Platform**: CUDA GPU (autotune handles refiner compute cost)

**Project Type**: Training-only add-on to existing V16.1.1 normal trainer

**Performance Goals**: Refiner eval at validation time only; <5% overhead on training step when distillation active

**Constraints**: No architecture changes to main model. No inference-time changes. No new dataset arrays. Separate runs folder from V16.1.1.

## Constitution Check

- **Article IV (Residual Model Chain)**: The refiner is not a V14+ residual terrain model — it's a training-time aux network directly supervised by ground truth. No constitution conflict.
- **Article V (Streaming Dataset)**: No new datasets or harvest pipeline changes.
- **All other articles**: No violations — all code in `wow-viewer/data-harvester/`, no repo-independence issues.

## Project Structure

```
wow-viewer/specs/015-v16-1-2-height-derived-normal-refiner/
├── spec.md       # Feature specification
├── plan.md       # This file
└── tasks.md      # Task breakdown (speckit-tasks)

wow-viewer/data-harvester/
├── src/harvester/
│   └── v16_1_models.py           # + V161NormalRefiner class
└── scripts/
    └── train_v16_1_common.py     # + refiner model, refiner loss, refiner eval + distillation loop
```

## Implementation Phases

### Phase 1: Refiner Model Definition

Add `V161NormalRefiner` to `v16_1_models.py`:

- Input: 4ch (pred_normals 3ch + height_norm 1ch)
- 3 residual blocks: Conv2d→BN→ReLU→Conv2d→BN, skip connection
- Output head: Conv2d(32→3, 1x1) + Tanh
- ~50K parameters
- No BatchNorm running stats needed (training-only, small)

**Goal**: A small conv net that can refine normals given height as structural hint.

### Phase 2: Refiner Loss and Eval in Train Loop

Add to `train_v16_1_common.py`:

- `_refiner_loss(model, refiner, batch, device, args)` — evaluates refiner output against gt_normals using the same `_masked_mean(cosine, train_mask)` as `L_main`. No masking args needed in refiner signature; reuses `train_mask` from main loss outputs.
- At best-epoch trigger: run refiner on entire validation set, compare full-val `L_main(refined, gt)` vs `L_main(pred, gt)`. Log `refiner_improved`, `refiner_loss`, `raw_loss`.
- Only if refiner wins: save refiner checkpoint, set `refiner_active = True`.

**Goal**: Quantifiable evidence that the refiner improves normals before any distillation is attempted.

### Phase 3: Distillation Loop

When `refiner_active = True`:

- Each training step: `teacher = refiner(pred.detach(), height_norm)`
- `L_distill = L_cos(pred, teacher, train_mask)` using same cosine loss as L_main
- `L_total = L_main + w_distill * L_distill`
- CLI flag: `--refiner-distill-weight` default 0.25
- Refiner weights frozen during distillation (no gradients into refiner)

**Goal**: Main model absorbs the interpolation structure from the refiner's height-conditioned output.

### Phase 4: Validation Preview + CLI Wiring

- Preview panel: add `refined_gt` column showing refiner output alongside `normal_gt` and `normal_pred`
- CLI flags: `--refiner-distill-weight`, `--refiner-eval-interval` (default same as val_interval)
- Config logging: log refiner checkpoint path, refiner parameters, distillation weight
- Separate runs folder: auto-nested under `normal/runs/v16_1_2_<name>` or overrideable via `--run-name`

### Phase 5: Resume from V16.1.1 Checkpoint

- `--resume-checkpoint` accepts V16.1.1 checkpoint (no refiner state in it — main model loads, refiner initializes fresh)
- Document the launch command: resume from `v16_1_1_normal_pool800_epoch256_autotune12_compile/checkpoints/v16_1_normal_last.pt`
- Validate that resumed training produces identical loss trajectory to mask-free baseline (refiner-disabled mode)