# Feature Specification: V16.1.2 Height-Derived Normal Refiner

**Feature Branch**: `015-v16-1-2-height-derived-normal-refiner`

**Created**: 2026-05-23

**Status**: Superseded — refiner approach failed; V16.1.3 height-channel normal model (spec 016) is the active normal lane. Reroute follow-up to `047-v18-distill-corpus-open-source-loop`.

**Input**: User observation that the V16.1.1 normal model's output is a full-resolution raster (every pixel independently computed by Tanh head), while the ground truth `normal_xyz` from ADT MCNR chunks is interpolated 17x17-per-chunk vertex data. The model never learns the interpolation structure. The height field shares the same interpolation structure, so a refiner model conditioned on height can teach it.

## Problem Statement

The V16.1.1 normal model (`minimap_rgb_256 → normal_xyz`) uses:

- per-pixel cosine + L1 + NZ loss on a Tanh(3ch) head
- terrain-valid masking, object/liquid downweighting
- hard-region weighting from height gradients, normal gradients, alpha/MCLY transitions

After the V16.1.1 run, the researcher notices an artifact in validation images: the ground truth normals show a smooth half-tone gradient structure, while the prediction is a fully saturated raster of independent RGB values. This is not a rendering bug. It is a real training artifact.

**Root cause**: The 3-channel Tanh head has no spatial prior — every output pixel is independent. The ground truth `normal_xyz` comes from MCNR vertex normals (17x17 per MCNK) interpolated to 257x257 float32. The two live in different distributional families: interpolated smooth vs per-pixel independent.

The height field `height_257` shares the same 17x17→257x257 interpolation path. A model that sees height as input can learn the interpolation structure and teach it back to the normal model.

## Architecture

The V16.1.2 system adds a single small model and a training-time only distillation loop:

```
┌─────────────────────────────────────────────────────────┐
│                   V16.1.2 Training Loop                  │
│                                                         │
│  main_model(minimap) → pred (3ch Tanh)                  │
│       │                                                 │
│       ├── L_main(pred, gt, mask)  ─── same as V16.1.1  │
│       │                                                 │
│       └── refiner(cat(pred.detach(), height)) → refined  │
│                                                         │
│  After validation:                                       │
│    if best_epoch:                                        │
│      compare L_main(refined, gt, mask) vs L_main(pred,gt,mask)
│      if refined wins → save refiner, mark active         │
│                                                         │
│  Next epoch (if refiner active):                         │
│    teacher = refiner(pred.detach(), height)              │
│    L = L_main + w_distill * L_cos(pred, teacher, mask)   │
│                                                         │
│  → main model learns to produce "interpolated-looking"  │
│    normals that approximate the refiner's output         │
└─────────────────────────────────────────────────────────┘
```

### Refiner Model

- **Input**: `cat(pred_normals, height_norm)` = 4ch (float32)
- **Output**: `refined_normals(3ch)` with Tanh
- **Architecture**: Small feed-forward conv net — 3 residual blocks, skip connection from input normals to output
- **Params**: ~50-100K
- **No masks, no object gating, no liquid weighting, no terrain-valid guards**

### Distillation Term

Only active after the refiner has proven useful at a best-epoch trigger:

```
L_distill = cos(pred_normals, teacher_normals)
L_distill = mask * L_distill  (reuse same terrain-valid mask as L_main)

L_total = L_main + w_distill * L_distill
```

Default `w_distill`: 0.25

### Why Height Works

Both `height_257` and `normal_xyz` originate from per-MCNK vertex data (17x17 grid) interpolated to 257x257. The interpolation kernel (bilinear or bicubic upsampling) creates the same gradient structure in both signals. The refiner sees height and learns: "where height changes smoothly, normals should also change smoothly." This gives the model a spatial-frequency prior it otherwise lacks.

## Relationship to V16.1.1

This is additive, not a reset:
- Resumes from the V16.1.1 checkpoint — same model weights, same optimizer state
- Uses the same dataset contract — no new Zarr arrays needed
- Does not change the main model architecture
- Does not change inference — the refiner is training-time only
- Runs in a separate runs folder (`v16_1_2_<name>`) alongside the ongoing V16.1.1 run

## User Scenarios & Testing

### User Story 1 — Refiner Activates After Best Epoch and Improves Loss (Priority: P1)

A terrain researcher resumes the V16.1.1 checkpoint into a V16.1.2 run. After the first complete validation epoch, the refiner runs on the held-out batch. The refiner's loss is lower than the raw prediction's loss. The run logs this comparison and marks the refiner as active.

**Why this priority**: The refiner must actually improve things before the distillation loop is worth running. Without this check, we might distill noise.

**Independent Test**: A V16.1.2 run on a bounded pool (400 train tiles, 48 val tiles) completes epoch 1 and the evidence shows `refiner_improved=true` with quantitative before/after loss.

**Acceptance Scenarios**:

1. **Given** a loaded V16.1.1 checkpoint, **When** the V16.1.2 validator runs the refiner for the first time, **Then** it logs `L_main(refined, gt)` vs `L_main(pred, gt)` for the entire validation set, not just one batch.
2. **Given** the refiner produces lower validation loss, **When** distillation is enabled, **Then** the training loss includes L_distill with the configured weight.
3. **Given** the refiner does NOT improve validation loss, **When** training continues, **Then** distillation is skipped and the run behaves like V16.1.1 (graceful fallback).

---

### User Story 2 — Distillation Progressively Improves Main Model Output (Priority: P1)

Over subsequent epochs, the main model's raw predictions (without refiner) begin to show smoother, more interpolated-looking normals that better match the ground truth gradient structure.

**Why this priority**: The whole point is to change what the main model produces, not just layer a post-processing network on top.

**Independent Test**: Compare validation images from the V16.1.2 run at epochs 1, 10, 25, and 50. The `normal_pred` panels show progressively less per-pixel noise and more coherent gradient structure.

**Acceptance Scenarios**:

1. **Given** distillation is active for 10+ epochs, **When** the researcher inspects validation `normal_pred` panels, **Then** the output normals show visible gradient coherence (fewer isolated pixel outliers) compared to the starting checkpoint.
2. **Given** the same validation tiles, **When** comparing before/after the refiner was introduced, **Then** angular error on terrain-only pixels is not worse than the pre-refiner checkpoint (regression guard).

---

### User Story 3 — No-Object Refiner Works Without Any Masking Signals (Priority: P2)

The refiner model's loss computation uses no terrain-valid masks, no object-presence downweighting, and no liquid gating. It simply compares refined normals to ground truth normals everywhere, trusting that height is a strong enough prior to ignore non-terrain regions.

**Why this priority**: Simplicity. If the refiner works without masks, it proves the height signal is inherently robust to object/liquid pollution.

**Independent Test**: Run the refiner comparison on tiles with heavy object coverage. The refiner's per-pixel loss on object pixels should not diverge significantly from the masked loss — because height is unaffected by whether a building sits on the terrain.

**Acceptance Scenarios**:

1. **Given** a tile with 30%+ object coverage (buildings, doodads), **When** the refiner computes loss, **Then** the unmasked refiner loss is within 10% of the terrain-masked refiner loss.
2. **Given** a tile with liquid coverage, **When** the refiner computes loss, **Then** liquid-region normals from the refiner are not systematically worse than from the main model.

---

### Edge Cases

- What if the refiner improves loss on the validation batch but generalizes differently per epoch? The refiner is evaluated at each best-epoch trigger; it can be re-disabled later if performance degrades.
- What if distillation causes the main model to regress on hard terrain details? The hard-region weighting in `L_main` is preserved — `L_distill` uses the same mask, so the model cannot ignore hard regions to chase smoothness.
- What if the refiner overfits to the epoch's validation batch? The best-epoch guard uses the entire validation set, not a single batch.
- What if the refiner adds meaningful latency to training? Autotune handles extra compute — the refiner is ~50K params, negligible compared to the U-Net backbone.

## Requirements

### Functional Requirements

- **FR-001**: A refiner model MUST accept 4-channel input: `cat(pred_normals, height_norm)` and produce 3-channel output normals with Tanh.
- **FR-002**: The refiner MUST be a small conv net (≤100K parameters) with no masking, object, liquid, or terrain-valid inputs.
- **FR-003**: The refiner MUST be evaluated at each best-epoch trigger by comparing `L_main(refined, gt)` vs `L_main(pred, gt)` over the entire validation set.
- **FR-004**: Distillation MUST only activate when the refiner's loss is lower than the raw prediction's loss at a best-epoch checkpoint.
- **FR-005**: The distillation term MUST reuse the same terrain-valid mask so hard-region weighting is not bypassed.
- **FR-006**: The distillation weight MUST be configurable via a CLI flag (`--refiner-distill-weight`, default 0.25).
- **FR-007**: The refiner MUST NOT be used at inference time — the main model runs standalone.
- **FR-008**: The V16.1.2 run MUST resume from a V16.1.1 checkpoint (same main model architecture, no architecture changes).
- **FR-009**: The V16.1.2 run MUST target a separate `runs/v16_1_2_<name>` directory.
- **FR-010**: The V16.1.1 run MUST be left in place and continue independently.
- **FR-011**: Validation preview panels MUST include a `refined_gt` panel showing the refiner's output alongside `normal_gt` and `normal_pred` for the same samples.
- **FR-012**: The run MUST log `refiner_improved`, `refiner_loss`, and `raw_loss` at every best-epoch evaluation.

### Key Entities

- **Height-Derived Normal Refiner**: Small conv net (4ch in, 3ch Tanh out) that refines raw normals using height as a spatial-structure hint.
- **Distillation Term**: `L_cos(pred, teacher_normals, train_mask)` added to `L_main` when refiner is active.
- **Best-Epoch Trigger**: The point at which `val_loss` reaches a new minimum; triggers refiner evaluation and potential activation.

## Success Criteria

### Measurable Outcomes

- **SC-001**: A V16.1.2 run on a bounded pool (400 train, 48 val) completes with evidence that the refiner was compared, and `refiner_improved=true/false` is logged.
- **SC-002**: If the refiner improves, the run shows progressively smoother normal predictions over the next 10+ epochs compared to the starting checkpoint.
- **SC-003**: The per-pixel angular error on terrain-only pixels does NOT increase relative to the pre-refiner baseline (regression guard passes).
- **SC-004**: The refiner works without any masking signals — the unmasked refiner loss stays within 10% of the terrain-masked version on object-heavy tiles.

## Assumptions

- The V16.1.1 checkpoint is the starting point; no fresh training from scratch.
- Height-derived normals provide a useful structural prior for the refiner to learn from.
- Autotune (`--autotune-batch-size --target-vram-gb`) can absorb the refiner's extra compute without manual batch-size tuning.
- The refiner does not need its own training phase — it is trained jointly with the main model through the distillation loss, or optionally with a small number of dedicated refiner-only update steps at best-epoch triggers.