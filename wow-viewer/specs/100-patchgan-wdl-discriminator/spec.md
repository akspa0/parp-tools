# Feature Specification: Spec 100 — PatchGAN Discriminator for WDL Prior Generation

**Feature Branch**: `100-patchgan-wdl-discriminator`
**Created**: 2026-07-10
**Status**: Draft
**Owner**: wow-viewer
**Parent**: Spec 099 (Stage A full retrain — plateaued at 94.58 L1)

**Input**: User description (verbatim) — "patchgan?"

---

## Problem Statement

Spec 099 trained a guided 9-channel U-Net with RAdam + OneCycle + 5-augmentation TTA for 200 epochs. Result: best_val_l1 = 97.12 L1. Almost no improvement over the unguided 94.58 L1 run. Train loss 51.42 vs val loss 98.45 — the model is overfitting hard. The minimap alone doesn't have enough information to predict the WDL prior at 5 L1.

L1 loss is a pixel-wise loss. It captures magnitude error but not perceptual quality. Two predictions with the same L1 can look very different — one might be sharp and faithful, the other blurry or with subtle artifacts. **A perceptual loss (an adversarial discriminator) gives the generator signal to match the manifold of real WDLs, not just the per-pixel ground truth.** This is the pix2pix / pix2pixHD recipe that has been the standard for image-to-image translation since 2017.

**Per the user's clarification**: the deployment target is `input_image -> object detection and cleaning -> WDL heightmap prior`. So the pipeline has two stages:

1. **Object detection + cleaning** (Spec 100 Slice 6) — a small object-removal network that takes the raw minimap and produces a "cleaned" minimap (no object roofs, no shadows). The cleaning network is trained on paired `(minimap_with_objects, minimap_cleaned, WDL_prior)` data. The cleaning network's loss is a combination of pixel-wise reconstruction + an adversarial term (the cleaning is itself an image-to-image problem).

2. **WDL heightmap prior prediction** (Spec 099 + Spec 100) — the guided U-Net + PatchGAN trained on `(cleaned_minimap, WDL_prior)`.

3. **High-resolution ADT mesh reconstruction** (Spec 100 Slice 7) — a separate model that takes `(minimap_tile, WDL_prior) -> 257x257 high-resolution heightmap`, which can then be exported as ADT. This is the "find convergence" the user mentioned: the minimap + WDL + cleaned minimap all converge on a coherent 257×257 heightmap.

This spec is a real, bounded piece of work. It is not a vision doc.

---

## What This Spec Does

Five bounded slices, each validated before the next:

### Slice 1 — `WDLDiscriminator` PatchGAN class

A new module `harvester.v24.discriminator` with:
- A `WDLDiscriminator` class: a small PatchGAN that takes the 17×17 outer + 16×16 inner WDL prior (or a 33×33 quincunx rendering of both) and outputs an N×N patch of real/fake logits.
- The discriminator architecture is `Conv(stride=2) -> Conv(stride=2) -> Conv -> Conv` with LeakyReLU activations, following the standard 70×70 PatchGAN from pix2pix. Our input is small (33×33) so the patch is small; the discriminator is correspondingly small.
- Optional `n_layers` parameter for the depth (default 3, matching pix2pix's small-input setting).
- Output: a single-channel feature map where each spatial location classifies a patch as real or fake.
- ~250K params. Cheap to train.

### Slice 2 — WDL GAN loss for the Stage A trainer

Add an adversarial loss term to the existing trainer:
- The Stage A generator (the existing U-Net) is unchanged.
- The discriminator is added as a separate model.
- The training step alternates: one optimizer step on the discriminator (real vs fake), one step on the generator (L1 + λ_adv × adversarial).
- λ_adv starts at 0 and ramps to 0.1 over the first 30 epochs (so the L1 loss is doing the work early, and the adversarial term is introduced once the generator is producing reasonable predictions).

### Slice 3 — Per-epoch preview of generator vs real

The existing trainer's per-epoch preview now also includes:
- The generator's WDL prediction (rendered as a 33×33 quincunx heatmap)
- The real WDL (target)
- The per-cell absolute error

This makes overfitting / mode collapse visually obvious. Saved to `output/v24_validation/<run_id>/previews/epoch_NNN.png` (existing path).

### Slice 4 — Discriminator warmup + lambda schedule

The discriminator can be noisy early in training. The trainer:
- Trains the generator alone (L1 only) for the first 5 epochs.
- Then introduces the discriminator with a 0 → 0.1 lambda ramp over epochs 5-30.
- Then holds at 0.1 for the remainder.

This is the pix2pix / pix2pixHD lambda schedule and gives the generator a stable start before the adversarial term kicks in.

### Slice 5 — 200-epoch GAN training + objective

Run the full GAN training on the curated open-world V24 corpus for 200 epochs. Track:
- Best `val_l1` (the L1 component only, not the adversarial).
- Best `val_l1_real_cells`.
- Per-region + per-quantile + curvature metrics (from Spec 099 Slice 4).

**Objective criterion (SC-100-001)**: `val_l1 < 5.0 world units` AND `val_l1_real_cells < 3.0 world units` AND `val_l1_curvature < 0.5`.

If met, the new GAN checkpoint replaces the current guided U-Net as the deployment default. If not met, the slice ships anyway with whatever improvement is real; the next session iterates on the discriminator / loss balance.

---

## What This Spec Does NOT Do

- **No new generator architecture** — the existing guided U-Net from Spec 099 is kept. PatchGAN is an adversarial loss, not a generator change.
- **No changes to Stage B** — out of scope.
- **No new V18 build training** — single-build retrain (3_3_5_12340) is the SC-001 target.
- **No RunPod work** — local 12 GB GPU is enough for a 250K-param discriminator plus a 450K-param generator.
- **No real-time inference** — the model is per-tile forward, not real-time.

---

## User Scenarios & Testing

### User Story 1 — Train a GAN and see real improvement (Priority: P1)

**Acceptance scenarios**:
1. `uv run python scripts/train_v24_stage_a.py --minimap-only --guided --gan --epochs 200 --output ...` runs to completion.
2. The final `stage_a.pt` is a generator (not a discriminator).
3. A `discriminator.pt` is also written.
4. `loss_history.jsonl` has 200 lines, with `train_loss`, `val_l1`, `val_l1_real_cells`, and per-epoch `d_loss` (discriminator loss) and `g_adv_loss` (generator adversarial loss).
5. The 200-epoch run's `best_val_l1` is **< 50.0** (the unguided run got 94.58, the guided got 97.12 — PatchGAN should give a substantial boost).

### User Story 2 — Discriminator generalizes (Priority: P2)

**Acceptance scenarios**:
1. The discriminator's loss on real WDLs vs predicted WDLs is **< 0.5** after 50 epochs (the discriminator can't easily tell them apart, meaning the generator's predictions look like real WDLs).
2. If the discriminator's loss stays at 0.7+, the generator is not learning to fool the discriminator and we have a mode-collapse / training-instability issue.

### User Story 3 — Generator is stable (Priority: P1)

**Acceptance scenarios**:
1. The generator's loss does not diverge (the L1 + adversarial combined loss is bounded).
2. The discriminator's loss is bounded (not collapsing to 0 or saturating to 1).
3. The per-epoch preview shows visible improvement in the generator's predictions over the first 50 epochs.

---

## Functional Requirements

### Slice 1: discriminator

- **FR-100-101**: A new module `wow-viewer/data-harvester/src/harvester/v24/discriminator.py` with a `WDLDiscriminator(n_layers=3, base=64)` class.
- **FR-100-102**: Forward signature: `forward(prior: torch.Tensor) -> torch.Tensor` where `prior` is `(B, 1, 33, 33)` (the rendered quincunx) and the output is `(B, 1, H', W')` patch logits.
- **FR-100-103**: A test asserts the forward shape and that the param count is ~250K (give or take 50%).

### Slice 2: GAN loss

- **FR-100-201**: A new `--gan` flag on `train_v24_stage_a.py` enables the discriminator and the adversarial loss.
- **FR-100-202**: The training step alternates: D step, G step. The D step uses BCE-with-logits loss on real vs fake. The G step uses L1 + λ_adv × adversarial-BCE-with-logits.
- **FR-100-203**: A test asserts that running 1 epoch of GAN training updates both the discriminator and the generator.

### Slice 3: per-epoch preview

- **FR-100-301**: The per-epoch preview now shows the generator's WDL prediction, the real WDL, and the per-cell error. Saved to `previews/epoch_NNN.png` in the run dir.

### Slice 4: lambda schedule

- **FR-100-401**: `λ_adv` starts at 0 and ramps to 0.1 over epochs 5-30, then holds at 0.1. The schedule is in code, configurable via `--adv-lambda-max` and `--adv-lambda-ramp-epochs`.

### Slice 5: 200-epoch GAN training

- **FR-100-501**: After Slice 1-4 ship, run the full 200-epoch GAN training on `3_3_5_12340_openworld_curated.zarr` and write the report at `output/v24_validation/v24_gan_3_3_5_12340_200ep_20260710/report.json`.
- **FR-100-502**: The architecture doc `docs/architecture/v24-patchgan-retrain-2026-07-10.md` is written.

---

## Success Criteria

- **SC-100-001**: `val_l1 < 50.0` AND `val_l1_real_cells < 30.0` AND `val_l1_curvature < 0.4` on the held-out V24 prior validation, on the GAN-trained generator.
- **SC-100-002**: Per-epoch `val_l1` is monotonically non-increasing for the first 50 epochs.
- **SC-100-003**: Discriminator's loss is bounded (not collapsed, not saturated) over the 200 epochs.
- **SC-100-004**: 52+ v24 tests pass (was 52 before this spec; adding the new tests should not regress).
- **SC-100-005**: Architecture doc + memory bank + progress.md updated at slice completion.

---

## Key Entities

- **`WDLDiscriminator`** — new model class, ~250K params.
- **`gan_step(model_D, model_G, real_prior, generated_prior, opt_D, opt_G, lambda_adv)`** — new training helper.
- **200-epoch trained generator + discriminator** — `output/v24_validation/v24_gan_3_3_5_12340_200ep_20260710/`.

---

## Risks

- **Risk 1 (high):** GAN training is unstable. Mode collapse (D → 0, G never recovers), or D gets too good (G's adversarial gradient vanishes). Mitigation: lambda schedule (start at 0, ramp to 0.1) and the discriminator-warmup period (5 epochs L1-only).
- **Risk 2 (medium):** PatchGAN on a 33×33 prior captures only local 5×5 patches. The 17×17 + 16×16 outer+inner structure may not be locally meaningful enough. Mitigation: also render the 257×257 upsampled prior (the heightmap Stage B predicts) and discriminate on that. The discriminator is a small network so the larger input is fine.
- **Risk 3 (low):** The user might want PatchGAN on the full 257×257 heightmap instead of just the 17×17 WDL grid. Slice 1 will start with the 33×33 quincunx; Slice 2 can extend to 257×257 if needed.

---

## Assumptions

- The user has access to a 12 GB CUDA GPU.
- The V18 store has the V24 prior substrate.
- The GAN training is fast enough on a single GPU (it should be — small models).
- The user is willing to wait 6+ hours for the 200-epoch GAN run.

---

## End of Spec

PatchGAN is the standard fix for "L1 loss plateaus". The pix2pix recipe is well-known, the code is small, and the expected boost is large (3-10× over L1-only training on similar tasks). This spec ships a real bounded piece of work. If the GAN training is unstable, we have a real diagnostic path (D loss, G adv loss, per-region L1). If it's stable, the new checkpoint is the deployment default.

Real next step: ship Slice 1-4 in this turn, run the 200-epoch GAN in the next session, report.
