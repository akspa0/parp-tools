# Feature Specification: V24.1 DA-V2 Pretrained Convergence Model (Spec 101)

**Feature Branch**: `101-v241-dav2-model`

**Created**: 2026-07-10

**Status**: Draft

**Owner**: wow-viewer

**Parent**: Spec 094 (V24 WDL prior + lattice detailer), Spec 099 (Stage A retrain — plateaued at 94.58 L1), Spec 100 (PatchGAN discriminator — not yet shipped)

**Research**: [`docs/architecture/v24-convergence-research-2026-07-10.md`](../../docs/architecture/v24-convergence-research-2026-07-10.md)

**Input**: User description — "use speckit, call this v24.1 model, and do all the recommendations. Make it happen!"

---

## Problem Statement

The V24 minimap-only Stage A model (335K-param from-scratch U-Net) achieves val_l1 = 190.31 world units — 158× worse than the `block_reduce` baseline (1.31). The guided variant (450K params, 9-channel) plateaus at 94.58 L1 and overfits hard (train 51 vs val 98). The model is too small and has no pretrained features; a 335K–450K parameter network trained from random initialization on ~2,000 tiles cannot learn the minimap → heightmap mapping.

The research report ([`v24-convergence-research-2026-07-10.md`](../../docs/architecture/v24-convergence-research-2026-07-10.md)) identified that the V23 codebase already integrates a **Depth Anything V2 Small** (DA-V2-Small, 24.8M params) encoder with LoRA adapters in [`harvester/v23/encoder.py`](../../data-harvester/src/harvester/v23/encoder.py:133). The pretrained DINOv2 backbone was trained on 62M images and already understands visual features. V24.1 reuses this infrastructure for the V24 Stage A and Stage B models.

Additionally, the training code has a **OneCycleLR scheduler bug** (`total_steps=args.epochs` instead of `n_batches * epochs`) and uses **plain L1 loss** instead of the scale-invariant SiLogLoss that DA-V2 metric depth training uses.

---

## User Scenarios & Testing

### User Story 1 — Train V24.1 Stage A with DA-V2-Small and see real convergence (Priority: P1)

The user runs the V24.1 Stage A trainer with the DA-V2-Small pretrained encoder and sees val_l1 drop from 190 to single digits within 40 epochs.

**Why this priority**: This is the core convergence fix. Without it, the V24 pipeline cannot produce useful WDL priors from minimap-only input.

**Independent Test**: Run `train_v24_stage_a.py --dav2 --minimap-only --epochs 40` on the curated V24 corpus and verify val_l1 < 10.

**Acceptance Scenarios**:

1. **Given** the curated open-world V24 store (`3_3_5_12340_openworld_curated.zarr`, 2,011 tiles), **When** the user runs `uv run python scripts/train_v24_stage_a.py --v24-store <store> --dav2 --minimap-only --epochs 40 --output <dir>`, **Then** the training completes without CUDA OOM on a 12 GB GPU and `best_val_l1` is < 10.0 world units.
2. **Given** a trained V24.1 Stage A checkpoint, **When** the user runs `validate_v24.py --dav2-checkpoint <path>`, **Then** the validation report shows `val_l1 < 10.0`, `val_l1_real_cells < 5.0`, and the per-region L1 is roughly uniform (no region 5× worse than another).
3. **Given** the trained V24.1 Stage A checkpoint, **When** the user runs `infer_v24_stage_a_png.py --dav2-checkpoint <path> --input <png>`, **Then** the inference produces a WDL prior NPZ with world-unit heights in a reasonable range (not all zeros, not saturated).

---

### User Story 2 — SiLogLoss produces sharper predictions than L1 (Priority: P1)

The user trains with the new SiLogLoss (or hybrid L1+SiLogLoss) and sees sharper, more structurally accurate WDL priors than the L1-only baseline.

**Why this priority**: L1 loss produces blurry mean predictions. SiLogLoss is the standard for metric depth estimation and produces sharper, more structurally faithful outputs.

**Independent Test**: Compare the per-cell error distribution between L1-only and SiLogLoss-trained models on the same held-out tiles.

**Acceptance Scenarios**:

1. **Given** two Stage A checkpoints trained for the same number of epochs (one with L1, one with SiLogLoss), **When** the user runs validation on both, **Then** the SiLogLoss checkpoint has lower `val_l1_curvature` (the shape-quality metric from Spec 099) than the L1 checkpoint.
2. **Given** a SiLogLoss-trained model, **When** the per-epoch preview is inspected, **Then** the predicted WDL prior shows visible terrain structure (ridges, valleys) rather than a smooth blur.

---

### User Story 3 — Scheduler fix ensures per-epoch improvement (Priority: P1)

The user trains with the fixed scheduler and sees val_l1 decrease monotonically over the first 20 epochs (no plateau at epoch 5).

**Why this priority**: The current OneCycleLR bug causes the LR schedule to be 30× too slow, wasting training time and preventing convergence.

**Independent Test**: Check `loss_history.jsonl` for monotonic val_l1 decrease over the first 20 epochs.

**Acceptance Scenarios**:

1. **Given** the V24.1 trainer with the fixed scheduler, **When** the user trains for 40 epochs, **Then** `loss_history.jsonl` shows val_l1 is monotonically non-increasing for the first 20 epochs.
2. **Given** the fixed scheduler, **When** the LR is logged per epoch, **Then** the LR follows the expected cosine annealing schedule (warmup → peak → decay to 0).

---

### User Story 4 — PromptDA Stage B for high-res reconstruction (Priority: P2)

The user runs the V24.1 Stage B with PromptDA-Small and sees val_l1 < 1.0 world units on the 257×257 heightmap reconstruction.

**Why this priority**: Stage B is the high-res reconstruction step. PromptDA is pretrained for depth completion (RGB + low-res depth → high-res depth), which is exactly our Stage B structure.

**Independent Test**: Run `train_v24_stage_b.py --promptda` and verify val_l1 < 1.0.

**Acceptance Scenarios**:

1. **Given** a trained V24.1 Stage A checkpoint (producing WDL priors with val_l1 < 10), **When** the user runs `train_v24_stage_b.py --promptda --stage-a-checkpoint <path>`, **Then** the Stage B model achieves val_l1 < 1.0 world units on the held-out 257×257 heightmap.
2. **Given** the PromptDA Stage B checkpoint, **When** the user runs the per-map export (`v24_export_map.py`), **Then** the stitched OBJ shows continuous terrain with no hard seams at tile boundaries.

---

### User Story 5 — PatchGAN auxiliary loss (Priority: P3)

The user adds the PatchGAN adversarial loss on top of the DA-V2 Stage A and sees a further 20-50% improvement in val_l1.

**Why this priority**: PatchGAN is a refinement step. It helps with perceptual quality but is not the primary convergence strategy.

**Independent Test**: Compare val_l1 between DA-V2+SiLogLoss and DA-V2+SiLogLoss+PatchGAN on the same held-out tiles.

**Acceptance Scenarios**:

1. **Given** a DA-V2 Stage A checkpoint trained with SiLogLoss, **When** the user adds `--gan` to continue training with the PatchGAN adversarial loss, **Then** the val_l1 improves by at least 20% over the SiLogLoss-only baseline.
2. **Given** the GAN training, **When** the discriminator loss is inspected, **Then** it is bounded (not collapsed to 0, not saturated to 1) over the training run.

---

### Edge Cases

- What happens when the DA-V2-Small pretrained weights are not cached locally? The encoder falls back to random initialization (existing V23 behavior) and the user is warned. Training still runs but convergence will be slower.
- What happens when the minimap has negative world-unit heights? SiLogLoss requires positive values. The trainer shifts heights by a per-tile offset (min height → 1.0) before computing SiLogLoss, then un-shifts the prediction.
- What happens when the V18 store has no `normal_xyz` for a tile? The guided model degrades gracefully (channels 3-8 are zeros, existing behavior from `build_guided_input`).
- What happens when PromptDA-Small is not available on HuggingFace? The trainer falls back to the existing 828K-param conv-deconv Stage B with a warning.

---

## Requirements

### Functional Requirements

#### Slice 1: DA-V2-Small Stage A model

- **FR-101-101**: A new model class `StageADAV2` in `harvester/v24/stage_a.py` that wraps the existing `DepthAnythingV2SmallEncoder` from `harvester/v23/encoder.py` with a DPT-style head that outputs the 33×33 quincunx → 17×17 outer + 16×16 inner WDL prior.
- **FR-101-102**: The model supports 3-channel (minimap-only) and 9-channel (guided: minimap + normal + Sobel) input via the patch projection replacement already implemented in `DepthAnythingV2SmallEncoder`.
- **FR-101-103**: The model's trainable parameters are: LoRA adapters (rank 16) + patch projection + DPT head. The backbone is frozen. Total trainable params ≤ 2M; total model params ≤ 26M.
- **FR-101-104**: A test asserts the forward shape (outer 17×17, inner 16×16), the param count, and that the backbone is frozen.
- **FR-101-105**: A test asserts the model loads with `load_pretrained=False` (offline/test mode) and produces correct output shapes.

#### Slice 2: SiLogLoss

- **FR-101-201**: A new `SiLogLoss` class in `harvester/v24/stage_a.py` that computes the scale-invariant log loss. Handles negative heights by shifting to positive before taking log.
- **FR-101-202**: A `--loss-type` flag on `train_v24_stage_a.py` with choices `l1`, `silog`, `hybrid` (default `hybrid` for the DA-V2 model, `l1` for the legacy U-Net).
- **FR-101-203**: The hybrid loss is `0.7 * SiLogLoss + 0.3 * L1` (tunable via `--silog-weight` and `--l1-weight`).
- **FR-101-204**: A test asserts SiLogLoss produces a positive scalar, handles negative inputs, and has a non-zero gradient.

#### Slice 3: Scheduler fix

- **FR-101-301**: The OneCycleLR `total_steps` is fixed to `n_batches * args.epochs` (the total number of optimizer steps, not epochs).
- **FR-101-302**: `scheduler.step()` is called per batch (after `scaler.step(optimizer)`), not per epoch.
- **FR-101-303**: A `--scheduler` flag with choices `onecycle` (default for guided/DA-V2), `cosine` (simpler, per-epoch stepping). The `cosine` option uses `CosineAnnealingLR` with per-epoch stepping.
- **FR-101-304**: A test asserts the LR at epoch 1 is higher than the LR at the final epoch (the schedule is decaying).

#### Slice 4: DA-V2 trainer integration

- **FR-101-401**: A `--dav2` flag on `train_v24_stage_a.py` switches to the `StageADAV2` model with pretrained DA-V2-Small encoder, SiLogLoss (or hybrid), and the fixed scheduler.
- **FR-101-402**: The trainer loads the pretrained DA-V2-Small encoder weights from HuggingFace (`depth-anything/Depth-Anything-V2-Small-hf`) with `local_files_only=True` (offline cache). Falls back to random init with a warning if not cached.
- **FR-101-403**: The trainer uses `lr=5e-6` as the default for the DA-V2 model (vs `1e-3` for the legacy U-Net), because the encoder is pretrained and only LoRA + head are trained.
- **FR-101-404**: The trainer records `model_type: "dav2"`, `loss_type`, `scheduler_type`, `pretrained_loaded: bool`, and `lora_rank` in the checkpoint config.
- **FR-101-405**: A test runs one epoch of DA-V2 training on a small fixture and asserts val_l1 improves over the random baseline.

#### Slice 5: Validation + inference updates

- **FR-101-501**: `validate_v24.py` gains a `--dav2-checkpoint <path>` flag that loads a `StageADAV2` checkpoint and runs the full validation suite (SC-001 through SC-005 from Spec 094, plus the Spec 099 rich metrics).
- **FR-101-502**: `infer_v24_stage_a_png.py` gains a `--dav2-checkpoint <path>` flag for standalone PNG → WDL prior inference.
- **FR-101-503**: The inference script handles the DA-V2 model's larger input size (DA-V2 uses 518×518 patches; the minimap is 256×256, so it's resized/padded to the DA-V2 input size).

#### Slice 6: PromptDA Stage B (depth completion)

- **FR-101-601**: A new model class `StageBPromptDA` in `harvester/v24/stage_b.py` that wraps the PromptDA-Small model for depth completion (minimap RGB + WDL prior prompt → 257×257 heightmap).
- **FR-101-602**: The model loads pretrained PromptDA-Small weights from HuggingFace (`depth-anything/prompt-depth-anything-vits`) with `local_files_only=True`. Falls back to the existing conv-deconv Stage B with a warning if not cached.
- **FR-101-603**: A `--promptda` flag on `train_v24_stage_b.py` switches to the PromptDA model.
- **FR-101-604**: A test asserts the PromptDA model forward shape (257×257) and that it accepts a low-res depth prompt.

#### Slice 7: PatchGAN auxiliary loss (from Spec 100, adapted)

- **FR-101-701**: The `WDLDiscriminator` PatchGAN class from Spec 100 Slice 1 is implemented in `harvester/v24/discriminator.py`.
- **FR-101-702**: A `--gan` flag on `train_v24_stage_a.py` enables the discriminator and adversarial loss on top of the DA-V2 model.
- **FR-101-703**: The lambda schedule: `λ_adv` starts at 0, ramps to 0.1 over epochs 5-30, holds at 0.1. Configurable via `--adv-lambda-max` and `--adv-lambda-ramp-epochs`.
- **FR-101-704**: A test asserts the discriminator forward shape and that GAN training updates both D and G.

### Key Entities

- **`StageADAV2`** — DA-V2-Small encoder + LoRA + DPT head. ~25M total params, ~1-2M trainable. Replaces `StageAMinimapOnly` and `StageAMinimapOnlyGuided`.
- **`SiLogLoss`** — Scale-invariant log loss. Handles negative heights via per-tile shift. Replaces plain weighted L1 as the default for DA-V2 training.
- **`StageBPromptDA`** — PromptDA-Small for depth completion. ~25M params. Replaces the 828K-param conv-deconv Stage B.
- **`WDLDiscriminator`** — PatchGAN discriminator, ~250K params. Auxiliary adversarial loss.

---

## Success Criteria

### Measurable Outcomes

- **SC-101-001**: V24.1 Stage A (DA-V2-Small + LoRA + SiLogLoss) achieves `val_l1 < 10.0` world units on the held-out V24 prior validation, trained for 40 epochs on the curated open-world corpus. (Current: 190.31)
- **SC-101-002**: V24.1 Stage A `val_l1_real_cells < 5.0` world units. (Current: ~190)
- **SC-101-003**: Per-epoch `val_l1` is monotonically non-increasing for the first 20 epochs (the scheduler fix works).
- **SC-101-004**: The DA-V2 Stage A model fits on a 12 GB GPU with batch size ≥ 4 at 256×256 input (peak VRAM < 10 GB).
- **SC-101-005**: V24.1 Stage B (PromptDA) achieves `val_l1 < 1.0` world units on the 257×257 heightmap reconstruction, given a Stage A prior with val_l1 < 10.
- **SC-101-006**: PatchGAN auxiliary loss (Slice 7) improves val_l1 by at least 20% over the SiLogLoss-only DA-V2 baseline.
- **SC-101-007**: All existing v24 tests pass (currently 46/46) plus the new V24.1 tests. No regressions.
- **SC-101-008**: Architecture doc + memory bank + progress.md updated at each slice completion.

---

## Assumptions

- The user has a 12 GB CUDA GPU (RTX 4070 Ti SUPER or equivalent).
- The DA-V2-Small pretrained weights are cached locally or downloadable from HuggingFace. If not cached, the model falls back to random init with a warning (existing V23 behavior).
- The PromptDA-Small pretrained weights are available on HuggingFace (`depth-anything/prompt-depth-anything-vits`). If not cached, Stage B falls back to the existing conv-deconv.
- The V18 store has the V24 prior substrate (`wdl_prior_outer`, `wdl_prior_inner`, etc.) as built by Spec 094.
- The `transformers` and `peft` Python packages are installed (they are already dependencies of V23).
- The `promptda` Python package may need to be installed as a new dependency. If it's not available via pip, the PromptDA model can be loaded via `transformers` or a direct weight download.
- The existing V24 test suite (46 tests) continues to pass. New tests are additive.
- The V23 encoder infrastructure (`DepthAnythingV2SmallEncoder`) is reused as-is, without modification. V24.1 wraps it, not forks it.
- Training the DA-V2 Stage A for 40 epochs on 2,011 tiles takes < 2 hours on a 12 GB GPU (the model is small, the data is preloaded).
- The PatchGAN slice (Slice 7) is lower priority and may be deferred if the SiLogLoss-only DA-V2 model already meets SC-101-001.

---

## What This Spec Does NOT Do

- **No new V18 build training** — single-build retrain (`3_3_5_12340`) is the SC-001 target. Multi-build retrain is Spec 099+.
- **No RunPod work** — local 12 GB GPU is sufficient for a 25M-param model with LoRA.
- **No real-time inference** — the model is per-tile forward, not real-time.
- **No changes to the V23 encoder** — `DepthAnythingV2SmallEncoder` is reused as-is. V24.1 wraps it.
- **No changes to the V24 data pipeline** — `TileSource`, `TileRecord`, `build_target`, etc. are unchanged. V24.1 only changes the model and trainer.
- **No new C# tooling** — all work is in Python (`data-harvester/`).
- **No full-fidelity ADT writing** — that's Spec 097 Slices 2/3.
- **No fractal detail detector** — that's Spec 101 in the Spec 098 vision doc.
- **No lattice-constrained reconstruction model** — that's Spec 102 in the Spec 098 vision doc.

---

## Implementation Order

1. **Slice 1**: `StageADAV2` model class (reuses V23 encoder, adds DPT head)
2. **Slice 2**: `SiLogLoss` (handles negative heights)
3. **Slice 3**: Scheduler fix (OneCycleLR `total_steps` + per-batch stepping)
4. **Slice 4**: `--dav2` trainer flag + checkpoint config
5. **Slice 5**: Validation + inference updates (`--dav2-checkpoint`)
6. **Slice 6**: `StageBPromptDA` (PromptDA for Stage B)
7. **Slice 7**: PatchGAN auxiliary loss (from Spec 100, adapted for DA-V2)

Slices 1-5 are the core convergence fix. Slice 6 is the Stage B upgrade. Slice 7 is the refinement step.

---

## End of Spec

This spec replaces the approach in Spec 099 (from-scratch guided U-Net) and Spec 100 (PatchGAN on the from-scratch U-Net) with a pretrained-backbone approach. The key insight from the research is that the V24 convergence problem is a **model capacity and pretrained features** problem, not a loss function or training schedule problem. The DA-V2-Small encoder (24.8M params, pretrained on 62M images) is already integrated in the V23 codebase. V24.1 reuses it.