# Feature Specification: DA-V2-Small LoRA Height Predictor with Cross-Tile Consistency

**Feature Branch**: `089-dav2-height-predictor`

**Created**: 2026-07-03

**Status**: Draft

**Input**: User description: "Train a depth-foundation-backed height predictor on top of the V22 dataset. Use DepthAnything-V2-Small + LoRA-r16 on the encoder, fresh DPT-style decoder head for `height_257`. No WDL priors — outputs must not depend on incomplete or absent WDL data. Must train on RunPod 24 GB VRAM cards (RTX 4090 / A10 / A16 / L4 class) and infer on smaller cards. Cross-tile grayscale-drift and hallucinated-detail problems must be solved structurally, not by seed-fixing. Same input must always produce byte-identical output."

## Problem Statement

Three concrete blockers on prior height-regressor routes (V19/V20/V21/speccit-066/speckit-077) must be solved for the next training pass to be production-viable:

1. **Stochastic drift is a diffusion-model problem that has leaked into our depth-model intuition.** Deterministic depth predictors (DepthAnything-V2, Metric3D, ZoeDepth, MiDaS) are single-forward-pass networks with no noise sampling; under `torch.manual_seed` + `model.eval()` + `torch.use_deterministic_algorithms(True)`, the same input produces byte-identical output across runs, cards, and PyTorch versions on the same CUDA arch. The "same seed, different colored output" pain is LDM-Marigold-TerraFusion-MESA-GeoWizard behavior, not DepthAnything-family behavior. Pick the deterministic family.

2. **Cross-tile boundary drift is structural, not seed-fixable.** Even a deterministic tile-wise inference produces patch seams when adjacent tiles are inferred independently — the per-tile scale/shift of an affine-invariant predictor is not anchored to anything across tiles. The fix is documented in two CVPR/2024–2025 papers: PatchFusion's Consistency-Aware Inference (CAI running mean) and PRO's Grouped Patch Consistency Training (GPCT). GPCT trains a consistency loss on overlapping sub-tiles in one forward pass; CAI averages predictions across overlapping tile shifts at inference. They combine cleanly: GPCT natively encourages consistent features in overlaps, CAI averages out residual disagreement.

3. **Hallucinated detail is the wrong loss function, not wrong seed.** Affine-invariant loss (DepthAnything native) mathematically removes global-scale hallucination by predicting disparity up to unknown scale+shift and least-squares-aligning against ground truth. Gradient-matching loss (DepthAnything native preserves object boundaries without inventing edges. Spatial Distance Constraint (DepthAnything-AC, July 2025) enforces relative patch-level geometry, eliminating the "looks like a mountain, paint rock texture" hallucination mode. Bias-Free Masking (PRO 2025) prevents overfit to dataset-specific palette/artifact bias. These three together kill hallucinations far better than any seed tuning.

This spec specifies one terrain model — `V23-HeightPredictor` — that uses the DepthAnything-V2-Small encoder as its foundation, runs in the V22 Zarr dataset contract, fits within a 24 GB RunPod training envelope, and produces deterministic, bit-reproducible, cross-tile-consistent `height_257` predictions.

## User Scenarios & Testing

### User Story 1 - V23 Dataset Adapter (Priority: P1)

As a ML engineer, I can load V22 Zarr data through a `V23HeightDataset` adapter that produces the documented multi-channel input plus the `height_257` target, with no WDL dependency, so training can run against any V22 store that has the required signal arrays.

**Why this priority**: The dataset adapter defines the model input/output contract. Nothing else can be built or tested without it.

**Acceptance Scenarios**:
1. **Given** a populated V22 Zarr store, **When** `V23HeightDataset` loads one tile, **Then** the input tensor has shape `[C, 256, 256]` where `C` is the documented channel count for the configured input mode and the target tensor has shape `[1, 257, 257]`.
2. **Given** a V22 store with `liquid_mask > 0` on some tile, **When** loaded, **Then** the height target at those pixels uses `liquid_height_257` (resampled from `liquid_height_256`), not raw `height_257`.
3. **Given** a V22 store missing `mcnr_normal_xyz` or `alpha_256` arrays, **When** loaded in a degraded-mode config, **Then** the adapter zero-fills missing channels and exposes a `valid_mask` per channel instead of crashing.
4. **Given** the adapter, **When** asked for the channel contract, **Then** it documents every channel index, source array, dtype, normalization, and fill policy.

---

### User Story 2 - DepthAnything-V2-Small Backbone Module (Priority: P1)

As a ML engineer, I can instantiate a `DepthAnythingV2SmallEncoder` that loads the official HF DA-V2-Small checkpoint, exposes the LoRA-r16 adapter hooks on every transformer block's Q/K/V/O projections, and accepts a configurable input channel count by replacing the first patch-embed conv (trained at full LR, not LoRA).

**Independent Test**: Instantiate the encoder with `in_channels=11` (documented input contract). Run a forward pass on a random `[2, 11, 518, 518]` tensor. Confirm output shape matches the documented feature pyramid. Confirm LoRA params are < 2M total and non-LoRA encoder params are frozen.

**Acceptance Scenarios**:
1. **Given** the encoder is instantiated, **When** parameter counts are inspected, **Then** non-LoRA base weights are frozen (`requires_grad=False`), the first patch-embed conv is fully trainable, LoRA adapter weights across Q/K/V/O of every transformer block sum to < 2M params.
2. **Given** the encoder, **When** a forward pass runs on a `[B, in_channels, 518, 518]` input, **Then** the output is the documented DPT feature pyramid matching DA-V2-Small's native intermediate shapes.
3. **Given** the encoder, **When** the LoRA adapters are disabled, **Then** the encoder forward pass is bit-identical to a stock DA-V2-Small forward pass on the same input.

---

### User Story 3 - V23 Height Decoder Head with Affine-Invariant Output (Priority: P1)

As a ML engineer, I can instantiate `V23HeightHead` — a small DPT-style decoder that consumes the encoder feature pyramid and predicts an affine-invariant disparity map at 257×257, plus a 2-channel per-tile scale+shift head for anchoring to metric at inference.

**Independent Test**: Instantiate the head. Feed the documented feature pyramid. Confirm output is `[B, 1, 257, 257]` (disparity) and `[B, 2]` (scale, shift). Confirm head parameter count is < 5M.

**Acceptance Scenarios**:
1. **Given** the head, **When** fed the encoder feature pyramid, **Then** the disparity output shape is `[B, 1, 257, 257]` and the affine-anchor output is `[B, 2]` (per-tile scale, per-tile shift).
2. **Given** a trained model, **When** the anchor head is applied to disparity, **Then** the resulting metric height is `disparity * scale + shift` and the per-tile L1 against `height_257` is the documented validation metric.
3. **Given** the head, **When** parameter count is measured, **Then** it is < 5M trainable params.

---

### User Story 4 - Training Script with Consistency Loss Stack (Priority: P1)

As a ML engineer, I can run `train_v23_height.py --dataset-dir <V22-Zarr> --builds ...` and have the loss compute: affine-invariant Lssi + gradient-matching Lgm + Spatial Distance Constraint + GPCT overlap-consistency. The script supports `--gpct-K`, `--spectral-weight`, `--sdc-weight`, `--gpct-weight`, `--bias-free-mask-ratio`.

**Independent Test**: Run `train_v23_height.py` for 2 epochs on a 50-tile V22 subset. Confirm it completes without errors, logs all loss components separately, saves a checkpoint, and emits a validation preview image.

**Acceptance Scenarios**:
1. **Given** the script, **When** run with `--gpct-weight 0`, **Then** GPCT is fully bypassed and a single-tile-per-batch path runs (compatibility baseline).
2. **Given** the script, **When** run with `--gpct-K 4 --gpct-weight 0.1`, **Then** four overlapping sub-tiles are processed per tile per step and the overlap-consistency L2 is added to the loss.
3. **Given** the script, **When** run with `--bias-free-mask-ratio 0.15`, **Then** 15% of input minimap patches are masked at the patch level during training without breaking the loss.
4. **Given** the script, **When** inspected for cross-repo imports, **Then** zero references to `gillijimproject_refactor` exist and all imports resolve inside `wow-viewer/data-harvester/src/` or third-party packages.

---

### User Story 5 - Deterministic Inference with CAI Stitching (Priority: P2)

As an inference operator, I can run `infer_v23_height.py` on a single tile or a tile grid, and the output is *byte-reproducible* across runs given the same checkpoint, input, and CUDA arch. When run on a tile grid, CAI's running-mean averaging with a configurable overlap budget (default R=16) is applied to remove tile-boundary seams.

**Independent Test**: Run `infer_v23_height.py --seed 42` and `infer_v23_height.py --seed 12345` on the same input. Confirm output arrays are bit-identical (`torch.allclose` with `atol=0, rtol=0`). Confirm CAI-R=16 produces no visible seam in the validation preview.

**Acceptance Scenarios**:
1. **Given** an inference run with `--seed 42` and another with `--seed 12345`, **When** the outputs are compared bit-for-bit, **Then** they are identical (`atol=0, rtol=0`).
2. **Given** a 3×3 tile grid from any V22 store, **When** inference is run with `--cai-r 16`, **Then** the stitched output has no visible seam at tile boundaries in a saved preview PNG.
3. **Given** a 6 GB consumer GPU, **When** a single-tile inference runs in fp16, **Then** peak VRAM stays below 4 GB and wall-time stays below 3 seconds.

---

### User Story 6 - RunPod-Targeted Training Bundle (Priority: P2)

As a project operator, I can package the V23 training code + a V22 Zarr store into a RunPod-ready bundle, ship it to a 24 GB Pod via the patterns defined in Spec 079, and start training with a single `bash runpod/train.sh`.

**Independent Test**: Package the bundle per Spec 079. Run `verify_bundle.sh` on a fresh Pod. Run `smoke.sh` (1-epoch micro-tile training). Pass.

**Acceptance Scenarios**:
1. **Given** the bundle packager, **When** it runs against the V23 training code and a V22 Zarr subset, **Then** it emits a `.tar` archive following Spec 079's manifest contract with `contains_game_client_files: false`.
2. **Given** a freshly provisioned 24 GB RunPod Pod with the `runpod/pytorch:2.3.1-cuda12.1-cudnn8-devel` image, **When** the bundle is unpacked and `install_deps.sh` runs, **Then** all V23 training dependencies install successfully and `verify_bundle.sh` reports OK.
3. **Given** the bundle on the Pod, **When** `smoke.sh` runs, **Then** it triggers a 2-epoch Glzhhhh micro-tile training pass that completes without CUDA OOM and without exceeding 22 GB VRAM peak.

---

### Edge Cases

- What happens when a V22 store has missing `tilesets/texture_rgb/` payloads (unloadable textures)? The dataset adapter records the tileset id in its one-hot input but reports `load_error=1` and zero-fills the decoded tileset RGB channels.
- What happens when a tile is entirely liquid? The target is entirely `liquid_height`-derived and `terrain_valid_mask` zeroes the loss at every pixel; the model still receives a forward pass.
- What happens when a tile has no minimap data? The dataset adapter raises a `MissingMinimapError` with the tile path; the trainer skips the tile and emits a counter to the run log.
- What happens when the LoRA adapter is removed at inference? The base DA-V2-Small encoder still produces a sensible (if less adapted) disparity map; the affine head re-aligns per-tile.
- What happens when CAI runs with R=1 (no overlap)? The output is identical to single-pass tile inference — CAI disabled is the safe fallback path.
- What happens when training hits CUDA OOM on the RunPod Pod? The training script catches the OOM via PyTorch's `torch.cuda.OutOfMemoryError`, halves the effective batch size, retries once, and emits a warning. Persistently failing OOM is a fatal error reported to the run log.

## Requirements

### Functional Requirements

- **FR-001**: The model MUST be `V23-HeightPredictor`, a single-signal terrain height predictor. It MUST NOT jointly predict normals, liquid type, or object placement. It MUST predict only `height_257` (via affine-invariant disparity + per-tile affine anchor).
- **FR-002**: The encoder MUST be `DepthAnything-V2-Small` (24.8M params) loaded from HuggingFace `depth-anything/Depth-Anything-V2-Small-hf`. Base encoder weights MUST be frozen.
- **FR-003**: Fine-tuning of the encoder MUST be via LoRA adapters (rank 16, alpha 32, dropout 0.05) on the Q/K/V/O projections of every transformer block. Total LoRA params MUST be < 2M.
- **FR-004**: The first patch-embed conv of the encoder MUST be replaced with a conv that accepts the documented input channel count. This replaced conv MUST be trained at full learning rate (not LoRA-constrained), as first-layer adaptation to a new input modality requires capacity beyond LoRA's low-rank update.
- **FR-005**: The decoder MUST be a small DPT-style head with parameter count < 5M. It MUST output a 257×257 affine-invariant disparity map and a 2-channel per-tile affine anchor `(scale, shift)`.
- **FR-006**: The dataset adapter `V23HeightDataset` MUST consume V22 Zarr stores directly. It MUST NOT require WDL priors of any form.
- **FR-007**: The documented input channel contract (see "Input Channel Contract" below) MUST be the only permitted default. A degraded config that drops the `tileset_one_hot`, `alpha_256`, or `normal_xyz` channels MUST be supported via flags so the model can be tested incrementally.
- **FR-008**: Where `liquid_mask > 0`, the height target MUST be `liquid_height` resampled to 257×257, not raw `height_257`.
- **FR-009**: The training loss MUST include, in this order of importance: (1) affine-invariant Lssi from DepthAnything-V2's training code, (2) gradient-matching Lgm from DepthAnything-V2's training code, (3) Spatial Distance Constraint term from DepthAnything-AC, (4) GPCT overlap-consistency L2 (controlled by `--gpct-weight`).
- **FR-010**: Bias-Free Masking MUST be implemented as patch-level input masking with ratio controlled by `--bias-free-mask-ratio` (default 0.15). Masked patches are dropped before the encoder forward pass, not patched with synthetic content.
- **FR-011**: The training script MUST be deterministic when run with `torch.manual_seed(N)` + `model.eval()` + `torch.use_deterministic_algorithms(True)`. The same seed, input, and CUDA arch MUST produce bit-identical gradient updates and weights.
- **FR-012**: GPCT MUST process K overlapping sub-tiles per tile per training step (K configurable, default 4). The overlap-consistency loss is the L2 distance between predictions on overlapping regions across the K sub-tiles, with the documented feature-level constraint active when `--gpct-feature-loss` is enabled (default true).
- **FR-013**: The inference script MUST implement CAI as a running mean over R overlapping tile shifts (R configurable, default 16). When R=1, the inference MUST be identical to single-pass tile inference.
- **FR-014**: The model MUST train on a single RunPod 24 GB GPU (RTX 4090 / A10 / A16 / L4 class) without exceeding 22 GB peak VRAM at the documented batch size and K=4 GPCT setting. The optimizer MUST be 8-bit AdamW (bitsandbytes) to fit the optimizer state. Mixed precision MUST be bf16.
- **FR-015**: The model SHOULD infer on a 6 GB consumer GPU at fp16 with peak VRAM below 4 GB and wall-time per tile below 3 seconds.
- **FR-016**: The RunPod bundle MUST conform to Spec 079's manifest contract. The bundle MUST contain only Python training code, the V22 Zarr subset, manifests, requirements, and pod-side helper scripts. It MUST NOT contain game client data.
- **FR-017**: The training script MUST have zero imports that resolve outside `wow-viewer/data-harvester/src/`, third-party packages, or the standard library. No `gillijimproject_refactor` references anywhere in the V23 work tree.
- **FR-018**: All model code MUST live under `wow-viewer/data-harvester/src/harvester/v23/`. All scripts MUST live under `wow-viewer/data-harvester/scripts/`. RunPod helper scripts MUST live under `wow-viewer/data-harvester/runpod/v23/`.
- **FR-019**: Training and inference scripts MUST record the full config (seed, commit SHA, hyperparameters, channel contract, data paths) into the checkpoint metadata. A second run with the same config + same commit MUST produce bit-identical weights (determinism verification).
- **FR-020**: Validation artifacts MUST include: input minimap RGB, ground-truth `height_257`, predicted disparity, predicted metric height (anchored), per-tile absolute error map, and (when CAI is enabled) the stitched multi-tile preview.

### Input Channel Contract (Default)

Channel indices into the encoder input tensor. Order is fixed so checkpoints can be loaded against any data store that emits the same channel contract.

| Index | Source V22 array | Channels | Dtype | Notes |
|------:|------|------:|------|------|
| 0–2 | `minimap_rgb` | 3 | uint8→float32 | normalized to [0,1], ImageNet std deviation |
| 3–6 | `alpha_256` | 4 | float32 | MCLY blend weights for the 4 texture layers |
| 7–10 | one-hot of `mcly_tileset_ids` (top layer per pixel) | 4 | float32 | 0/1 encoding of the dominant tileset per pixel; pruned to the top-K most common tilesets across the build, default K=256; pruned table shipped alongside the checkpoint |
| 11–13 | `normal_xyz` (resampled 257→256) | 3 | float32 | MCNR-derived; zero-filled + valid mask channel if absent |
| 14 | `terrain_valid_mask` | 1 | bool→float32 | derived from `mcnr_mask_257` + (1 - liquid_mask_256) + (1 - object_mask_257_binarized) |

Total: 15 channels by default. Degraded modes drop indices 7–10, 11–13, or both via the `--input-mode` flag. The model layer1 conv is sized to the active channel count.

### Loss Components

| Loss | Source | Weight Flag | Default | Purpose |
|------|------|------|------:|------|
| Affine-invariant Lssi | DepthAnything-V2 training code | (constant) | 1.0 | removes global-scale hallucination structurally |
| Gradient-matching Lgm | DepthAnything-V2 training code | (constant) | 0.5 | sharpens boundaries, prevents blurry predictions |
| Spatial Distance Constraint (SDC) | DepthAnything-AC paper | `--sdc-weight` | 0.1 | patch-level relative geometry, kills texture-template hallucination |
| GPCT overlap consistency | PRO paper (KAIST 2025) | `--gpct-weight` | 0.1 | cross-tile feature agreement, native consistency |
| Bias-Free Masking | PRO paper (KAIST 2025) | `--bias-free-mask-ratio` | 0.15 | input dropout, prevents dataset-bias overfit |

### RunPod Training Envelope (Concrete)

- GPU: 1× RTX 4090 24 GB (preferred), A10 24 GB (acceptable), L4 24 GB (acceptable), L40S 48 GB (oversized-headroom option)
- Image: `runpod/pytorch:2.3.1-cuda12.1-cudnn8-devel`
- Optimizer: bitsandbytes `PagedAdamW8bit`
- Mixed precision: bf16 (matmul autocast enabled)
- Gradient checkpointing: enabled on encoder
- Effective batch size: 16 (= 4 GPCT sub-tiles × 4 tiles per GPU step) with grad-accum 1
- Storage: 100 GB container, 200 GB network volume (V22 Zarr + checkpoints + logs)
- Estimated training wall-time: 12–24 hr per epoch over ~3 builds × ~10K tiles; 4–6 epochs to converge

### Inference Envelope (Concrete)

- Production server: any 6 GB+ VRAM card, fp16
- Peak VRAM: < 4 GB single forward pass
- Wall-time per tile: < 3 seconds (no CAI), < 15 seconds with CAI-R=16
- Optional batch wrapper for whole-map inference released as a CLI tool

### Key Entities

- **V23-HeightPredictor**: the full model = DA-V2-Small encoder (frozen + LoRA-r16 + replaced patch-embed) + V23HeightHead decoder + affine anchor head.
- **V23HeightDataset**: PyTorch Dataset adapter over V22 Zarr stores. Produces the documented Input Channel Contract + `height_257` target.
- **GPCTLoss**: Grouped Patch Consistency Training loss. L2 on overlapping region predictions across K sub-tiles, optionally with feature-level constraint.
- **CAIDataset / CAIInference**: inference-time dataset that emits overlapping sub-tile shifts and a running-mean accumulator.
- **V23Checkpoint**: checkpoint format with full config in metadata (seed, commit SHA, channel contract, hyperparameters, V22 store path hash).
- **RunPodBundle**: tar archive containing Python code + V22 Zarr subset + RunPod helper scripts per Spec 079.

## Success Criteria

### Measurable Outcomes

- **SC-001**: `V23-HeightPredictor` trains to completion without CUDA OOM on a single 24 GB RunPod GPU at the documented batch size + K=4 GPCT.
- **SC-002**: `infer_v23_height.py` produces bit-identical output across two runs differing only in seed (`torch.allclose(atol=0, rtol=0`).
- **SC-003**: Per-tile L1 height error on validation set drops by at least 25% vs the V21 baseline (the most recent trunk height model before this spec).
- **SC-004**: A 3×3 CAI-R=16 stitched validation preview shows no visible seam at tile boundaries (manual inspection of the saved preview PNG).
- **SC-005**: Cross-tile L1 difference (L1 between adjacent tile-edge pixel rows) is at least 50% lower with CAI than without.
- **SC-006**: RunPod-augmented determinism check: two separate Pods running the same config + same commit produce bit-identical final weights.
- **SC-007**: The RunPod bundle's `manifest.json` reports `contains_game_client_files: false` and no path inside the bundle resolves to a client path under `output/tmp/wowarchive-clients/` or any WoWArchive source.
- **SC-008**: Total trainable parameter count for `V23-HeightPredictor` is < 8M (DA-V2-Small patch-embed + LoRA adapters + decoder head + affine anchor head).

## Assumptions

- DepthAnything-V2-Small HuggingFace weights are loadable under the Apache-compatible license interpretation used by the DA-V2 release; using the encoder as a frozen backbone + LoRA fine-tune does not violate the model's non-commercial clause for this internal-research workflow.
- The V22 Zarr stores produced by Spec 088 will be available under `wow-viewer/output/datasets/v22/<build>.zarr` with the documented signal arrays.
- The 24 GB RunPod envelope is the primary training target; CUDA arch target is Ampere (RTX 4090 / A10) or Ada (RTX 4090 if using the "Ada" subclass of 4090) or Hopper (H100) or Ampere+Turing fallbacks.
- bitsandbytes `PagedAdamW8bit` produces deterministic optimizer steps under `torch.use_deterministic_algorithms(True)` on the same CUDA arch, host PyTorch version, and CUDA toolkit version. (Cross-arch reproducibility is a non-goal.)
- PatchFusion's CAI + PRO's GPCT combine without architectural conflict because GPCT is a training-time loss and CAI is an inference-time running mean; this assumption is documented in the spec body and confirmed by the absence of contract overlap between the two papers.
- All input channels can be sampled to 256×256 (with the height target upsampled to 257×257) without incurring significant aliasing on the 257-vertex grid. This matches V18/V19's existing behavior.
- bf16 mixed precision is sufficient for training-stability and final L1 quality. fp32 fallback exists but is documented as not required on RunPod.
- All V22 stores the model trains on include the documented Input Channel Contract arrays at the required shapes/dtypes. Missing-array fallback at the dataset level is supported, but disabling a channel entirely must be explicit via `--input-mode`.

## Relationship to Other Specs

- **Supersedes for the trunk height-regressor route**: Spec 066 (V19) — kept as historical context; its MCNR checkerboard fix work must be reused, but the height-regressor route is now V23 under this spec, not V19.
- **Builds on**: Spec 088 (V22 Enrichment From V18) — V23 consumes the V22 Zarr store as its only data interface.
- **Builds on**: Spec 079 (RunPod Integration Guide) — V23's bundle follows Spec 079's contract.
- **Complements**: Spec 077 (Minimap Deconstruction Engine) — V23 is the height-only terrain model that 077's User Story 3 anticipated; V23 may consume processed-minimap priors from 077's User Story 2 in a future integration phase, but its initial delivery uses raw V22 inputs.
- **Pauses**: Spec 068 (Fractal-Aware Height Loss) remains paused; V23's loss stack does not include spectral or fractal-dimension terms. Those can be added later if the affine-invariant + SDC + GPCT stack underperforms.