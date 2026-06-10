# Implementation Plan: Synthetic Terrain Guidance Model

**Spec**: `008-synthetic-terrain-guidance-model/spec.md`
**Created**: 2026-05-22

## Phase 1: Model Architecture Decision

**Goal**: Choose the generative modeling approach and define the training contract before writing code.

### Step 1.1 — Survey generative approaches
Evaluate candidate architectures: latent diffusion, VAE, score-based model, autoregressive. Document tradeoffs for multi-signal terrain data (variable resolutions, mix of continuous/categorical targets, conditioning requirements).
**Validation**: A brief architecture decision record in the spec folder.

### Step 1.2 — Define the joint signal contract
Specify which V16 Zarr arrays the guidance model consumes/produces, how they are preprocessed (normalization, encoding for categoricals), and the loss function families for each signal type.
**Validation**: A signal mapping table with shapes, dtypes, normalization, and loss function per signal.

### Step 1.3 — Define the conditioning interface
Specify how partial signal subsets are passed as conditioning (masking strategy, embedding for missing signals), how difficulty parameters are injected, and how the consistency score is derived.
**Validation**: The conditioning interface is documented and can be reviewed before implementation.

### Step 1.4 — Choose initial architecture and scope
Pick one architecture for the first proof. Default to latent diffusion unless evidence rules it out. Define what "done" means for the proof (synthetic minimap passes visual inspection).
**Validation**: Architecture choice is documented with rationale. The proof gate is written down.

## Phase 2: Joint-Distribution Dataset

**Goal**: Prepare the V16 Zarr corpus for joint-distribution training — all signals, aligned, preprocessed, loadable as a single batch.

### Step 2.1 — Create `GuidanceDataset` loader
Write `wow-viewer/data-harvester/src/harvester/guidance_dataset.py` — loads all V16 signals for a tile simultaneously: `minimap_rgb_256`, `height_257`, `normal_xyz`, `alpha_256`, `holes_16`, `liquid_mask_256`, `liquid_type`, `mcly_texture_ids`, `mcly_layer_mask`. Must support the same curation manifest filtering as V16.1.
**Validation**: A smoke script loads 10 tiles from a V16 Zarr store and prints shapes/dtypes for every signal.

### Step 2.2 — Define preprocessing transforms
Implement per-signal normalization: minimap to `[0,1]`, height z-score, normals unit-vector `[-1,1]`, alpha `[0,1]`, holes binary, liquid mask `[0,1]`, liquid type one-hot or embed, MCLY IDs as categorical, MCLY mask `[0,1]`.
**Validation**: Preprocessed batch shapes and value ranges are verified against the spec contract.

### Step 2.3 — Implement train/val split and curation
Reuse the V16.1 curation manifest infrastructure to filter and split tiles. The guidance model should train on the same curated tile set as the V16.1 trainers.
**Validation**: A curation-filtered tile list can be loaded and split without errors.

### Step 2.4 — Add 8-way augmentation support
Port the V16.1 8-way geometric augmentation (flip, rotate 90, transpose) to the guidance dataset. Augmentation must consistently transform all signals jointly so they remain aligned.
**Validation**: Augmented samples maintain pixel-perfect alignment across all signal arrays.

### Step 2.5 — Write dataset smoke proof
Run a 1-epoch CPU smoke that iterates the entire training set through the dataset loader and verifies no crashes, shape mismatches, or NaN values.
**Validation**: Smoke root exists at `wow-viewer/output/datasets/guidance/smoke_dataset/`.

## Phase 3: Guidance Model Implementation

**Goal**: Implement the chosen generative model architecture.

### Step 3.1 — Implement model backbone
Write `wow-viewer/data-harvester/src/harvester/guidance_model.py` with the chosen architecture. Include encoder, latent/sampling path (if diffusion/VAE), decoder, and conditioning injection points.
**Validation**: Model forward pass runs on a dummy batch with correct output shapes.

### Step 3.2 — Implement training loop
Write `wow-viewer/data-harvester/scripts/train_guidance.py`. Include loss computation for all signal types, validation metrics, checkpointing, and log/metrics.
**Validation**: Training loop starts and completes 10 steps without errors.

### Step 3.3 — Implement generation/sampling
Write the generation entrypoint: given conditioning signals (or none), produce a complete synthetic tile. For latent diffusion: implement the denoising loop. For VAE: implement the decoder pass from sampled latent.
**Validation**: A synthetic tile can be generated and saved to disk without runtime errors.

### Step 3.4 — Implement consistency scoring
Write the consistency score function: given a complete set of signals, compute the model's likelihood/energy/score. For diffusion: the negative log-likelihood estimate. For VAE: the ELBO reconstruction term.
**Validation**: Two tiles (one real, one deliberately misaligned) produce distinguishable scores.

### Step 3.5 — Port V16.1 runtime seams
Carry forward `torch.compile`, worker auto-resolution, persistent workers, prefetch factor, and gradient accumulation from the V16.1 shared trainer.
**Validation**: A compile-enabled smoke run completes without graph-break warnings.

## Phase 4: Training & Smoke Proof

**Goal**: Train the guidance model and verify it produces visually plausible synthetic tiles.

### Step 4.1 — Run short GPU smoke
Train `guidance` model on a 400-tile scouting pool for 50 epochs on GPU. Monitor loss curves for all signal types. Verify no divergence or NaN.
**Validation**: Smoke root exists at `wow-viewer/models/guidance/runs/smoke_scouting/`. Loss curves show stable convergence.

### Step 4.2 — Generate first synthetic tiles
Sample 50 unconditional synthetic tiles from the smoke checkpoint. Save as a synthetic Zarr store. Inspect visually — minimaps should show plausible terrain colors and structure.
**Validation**: Synthetic minimaps pass initial visual inspection (no obvious garbage, recognizable terrain shapes). At least 40/50 tiles are visually plausible.

### Step 4.3 — Generate conditional synthetic tiles
Sample 50 tiles conditioned on only `(height_257)` — the model should fill in plausible minimap, normals, alpha, etc. Sample 50 conditioned on `(minimap_rgb_256, normal_xyz)` — the model should fill in the rest.
**Validation**: Conditional samples are consistent with their conditioning signals. Height-conditioned tiles produce minimaps that correlate with the input height field.

### Step 4.4 — Run full-corpus training
Launch a full training run on the complete V16 six-build corpus (if smoke passes). Target: 200+ epochs with best-gated checkpointing.
**Validation**: Full run completes and best checkpoint is saved.

### Step 4.5 — Evaluate diversity
Compute diversity metrics across 1000 synthetic samples: pairwise distance in signal space, number of unique clusters, coverage of the real data's PCA manifold.
**Validation**: Diversity metrics are reported. Mode collapse is flagged if present.

## Phase 5: Synthetic Pair Generation Pipeline

**Goal**: Export synthetic pairs in a format the V16.1 trainers can consume, enabling data-independent training.

### Step 5.1 — Create synthetic Zarr store builder
Write a script that generates N synthetic tiles from the guidance model and writes them as a V16-compatible Zarr store (`synthetic_guidance.zarr`) with matching array names/shapes/dtypes.
**Validation**: `V161Dataset` can load from the synthetic Zarr store without errors.

### Step 5.2 — Condition on difficulty parameters
Wire the difficulty conditioning surface (deformation level, gradient strength, alpha complexity) so the operator can steer generation toward `hard`/`pathological` terrain.
**Validation**: 100 `hard`-conditioned tiles show measurably higher deformation metrics than 100 unconditioned tiles.

### Step 5.3 — Run V16.1 training on synthetic data
Train a V16.1 normal model on 5000 synthetic `(minimap, normal)` pairs. Evaluate on 100 held-out real tiles. Report `val_normal_angle`.
**Validation**: SC-002 from the spec: synthetic-trained model achieves `val_normal_angle` within 15% of real-data-trained model.

### Step 5.4 — Write operator generation commands
Document generation commands in `data-harvester/README.md`: unconditional, conditional on arbitrary subsets, difficulty-steered, batch size, GPU fallback.
**Validation**: Commands are documented and produce repeatable results.

## Phase 6: Consistency Critic Integration

**Goal**: Wire the guidance model's plausibility scoring into the V16.1 inference pipeline.

### Step 6.1 — Implement post-inference scoring
Write a script that takes a V16.1 stitched inference `.pred.zarr` store and scores each tile through the guidance model, writing a `guidance_score` array to the output.
**Validation**: Scores are non-NaN across all tiles. Distribution is visible (not all identical).

### Step 6.2 — Correlate scores with visible errors
Run scoring on an inference run with known failure modes. Verify that low scores correlate with visible cross-signal inconsistencies (manual inspection of top/bottom decile).
**Validation**: Bottom-decile tiles show visible inconsistencies; top-decile tiles look consistent.

### Step 6.3 — Write operator scoring commands
Document the consistency scoring workflow in README: how to score an inference run, how to inspect low-scoring tiles, the reporting format.
**Validation**: Commands are documented and produce a report with actionable results.

## Phase 7: Documentation & Handoff

**Goal**: Capture the final guidance model contract, training commands, and known limitations.

### Step 7.1 — Write architecture doc
Write `wow-viewer/docs/architecture/synthetic-terrain-guidance-model-2026-05-22.md` covering model architecture, training data, hyperparameters, and generation surface.
**Validation**: Doc exists and matches the implemented model.

### Step 7.2 — Update memory bank
Record the guidance model's status in `activeContext.md` and `progress.md`. Note what is proven and what remains.
**Validation**: Memory bank files mention the guidance model with current status.

### Step 7.3 — Sync continuity and stop
Ensure fresh chats route to the next bounded implementation slice for the guidance model.
**Validation**: Continuity docs point to the right next step.
