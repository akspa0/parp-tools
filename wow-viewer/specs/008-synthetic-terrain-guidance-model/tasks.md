# Tasks: Synthetic Terrain Guidance Model

**Plan**: `008-synthetic-terrain-guidance-model/plan.md`

---

## Phase 1: Model Architecture Decision

- [ ] **1.1** Survey generative approaches and document tradeoffs
  - latent diffusion, VAE, score-based, autoregressive
  - variable resolutions, continuous vs categorical signals, conditioning
- [ ] **1.2** Define the joint signal contract
  - signal mapping table: shapes, dtypes, normalization, loss function
- [ ] **1.3** Define the conditioning interface
  - partial signal masking, difficulty parameter injection, consistency score derivation
- [ ] **1.4** Choose initial architecture and scope
  - decision record with rationale, proof gate definition

## Phase 2: Joint-Distribution Dataset

- [ ] **2.1** Create `GuidanceDataset` loader
  - `wow-viewer/data-harvester/src/harvester/guidance_dataset.py`
  - loads all V16 signals jointly, supports curation manifest filtering
- [ ] **2.2** Define preprocessing transforms
  - per-signal normalization, categorical encoding
- [ ] **2.3** Implement train/val split and curation
  - reuse V16.1 curation manifest infrastructure
- [ ] **2.4** Add 8-way geometric augmentation
  - consistent joint transforms across all signals
- [ ] **2.5** Write dataset smoke proof
  - 1-epoch CPU smoke, verify no crashes or NaN

## Phase 3: Guidance Model Implementation

- [ ] **3.1** Implement model backbone
  - `wow-viewer/data-harvester/src/harvester/guidance_model.py`
  - encoder, latent/sampling path, decoder, conditioning injection
- [ ] **3.2** Implement training loop
  - `wow-viewer/data-harvester/scripts/train_guidance.py`
  - multi-signal loss, validation, checkpointing
- [ ] **3.3** Implement generation/sampling
  - unconditional and conditional generation entrypoint
- [ ] **3.4** Implement consistency scoring
  - likelihood/energy/score from the model for arbitrary signal sets
- [ ] **3.5** Port V16.1 runtime seams
  - `torch.compile`, worker auto-resolution, grad accumulation

## Phase 4: Training & Smoke Proof

- [ ] **4.1** Run short GPU smoke on 400-tile scouting pool
  - 50 epochs, verify stable convergence
- [ ] **4.2** Generate and inspect first synthetic tiles
  - 50 unconditional samples, visual plausibility check (40/50 pass)
- [ ] **4.3** Generate conditional synthetic tiles
  - 50 height-conditioned, 50 minimap+normal-conditioned
- [ ] **4.4** Run full-corpus training on six-build V16 corpus
  - 200+ epochs with best-gated checkpointing
- [ ] **4.5** Evaluate diversity across 1000 synthetic samples
  - pairwise distances, cluster count, PCA manifold coverage

## Phase 5: Synthetic Pair Generation Pipeline

- [ ] **5.1** Create synthetic Zarr store builder
  - V16-compatible output, consumable by `V161Dataset`
- [ ] **5.2** Condition on difficulty parameters
  - steer generation toward hard/pathological terrain
- [ ] **5.3** Train V16.1 normal on 5000 synthetic pairs
  - compare against real-data-trained baseline
- [ ] **5.4** Write operator generation commands
  - document in `data-harvester/README.md`

## Phase 6: Consistency Critic Integration

- [ ] **6.1** Implement post-inference scoring script
  - score V16.1 `.pred.zarr` stores, write `guidance_score` array
- [ ] **6.2** Correlate scores with visible cross-signal errors
  - manual inspection of top/bottom decile
- [ ] **6.3** Write operator scoring commands
  - document workflow in README

## Phase 7: Documentation & Handoff

- [ ] **7.1** Write architecture doc
  - `wow-viewer/docs/architecture/synthetic-terrain-guidance-model-2026-05-22.md`
- [ ] **7.2** Update memory bank
  - `activeContext.md`, `progress.md`
- [ ] **7.3** Sync continuity and stop
