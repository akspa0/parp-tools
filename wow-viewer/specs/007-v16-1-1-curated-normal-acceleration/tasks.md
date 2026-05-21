# Tasks: V16.1.1 Curated Normal Acceleration

**Plan**: `007-v16-1-1-curated-normal-acceleration/plan.md`

---

## Phase 1: Contract And Routing

- [x] **1.1** Create the V16.1.1 spec pack
  - add `spec.md`
  - add `plan.md`
  - add `tasks.md`
- [x] **1.2** Update continuity routing
  - record V16.1.1 as the next normal-lane slice
  - keep V16.1 as the landed base

## Phase 2: Difficulty-Aware Curation

- [ ] **2.1** Add per-tile usefulness scoring to the normal curation profile
  - deformation richness
  - terrain-only validity
  - painted-alpha / MCLY presence
  - minimap-vs-target usefulness
- [ ] **2.2** Add difficulty buckets
  - `easy`
  - `medium`
  - `hard`
  - `pathological`
- [ ] **2.3** Preserve hard rejection for blank genesis and other known garbage
  - keep `blank_what_plate_tile` explicit
- [ ] **2.4** Publish bounded scouting-manifest guidance
  - mixed `400`-tile train pool
  - smaller validation pool

## Phase 3: Bucket-Aware Epoch Sampling

- [ ] **3.1** Extend manifest ingestion for bucket metadata
  - dataset/trainer startup sees bucket counts
- [ ] **3.2** Add bucket-biased epoch sampling
  - oversample `hard`
  - preserve some `easy` / `medium`
- [ ] **3.3** Emit sampler evidence
  - per-epoch bucket usage
  - selected tile mix logs

## Phase 4: Hard-Region Normal Loss

- [ ] **4.1** Refine the hard-region weight map
  - current detail boost becomes a richer region-weight surface
- [ ] **4.2** Keep terrain-only masking authoritative
  - object/liquid/invalid areas stay downweighted
- [ ] **4.3** Run focused comparison against the current detail-boost baseline
  - evidence roots for both runs

## Phase 5: Optional Uncertainty-Guided Normal Training

- [ ] **5.1** Add optional uncertainty head
  - CLI/config toggle
- [ ] **5.2** Add uncertainty-aware loss weighting
  - log uncertainty metrics
- [ ] **5.3** Add uncertainty review artifacts
  - separate from normal RGB validation

## Phase 6: Geometry-Consistency Supervision

- [ ] **6.1** Add local-relative normal consistency supervision
- [ ] **6.2** Record operator comparison surfaces
  - baseline
  - hard-region
  - uncertainty / consistency

## Phase 7: Operator Commands And Fresh-Chat Handoff

- [ ] **7.1** Publish V16.1.1 curation commands
  - full-corpus
  - scouting mode
- [ ] **7.2** Publish V16.1.1 normal-training commands
  - bucket-aware scouting run
  - longer resumed run
- [x] **7.3** Sync continuity and stop
  - memory-bank files updated
  - fresh chat can implement from this pack directly
