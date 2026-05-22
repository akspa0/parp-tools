# Tasks: V16.2 Patched Signal Expansion

**Plan**: `011-v16-2-patched-signal-expansion/plan.md`

---

## Phase 1: Contract And Signal Inventory

- [ ] **1.1** Promote the active dataset contract from V16 to V16.2
  - update the active dataset/model docs
  - make `V16.2` the named compact-corpus contract
- [ ] **1.2** Enumerate the upgraded signal inventory
  - precise-mask family
  - renderer-truth visibility surfaces
  - terrain-only guidance surfaces such as `no_object_minimap`
- [ ] **1.3** Define compatibility rules for old and new object-mask surfaces
  - keep coarse compatibility semantics explicit
  - define richer precise-mask semantics explicitly
- [ ] **1.4** Freeze the current proof boundary
  - document that real renderer-truth capture proof exists today for `0_5_3_3368`
  - document that real renderer-truth capture proof exists today for `3_3_5_12340`
  - avoid implying six-build closure before it exists

## Phase 2: Sidecar Store Schema

- [ ] **2.1** Define the compact sidecar-store layout for `V16.2` arrays
  - shapes
  - dtypes
  - optional-coverage rules
- [ ] **2.2** Extend metadata and presence flags
  - `index.parquet`
  - validation metadata
  - provenance / coverage reporting
- [ ] **2.3** Define resumable sidecar patch-state rules
  - interrupted patch behavior
  - partial-coverage metadata behavior

## Phase 3: Sidecar Patch-And-Reindex Workflow

- [ ] **3.1** Add bounded signal patch commands
  - patch precise-mask arrays into sidecar stores
  - patch terrain-guidance arrays into sidecar stores
- [ ] **3.2** Reindex metadata after patching
  - refresh presence flags
  - refresh validation summaries
  - keep stores directly consumable after patch
- [ ] **3.3** Measure compactness after upgrade
  - before/after store size comparison
  - confirm storage growth stays acceptable

## Phase 4: Cross-Build Validation Matrix

- [ ] **4.1** Run the remaining real capture proofs
  - `0_5_5_3494`
  - `0_7_0_3694`
  - `3_0_1_8303`
  - `4_0_0_11927`
- [ ] **4.2** Gate later base-store merge on matrix completion
  - define when sidecar-only is sufficient
  - define when merge-back into canonical stores is allowed

## Phase 5: Signal Generation And Ingestion

- [ ] **5.1** Ingest precise-mask sources into the V16.2 patch workflow
  - map current precise-mask artifacts into store arrays
  - emit coverage metadata for patched tiles
- [ ] **5.2** Ingest renderer-derived terrain guidance
  - map `no_object_minimap` into the sidecar contract
  - keep spatial alignment explicit and reviewable
- [ ] **5.3** Define mixed-coverage fallback behavior
  - patched tiles remain usable
  - unpatched tiles remain usable
  - mixed-build coverage stays valid

## Phase 6: Loader And Trainer Upgrade

- [ ] **6.1** Upgrade dataset loaders to read V16.2 signals
  - precise masks
  - terrain-guidance arrays
  - upgraded presence metadata
- [ ] **6.2** Upgrade the normal lane as the first bounded consumer
  - consume new precise-mask surfaces
  - consume terrain-only guidance without replacing raw terrain truth
- [ ] **6.3** Keep compatibility fallbacks explicit
  - mixed-coverage smoke read
  - mixed-coverage smoke training proof

## Phase 7: Corpus Validation On Existing Stores

- [ ] **7.1** Patch one representative existing build sidecar
  - real compact store
  - no full rebuild
  - metadata refreshed
- [ ] **7.2** Validate spatial alignment and review artifacts
  - inspect precise-mask alignment
  - inspect terrain-guidance alignment
  - confirm review outputs are credible
- [ ] **7.3** Expand patch-and-reindex across remaining target stores
  - run the same bounded path on the retained corpus
  - confirm upgraded coverage metadata on all target stores

## Phase 8: Operator Workflow And Handoff

- [ ] **8.1** Publish operator patch commands
  - locate or generate new signal artifacts
  - patch an existing store
  - reindex and validate it
- [ ] **8.2** Publish training guidance for upgraded stores
  - required versus optional new signals
  - fallback behavior when signals are missing
- [ ] **8.3** Document the first orchestration seam
  - choose builds
  - choose signal families
  - choose output mode for model-facing data prep
- [ ] **8.4** Sync continuity and stop
  - memory-bank routing
  - `V16.2` as the active corpus-upgrade lane
