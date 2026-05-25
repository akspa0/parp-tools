# Tasks: V18 Canvas Paste Refinement Layer

## Phase 1 — Canvas Mining Surface

- [ ] T001 Create `scripts/mine_v18_pastes_canvas.py` with bounded CLI (`--builds`, `--maps`, `--max-tiles`, `--seed`).
- [ ] T002 Add stitched map-canvas assembly from V16 dataset rows with tile coordinate placement.
- [ ] T003 Add canvas-space signal generation (`alpha`/`transition`/`hard_region`/mask composites).
- [ ] T004 Add canvas candidate detection that emits multi-tile regions with `canvas_bbox` and `tile_coverage`.
- [ ] T005 Add debug overlay outputs and machine-readable evidence files (`summary.json`, `candidates.jsonl`, config snapshot).
- [ ] T006 Validate bounded run on at least one map with confirmed multi-tile candidate output.

## Phase 2 — Cross-Build Dedupe + Library

- [ ] T007 Add deterministic fingerprinting for canvas candidates.
- [ ] T008 Add alpha-layer-aware signatures (MCAL-layer descriptors) to candidate and dedupe metadata.
- [ ] T009 Add cross-build cluster assignment and canonical exemplar selection.
- [ ] T010 Add variant lineage metadata fields (`cluster_id`, `canonical_id`, `variant_rank`).
- [ ] T011 Emit deduped outputs (`candidates_deduped.jsonl`, cluster summaries, duplicate stats).
- [ ] T012 Add cluster atlas outputs for manual QA.
- [ ] T013 Validate dedupe stability across two reruns with identical seed/config.

## Phase 3 — Refined Manifest Generation

- [ ] T014 Create `scripts/build_v18_refined_manifest.py` consuming deduped candidate clusters.
- [ ] T015 Add normal-aware quality gates and cluster-balancing selection knobs.
- [ ] T016 Emit trainer-compatible refined manifest format + evidence (`selection_hash`, cluster distribution, duplicate ratio).
- [ ] T017 Validate manifest loads in `train_v16_1_normal.py` with bounded dry run.

## Phase 4 — Composition Graph Layer

- [ ] T018 Build composition-graph generator from deduped canvas candidates (adjacency/co-occurrence edges).
- [ ] T019 Add MCNK AreaID overlap extraction and dominant AreaID labeling per candidate/group.
- [ ] T020 Emit stable composition-family IDs and macro-style summaries.
- [ ] T021 Integrate composition-family balancing metadata into refined manifest outputs.
- [ ] T022 Validate deterministic graph stats across reruns.

## Phase 5 — Auto-Naming + Library Metadata

- [ ] T023 Add deterministic paste-family naming generator from role/shape/layer/AreaID descriptors.
- [ ] T024 Add alias and naming confidence metadata fields plus review state (`auto`, `reviewed`, `locked`).
- [ ] T025 Emit/validate paste-library catalog outputs with stable IDs and naming metadata.

## Phase 6 — V18 Baseline Launch Contract

- [ ] T026 Define baseline run profiles from refined manifests (small/medium/large pool).
- [ ] T027 Execute first bounded V18 normal run and collect throughput/convergence evidence.
- [ ] T028 Write comparison report vs prior non-refined curation runs.
