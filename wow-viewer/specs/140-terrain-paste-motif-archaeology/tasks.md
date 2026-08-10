# Tasks: Terrain Paste and Fractal Motif Archaeology

## Dependencies

```text
Setup -> Foundation -> US1 -> Paint-order evidence -> US2 -> US3 -> US4 -> US5 -> Alpha fan-out -> Scale join -> Curriculum -> Polish
                                                   \-> US3/US4 may run in parallel after US2
```

US1 is the MVP: a deterministic corpus and visual atlas. US2 must pass before any neural motif
model is considered. US4 is blocked until the retrieval contract is stable. US5 remains optional.

## Phase 1: Setup

- [ ] T001 Create the Spec 140 package directories under `wow-viewer/data-harvester/src/harvester/v60/`, `wow-viewer/data-harvester/scripts/`, and `wow-viewer/data-harvester/tests/v60/`.
- [ ] T002 Add the Spec 140 module exports and script entrypoint stubs in `wow-viewer/data-harvester/src/harvester/v60/__init__.py` and `wow-viewer/data-harvester/scripts/`.
- [ ] T003 [P] Add the initial deterministic configuration fixture at `wow-viewer/data-harvester/tests/v60/fixtures/terrain_motif_config.json`.

## Phase 2: Foundation

- [ ] T004 Define `ObservationWindow`, `SignalBundle`, and availability states in `wow-viewer/data-harvester/src/harvester/v60/terrain_motif_corpus.py`.
- [ ] T005 Define `TilesetProfile`, `MotifCandidate`, `PasteFamily`, `BrushScaleRecord`, `DifficultyGuidance`, and `GuidanceBundle` in `wow-viewer/data-harvester/src/harvester/v60/terrain_motif_models.py`.
- [ ] T006 Implement deterministic content hashes and fail-closed manifest validation in `wow-viewer/data-harvester/src/harvester/v60/terrain_motif_contract.py`.
- [ ] T007 [P] Add contract tests for missing signals, validation-only height, provenance, and hash requirements in `wow-viewer/data-harvester/tests/v60/test_terrain_motif_contract.py`.
- [ ] T008 Implement arbitrary-origin window extraction with tile/chunk boundary metadata in `wow-viewer/data-harvester/src/harvester/v60/terrain_motif_corpus.py`.
- [ ] T009 Add source-group and paste-family split checks that reject overlapping leakage in `wow-viewer/data-harvester/src/harvester/v60/terrain_motif_splits.py`.

## Phase 3: User Story 1 — Build a motif atlas

**Goal**: Produce a deterministic synthetic/real corpus and visual atlas that makes signal variety and availability reviewable.

**Independent test**: Build the control-only corpus and confirm every required terrain family, variant, boundary case, signal availability state, and provenance field appears in the manifest and atlas.

- [ ] T010 [P] [US1] Implement observation/albedo-confidence and gradient summaries in `wow-viewer/data-harvester/src/harvester/v60/terrain_motif_descriptors.py`.
- [ ] T011 [P] [US1] Implement height, curvature, and transition summaries in `wow-viewer/data-harvester/src/harvester/v60/terrain_motif_descriptors.py`.
- [ ] T012 [P] [US1] Implement alpha-layer, texture-layer, and tileset-ID summaries in `wow-viewer/data-harvester/src/harvester/v60/terrain_motif_descriptors.py`.
- [ ] T013 [P] [US1] Implement optional auxiliary-channel and normalized object-slot summaries in `wow-viewer/data-harvester/src/harvester/v60/terrain_motif_descriptors.py`.
- [ ] T014 [US1] Implement the corpus builder CLI in `wow-viewer/data-harvester/scripts/v60_build_terrain_motif_corpus.py`.
- [ ] T015 [US1] Implement the corpus validator CLI in `wow-viewer/data-harvester/scripts/v60_validate_terrain_motif_corpus.py`.
- [ ] T016 [US1] Implement the multi-signal visual atlas CLI in `wow-viewer/data-harvester/scripts/v60_visualize_terrain_motif_corpus.py`.
- [ ] T017 [US1] Add corpus, descriptor, and atlas tests in `wow-viewer/data-harvester/tests/v60/test_terrain_motif_corpus.py` and `wow-viewer/data-harvester/tests/v60/test_terrain_motif_descriptors.py`.

## Phase 4: Paint-order and sculpt-intent evidence

**Goal**: Test whether ordered alpha additions and recurring paste regions provide an upstream terrain-intent scaffold.

**Independent test**: Synthetic controls with known opaque layer-0 base, layer-1 paste, later additions, and sculpt order recover the sequence; real samples report intact, retextured, resculpted, unknown, or insufficient-data status without fabricating `alpha_0`.

- [ ] T018 [P] [US1] Preserve opaque MCLY layer 0, first alpha-bearing layer 1, later layer order, MCAL offsets, and alpha availability in `wow-viewer/data-harvester/src/harvester/v60/terrain_motif_corpus.py`.
- [ ] T019 [US1] Implement cumulative and incremental alpha occupancy descriptors in `wow-viewer/data-harvester/src/harvester/v60/terrain_paint_order.py`.
- [ ] T020 [US1] Implement paint/paste-to-relief relationship scoring and status classification in `wow-viewer/data-harvester/src/harvester/v60/terrain_paint_order.py`.
- [ ] T021 [US1] Implement the paint-order analysis CLI in `wow-viewer/data-harvester/scripts/v60_analyze_terrain_paint_order.py`.
- [ ] T022 [US1] Add synthetic known-order and real missing-base-alpha tests in `wow-viewer/data-harvester/tests/v60/test_terrain_paint_order.py`.

## Phase 5: User Story 2 — Retrieve recurring paste families

**Goal**: Prove that known transformed motifs can be retrieved across arbitrary tile/chunk boundaries and remain separate from generic fractal complexity.

**Independent test**: Run the transformed synthetic benchmark and verify family ranking, transform estimates, boundary support, and no source-family leakage.

- [ ] T023 [US2] Add deterministic transformed motif query/reference generation in `wow-viewer/data-harvester/src/harvester/v60/terrain_motif_benchmark.py`.
- [ ] T024 [US2] Implement multiscale spatial descriptor distance and correlation baselines in `wow-viewer/data-harvester/src/harvester/v60/terrain_motif_retrieval.py`.
- [ ] T025 [US2] Implement cross-boundary matching and transform estimation in `wow-viewer/data-harvester/src/harvester/v60/terrain_motif_retrieval.py`.
- [ ] T026 [US2] Implement recurring/unconfirmed/rejected family promotion rules in `wow-viewer/data-harvester/src/harvester/v60/terrain_motif_retrieval.py`.
- [ ] T027 [US2] Implement the retrieval report and match-atlas CLI with separate atomic, block, and macro metrics in `wow-viewer/data-harvester/scripts/v60_retrieve_terrain_motifs.py`.
- [ ] T028 [US2] Add retrieval and leakage tests in `wow-viewer/data-harvester/tests/v60/test_terrain_motif_retrieval.py`.

## Phase 6: User Story 3 — Separate tileset identity from geometry

**Goal**: Measure tileset/alpha/auxiliary evidence independently from height and avoid treating a correlated appearance signal as a geometry target.

**Independent test**: Hold geometry fixed while changing tileset profiles, then hold tileset profiles fixed while changing geometry; verify independent descriptor/report behavior.

- [ ] T029 [P] [US3] Implement tileset profile assembly and build-family provenance in `wow-viewer/data-harvester/src/harvester/v60/terrain_tileset_profiles.py`.
- [ ] T030 [P] [US3] Implement per-channel geometry correlation reports in `wow-viewer/data-harvester/src/harvester/v60/terrain_tileset_profiles.py`.
- [ ] T031 [US3] Add tileset profile validation tests in `wow-viewer/data-harvester/tests/v60/test_terrain_tileset_profiles.py`.

## Phase 7: User Story 4 — Feed bounded guidance into Spec 139

**Goal**: Compare clean-signal reconstruction with and without validated motif or tileset guidance.

**Independent test**: Run parity, motif-guided, tileset-guided, and combined manifests on held-out synthetic families and report per-signal and seam metrics.

- [ ] T032 [US4] Implement the Spec 139 guidance-bundle adapter in `wow-viewer/data-harvester/src/harvester/v60/terrain_motif_guidance.py`.
- [ ] T033 [US4] Add parity/motif/tileset/combined ablation manifest generation in `wow-viewer/data-harvester/scripts/v60_build_terrain_guidance.py`.
- [ ] T034 [US4] Add per-signal, seam, confidence, and baseline metrics in `wow-viewer/data-harvester/src/harvester/v60/terrain_motif_guidance.py`.
- [ ] T035 [US4] Add a validation-only 0.x/1.x transfer adapter with explicit client-backed provenance in `wow-viewer/data-harvester/scripts/v60_build_terrain_guidance.py`.
- [ ] T036 [US4] Add guidance contract and ablation tests in `wow-viewer/data-harvester/tests/v60/test_terrain_motif_guidance.py`.

## Phase 8: User Story 5 — Optional object-slot evidence

**Goal**: Preserve normalized object-placement structure as a separable future signal without making exact object identity part of terrain reconstruction.

**Independent test**: Controls with none/sparse/dense/overlap/boundary-crossing placements produce optional slot evidence without changing terrain or motif hashes.

- [ ] T037 [US5] Add normalized object-slot encoding to `wow-viewer/data-harvester/src/harvester/v60/terrain_motif_descriptors.py`.
- [ ] T038 [US5] Add object-slot isolation tests in `wow-viewer/data-harvester/tests/v60/test_terrain_motif_descriptors.py`.

## Phase 9: Alpha evidence fan-out

**Goal**: Preserve every available alpha layer before deriving competing interpretations of its
structure.

**Independent test**: A corpus audit reports complete source-layer provenance and reproduces raw,
transition, atomic, block, macro, ordered-layer, and cross-tile views without silently replacing
missing or opaque data with empty masks.

- [ ] T039 [US1] Add lossless alpha-layer references and MCLY/MCAL/tile/map/build provenance to `wow-viewer/data-harvester/src/harvester/v60/terrain_motif_models.py`.
- [ ] T040 [US1] Implement independent alpha occupancy, transition, ordered-layer, and cross-tile view availability in `wow-viewer/data-harvester/src/harvester/v60/terrain_alpha_evidence.py`.
- [ ] T041 [US1] Add alpha coverage audit and no-fabricated-empty-mask tests in `wow-viewer/data-harvester/tests/v60/test_terrain_alpha_evidence.py`.

## Phase 10: Complementary brush-scale join

**Goal**: Preserve the early Python atomic brush evidence and later C# blocky/macro segmentation
as linked but independently reviewable scales.

**Independent test**: Synthetic cross-tile controls and a bounded real map produce atomic,
paste-block, and macro-prefab-context records with stable parent/child links, while one-off,
unlinked, and boundary-truncated records remain valid.

- [ ] T042 [US1] Add atomic-brush, paste-block, and macro-prefab-context records to `wow-viewer/data-harvester/src/harvester/v60/terrain_motif_models.py` with independent status and confidence.
- [ ] T043 [US1] Implement spatial/provenance parent-child linking across alpha components, blocky regions, and full-map macro regions in `wow-viewer/data-harvester/src/harvester/v60/terrain_motif_hierarchy.py`.
- [ ] T044 [US1] Add hierarchy atlas rows and per-scale validation metrics to `wow-viewer/data-harvester/scripts/v60_visualize_terrain_motif_corpus.py` and `wow-viewer/data-harvester/scripts/v60_validate_terrain_motif_corpus.py`.
- [ ] T045 [US1] Add tests proving atomic records are not promoted to prefabs and macro/block records are not discarded as invalid connected components in `wow-viewer/data-harvester/tests/v60/test_terrain_motif_hierarchy.py`.

## Phase 11: Curriculum difficulty guidance

**Goal**: Use a frozen synthetic reference model to prioritize learnable-hard controls without
turning validation error into staleness or truth labeling.

**Independent test**: The same checkpoint, corpus, scoring configuration, and seed reproduce
identical per-signal scores, bands, and sampling weights without changing labels or provenance.

- [ ] T046 [US4] Implement frozen-reference per-signal, seam/boundary, confidence, and coverage scoring in `wow-viewer/data-harvester/src/harvester/v60/terrain_difficulty_guidance.py`.
- [ ] T047 [US4] Implement deterministic `easy`, `learnable_hard`, and `pathological` banding with `not_staleness: true` in `wow-viewer/data-harvester/scripts/v60_score_terrain_difficulty.py`.
- [ ] T048 [US4] Add curriculum guidance invariance tests proving scores cannot alter labels, provenance, split ownership, or signal availability in `wow-viewer/data-harvester/tests/v60/test_terrain_difficulty_guidance.py`.

## Phase 12: Polish and handoff

- [ ] T049 [P] Update `wow-viewer/specs/140-terrain-paste-motif-archaeology/quickstart.md` with only validated commands after G0/G1.
- [ ] T050 [P] Add deterministic run-summary and visual-review paths to `wow-viewer/specs/140-terrain-paste-motif-archaeology/contracts/motif-pipeline.schema.md`.
- [ ] T051 Update `wow-viewer/memory-bank/activeContext.md`, `wow-viewer/memory-bank/workstream-terrain-ml.md`, and `wow-viewer/memory-bank/progress.md` after each completed phase.

## Implementation strategy

1. Deliver US1 as the smallest useful artifact: corpus manifest plus atlas.
2. Prove paint-order evidence on synthetic known-order controls before making it a model input.
3. Prove US2 with classical retrieval before adding a neural motif model.
4. Keep US3 and US4 independently ablatable; a better aggregate score cannot hide a dead signal.
5. Add US5 only after terrain guidance has a measured benefit.
6. The user owns corpus generation, real-client runs, and all GPU training. Codex prepares and validates the bounded code path only.
