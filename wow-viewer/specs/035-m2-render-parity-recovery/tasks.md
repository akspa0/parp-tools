# Tasks: M2 Render Parity Recovery

**Input**: Design documents from `specs/035-m2-render-parity-recovery/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

## Phase 1: Setup (Shared Infrastructure)

- [x] T001 Create parity evidence root at `wow-viewer/output/tmp/m2-parity/` and add run naming conventions in `specs/035-m2-render-parity-recovery/quickstart.md`.
- [x] T002 Add a feature-scoped parity sample manifest doc at `specs/035-m2-render-parity-recovery/parity-samples.md` with initial 3.3.5 model/tile sample IDs.
- [x] T003 [P] Add a small route-decision contract note in `wow-viewer/docs/architecture/m2-native-client-research-2026-03-31.md` linking this feature.

## Phase 2: Foundational (Blocking Prerequisites)

- [x] T004 Implement a shared `M2RouteDecision` record and enums in `wow-viewer/src/viewer/WoWViewer/Rendering/` for world-route outcomes.
- [x] T005 Implement a shared `M2MaterialPassProfile` diagnostic record in `wow-viewer/src/viewer/WoWViewer/Rendering/`.
- [x] T006 Refactor world M2 load path to return route decision metadata from `wow-viewer/src/viewer/WoWViewer/Terrain/WorldAssetManager.cs`.
- [x] T007 Refactor WMO doodad M2 load path to return route decision metadata from `wow-viewer/src/viewer/WoWViewer/Rendering/WmoRenderer.cs`.
- [x] T008 [P] Extend probe output plumbing in `wow-viewer/src/viewer/WoWViewer/AssetProbe.cs` to print route decision metadata.
- [x] T009 Add a single formatter utility for route diagnostics in `wow-viewer/src/viewer/WoWViewer/Rendering/` and route all M2 probe/runtime logs through it.

## Phase 3: User Story 1 - Stable World Doodad Visibility (Priority: P1) 🎯 MVP

**Goal**: World M2 placements that are visible/selectable also render geometry reliably for tree and cutout-heavy assets.

**Independent Test**: Load a known 3.3.5 tree-heavy tile and verify sampled tree placements render geometry with bounds enabled.

- [ ] T010 [US1] Normalize adapted-M2 cutout threshold and depth-write policy in `wow-viewer/src/viewer/WoWViewer/Rendering/ModelRenderer.cs`.
- [ ] T011 [US1] Align transparent/cutout pass gating behavior for world M2 instances in `wow-viewer/src/viewer/WoWViewer/Terrain/WorldScene.cs` and `wow-viewer/src/core/WowViewer.Core.Runtime/World/Passes/WorldObjectPassCoordinator.cs`.
- [ ] T012 [US1] Ensure route-specific renderer wrappers in `wow-viewer/src/viewer/WoWViewer/Rendering/M2Renderer.cs` do not override or mask compatibility pass semantics.
- [ ] T013 [US1] Add focused tree/cutout runtime diagnostics (single-line per model) in `wow-viewer/src/viewer/WoWViewer/Rendering/ModelRenderer.cs`.
- [ ] T014 [US1] Validate on staged `3_3_5_12340` samples and record outputs in `wow-viewer/output/tmp/m2-parity/us1/`.

## Phase 4: User Story 2 - Deterministic M2 Load/Render Routing (Priority: P1)

**Goal**: Probe and runtime report the same deterministic world route decisions and fallback reasons.

**Independent Test**: Run `--probe-m2-adapter` and `--probe-m2-runtime` for each sample and compare route diagnostics.

- [ ] T015 [US2] Consolidate world route selection and fallback ordering in `wow-viewer/src/viewer/WoWViewer/Terrain/WorldAssetManager.cs`.
- [ ] T016 [US2] Consolidate WMO doodad route selection and fallback ordering in `wow-viewer/src/viewer/WoWViewer/Rendering/WmoRenderer.cs`.
- [ ] T017 [P] [US2] Add shared route-decision emission helper and use it in both loaders and probe code.
- [ ] T018 [US2] Add per-model fallback reason emission for adapter failure, skin resolution failure, and conversion fallback.
- [ ] T019 [US2] Capture deterministic route outputs for parity samples under `wow-viewer/output/tmp/m2-parity/us2/`.

## Phase 5: User Story 3 - Controlled Compatibility and Refactor Boundaries (Priority: P2)

**Goal**: Route policy changes are bounded, reviewable, and regression-detectable before manual world QA.

**Independent Test**: Run guard checks and verify route drift is reported when route policy changes.

- [ ] T020 [US3] Add route contract verification checks in `wow-viewer/src/viewer/WoWViewer/AssetProbe.cs` for probe modes.
- [ ] T021 [US3] Add a parity comparison script or command note in `specs/035-m2-render-parity-recovery/quickstart.md` with required output fields.
- [ ] T022 [US3] Add an explicit route-policy section to `wow-viewer/docs/architecture/m2-native-client-research-2026-03-31.md` summarizing approved primary/fallback paths.
- [ ] T023 [US3] Add a regression checklist entry in `specs/035-m2-render-parity-recovery/checklists/requirements.md` covering route drift checks.
- [ ] T024 [US3] Execute guard checks and store parity comparison evidence in `wow-viewer/output/tmp/m2-parity/us3/`.

## Phase 6: Polish & Cross-Cutting Concerns

- [ ] T025 [P] Update `specs/035-m2-render-parity-recovery/` docs with final evidence paths and known limitations.
- [ ] T026 Run `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` and record build result in feature notes.
- [ ] T027 Run targeted probes for all sample models and record a final pass/fail summary in `specs/035-m2-render-parity-recovery/parity-results.md`.

## Dependencies & Execution Order

- Phase 1 must complete first.
- Phase 2 blocks all user stories.
- US1 and US2 can proceed in parallel after Phase 2, but US1 validation should complete before declaring MVP.
- US3 depends on route and parity outputs from US1 and US2.
- Phase 6 depends on completion of selected user stories and validation evidence.

## Parallel Opportunities

- T003, T008, T017, and T025 are parallelizable.
- Probe evidence collection tasks across different models can run in parallel once route semantics are stable.

## Implementation Strategy

1. Land deterministic route contracts and shared diagnostics first.
2. Recover tree/cutout world visibility as the MVP gate.
3. Lock route boundaries with explicit guard checks and parity evidence.
