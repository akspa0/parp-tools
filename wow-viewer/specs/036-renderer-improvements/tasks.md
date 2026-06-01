# Tasks: Renderer Improvements Convergence

**Input**: Design documents from `specs/036-renderer-improvements/`

**Prerequisites**: spec.md, plan.md, research.md, data-model.md, contracts/

## Phase 1: Setup (Shared Infrastructure)

- [ ] T001 Create a source-spec mapping table in `specs/036-renderer-improvements/research.md` that links major slices from specs 030, 031, and 032 to convergence phases.
- [ ] T002 Add a final evidence-root convention for convergence work in `specs/036-renderer-improvements/quickstart.md`.
- [ ] T003 [P] Add convergence-owner notes to any remaining source-pack entrypoints in `specs/030-wmo-render-pass-architecture/`, `specs/031-terrain-cell-awareness/`, and `specs/032-native-renderer-parity/` if they still route readers away from `036`.
- [ ] T003A Add a `3.3.5.12340` runtime-controls inventory section in `specs/036-renderer-improvements/research.md` (terrain/light/fog/liquid/M2 controls extracted via Ghidra).

## Phase 2: Foundational (Blocking Prerequisites)

- [ ] T004 Define a shared capability-inventory artifact for renderer slices in `specs/036-renderer-improvements/contracts/renderer-capability-slice.schema.json` examples or companion docs.
- [ ] T005 Define a shared staged-client validation inventory in `specs/036-renderer-improvements/contracts/renderer-validation-scenario.schema.json` examples or companion docs.
- [ ] T006 Refine `specs/036-renderer-improvements/plan.md` with phase-level proof owners and required staged-client checkpoints.
- [ ] T006A Define a telemetry artifact contract (control snapshot table/log schema) in `specs/036-renderer-improvements/research.md` and reference it from `plan.md`.

## Phase 3: User Story 1 - Single Renderer Owner Plan (Priority: P1) 🎯 MVP

**Goal**: One feature pack becomes the active owner plan for renderer modernization across specs 030-032.

**Independent Test**: A maintainer can open `036` first and trace all major slices from specs 030-032 into one owner plan.

- [ ] T007 [US1] Add a source-to-phase traceability matrix to `specs/036-renderer-improvements/research.md`.
- [ ] T008 [US1] Add an explicit “active owner vs source slice” section to `specs/036-renderer-improvements/spec.md`.
- [ ] T009 [US1] Add or tighten convergence notes in `specs/030-wmo-render-pass-architecture/plan.md`, `specs/031-terrain-cell-awareness/plan.md`, and `specs/032-native-renderer-parity/plan.md`.
- [ ] T010 [US1] Review the convergence pack for scope collisions with `specs/035-m2-render-parity-recovery/` and record the boundary in `specs/036-renderer-improvements/research.md`.

## Phase 4: User Story 2 - Bounded Library-First Renderer Phases (Priority: P1)

**Goal**: The convergence owner plan defines a dependency-ordered library-first implementation sequence for terrain, WMO, lighting, sky/fog, liquid, and viewer wiring.

**Independent Test**: A renderer engineer can pick any slice and identify the owning phase, owner layer, and prerequisite validations.

- [ ] T011 [US2] Expand `specs/036-renderer-improvements/data-model.md` with concrete owner-layer examples for terrain, WMO, lighting, liquid, and viewer concerns.
- [ ] T012 [US2] Add per-phase owner-layer summaries to `specs/036-renderer-improvements/plan.md`.
- [ ] T013 [US2] Add explicit prerequisites and completion gates for each convergence phase in `specs/036-renderer-improvements/plan.md`.
- [ ] T014 [US2] Record the first recommended implementation slice after planning completion in `specs/036-renderer-improvements/quickstart.md`.
- [ ] T014A [US2] Add phase-owned runtime-control gates for `terrainLOD`, `terrainAlphaBitDepth`, `mapObjLightLOD`, `MaxLights`, `projectedTextures`, and `waterLOD` in `specs/036-renderer-improvements/plan.md`.
- [ ] T014B [US2] Add bounded M2 dependency gates (`M2UseZFill`, `M2UseClipPlanes`, `M2UseThreads`, `M2BatchDoodads`, `M2BatchParticles`, `M2ForceAdditiveParticleSort`) in `specs/036-renderer-improvements/plan.md` while preserving spec 035 ownership.

## Phase 5: User Story 3 - Shared Validation and Out-of-Scope Boundaries (Priority: P2)

**Goal**: Each convergence phase has staged-client proof guidance and explicit non-goals so unrelated regressions are not silently absorbed.

**Independent Test**: An operator can read `quickstart.md` and know how to validate a phase and what remains outside this convergence feature.

- [ ] T015 [US3] Add representative staged-client validation scenarios to `specs/036-renderer-improvements/quickstart.md`.
- [ ] T016 [US3] Add explicit out-of-scope and adjacent-track sections to `specs/036-renderer-improvements/spec.md` and `research.md`.
- [ ] T017 [P] [US3] Add example validation scenario records or documentation to `specs/036-renderer-improvements/contracts/README.md`.
- [ ] T018 [US3] Record phase-by-phase evidence expectations in `specs/036-renderer-improvements/plan.md`.
- [ ] T018A [US3] Add telemetry-first checkpoint requirements to `specs/036-renderer-improvements/quickstart.md` and `plan.md` so runtime-control snapshots are required before screenshot signoff.
- [ ] T018B [US3] Add liquid material-path validation tasks (water/no-spec/proc-water/magma) tied to staged-client evidence in `specs/036-renderer-improvements/plan.md` and `quickstart.md`.

## Phase 6: Polish & Cross-Cutting Concerns

- [ ] T019 [P] Update `wow-viewer/docs/architecture/m2-native-client-research-2026-03-31.md` and other renderer architecture docs only if they need direct routing back to the convergence owner plan.
- [ ] T020 Run `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` after any follow-on implementation changes and record the result in feature notes.
- [ ] T021 Add a final convergence summary in `specs/036-renderer-improvements/parity-results.md` or equivalent once execution phases begin.
- [ ] T021A Add a short RE evidence appendix in `specs/036-renderer-improvements/research.md` listing core function anchors and addresses used for the convergence updates.

## Dependencies & Execution Order

- Phase 1 must complete first.
- Phase 2 blocks all user stories.
- US1 should complete before using `036` as the default routing document for future renderer work.
- US2 depends on the owner-plan routing from US1.
- US3 depends on the phase structure and routing from US1 and US2.

## Parallel Opportunities

- T003, T005, T017, and T019 are parallelizable.
- Source-spec routing notes can be updated in parallel once the convergence owner wording is stable.

## Implementation Strategy

1. Establish `036` as the unambiguous owner plan first.
2. Tighten traceability and phase ownership next.
3. Add validation guidance and non-goal boundaries before starting execution slices.
