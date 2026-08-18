---
description: "Task list for Skybox Rendering (spec 160)"
---

# Tasks: Skybox Rendering

**Input**: Design documents from `/specs/160-skybox-rendering/`

**Prerequisites**: [plan.md](./plan.md), [spec.md](./spec.md), [research.md](./research.md),
[data-model.md](./data-model.md), [contracts/](./contracts/), [quickstart.md](./quickstart.md)

**Tests**: Included. The plan specifies unit coverage in every phase; rendering correctness is not
unit-testable and is covered by user-run real-client proof instead.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependency on an incomplete task)
- **[Story]**: `[US1]`–`[US5]`, mapping to the spec's user stories
- **👤 USER**: Task is user-run. Per AGENTS.md, real-client visual/FPS/rendering proof is
  user-owned — **do not launch these**; hand over the prepared command from
  [quickstart.md](./quickstart.md)

## Constitution gates applied throughout

- **Library-First**: resolution, provenance, band mapping, and classification land under
  `src/core/`. `WorldScene.cs` receives **wiring and draw order only** — no resolution logic.
- **One Phase at a Time**: a phase is done when **validated**, not when coded. Build/test success is
  not rendering proof.
- **Bite-Sized**: ≤ 10 tasks per phase, one concern per task.

---

## Phase 1: Setup — Frame-cost baseline (BLOCKING)

**Purpose**: Capture the pre-change baseline required by FR-022. **Once any sky code changes this
baseline is unrecoverable**, so it must complete before Phase 2.

- [ ] T001 👤 USER Capture `Sky` and `SkyboxBackdrop` p50/p99/max on a **dense** map with a
      continuously moving camera; confirm the window reports `CameraMovedDuringWindow` before
      recording (a static capture is invalid evidence — see contracts/frame-budget.md)
- [ ] T002 👤 USER Repeat T001 on a **sparse** map so the budget is not fitted to one scene
- [ ] T003 Record both baselines, client build identity, configured root, map, and frame counts in
      `specs/160-skybox-rendering/contracts/frame-budget.md`
- [ ] T004 Set the FR-022 budget in `contracts/frame-budget.md` as a delta gate + absolute ceiling +
      zero-new-hitch gate, allowing for the known steady-state increase Phase 5 will introduce

**Checkpoint**: `contracts/frame-budget.md` has no remaining `_(to fill)_` fields. Phase 2 is blocked
until this holds.

---

## Phase 2: Foundational — Provenance and source resolution (BLOCKING)

**Purpose**: The shared scaffold every user story reports through. **No user story work can begin
until this phase completes.**

**⚠️ Nothing rendered changes in this phase.** That is expected and is the point.

- [ ] T005 [P] Create `SkyProvenance` (SourceKind, SourceIdentity, RecordIdentity, IsAuthored,
      BuildIdentity) in `src/core/WowViewer.Core.Runtime/World/Sky/SkyProvenance.cs`
- [ ] T006 [P] Create `SkyBand` (Order, Color, HeightFactor) in
      `src/core/WowViewer.Core.Runtime/World/Sky/SkyBand.cs`
- [ ] T007 Create `SkyGradientSource` (Bands, FogColor, Provenance, AuthoredBandCount,
      ExpectedBandCount) in `src/core/WowViewer.Core.Runtime/World/Sky/SkyGradientSource.cs`
      (depends on T005, T006)
- [ ] T008 [P] Create `SkyModelReference` (AssetPath, Provenance, SelectionReason, LoadState) in
      `src/core/WowViewer.Core.Runtime/World/Sky/SkyModelReference.cs` (depends on T005)
- [ ] T009 Create `SkySourceSelection` joining gradient and model as **independently resolved**
      results per research R1, in
      `src/core/WowViewer.Core.Runtime/World/Sky/SkySourceSelection.cs` (depends on T007, T008)
- [ ] T010 Implement `SkySourceResolver` with single-source selection and precedence
      (override → map-scoped → global → authored-over-fallback → documented tiebreak) in
      `src/core/WowViewer.Core.Runtime/World/Sky/SkySourceResolver.cs` (contract C2, C3)
- [ ] T011 Unit-test the no-mixing invariant in
      `tests/WowViewer.Core.Tests/World/SkySourceResolverTests.cs`: for LIT-only, DBC-only, both, and
      neither, assert every band in a returned set shares one `SourceKind` **and** one
      `SourceIdentity` (contract C2)
- [ ] T012 [P] Unit-test provenance totality and determinism in
      `tests/WowViewer.Core.Tests/World/SkyProvenanceTests.cs`: every leaf reaches a non-null
      provenance; `IsAuthored` is false for every fallback kind; resolving twice yields identical
      selection (contracts C3, C4)
- [ ] T013 Surface resolved sky provenance in the diagnostics readout in
      `src/viewer/WoWViewer/ViewerApp_Sidebars.cs` (FR-003)

**Checkpoint**: resolver tests pass including the no-mixing assertion; provenance visible in the UI
for a real map; **no rendered output has changed**.

---

## Phase 3: User Story 1 — Authored sky colours reach the screen (P1) 🎯 MVP

**Goal**: Client-authored sky colours survive to the rendered frame instead of being overwritten
every frame.

**Independent Test**: Change an authored sky colour in client data, reload, and confirm the rendered
sky changes. Today it does not change at all.

**Why first**: the overwrite defect makes US2–US5 invisible. Nothing else can be visually validated
before this lands.

- [ ] T014 [US1] Change `SkyDomeRenderer.UpdateFromLighting` in
      `src/viewer/WoWViewer/Rendering/SkyDomeRenderer.cs` so it supplies values **only where no
      authored value was resolved**, instead of unconditionally overwriting `ZenithColor` and
      `HorizonColor` (FR-004)
- [ ] T015 [US1] Wire the resolved `SkyGradientSource` through the lighting step in
      `src/viewer/WoWViewer/Terrain/WorldScene.cs` so authored colours survive to the draw —
      **wiring only, no resolution logic in this file** (FR-004)
- [ ] T016 [US1] Make the hardcoded curve an explicit, reported fallback rather than an
      unconditional overwrite, provenanced as `HardcodedFallback` (FR-005, contract C5)
- [ ] T017 [US1] Drive sky colours from the world time-of-day clock through the selected source's
      timed samples in `src/viewer/WoWViewer/Terrain/WorldScene.cs` (FR-006)
- [ ] T018 [US1] Preserve the existing manual LIT override as an override, recorded with
      `IsManualOverride` in provenance, in `src/viewer/WoWViewer/Terrain/WorldScene.cs`
      (research R7)
- [ ] T019 [P] [US1] Unit-test that a resolved authored value is never replaced by a fallback value,
      in `tests/WowViewer.Core.Tests/World/SkyGradientResolutionTests.cs`
- [ ] T020 👤 USER [US1] Differential proof: change an authored sky colour, reload, confirm the
      rendered sky changes (SC-001); scrub time of day and confirm it follows authored samples
- [ ] T021 👤 USER [US1] Load a map with **no** resolvable profile; confirm a fallback sky renders
      **and reports itself as the fallback** in diagnostics (SC-007)

**Checkpoint**: authored colour is visible on screen for the first time. US2–US5 are now visually
validatable.

---

## Phase 4: User Story 2 — Skybox model visible across the whole day (P1)

**Goal**: A resolved skybox model renders across the full day/night cycle, animating on the world
clock.

**Independent Test**: Set time to midday on a map with a resolvable model and confirm it renders;
sweep the full cycle and confirm no pop-in or pop-out.

**Dependency note**: depends on Phase 2 only. **Independent of Phase 3** — gradient and model resolve
separately (research R1), so this can run in parallel with US1 if staffed.

- [ ] T022 [US2] Remove the `NightVisibility > 0.001f` gate from `RenderSkyboxBackdrop` in
      `src/viewer/WoWViewer/Terrain/WorldScene.cs` so a resolved model renders across the full cycle
      (FR-010)
- [ ] T023 [US2] Drive the active skybox model's animation from the world clock by setting
      `CurrentFrame`, instead of accumulating `DateTime.UtcNow` deltas, in
      `src/viewer/WoWViewer/Rendering/ModelRenderer.cs` (FR-011, research R3)
- [ ] T024 [US2] Handle sequence wrap at the midnight boundary **at the call site** — `M2RuntimeAnimator`
      clamps via `ClampFrame` while `MdxAnimator` assigns raw, so wrap cannot be assumed from the
      setter (research R3)
- [ ] T025 [US2] Confirm and pin draw order in `src/viewer/WoWViewer/Terrain/WorldScene.cs`: gradient
      first, model composited over it, both behind all world geometry, no depth write (FR-012)
- [ ] T026 [US2] Make candidate selection deterministic on a distance tie in
      `src/core/WowViewer.Core.Runtime/World/Sky/SkyModelReference.cs` (FR-013)
- [ ] T027 [US2] Make missing / unresolvable / still-loading models degrade to gradient-only,
      reported **once** and never retried per frame, without blocking the render thread
      (FR-014, FR-024, contract C7)
- [ ] T028 [P] [US2] Unit-test deterministic tie-break selection and once-only failure reporting in
      `tests/WowViewer.Core.Tests/World/SkyModelSelectionTests.cs`
- [ ] T029 👤 USER [US2] Confirm the model renders at midday and is continuous across a full day
      sweep with no threshold pop (SC-002)
- [ ] T030 👤 USER [US2] Scrub time of day and confirm model appearance advances **with the clock**;
      freeze time and confirm the sky stops rather than continuing to animate
- [ ] T031 👤 USER [US2] Point the model reference at a deliberately broken path; confirm the
      gradient still renders and the failure is reported once (SC-007)

**Checkpoint**: skybox models are visible all day and follow the world clock.

---

## Phase 5: User Story 3 — Five-band sky gradient (P2)

**Goal**: The rendered gradient reproduces every band the source authors, not a two-colour blend.

**Independent Test**: Change one mid-sky band in isolation and confirm the change appears in that
band's region while zenith and horizon hold.

**Dependency**: Phase 3. Until authored colour survives, band changes are invisible.

- [ ] T032 [US3] Extend the fragment shader in
      `src/viewer/WoWViewer/Rendering/SkyDomeRenderer.cs` from a two-colour `mix` to ordered N-band
      interpolation over `vHeight` (FR-007, contract G3)
- [ ] T033 [US3] Upload the resolved ordered band set as a uniform array in
      `src/viewer/WoWViewer/Rendering/SkyDomeRenderer.cs`. **Do not modify**
      `src/core/WowViewer.Core.Runtime/World/Sky/SkyDomeVertexBuilder.cs` (contract G7, research R2)
- [ ] T034 [US3] Map LIT colour tracks to band order in
      `src/core/WowViewer.Core.Runtime/World/Sky/SkyGradientSource.cs` — **order is the reverse of
      track index**: track 2 is zenith (Order 4), track 6 is horizon (Order 0). A direct
      index-to-order copy renders the sky upside down (contract G2)
- [ ] T035 [US3] Preserve the existing below-horizon fog blend (`smoothstep(0.15, 0.0, vHeight)`),
      applied **after** band interpolation, in `src/viewer/WoWViewer/Rendering/SkyDomeRenderer.cs`
      (contract G5)
- [ ] T036 [US3] Clamp below-horizon authored bands to `HeightFactor = 0.0` and count them in the
      shortfall report rather than discarding them silently (contract G4)
- [ ] T037 [US3] Handle short band sets — use what exists, report the shortfall, never zero-fill to a
      fixed length (FR-008, contract G6)
- [ ] T038 [P] [US3] Unit-test band ordering in
      `tests/WowViewer.Core.Tests/World/SkyBandMappingTests.cs`, asserting **explicitly** that LIT
      track 2 maps to zenith and track 6 to horizon (contract G2 — this test exists to catch the
      inversion)
- [ ] T039 [P] [US3] Unit-test the regression guard in
      `tests/WowViewer.Core.Tests/World/SkyBandMappingTests.cs`: a two-band set renders identically
      to the pre-change gradient (contract G6)
- [ ] T040 👤 USER [US3] Change one mid-sky band in isolation; confirm the change is confined to that
      band's region with zenith and horizon unmoved (SC-003), and inspect boundaries for seams
      (FR-009)

**Checkpoint**: the full authored gradient is on screen, right way up.

---

## Phase 6: User Story 4 — WMO interior skyboxes (P3)

**Goal**: Entering a WMO that declares a skybox swaps the visible sky; leaving restores it.

**Independent Test**: Enter a WMO known to declare a skybox, confirm the sky changes, leave, confirm
it reverts.

**Dependency**: Phase 4 — needs a working model path underneath.

- [ ] T041 [US4] Add the skybox name to `WmoSummary` alongside `HasSkybox` in
      `src/core/WowViewer.Core/Wmo/WmoSummary.cs` (FR-015)
- [ ] T042 [US4] Stop discarding the parsed `MOSB` name in
      `src/core/WowViewer.Core.IO/Wmo/WmoSummaryReader.cs` — keep it a **single** parse; do not add a
      second read of the WMO root (research R5)
- [ ] T043 [US4] Call the existing `WmoCameraVisibility.IsInsideRootOrGroup` where WMO instance
      transforms and bounds are already resident in `src/viewer/WoWViewer/Terrain/WorldScene.cs`;
      the caller owns the world→local transform and padding (research R4)
- [ ] T044 [US4] Make the declared interior skybox the active sky while inside, restoring the outdoor
      sky on exit, in `src/core/WowViewer.Core.Runtime/World/Sky/SkySourceSelection.cs` (FR-016)
- [ ] T045 [US4] Make interior selection deterministic and stable under nested and overlapping WMOs
      (FR-017)
- [ ] T046 [US4] Fall back to the outdoor sky on an unresolvable declared name, reported (FR-018,
      contract C7)
- [ ] T047 [US4] Add hysteresis (or equivalent) at the interior/exterior transition so repeated
      boundary crossings do not strobe the sky (SC-005)
- [ ] T048 [P] [US4] Unit-test nested-WMO determinism and unresolvable-name fallback in
      `tests/WowViewer.Core.Tests/World/SkyInteriorSelectionTests.cs`
- [ ] T049 [P] [US4] Unit-test that `WmoSummaryReader` preserves the `MOSB` name in
      `tests/WowViewer.Core.Tests/WmoSummaryReaderTests.cs`
- [ ] T050 👤 USER [US4] Enter/leave a WMO declaring a skybox; cross repeatedly and confirm no
      flicker; enter one with an unresolvable name and confirm the outdoor sky persists with the
      reference reported

**Checkpoint**: interior skyboxes swap correctly and stably.

---

## Phase 7: User Story 5 — Data-driven classification (P3)

**Goal**: Skyboxes are identified by client declaration, not by filename keywords.

**Independent Test**: A declared skybox whose filename lacks the keywords is classified as sky; a
non-sky asset whose filename contains one is not.

**Dependency**: Phase 4.

- [ ] T051 [US5] Expose declared skybox names from the `Light*` chain in
      `src/core/WowViewer.Core.IO/Lighting/LightDbcCatalog.cs` (FR-019)
- [ ] T052 [US5] Build the declared-name set as the union of `LightSkybox` names and `MOSB` names in
      `src/core/WowViewer.Core.Runtime/World/Sky/SkySourceResolver.cs` (contract C6)
- [ ] T053 [US5] Classify placements against declared names in
      `src/core/WowViewer.Core.Runtime/World/WorldSkyboxBackdropClassifier.cs` (FR-019)
- [ ] T054 [US5] Keep the filename heuristic as an **explicitly reported** fallback for builds with
      no declaration source in
      `src/core/WowViewer.Core.Runtime/World/WorldSkyboxBackdropClassifier.cs`. Per research R1 this
      includes LIT-era alpha outdoor models, so this path is load-bearing on this branch's target
      era — not a rare edge (FR-020)
- [ ] T055 [US5] Report which declaration classified each skybox, through provenance (FR-021)
- [ ] T056 [P] [US5] Unit-test both directions in
      `tests/WowViewer.Core.Tests/World/SkyClassificationTests.cs`: a declared skybox with a
      non-matching filename is classified; a non-sky asset with a matching filename is not
- [ ] T057 👤 USER [US5] Confirm both directions against real client data, and confirm the fallback
      path reports itself on a build with no declaration source

**Checkpoint**: classification is declaration-driven with a reported fallback.

---

## Phase 8: Polish — Non-regression and documentation

**Purpose**: Close the FR-022 budget gate and the constitution's terrain guard.

- [ ] T058 👤 USER Re-capture `Sky` and `SkyboxBackdrop` distributions on the **same two maps** as
      T001/T002, moving camera, and compare against the recorded budget (SC-008)
- [ ] T059 👤 USER Confirm hitch attribution shows no new hitches attributed to either stage
- [ ] T060 👤 USER Disable sky rendering; confirm both stages measure **zero**, not merely small
      (FR-023, SC-009)
- [ ] T061 👤 USER Walk the full failure matrix — no profile, missing asset, still-loading asset,
      unresolvable WMO reference — and confirm every case renders a sky (SC-007, contract C7)
- [ ] T062 👤 USER Re-check terrain fog on **both** Alpha-era and LK 3.3.5 terrain, since sky already
      shares fog colour (Constitution terrain risk-area guard)
- [ ] T063 Record measured post-change numbers and the budget verdict in
      `specs/160-skybox-rendering/contracts/frame-budget.md`
- [ ] T064 [P] Update affected architecture docs in `docs/architecture/` in the **same commit** as
      the code (Constitution: Spec Docs Are Source of Truth)
- [ ] T065 [P] Update `specs/STATUS.md` with the delivered state and remaining gaps
- [ ] T066 [P] Update `memory-bank/activeContext.md` and `memory-bank/progress.md`, preserving any
      negative results found along the way

---

## Dependencies & Execution Order

### Phase dependencies

```text
Phase 1 (Baseline) ──BLOCKS──► Phase 2 (Scaffold) ──BLOCKS──► all user stories
                                        │
                     ┌──────────────────┴──────────────────┐
                     ▼                                     ▼
              Phase 3 (US1, P1)                    Phase 4 (US2, P1)
                     │                                     │
                     ▼                          ┌──────────┴──────────┐
              Phase 5 (US3, P2)                 ▼                     ▼
                                         Phase 6 (US4, P3)    Phase 7 (US5, P3)
                     └───────────────┬───────────────┴─────────────────┘
                                     ▼
                              Phase 8 (Polish)
```

- **Phase 1 blocks everything.** The baseline is unrecoverable once sky code changes.
- **Phase 3 and Phase 4 are independent of each other** — gradient and model resolve separately
  (research R1). Both need only Phase 2.
- **Phase 5 depends on Phase 3**: band changes are invisible until authored colour survives.
- **Phases 6 and 7 depend on Phase 4**: both need a working model path.

### Parallel opportunities

- **Phase 2**: T005, T006 in parallel; then T008 alongside T007; T011 and T012 in parallel.
- **Phases 3 and 4** can run in parallel by different implementers after Phase 2.
- **Phases 6 and 7** can run in parallel after Phase 4.
- **Phase 8**: T064, T065, T066 in parallel after the user-run gates return.
- All `[P]` unit tests within a phase are parallel — they are separate files.

### Within each phase

- Library types before the services that use them.
- Library logic before viewer wiring (Constitution: Library-First).
- Code before its unit test only where the test targets a new type; the no-mixing test (T011) is the
  gate on T010 and should be written against the contract, not the implementation.
- **User-run validation is last in each phase and is what makes the phase done.**

---

## Implementation Strategy

### MVP scope

**Phase 1 → Phase 2 → Phase 3.** That delivers the single change that makes the feature real:
client-authored sky colour reaching the screen for the first time. Everything after it is fidelity
and coverage.

**Stop and validate after Phase 3.** T020's differential test is the honest measure — if changing an
authored colour does not change the render, nothing later in this spec will be verifiable.

### Incremental delivery

1. **Phase 1 + 2**: scaffold, nothing visible changes. Resist the urge to skip the baseline.
2. **Phase 3**: authored colour visible → MVP, independently valuable.
3. **Phase 4**: skybox models all day → the second P1, and what "skyboxes working" most directly
   means.
4. **Phase 5**: full gradient fidelity.
5. **Phases 6 + 7**: interior skyboxes and correct classification.
6. **Phase 8**: budget gate and docs.

### Expected surprises

Phase 3 makes previously-hidden data bugs suddenly visible — authored sky data has **never** reached
the screen, so anything odd that appears is newly-visible truth rather than a regression introduced
here. Record such findings; do not treat them as Phase 3 defects by default.

---

## Task summary

| Phase | Story | Tasks | Of which user-run |
|---|---|---|---|
| 1 Baseline | — | T001–T004 (4) | 2 |
| 2 Scaffold | — | T005–T013 (9) | 0 |
| 3 Authored colours | US1 | T014–T021 (8) | 2 |
| 4 Model all day | US2 | T022–T031 (10) | 3 |
| 5 Five-band gradient | US3 | T032–T040 (9) | 1 |
| 6 WMO interiors | US4 | T041–T050 (10) | 1 |
| 7 Classification | US5 | T051–T057 (7) | 1 |
| 8 Polish | — | T058–T066 (9) | 5 |
| **Total** | | **66** | **15** |

Every phase is ≤ 10 tasks per the constitution's bite-sized-plans rule.
