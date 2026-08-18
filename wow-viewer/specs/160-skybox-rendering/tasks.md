---
description: "Task list for Skybox Rendering (spec 160)"
---

# Tasks: Skybox Rendering

**Input**: Design documents from `/specs/160-skybox-rendering/`

**Prerequisites**: [plan.md](./plan.md), [spec.md](./spec.md), [research.md](./research.md),
[data-model.md](./data-model.md), [contracts/](./contracts/), [quickstart.md](./quickstart.md)

**Tests**: Included — unit tests for library logic, **automated capture + pixel diff** for rendering
proof.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependency on an incomplete task)
- **[Story]**: `[US1]`–`[US5]`, mapping to the spec's user stories
- **🤖 AUTO**: Agent-run through the project's existing capture automation. Rendering proof is
  produced by rendering and diffing images, not by asking a human to look at the screen.

## Rendering proof is automated

The viewer has startup capture automation and there is a headless production-scene profiler. Both
are used directly by this spec:

| Capability | Entry point | Used for |
|---|---|---|
| Headless screenshot of the real scene | `WoWViewer.csproj` + `--capture-shot current --capture-no-ui --capture-after-frames N --exit-after-capture` | Every visual acceptance test (US1–US5) |
| Headless production `WorldScene` render + stage diagnostics JSON | `WowViewer.Tool.ValidationCapture profile-render` | Frame-cost baseline and the FR-022 budget gate |

**Two automation gaps must be closed first** (T001, T002) — without them the day-cycle tests and the
movement-valid frame window cannot be produced at all. They are small additions to existing tools,
not new infrastructure.

Human review remains welcome as a sanity check on any capture, but **it is not a gate** and no task
below blocks on it.

## Constitution gates applied throughout

- **Library-First**: resolution, provenance, band mapping, and classification land under
  `src/core/`. `WorldScene.cs` receives **wiring and draw order only**.
- **One Phase at a Time**: a phase is done when **validated**, not when coded.
- **Bite-Sized**: ≤ 10 tasks per phase, one concern per task.

---

## Phase 1: Close automation gaps, then capture the baseline (BLOCKING)

**Purpose**: Make the sky testable without a human in the loop, then capture the pre-change
frame-cost baseline required by FR-022. **Once any sky code changes the baseline is unrecoverable**,
so this phase completes before Phase 2.

- [ ] T001 🤖 AUTO Add a `--time-of-day <0..1|hours>` startup flag in
      `src/viewer/WoWViewer/ViewerApp_StartupAutomation.cs` that pins the world clock before capture,
      so day-cycle and time-scrub tests are reproducible. Without it US1's scrub test and US2's day
      sweep cannot be automated at all
- [ ] T002 🤖 AUTO Add camera motion across sampled frames to
      `tools/validation-capture/WowViewer.Tool.ValidationCapture/ProductionWorldSceneProfiler.cs`.
      `ResolveCameraPosition` is currently computed **once** (line 92) and reused for every frame, so
      the window is static and its p99/max are not valid evidence. Add a `--camera-path` or
      per-frame offset and emit `CameraMovedDuringWindow` in the JSON report
- [ ] T003 🤖 AUTO Emit per-stage p50/p99/max distributions for `Sky` and `SkyboxBackdrop` in the
      `profile-render` JSON report, not just per-frame totals (contracts/frame-budget.md)
- [ ] T004 🤖 AUTO Add a reusable capture+diff helper script under `scripts/` that renders two
      configurations and reports per-pixel delta, so every differential test below is one command
- [ ] T005 🤖 AUTO Run `profile-render` on a **dense** map with camera motion enabled; record
      `Sky`/`SkyboxBackdrop` p50/p99/max and confirm the report says the camera moved
- [ ] T006 🤖 AUTO Run `profile-render` on a **sparse** map so the budget is not fitted to one scene
- [ ] T007 Record both baselines, client build identity, configured root, map, and frame counts in
      `specs/160-skybox-rendering/contracts/frame-budget.md`
- [ ] T008 Set the FR-022 budget in `contracts/frame-budget.md` as a delta gate + absolute ceiling +
      zero-new-hitch gate, allowing for the known steady-state increase Phase 5 introduces

**Checkpoint**: `contracts/frame-budget.md` has no remaining `_(to fill)_` fields, and both windows
report camera movement. Phase 2 is blocked until this holds.

---

## Phase 2: Foundational — Provenance and source resolution (BLOCKING)

**Purpose**: The shared scaffold every user story reports through. **No user story work can begin
until this phase completes.**

**⚠️ Nothing rendered changes in this phase.** That is expected.

- [ ] T009 [P] Create `SkyProvenance` (SourceKind, SourceIdentity, RecordIdentity, IsAuthored,
      BuildIdentity) in `src/core/WowViewer.Core.Runtime/World/Sky/SkyProvenance.cs`
- [ ] T010 [P] Create `SkyBand` (Order, Color, HeightFactor) in
      `src/core/WowViewer.Core.Runtime/World/Sky/SkyBand.cs`
- [ ] T011 Create `SkyGradientSource` (Bands, FogColor, Provenance, AuthoredBandCount,
      ExpectedBandCount) in `src/core/WowViewer.Core.Runtime/World/Sky/SkyGradientSource.cs`
      (depends on T009, T010)
- [ ] T012 [P] Create `SkyModelReference` (AssetPath, Provenance, SelectionReason, LoadState) in
      `src/core/WowViewer.Core.Runtime/World/Sky/SkyModelReference.cs` (depends on T009)
- [ ] T013 Create `SkySourceSelection` joining gradient and model as **independently resolved**
      results per research R1, in
      `src/core/WowViewer.Core.Runtime/World/Sky/SkySourceSelection.cs` (depends on T011, T012)
- [ ] T014 Implement `SkySourceResolver` with single-source selection and precedence
      (override → map-scoped → global → authored-over-fallback → documented tiebreak) in
      `src/core/WowViewer.Core.Runtime/World/Sky/SkySourceResolver.cs` (contracts C2, C3)
- [ ] T015 Unit-test the no-mixing invariant in
      `tests/WowViewer.Core.Tests/World/SkySourceResolverTests.cs`: for LIT-only, DBC-only, both, and
      neither, assert every band in a returned set shares one `SourceKind` **and** one
      `SourceIdentity` (contract C2)
- [ ] T016 [P] Unit-test provenance totality and determinism in
      `tests/WowViewer.Core.Tests/World/SkyProvenanceTests.cs` (contracts C3, C4)
- [ ] T017 Surface resolved sky provenance in the diagnostics readout in
      `src/viewer/WoWViewer/ViewerApp_Sidebars.cs`, and emit it into the `profile-render` JSON so
      provenance is assertable from automation, not only visible in the UI (FR-003)

**Checkpoint**: resolver tests pass including the no-mixing assertion; provenance appears in the
`profile-render` JSON for a real map; **no rendered output has changed**.

---

## Phase 3: User Story 1 — Authored sky colours reach the screen (P1) 🎯 MVP

**Goal**: Client-authored sky colours survive to the rendered frame instead of being overwritten.

**Independent Test** (automated): capture the sky, change an authored colour, capture again, diff.
A non-zero diff is the pass condition. Today the diff is exactly zero.

- [ ] T018 [US1] Change `SkyDomeRenderer.UpdateFromLighting` in
      `src/viewer/WoWViewer/Rendering/SkyDomeRenderer.cs` so it supplies values **only where no
      authored value was resolved**, instead of unconditionally overwriting `ZenithColor` and
      `HorizonColor` (FR-004)
- [ ] T019 [US1] Wire the resolved `SkyGradientSource` through the lighting step in
      `src/viewer/WoWViewer/Terrain/WorldScene.cs` — **wiring only, no resolution logic** (FR-004)
- [ ] T020 [US1] Make the hardcoded curve an explicit, reported fallback provenanced as
      `HardcodedFallback` (FR-005, contract C5)
- [ ] T021 [US1] Drive sky colours from the world time-of-day clock through the selected source's
      timed samples in `src/viewer/WoWViewer/Terrain/WorldScene.cs` (FR-006)
- [ ] T022 [US1] Preserve the existing manual LIT override as an override, recorded with
      `IsManualOverride` in provenance (research R7)
- [ ] T023 [P] [US1] Unit-test that a resolved authored value is never replaced by a fallback value,
      in `tests/WowViewer.Core.Tests/World/SkyGradientResolutionTests.cs`
- [ ] T024 🤖 AUTO [US1] Differential capture proof (SC-001): capture sky → modify an authored sky
      colour → capture again → assert non-zero pixel delta in the sky region. Assert the pre-change
      build produces a **zero** delta, so the test is proven capable of detecting the change it
      claims to detect
- [ ] T025 🤖 AUTO [US1] Time-scrub proof: capture at several pinned `--time-of-day` values and
      assert the sky differs between them and tracks the authored samples (FR-006)
- [ ] T026 🤖 AUTO [US1] Capture on a map with **no** resolvable profile; assert a sky renders and
      that `profile-render` JSON reports provenance `HardcodedFallback` with `IsAuthored=false`
      (SC-007, contract C5)

**Checkpoint**: authored colour is visible on screen for the first time, proven by pixel diff.

---

## Phase 4: User Story 2 — Skybox model visible across the whole day (P1)

**Goal**: A resolved skybox model renders across the full day/night cycle, animating on the world
clock.

**Dependency**: Phase 2 only. **Independent of Phase 3** (research R1) — can run in parallel.

- [ ] T027 [US2] Remove the `NightVisibility > 0.001f` gate from `RenderSkyboxBackdrop` in
      `src/viewer/WoWViewer/Terrain/WorldScene.cs` (FR-010)
- [ ] T028 [US2] Drive the active skybox model's animation from the world clock by setting
      `CurrentFrame`, instead of accumulating `DateTime.UtcNow` deltas, in
      `src/viewer/WoWViewer/Rendering/ModelRenderer.cs` (FR-011, research R3)
- [ ] T029 [US2] Handle sequence wrap at the midnight boundary **at the call site** —
      `M2RuntimeAnimator` clamps via `ClampFrame` while `MdxAnimator` assigns raw (research R3)
- [ ] T030 [US2] Confirm and pin draw order in `src/viewer/WoWViewer/Terrain/WorldScene.cs`:
      gradient first, model over it, both behind all world geometry, no depth write (FR-012)
- [ ] T031 [US2] Make candidate selection deterministic on a distance tie in
      `src/core/WowViewer.Core.Runtime/World/Sky/SkyModelReference.cs` (FR-013)
- [ ] T032 [US2] Make missing / unresolvable / still-loading models degrade to gradient-only,
      reported **once**, never blocking the render thread (FR-014, FR-024, contract C7)
- [ ] T033 [P] [US2] Unit-test deterministic tie-break and once-only failure reporting in
      `tests/WowViewer.Core.Tests/World/SkyModelSelectionTests.cs`
- [ ] T034 🤖 AUTO [US2] Day-sweep capture: capture at N pinned `--time-of-day` values across the
      full cycle and assert the model is present in **every** frame with no discontinuity at the
      former night threshold (SC-002)
- [ ] T035 🤖 AUTO [US2] Clock-driven animation proof: capture two frames at different pinned times
      and assert the model differs; capture two frames at the **same** pinned time and assert they
      are identical, proving animation follows the world clock and not wall-clock time
- [ ] T036 🤖 AUTO [US2] Broken-model proof: point the model reference at an invalid path, capture,
      and assert a gradient sky still renders and the failure is reported once (SC-007)

**Checkpoint**: skybox models visible all day, clock-driven, proven by capture.

---

## Phase 5: User Story 3 — Five-band sky gradient (P2)

**Goal**: The rendered gradient reproduces every authored band.

**Dependency**: Phase 3. Until authored colour survives, band changes are invisible.

- [ ] T037 [US3] Extend the fragment shader in
      `src/viewer/WoWViewer/Rendering/SkyDomeRenderer.cs` to ordered N-band interpolation over
      `vHeight` (FR-007, contract G3)
- [ ] T038 [US3] Upload the resolved ordered band set as a uniform array. **Do not modify**
      `src/core/WowViewer.Core.Runtime/World/Sky/SkyDomeVertexBuilder.cs` (contract G7, research R2)
- [ ] T039 [US3] Map LIT colour tracks to band order in
      `src/core/WowViewer.Core.Runtime/World/Sky/SkyGradientSource.cs` — **order is the reverse of
      track index**: track 2 is zenith (Order 4), track 6 is horizon (Order 0) (contract G2)
- [ ] T040 [US3] Preserve the existing below-horizon fog blend, applied **after** band
      interpolation (contract G5)
- [ ] T041 [US3] Clamp below-horizon authored bands to `HeightFactor = 0.0` and count them in the
      shortfall report rather than discarding silently (contract G4)
- [ ] T042 [US3] Handle short band sets — use what exists, report the shortfall, never zero-fill
      (FR-008, contract G6)
- [ ] T043 [P] [US3] Unit-test band ordering in
      `tests/WowViewer.Core.Tests/World/SkyBandMappingTests.cs`, asserting **explicitly** that LIT
      track 2 maps to zenith and track 6 to horizon — this test exists to catch the inversion
      (contract G2)
- [ ] T044 [P] [US3] Unit-test the regression guard: a two-band set matches the pre-change gradient
      (contract G6)
- [ ] T045 🤖 AUTO [US3] Band-isolation proof (SC-003): modify one mid-sky band, capture, and assert
      the pixel delta is **confined to that band's height range** while the zenith and horizon rows
      are unchanged. Also assert the rendered zenith row matches the authored zenith colour, which
      catches the G2 inversion in the render as well as in the unit test
- [ ] T046 🤖 AUTO [US3] Seam check: scan a vertical pixel column and assert no discontinuity in
      colour derivative at band boundaries (FR-009)

**Checkpoint**: the full authored gradient is on screen, right way up, proven by capture.

---

## Phase 6: User Story 4 — WMO interior skyboxes (P3)

**Dependency**: Phase 4.

- [ ] T047 [US4] Add the skybox name to `WmoSummary` alongside `HasSkybox` in
      `src/core/WowViewer.Core/Wmo/WmoSummary.cs` (FR-015)
- [ ] T048 [US4] Stop discarding the parsed `MOSB` name in
      `src/core/WowViewer.Core.IO/Wmo/WmoSummaryReader.cs` — keep it a **single** parse
      (research R5)
- [ ] T049 [US4] Call the existing `WmoCameraVisibility.IsInsideRootOrGroup` where WMO instance
      transforms and bounds are already resident in `src/viewer/WoWViewer/Terrain/WorldScene.cs`
      (research R4)
- [ ] T050 [US4] Make the declared interior skybox the active sky while inside, restoring the
      outdoor sky on exit, in
      `src/core/WowViewer.Core.Runtime/World/Sky/SkySourceSelection.cs` (FR-016)
- [ ] T051 [US4] Make interior selection deterministic under nested and overlapping WMOs (FR-017)
- [ ] T052 [US4] Fall back to the outdoor sky on an unresolvable declared name, reported
      (FR-018, contract C7)
- [ ] T053 [US4] Add hysteresis at the interior/exterior transition so repeated crossings do not
      strobe the sky (SC-005)
- [ ] T054 [P] [US4] Unit-test nested-WMO determinism and unresolvable-name fallback in
      `tests/WowViewer.Core.Tests/World/SkyInteriorSelectionTests.cs`
- [ ] T055 [P] [US4] Unit-test that `WmoSummaryReader` preserves the `MOSB` name in
      `tests/WowViewer.Core.Tests/WmoSummaryReaderTests.cs`
- [ ] T056 🤖 AUTO [US4] Interior swap proof: capture from an inside-WMO camera position and an
      outside one, assert the skies differ; capture the outside position twice across a simulated
      re-entry and assert identical output (no strobe, SC-005)

**Checkpoint**: interior skyboxes swap correctly and stably, proven by capture.

---

## Phase 7: User Story 5 — Data-driven classification (P3)

**Dependency**: Phase 4.

- [ ] T057 [US5] Expose declared skybox names from the `Light*` chain in
      `src/core/WowViewer.Core.IO/Lighting/LightDbcCatalog.cs` (FR-019)
- [ ] T058 [US5] Build the declared-name set as the union of `LightSkybox` and `MOSB` names in
      `src/core/WowViewer.Core.Runtime/World/Sky/SkySourceResolver.cs` (contract C6)
- [ ] T059 [US5] Classify placements against declared names in
      `src/core/WowViewer.Core.Runtime/World/WorldSkyboxBackdropClassifier.cs` (FR-019)
- [ ] T060 [US5] Keep the filename heuristic as an **explicitly reported** fallback for builds with
      no declaration source. Per research R1 this includes LIT-era alpha outdoor models, so this
      path is load-bearing on this branch's target era (FR-020)
- [ ] T061 [US5] Report which declaration classified each skybox, through provenance (FR-021)
- [ ] T062 [P] [US5] Unit-test both directions in
      `tests/WowViewer.Core.Tests/World/SkyClassificationTests.cs`: a declared skybox with a
      non-matching filename is classified; a non-sky asset with a matching filename is not
- [ ] T063 🤖 AUTO [US5] Assert against real client data via `profile-render` JSON that
      classification counts and reported rules match expectation on both a declaration-bearing build
      and a LIT-era build where the fallback must engage and say so

**Checkpoint**: classification is declaration-driven with a reported fallback.

---

## Phase 8: Polish — Non-regression and documentation

- [ ] T064 🤖 AUTO Re-run `profile-render` on the **same two maps** as T005/T006 with camera motion
      and compare `Sky`/`SkyboxBackdrop` against the recorded budget (SC-008)
- [ ] T065 🤖 AUTO Assert no new hitches attributed to either stage in the report
- [ ] T066 🤖 AUTO Run with sky rendering disabled and assert both stages measure **zero**, not
      merely small (FR-023, SC-009)
- [ ] T067 🤖 AUTO Walk the full failure matrix by capture — no profile, missing asset, still-loading
      asset, unresolvable WMO reference — and assert every case renders a non-black sky
      (SC-007, contract C7)
- [ ] T068 🤖 AUTO Terrain fog guard: capture terrain on **both** Alpha-era and LK 3.3.5 and assert
      no pixel delta versus a pre-change capture, since sky already shares fog colour (Constitution
      terrain risk-area guard)
- [ ] T069 Record measured post-change numbers and the budget verdict in
      `specs/160-skybox-rendering/contracts/frame-budget.md`
- [ ] T070 [P] Update affected architecture docs in `docs/architecture/` in the **same commit** as
      the code (Constitution: Spec Docs Are Source of Truth)
- [ ] T071 [P] Update `specs/STATUS.md` with the delivered state and remaining gaps
- [ ] T072 [P] Update `memory-bank/activeContext.md` and `memory-bank/progress.md`, preserving any
      negative results found along the way

---

## Dependencies & Execution Order

```text
Phase 1 (Automation + Baseline) ──BLOCKS──► Phase 2 (Scaffold) ──BLOCKS──► all user stories
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

- **Phase 1 blocks everything.** The baseline is unrecoverable once sky code changes, and T001/T002
  are what make the rest testable without a human in the loop.
- **Phases 3 and 4 are independent of each other** (research R1).
- **Phase 5 depends on Phase 3**; **Phases 6 and 7 depend on Phase 4**.

### Parallel opportunities

- **Phase 1**: T001, T002, T003, T004 are separate files and can proceed in parallel; T005/T006 need
  T002 and T003.
- **Phase 2**: T009, T010 in parallel; T012 alongside T011; T015 and T016 in parallel.
- **Phases 3 and 4** in parallel after Phase 2; **Phases 6 and 7** in parallel after Phase 4.
- All `[P]` unit tests within a phase are parallel — separate files.

---

## Implementation Strategy

### MVP scope

**Phase 1 → Phase 2 → Phase 3.** That delivers client-authored sky colour reaching the screen for
the first time, with an automated pixel-diff proving it.

**Stop and validate after Phase 3.** T024 is the honest measure, and it is deliberately built to
fail on the pre-change build — if the diff is not zero before and non-zero after, the test is not
measuring what it claims and nothing later in this spec is verifiable.

### Detector-first discipline

Several capture tests above assert their own power before asserting a result: T024 requires a zero
delta pre-change, T035 requires identical output at identical pinned times, T045 requires the delta
be confined to one band range. This is deliberate. A capture test that cannot demonstrate it would
have caught the bug is not evidence, and this project has been burned by exactly that — a static
camera producing false null results.

### Expected surprises

Phase 3 makes previously-hidden data bugs suddenly visible. Authored sky data has **never** reached
the screen, so anything odd that appears is newly-visible truth rather than a regression introduced
here. Record such findings; do not treat them as Phase 3 defects by default.

---

## Task summary

| Phase | Story | Tasks | Automated proof |
|---|---|---|---|
| 1 Automation + baseline | — | T001–T008 (8) | 4 |
| 2 Scaffold | — | T009–T017 (9) | 0 |
| 3 Authored colours | US1 | T018–T026 (9) | 3 |
| 4 Model all day | US2 | T027–T036 (10) | 3 |
| 5 Five-band gradient | US3 | T037–T046 (10) | 2 |
| 6 WMO interiors | US4 | T047–T056 (10) | 1 |
| 7 Classification | US5 | T057–T063 (7) | 1 |
| 8 Polish | — | T064–T072 (9) | 5 |
| **Total** | | **72** | **19** |

Every phase is ≤ 10 tasks per the constitution's bite-sized-plans rule. **No task blocks on human
visual review.**
