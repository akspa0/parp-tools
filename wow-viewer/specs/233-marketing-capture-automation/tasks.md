# Tasks: Renderer Marketing Capture Automation

**Input**: Design documents from `/specs/233-marketing-capture-automation/`

**Prerequisites**: `spec.md`, `plan.md`, `research.md`, `data-model.md`, `contracts/`, `quickstart.md`

**Tests**: Required by SC-003. Unit tests are contract/source evidence only; real client/UI/video/FPS/Comfy validation remains operator-owned.

## Phase 1: SpecKit Design and Baseline

**Purpose**: Freeze the user-directed contract before touching renderer capture behavior.

- [x] T001 Record the design receipt in `specs/233-marketing-capture-automation/evidence/t001-design-receipt.md`: files, JSON-contract parse result, and FR/SC-to-artifact mapping. Do not claim runtime behavior.
- [x] T002 Confirm `WowViewer.Core.Runtime` is already referenced by `tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj`; add no project, package, `ViewerApp` field, or `ViewerApp_*` partial class.

**Checkpoint**: Feature contract is explicit; implementation may begin only in the owned Core Runtime marketing namespace.

---

## Phase 2: Foundational Marketing-Capture Contract

**Purpose**: Create the deterministic, library-owned model required by every user story.

- [x] T003 [P] Add failing recipe/beat validation tests in `tests/WowViewer.Core.Tests/MarketingCapture/FeatureTourRecipeTests.cs` for required identity, FPS bounds, sorted non-overlapping beats, and supported presentation kinds.
- [x] T004 [P] Add failing managed-output and handoff tests in `tests/WowViewer.Core.Tests/MarketingCapture/MarketingCaptureOutputPolicyTests.cs` for relative paths, traversal/out-of-root rejection, and no client-root field.
- [x] T005 Implement `FeatureTourRecipe`, `FeatureTourBeat`, presentation kind, and validation result types in `src/core/WowViewer.Core.Runtime/Marketing/FeatureTourRecipe.cs`.
- [x] T006 Implement `MarketingCaptureOutputPolicy` and `AuthoringHandoff` validation in `src/core/WowViewer.Core.Runtime/Marketing/MarketingCaptureOutputPolicy.cs` and `src/core/WowViewer.Core.Runtime/Marketing/AuthoringHandoff.cs`.
- [x] T007 Add a built-in, versioned camera-path feature-tour recipe factory in `src/core/WowViewer.Core.Runtime/Marketing/BuiltinFeatureTourRecipes.cs`; it may identify a loaded path but must never persist a client-root path.
- [x] T008 Run `dotnet test I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~MarketingCapture"` and capture the exact result in `specs/233-marketing-capture-automation/evidence/t003-t008-foundation-receipt.md`.

**Checkpoint**: Recipes and external artifact references are deterministic, testable, and safely contained; no renderer/UI behavior has changed yet.

---

## Phase 3: User Story 1 - Record a Scripted Renderer Feature Tour (Priority: P1) 🎯 MVP

**Goal**: An operator can start a warm-and-record camera-path tour with timed callouts and no manual option-bar interaction.

**Independent Test**: With a real loaded client/world and `FlybyUndead` path, operator starts the named tour, observes warmup then playback, and verifies that ordinary chrome remains hidden while scheduled callouts appear in the recorded video.

- [x] T009 [P] [US1] Add pure tour-session transition tests in `tests/WowViewer.Core.Tests/MarketingCapture/FeatureTourAttemptTests.cs` for ordered beat activation, cancelled paths, and invalid recipe rejection.
- [x] T010 [US1] Implement `MarketingTourAttempt` and immutable presentation snapshots in `src/core/WowViewer.Core.Runtime/Marketing/MarketingTourAttempt.cs`; it must not allocate while advancing a frame.
- [x] T011 [US1] Implement `Capture/MarketingTourOverlayRenderer.cs` as an owned, stateless ImGui adapter that draws only a validated active callout.
- [x] T012 [US1] Modify the existing composition seam in `src/viewer/WoWViewer/ViewerApp_CameraPaths.cs` to provide **Feature Tour + Video** beside existing **Play + Video**, carry the attempt through the existing warmup/active-recording objects, and force clean full-frame capture. Do not add a `ViewerApp` field, method, or partial class.
- [x] T013 [US1] Modify the existing capture/UI seams in `src/viewer/WoWViewer/ViewerApp_CaptureAutomation.cs` and `src/viewer/WoWViewer/ViewerApp.cs` so an active attempt advances from camera-path time and its callout is drawn before the existing with-UI capture tap while ordinary chrome remains hidden.
- [x] T014 [US1] Run the focused marketing tests and `dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`; record both source results in `specs/233-marketing-capture-automation/evidence/t009-t014-us1-source-receipt.md`.
- [ ] T015 [US1] Operator runs quickstart steps 1–5 and records a real video/UI timing witness in `specs/233-marketing-capture-automation/evidence/t015-operator-tour-witness.md`, including client build/root fingerprint, map, camera-path identity, output path, and no-manual-click observation.

**Checkpoint**: The first real feature tour is recorded directly from the renderer. T015 is required before any runtime/visual claim.

---

## Phase 4: User Story 2 - Inspect a Capture and Renderer Benchmark Receipt (Priority: P2)

**Goal**: Every tour attempt receives a terminal receipt that makes performance behavior and failure state inspectable.

**Independent Test**: Fixture completions, aborted warmup, encoder failure, and retained frame-history hitches produce distinct, valid receipt outcomes.

- [ ] T016 [P] [US2] Add receipt serialization tests in `tests/WowViewer.Core.Tests/MarketingCapture/TourAttemptReceiptTests.cs` for all terminal outcomes and required provenance/performance fields.
- [ ] T017 [US2] Implement receipt model, serializable frame-history projection, and atomic writer in `src/core/WowViewer.Core.Runtime/Marketing/TourAttemptReceipt.cs`.
- [ ] T018 [US2] Extend the existing recording start/stop seam in `src/viewer/WoWViewer/ViewerApp_CaptureAutomation.cs` to reset/snapshot the existing `WorldScene.FrameHistory` at actual record start/terminal state and write a managed-root receipt. Do not modify `WorldScene`.
- [ ] T019 [US2] Record focused-test/build results in `specs/233-marketing-capture-automation/evidence/t016-t019-us2-source-receipt.md`; include criterion-to-output table and do not mark the runtime criterion complete.
- [ ] T020 [US2] Operator inspects a real completed/degraded/failed receipt from a tour and records video playback plus measured hitch/FPS evidence in `specs/233-marketing-capture-automation/evidence/t020-operator-benchmark-witness.md`.

**Checkpoint**: An encoded file is not mistaken for a successful benchmark; runtime evidence remains clearly attributed.

---

## Phase 5: User Story 3 - Hand Captures to External Authoring Automation (Priority: P3)

**Goal**: A future MCP/Comfy workflow can consume a safe, versioned handoff without reading viewer internals.

**Independent Test**: A completed fixture receipt produces a valid descriptor; absent receipt, unavailable transport, or an out-of-root path is a typed refusal with no artifact mutation.

- [ ] T021 [P] [US3] Add handoff-from-receipt tests in `tests/WowViewer.Core.Tests/MarketingCapture/AuthoringHandoffTests.cs` for valid descriptors, no client-root leakage, missing artifacts, and out-of-root refusal.
- [ ] T022 [US3] Extend `src/core/WowViewer.Core.Runtime/Marketing/AuthoringHandoff.cs` to derive a descriptor only from a valid completed/degraded receipt and serialize it atomically beside the receipt.
- [ ] T023 [US3] Select and document the actual callable external MCP schema in `specs/233-marketing-capture-automation/research.md` before adding any transport adapter. Direct HTTP to ComfyUI is out of scope.
- [ ] T024 [US3] After T023, implement an opt-in MCP adapter in its selected owned service location; unavailable/rejected transport must return a typed result and preserve local artifacts.
- [ ] T025 [US3] Record source tests/build and, separately, the operator's real Comfy workflow witness in `specs/233-marketing-capture-automation/evidence/`; do not claim authored video without the latter.

**Checkpoint**: The boundary is useful externally without making a hidden local service or a publishing claim.

---

## Phase 6: Publishing and Continuity Gates

**Purpose**: Publish only reviewed facts and keep the implementation state recoverable.

- [ ] T026 Add real, operator-reviewed video/still links or embeds to the repository README only after T015/T020 receipts identify their exact artifacts; never add placeholders.
- [ ] T027 Update `specs/STATUS.md`, `memory-bank/activeContext.md`, and `memory-bank/progress.md` with completed tasks, source receipts, and remaining operator gates.
- [ ] T028 Run `git diff --check`, inspect `git status --short`, `git diff`, and `git diff --cached`, then selectively stage only Spec 233 files plus their owned source/tests/evidence and create a bounded commit.
- [ ] T029 Scope Patreon into a separate operator-approved SpecKit feature if desired; it is not part of this implementation contract.

## Dependencies & Execution Order

- T001–T002 precede the Core Runtime contract.
- T003–T008 are foundational and block all user stories.
- US1 (T009–T015) precedes actual receipts because an attempt must exist before it can be recorded.
- US2 (T016–T020) supplies the durable evidence required by US3 handoff.
- US3 transport is explicitly blocked on T023, a concrete callable MCP schema and operator-selected workflow.
- README publication (T026) is blocked on real operator artifacts; it cannot be satisfied by source tests.

## Implementation Strategy

1. Complete Phase 2 and validate its pure model/tests.
2. Complete the P1 viewer composition as one bounded change, then stop for the operator visual/video witness.
3. Add receipts, then external handoff/transport only after evidence and a selected MCP contract.
