# Tasks: Minimap Interaction, Fog-Bounded Residency, and Doodad Instancing

**Input**: Design documents from `wow-viewer/specs/147-minimap-fog-instancing/`

**Tests**: Focused deterministic tests are required before integration; solution build and
real-client visual/FPS/capture proof remain separate gates owned by the user.

## Phase 1: Contract setup and baseline protection

- [ ] T001 Record the current dirty-tree boundary in `wow-viewer/specs/147-minimap-fog-instancing/research.md`; preserve `wow-viewer/imgui.ini` and unrelated changes.
- [ ] T002 Review the existing Spec 136, 137, and 142 task ownership and add cross-links in `wow-viewer/specs/147-minimap-fog-instancing/plan.md` without changing those specs' completed claims.
- [ ] T003 Add the pure contract types described in `wow-viewer/specs/147-minimap-fog-instancing/data-model.md` only where an existing core/runtime owner does not already provide the same state; do not create duplicate format or lighting readers.
- [ ] T004 Add a focused validation entrypoint or test naming convention for Spec 147 in `wow-viewer/specs/147-minimap-fog-instancing/quickstart.md`, so a zero-test filter cannot be mistaken for proof.

## Phase 2: User Story 1 — Full-screen minimap navigation (P1)

**Goal**: Fullscreen and docked minimaps share one deterministic drag-versus-triple-click contract, and fullscreen rendering has one interaction owner.

**Independent test**: Pure pointer-event tests prove pan, reset, target changes, timeout, invalid targets, and exactly-one teleport on the third same-target click.

### Tests first

- [ ] T005 [P] [US1] Add failing pure gesture tests for drag classification, pan deltas, release reset, same-target click counts, timeout, changed target, invalid target, and exactly-once third-click execution in `wow-viewer/tests/WowViewer.Core.Tests/MinimapInteractionTests.cs`.

### Implementation

- [ ] T006 [US1] Implement the pure minimap gesture state/decision contract in `wow-viewer/src/core/WowViewer.Core.Runtime/World/Minimap/MinimapInteractionState.cs` (or the existing runtime minimap owner if the audit finds one), following `contracts/minimap-interaction.md`.
- [ ] T007 [US1] Make fullscreen minimap rendering single-owner by removing the duplicate call path and preserving one unique interaction ID in `wow-viewer/src/viewer/WoWViewer/ViewerApp.cs` and `wow-viewer/src/viewer/WoWViewer/ViewerApp_MinimapAndStatus.cs`.
- [ ] T008 [US1] Adapt the shared ImGui minimap surface to feed the pure gesture contract, preserve drag capture while held, reset teleport after a drag, and call the existing map/world transform only for `TeleportExecuted` in `wow-viewer/src/viewer/WoWViewer/ViewerApp_MinimapAndStatus.cs`.
- [ ] T009 [US1] Run the focused minimap tests and a Debug build; stop and repair any fullscreen/docked interaction divergence before starting fog work.

## Phase 3: User Story 2 — Fog-bounded coverage (P1)

**Goal**: Normal detailed ADT/object coverage follows the same effective fog snapshot used by rendering, while nearby bounds remain protected and capture/full-load exceptions stay explicit.

**Independent test**: Deterministic tile fixtures prove bounds intersection, near-field protection, directional ordering without nearby exclusion, invalid-fog fallback, hysteresis, and named preload/full-load exceptions.

### Tests first

- [ ] T010 [P] [US2] Add failing fog-window tests for tile-bounds intersection, tile edges, nearby side/rear tiles, outside-window exclusion, invalid fog, revision changes, and preload/full-load reasons in `wow-viewer/tests/WowViewer.Core.Tests/FogCoverageTileSelectorTests.cs`.
- [ ] T011 [P] [US2] Add failing residency-state tests for selected/retained/resident/drawable/preloaded separation and release hysteresis in `wow-viewer/tests/WowViewer.Core.Tests/TileResidencyStateTests.cs`.

### Implementation

- [ ] T012 [US2] Implement the pure fog coverage window and deterministic tile-bounds selector in `wow-viewer/src/core/WowViewer.Core.Runtime/World/Terrain/FogCoverageTileSelector.cs`, consuming active fog values and existing map/tile coordinate contracts.
- [ ] T013 [US2] Define the per-frame active fog snapshot/order handoff so `TerrainManager.UpdateAOI` consumes the same effective `fogEnd` that `WorldScene.Render` uses, in `wow-viewer/src/viewer/WoWViewer/Terrain/WorldScene.cs` and `wow-viewer/src/viewer/WoWViewer/Terrain/TerrainManager.cs`.
- [ ] T014 [US2] Replace the streaming target path that discards `fogEnd` in `wow-viewer/src/viewer/WoWViewer/Terrain/TerrainManager.cs`; retain manual detail/quality controls as explicit policy inputs and preserve the separate retained camera window.
- [ ] T015 [US2] Apply the normal fog coverage gate to tile-owned detailed terrain, liquids, WMO admission, and MDX/M2 admission without changing WDL underlay behavior, capture preload leases, full-load diagnostics, or WMO containment fallbacks in `wow-viewer/src/viewer/WoWViewer/Terrain/TerrainManager.cs` and `wow-viewer/src/viewer/WoWViewer/Terrain/WorldScene.cs`.
- [ ] T016 [US2] Add deterministic admission/eviction reason codes and hysteresis state to the existing tile/runtime diagnostics owner; keep selected, retained, resident, drawable, and preloaded counts separate.
- [ ] T017 [US2] Run focused fog/residency tests and a Debug build; stop before doodad batching if any near-field or capture-preload invariant fails.

## Phase 4: User Story 3 — Shared doodad assets and compatible instance batches (P1)

**Goal**: Static compatible doodad placements share immutable asset resources and grouped instance submissions, while correctness-sensitive paths remain explicit fallbacks.

**Independent test**: Deterministic placement fixtures prove grouping, splitting, asset lifetime across tiles, fallback routing, and count reconciliation.

### Tests first

- [ ] T018 [P] [US3] Add failing compatibility-key tests for asset/backend/pass/material/alpha/fade/animation/effect/WMO-context mismatches in `wow-viewer/tests/WowViewer.Core.Tests/DoodadBatchPlanningTests.cs`.
- [ ] T019 [P] [US3] Add failing lifetime/count tests proving shared geometry is loaded once across tiles, tile release does not destroy referenced assets, and placements/batches/fallbacks/submissions reconcile in `wow-viewer/tests/WowViewer.Core.Tests/DoodadBatchLifetimeTests.cs`.

### Implementation

- [ ] T020 [US3] Implement the deterministic doodad compatibility key and batch planning contract in `wow-viewer/src/core/WowViewer.Core.Runtime/World/Passes/DoodadBatchKey.cs` and `wow-viewer/src/core/WowViewer.Core.Runtime/World/Passes/DoodadBatchPlanner.cs` (reuse an existing owner if equivalent types already exist).
- [ ] T021 [US3] Extend the visible-object pass preparation in `wow-viewer/src/viewer/WoWViewer/Terrain/WorldScene.cs` to collect fog-visible external MDX/M2 placements into compatible asset buckets without traversing inactive tiles.
- [ ] T022 [US3] Extend the WMO-internal doodad preparation in `wow-viewer/src/viewer/WoWViewer/Rendering/WmoRenderer.cs` and its caller so repeated WMO placements can reuse immutable compatible asset state while preserving group, portal, placement-transform, animation, and transparent-order semantics.
- [ ] T023 [US3] Route static compatible buckets through the existing `IGpuInstancedModelRenderer`/batch interfaces in `wow-viewer/src/viewer/WoWViewer/Rendering/M2Renderer.cs`, `wow-viewer/src/viewer/WoWViewer/Rendering/ModelRenderer.cs`, and `wow-viewer/src/viewer/WoWViewer/Terrain/WorldScene.cs`; retain named fallback routes for native/legacy, transparent, animated, particle, ribbon, effect, and unsupported cases.
- [ ] T024 [US3] Deduplicate asset preparation and animation updates only where placement-local state does not require duplication, and add fallback reason logging without reintroducing per-placement client I/O in `wow-viewer/src/viewer/WoWViewer/Terrain/WorldScene.cs` and `wow-viewer/src/viewer/WoWViewer/Rendering/WmoRenderer.cs`.
- [ ] T025 [US3] Add unique-asset, compatible-bucket, instance, fallback, animation-update, and draw-submission counters to the existing world frame statistics owner in `wow-viewer/src/core/WowViewer.Core.Runtime/World/WorldRenderFrameStats.cs` and its viewer accumulation sites.
- [ ] T026 [US3] Run focused batching/lifetime tests and a Debug build; stop before runtime proof if any transparent/effect fallback or WMO placement-local invariant regresses.

## Phase 5: User Story 4 — Diagnostics and proof handoff (P2)

**Goal**: A frame report explains fog coverage, residency exceptions, minimap ownership, doodad batching, and stage costs well enough for a user-run benchmark/capture.

**Independent test**: A deterministic frame report fixture reconciles tile and doodad counts and emits named invariant failures.

- [ ] T027 [P] [US4] Add structured fog/residency/doodad diagnostics and invariant checks for duplicate fullscreen surfaces, near-field eviction, out-of-window normal admission, incompatible batch merges, and capture-lease accounting in the existing runtime/frame diagnostics owner.
- [ ] T028 [US4] Expose the compact diagnostics in the existing viewer status/log path without creating a new sidebar surface or reintroducing horizontal overflow in `wow-viewer/src/viewer/WoWViewer/ViewerApp.cs` and the relevant log/status partial.
- [ ] T029 [US4] Update `wow-viewer/specs/147-minimap-fog-instancing/quickstart.md` with the final focused test names, user-run client/build capture fields, and explicit proof limitations.
- [ ] T030 [US4] Update `wow-viewer/specs/STATUS.md`, `wow-viewer/memory-bank/activeContext.md`, and `wow-viewer/memory-bank/progress.md` to route the next implementation slice to Spec 147 only after the current planning artifact is reviewed; retain Spec 142/136/137 ownership notes.
- [ ] T031 Run focused tests, then `dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`; do not launch a long real-client capture or claim FPS/visual/audio proof.

## Dependencies and execution order

- Phase 1 contract tests and fullscreen ownership must complete before fog or doodad work, because
  minimap interaction is needed to produce reliable camera movement/capture evidence.
- Phase 2 fog coverage and residency tests must complete before Phase 3 batching, because batching
  must consume the correct active placement set rather than optimize whole-map admission.
- Phase 3 batching tests must complete before Phase 4 runtime diagnostics can be trusted for dense
  doodad captures.
- Tasks marked `[P]` have disjoint write sets and may run in parallel; integration tasks in the same
  phase remain sequential.
- Stop after every phase's focused validation. Do not proceed on a build-only result when the phase
  requires visual or real-client proof.

## MVP scope

The first user-testable MVP is Phase 1: one fullscreen minimap owner, reliable drag, and exact
three-click teleport with focused tests. The renderer MVP is Phase 2: fog-driven bounded coverage
with named residency diagnostics. Doodad batching and real-client performance proof follow only
after those contracts are stable.
