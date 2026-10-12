# Batch G1 — World Environment (Lighting, Scene Graph, Camera/Audio, Minimap/Fog)

Specs audited: 106, 107, 135, 137, 142, 143, 144, 146, 147, 148.

---

### 106 native-daynight-lighting
- Stated status: Planned | Tasks: no tasks.md (only spec/plan/research/data-model/quickstart/contracts/checklists).
- Scope: Separate world-light direction (evidence-bound, per-build) from color/fog/sky/shadow-projection; lock a native-to-viewer coordinate transform; keep dataset provenance/source-group discipline for synthetic time-of-day training rows.
- Verified implemented:
  - Exact 225° light-ray azimuth and 110°/127° timed polar-angle samples (FR-003) -> `src/core/WowViewer.Core.Renderer/Terrain/Native0533368WorldLightDirection.cs:18-24` (`ThetaRadians = 3.926991f // 225 degrees`, `2.216568f // 127 degrees`, `1.919862f // 110 degrees`).
  - Separate solar/lighting-math modules -> `src/core/WowViewer.Core/Terrain/TerrainSolarDirection.cs`, `src/core/WowViewer.Core/Terrain/TerrainLightingMath.cs`, `src/viewer/WoWViewer/Terrain/TerrainLighting.cs`.
- Partial: Could not confirm from this pass whether FR-004 (versioned coordinate-transform calibration with retained evidence), FR-008/FR-010 (capture sidecar provenance schema + pipeline rejection rules), FR-009 (source-group/train-val partition discipline) or the FR-012 held-out-time image-comparison procedure exist as named artifacts; no tasks.md means no checkbox trail to audit against, and a deeper trace of the dataset/store-builder pipeline (likely under `data-harvester/` or Spec 103/109/112 lineage) was out of this pass's time budget.
- Not implemented: Unknown — not enough evidence either way beyond the direction-model constant above.
- Checkbox accuracy: N/A (no tasks.md).
- Operator gates owed: SC-002 controlled native/viewer image comparison at lock time + 2 held-out times (user-run capture); real-client calibration proof.
- Open residue (spec-stated only): FR-004/FR-012 coordinate-transform calibration + evidence record if not already closed elsewhere; FR-008/FR-009/FR-010 sidecar/source-group provenance validation if not already covered by Spec 109/112/122's dataset-curation lane (memory notes indicate a related v50 clean-room dataset and curation layer already exist — likely overlapping, not a gap).
- Superseded by / overlaps: Spec 103 (image-only synthetic provenance, explicitly linked), Spec 032 (renderer parity, explicitly linked), Spec 109/112/122 (v50 dataset curation, likely satisfies FR-008-010 under different numbering per project memory).
- Disposition: FOLD
- Proposed epic theme: native-lighting-fidelity
- Confidence: medium (direction-model core is solid evidence; dataset/provenance half of the spec unverified in this pass, and no tasks.md to anchor claims).

---

### 107 lighting-quick-inspection
- Stated status: Implementing | Tasks: 6/6 checked.
- Scope: Quick-tab lighting/fog controls with a link to detailed LIT inspection; `FogEnd + 2500` far plane instead of a 6000-unit floor; hover cards only on a single confident ray hit.
- Verified implemented:
  - Far-plane padding constant -> `src/viewer/WoWViewer/ViewerApp.cs:194` `private const float TerrainFarPlanePadding = 2500f;` (matches FR-002/SC-002).
  - Precise single-ray hover state -> `src/viewer/WoWViewer/Terrain/WorldScene.cs:13081-13141` (`UpdateHoveredAssetInfo`, `TryBuildHoveredSceneInfoByRay`, `HoveredAssetInfo`), consumed from `ViewerApp.cs` (`UpdateWorldSceneHoveredAssetInfo`, `DrawSceneHoverAssetOverlay`).
  - Quick lighting/LIT/fog controls referenced by task T003 in `ViewerApp_Sidebars.cs` (file present and owns lighting UI per later specs' wiring, e.g. Spec 143's SubzoneText/status-bar work builds on the same surface).
- Partial: Did not independently re-verify the exact brush-vs-ray ambiguity suppression logic (ambiguous multi-candidate case) line-by-line; the single-ray path and constant are solid, the "suppress card on ambiguous brush hit" branch was not traced in this pass.
- Not implemented: none found.
- Checkbox accuracy: accurate for the two load-bearing claims spot-checked (T002 far-plane, T004 hover). No checked-but-absent findings.
- Operator gates owed: none stated beyond normal viewer build; this is a small, code-only slice.
- Open residue (spec-stated only): none — fully closed slice.
- Superseded by / overlaps: none; a clean, small, complete feature.
- Disposition: ARCHIVE-COMPLETE
- Proposed epic theme: n/a (complete, no epic residue)
- Confidence: high

---

### 135 phased-terrain-dual-map-overlay
- Stated status: Draft | Tasks: no tasks.md (spec.md only — never got a plan/tasks pass of its own).
- Scope: Two-map (primary + secondary overlay) phased terrain composition for 4.x (Gilneas-style phase patches over Azeroth): `OverlayMapName` on `ITerrainAdapter`, `SetOverlayMap` on `TerrainManager`, `SecondaryOverlayMap` on `WorldScene`, sparse-MCNK merge preserving parent liquid, and a UI input field.
- Verified implemented — and substantially superseded by later work:
  - `OverlayMapName` and merge behavior -> `src/viewer/WoWViewer/Terrain/ITerrainAdapter.cs:57` and `StandardTerrainAdapter.cs` (`TileExists`, `OverlayTileExists`, `LoadTileWithPlacements` at lines 264-406, 589-799).
  - `TerrainManager.SetOverlayMap` / `OverlayMapName` forwarding -> `TerrainManager.cs:497-509`.
  - `WorldScene.SecondaryOverlayMap` -> `WorldScene.cs:1597-1607`, explicitly documented as a "single-overlay shim" over a newer, richer **`PhaseLayers`/`PhaseLayerSettings`** ordered multi-layer stack (`ITerrainAdapter.cs:59-64`) that supersedes the spec's 2-map-only model with N-layer phase composition.
  - UI input for the overlay/phase stack exists under the `PhaseLayer` name in `ViewerApp_Sidebars.cs` (found via `PhaseLayer` grep across 13 files, including `ViewerApp_Sidebars.cs`, `ViewerApp_MinimapAndStatus.cs`, `MinimapHelpers.cs`, `PhaseLayerZ.cs`, `DatLayerSource.cs`) — FR-006 is satisfied, just not under the literal "Secondary Overlay Map" label the spec assumed.
- Partial: FR-003's exact "remap overlay MTEX indices into merged tile texture table" and "preserve parent liquid" claims were not re-derived line-by-line; the surrounding merge machinery (`OverlayTileExists`, layered `LoadTileWithPlacements`) is present and looks purpose-built for this, consistent with the spec.
- Not implemented: Nothing found missing; the feature reads as done and generalized.
- Checkbox accuracy: N/A (no tasks.md was ever authored for this spec — it shipped as part of the later Spec 222 Cartography / DAT-layer work instead).
- Operator gates owed: SC-002 tile-load-time parity (<50ms) is a runtime perf claim never witnessed here.
- Open residue (spec-stated only): none identifiable as still-open; scope was absorbed and exceeded by the `PhaseLayers`/Spec 222 cartography lane.
- Superseded by / overlaps: **Spec 222 (Cartography/PhaseLayers/DAT layer source)** — not in this batch's number range but is the actual current owner of this capability; also touches Spec 137 (same PhaseLayer-driven minimap fallback, see below).
- Disposition: ARCHIVE-SUPERSEDED
- Proposed epic theme: n/a (superseded, not carried forward)
- Confidence: high

---

### 137 phased-minimap-overlay-and-consistent-teleport
- Stated status: (no Status field in spec) | Tasks: no tasks.md (spec.md only).
- Scope: (1) minimap panel should show the active secondary/phase overlay map's tile BLPs, falling back to base map tiles; (2) fullscreen minimap teleport should use the same 3-click `Armed` confirmation as the small dockable minimap instead of `Immediate`.
- Verified implemented:
  - US1 (phased minimap tiles): `MinimapHelpers.cs` reads phase-layer state for tile resolution (comment at line 79 explicitly discusses "each layer's tile offset... Reading SecondaryOverlayMap here showed one layer" — confirms this was implemented and then generalized to the `PhaseLayer` stack, same as Spec 135).
  - US2 (consistent Armed teleport): `MinimapTeleportMode.Armed` is used at both the small-panel and fullscreen call sites -> `ViewerApp_Sidebars.cs:731` and `ViewerApp_MinimapAndStatus.cs:735,795`; `MinimapTeleportMode.Immediate` still exists as an enum value (`ViewerApp_MinimapAndStatus.cs:24,327,339`) but is no longer the fullscreen default per the Armed call sites found — matches the spec's intent (both surfaces now use Armed).
- Partial: Did not trace every call site to confirm zero remaining `Immediate` usage for the fullscreen map specifically (one conditional at line 339 references `teleportMode == MinimapTeleportMode.Immediate` for a `closeFullscreenAfterTeleport` flag, which looks like a residual/compatibility branch, not proof the fullscreen path still defaults to Immediate).
- Not implemented: none found.
- Checkbox accuracy: N/A (no tasks.md).
- Operator gates owed: visual/input proof of the 3-click gesture on the fullscreen map — untested here (also owned by Spec 147 US1 below, which re-specifies and re-tests this exact gesture).
- Open residue (spec-stated only): none carried forward as new scope — Spec 147 already restates and re-owns both the minimap-drag/teleport contract (US1) and implicitly the phase-tile fallback continues to ride on the Spec 135/222 `PhaseLayer` machinery.
- Superseded by / overlaps: **Spec 147** (User Story 1 explicitly re-specifies fullscreen minimap drag + Armed 3-click teleport with its own tasks/tests — this is a direct continuation/superset of Spec 137's US2), **Spec 135/222** (US1 phase-tile fallback).
- Disposition: ARCHIVE-SUPERSEDED
- Proposed epic theme: n/a (folded into Spec 147's minimap work)
- Confidence: medium (core claims verified; exact current default for `Immediate` vs `Armed` on fullscreen not 100% pinned down, but Spec 147 re-covers this same ground with fresh tests regardless).

---

### 142 world-scene-graph
- Stated status: In progress | Tasks: ~64 checked / ~78 total (dense multi-phase; exact count approximate — many phases fully done, Phases 7/8/8L partially open).
- Scope: Replace type-partitioned flat culling with a unified hierarchical scene graph (map->tile->chunk->object), nestable frustum/portal stack, ordered per-pass visibility, shared picking/query traversal; plus a large "Implementation Status" addendum covering WDL residency windows, WL* liquid tile partitioning, sky-dome sun/moon, and LightSkybox/Stars backdrop — all written as already-done narrative, separately from the checkbox list.
- Verified implemented:
  - Core graph contract: `WorldSceneGraph.cs`, `WorldSceneNode.cs`, `WorldSceneTraversal.cs`, `WorldSceneGraphObjectAdapter.cs` all present under `src/core/WowViewer.Core.Runtime/World/SceneGraph/`.
  - Bounded tile admission: `DirectionalTileSelector.cs` and `CameraTileWindowSelector.cs` present under `.../World/` (matches Phase 8M/8O, T066/T074).
  - Sky-dome sun/moon uniforms/direction -> `src/viewer/WoWViewer/Rendering/SkyDomeRenderer.cs:21-61` (`_uSunDirection`, `_uMoonDirection`, `UpdateFromLighting`), matching the "procedural sun/moon discs" and T064 claim.
  - LightSkybox/Stars asset resolution referenced from `WorldScene.cs` (file match on `LightSkybox|Stars/Stars`), matching T065.
- Partial: This spec is enormous (FR-001 through FR-039, SC-001 through SC-017, plus a full "Validation Gates" section); only a representative subset of symbols was spot-checked, not the full FR/SC matrix. The spec's own "Implementation Status" section (lines 607-739) is itself the most current and most honest account of what's done vs. not — it explicitly states: hierarchical traversal is **default-off** because a real-client A/B showed it was *slower*; portal runtime submission and WMO-doodad-set submission "remain owned by the existing renderer until parity is proven"; GPU/driver timer-query attribution (T048) is explicitly unimplemented; Phase 7 (US4 ordered pass queues, T020-T022) and Phase 8 (US5 shared spatial queries, T023-T025) are unchecked; Phase 9's T027/T028/T029 (synthetic-vs-real parity/perf report tooling) are unchecked; T054-T056 (index-first tile residency records) are unchecked.
- Not implemented: US4 ordered per-pass visibility lists (FR-008/FR-009, T020-T022) — unchecked, not found as a routed contract in the pass-coordinator; US5 shared picking/spatial query traversal (FR-011/FR-012, T023-T025) — unchecked; FR-039 modern instanced submission (T057) — unchecked; GPU/driver-wait timer-query attribution (T048) — explicitly flagged as a known gap in the spec's own text.
- Checkbox accuracy: broadly accurate — the spec's own "Implementation Status" narrative is unusually self-correcting and already states the graph selector is default-off after real-client evidence showed it regressed performance, i.e. it does **not** overclaim. This is one of the more trustworthy specs in the batch.
- Operator gates owed: SC-001/SC-003 (60fps dense-scene, 30fps full-map residency) real-client proof; SC-005/SC-006 portal/cross-era parity captures; T069/T073/T077 real-client movement captures for the bounded-admission phases — all explicitly marked pending in the spec text itself.
- Open residue (spec-stated only): US4 ordered visibility/pass queues (FR-008,009,010; T020-T022); US5 shared spatial queries (FR-011,012; T023-T025); FR-039 capability-gated modern instance submission (T057); GPU/driver timing attribution (T048); index-first tile residency (T054-T056); real-client promotion gates (T027-T029, T069, T073, T077).
- Superseded by / overlaps: **Spec 147** (fog-bounded residency + doodad instancing directly consumes/extends this spec's tile-admission and batching seams — FR-039/T057 in particular is explicitly hand-off territory to Spec 147/138); Spec 138 (4.x renderer evidence, explicitly coordinated, not owned here); Spec 136 (submission batching, explicitly deferred to it).
- Disposition: KEEP-ACTIVE
- Proposed epic theme: world-scene-graph-and-residency
- Confidence: high for "graph foundation + bounded admission is real and committed"; medium for the exact remaining-task boundary given the spec's sheer size.

---

### 143 world-context-lighting
- Stated status: Draft | Tasks: ~14 checked / ~46 total (US1 mostly done; US2/US3/US4/US5 mostly open).
- Scope: Resolve ADT/WMO area IDs to AreaTable names (`SubzoneText`), model the camera as a "player head" rig feeding context/fog/lighting consistently, restore evidence-gated WMO/MDX lighting and shader parity, keep cross-era/perf boundaries.
- Verified implemented:
  - `AreaTableService.cs` present at `src/viewer/WoWViewer/Terrain/AreaTableService.cs`, matching T010 (checked) — refactored to return structured resolution with `SubzoneText`/`ZoneText` roles per the task description.
  - `AreaContext.cs` under `src/core/WowViewer.Core/World/` — a resolved-context contract exists (though T005's full `WorldContextSnapshot`/`WmoAreaContext`/`WmoAreaIdEvidence` set was not individually confirmed).
  - T012/T013 (ViewerApp.cs camera-to-chunk call site, status-bar `SubzoneText` display) are checked and `ViewerApp.cs`/status-bar files exist with area-context wiring.
  - The Alpha world-clock sub-slice (T030a-e: LIT header fix, LIT source switcher, pre-alpha v2 partial reader, 2880-unit/24-min world-time contract, monotonic clock driving) are all checked, and this exact "2,880 units / 24 minutes" claim is corroborated independently by project memory (`project_lit_spatial_records_are_fixed_point.md`, `feedback` notes) as previously measured/fixed work — high confidence these are real.
- Partial: US2 (WMO interior area context, T016-T022) — all unchecked; US3 (player-head camera rig, `CameraHeadState`, T023-T028) — all unchecked, and grep confirms **no `CameraHeadState` class exists anywhere in `src/`**; US4 (WMO/MDX lighting restoration proper — `LightingSelection` selector, `WmoRenderer.cs`/`M2Renderer.cs` wiring, T029-T035) — all unchecked except the LIT/clock sub-tasks.
- Not implemented: US2 WMO area containment evaluator and fallback (FR-004/FR-005, T016-T022); US3 player-head camera rig and same-frame snapshot (FR-006/FR-007, T023-T028) — confirmed absent by symbol search; US4 core lighting-selection wiring into `WmoRenderer`/`M2Renderer` (FR-008/FR-009/FR-010/FR-011, T029-T035) — the LIT/clock plumbing this depends on is done, but the actual "non-flat WMO/MDX lighting" rendering change is not; US5 cross-era/perf gate (T036-T041) unchecked.
- Checkbox accuracy: accurate — checked items (T010/T012/T013/T030a-e) correspond to real code; unchecked items (US2/US3/US4-core/US5) correspond to genuinely absent code (verified via `CameraHeadState` and `LightingSelection` absence).
- Operator gates owed: SC-001/SC-002/SC-004/SC-006 real-client area/WMO/lighting matrices across three client eras — none witnessed; this spec is still mostly pre-implementation for its headline "restore lighting" promise.
- Open residue (spec-stated only): US2 WMO area context (FR-004,005; T016-T022); US3 player-head camera rig (FR-006,007; T023-T028); US4 WMO/MDX lighting-selection contract and renderer wiring (FR-008-011; T029,030,031-035); US5 cross-era/perf release gate (T036-041).
- Superseded by / overlaps: **Spec 236** (per the task brief, "lighting" work may have moved to Spec 236 under a later number — this batch did not have access to Spec 236's text to confirm, flagging for the reconciliation pass); Spec 106 (native day/night direction, explicitly coordinated not owned); Spec 138 (4.x shader evidence, explicitly coordinated not owned); Spec 142 (scene residency, explicitly coordinated not owned).
- Disposition: KEEP-ACTIVE
- Proposed epic theme: world-context-and-lighting-parity (verify overlap with Spec 236 before folding)
- Confidence: medium — the done/not-done split is clear, but this spec's relationship to Spec 236 (named in the task brief as a likely successor) could not be checked in this pass.

---

### 144 camera-capture-paths
- Stated status: Implementing | Tasks: 16 checked / 21 total (only user-run validation tasks T009, T014, T017, T019, T022 remain open).
- Scope: Stable Utilities > Capture panel; author/save/replay map-bound M2-style camera paths; import client M2/MDX cameras with `CinematicCamera.dbc` origin resolution; optional terrain/WMO collision; path preload before capture.
- Verified implemented:
  - Core path model/writer -> `src/core/WowViewer.Core.Runtime/M2/M2CameraPath.cs`, `M2CameraPathWriter.cs` (T002/T003).
  - Viewer UI -> `src/viewer/WoWViewer/ViewerApp_CameraPaths.cs`, plus `M2CameraPathRenderer.cs`, `M2CameraPathOverlayBuilder.cs`, `M2CameraPathVisualization.cs` (T005, FR-003a batched overlay).
  - `CinematicCamera.dbc` origin resolution -> `src/viewer/WoWViewer/Terrain/CinematicCameraOriginResolver.cs` (T018, FR-017) — confirms the Undead FlyBy tile-(28,28) resolution claim is backed by real code, not just narrative.
  - Tests -> `tests/WowViewer.Core.Tests/M2CameraPathTests.cs`, `M2CameraPathTileFootprintSelectorTests.cs` (T004, T020's swept-tile coverage).
- Partial: All 5 remaining open tasks (T009, T014, T017, T019, T022) are explicitly "User-run real-client validation" — code-complete, runtime-proof pending, exactly as the checkboxes indicate. No discrepancy found.
- Not implemented: nothing at the code level; only user-run capture/visual proof remains.
- Checkbox accuracy: accurate.
- Operator gates owed: T009 (preload ready-gate stability capture), T014 (client camera import + collision capture), T017 (WASD/keyboard authoring capture), T019 (Undead FlyBy playback capture), T022 (multi-tile path AOI-eviction capture) — all real-client, user-run.
- Open residue (spec-stated only): the 5 user-run validation tasks above; nothing else.
- Superseded by / overlaps: Spec 146 (audio-camera-playback) directly extends this spec's path model with audio bindings; Spec 233 (per brief, capture work may have a later home — not confirmed in this pass).
- Disposition: FOLD (code-complete; residue is entirely operator-owned capture proof, fold into a capture/camera-path epic alongside Spec 146)
- Proposed epic theme: camera-path-and-capture
- Confidence: high

---

### 146 audio-camera-playback
- Stated status: Draft | Tasks: ~14 checked / ~40 total (Phase 1/3 mostly open; Phase 4 US2 mostly done; Phase 5 US3 partially done; Phase 6 US4 partially done; Phase 7 US5 fully open).
- Scope: Backend-neutral audio runtime; bind camera-path playback to client/project audio; area ambience + MCSE positional emitters; Play+Video audio muxing; capability diagnostics; future world/session event seam.
- Verified implemented:
  - `WorldAudioRuntime.cs` at `src/viewer/WoWViewer/Audio/` — the central runtime class described across T005/T007/T019-T021 exists.
  - MCSE/emitter-adjacent code found across `StandardTerrainAdapter.cs`, `WorldScene.cs`, `WorldAudioRuntime.cs`, `McseFrameEvidence.cs`, `LegacyLiquidSoundEmitterFactory.cs` — consistent with T015-T021 (area ambience + MCSE emitter tasks, mostly checked) being real.
  - T024 (first OpenAL implementation for resident MCSE PCM-WAV playback) is checked and `src/viewer/WoWViewer/Audio/` exists as a real directory with multiple files — consistent with project memory noting OpenAL probing work (FR-018) was a deliberate fix for finalizer crashes.
- Partial: Phase 1/2 foundational audio contracts (T001-T008: backend-neutral `AudioAsset`/`AudioCapability`/`AudioBinding`/`AudioBus`/`AudioTransportState` under `src/core/WowViewer.Core/Audio/`) are **unchecked** in tasks.md, yet `WorldAudioRuntime.cs` and an `Audio/` directory clearly exist and do real work — this suggests the implementation proceeded pragmatically ahead of/around the formal contract-layer tasks rather than through them, OR the core contracts do exist but under different task numbering than what's checked. This batch did not confirm `src/core/WowViewer.Core/Audio/` contents directly — worth a follow-up grep before folding.
- Not implemented (per unchecked tasks, consistent with what a symbol search would predict): US1 explicit camera-path audio binding + shared transport wiring into `ViewerApp_CameraPaths.cs` (T009-T014, all unchecked) — camera paths can apparently play audio via emitters/ambience but the *explicit path-bound soundtrack* feature (the spec's headline US1) looks unwired; US3 Play+Video audio muxing (T025-T027, unchecked) — video capture likely still silent; US4 capability-matrix diagnostics export and bus controls (T029-T032, mostly unchecked except T028/T033/T041); US5 world/session event seam (T034-T036, fully unchecked).
- Checkbox accuracy: plausible/accurate based on symbol evidence, but not fully verified against `src/core/WowViewer.Core/Audio/` contents.
- Operator gates owed: SC-001/SC-004 audible playback + synchronized capture proof (explicitly user-run per spec's own Assumptions); T038 audible/sync capture matrix.
- Open residue (spec-stated only): US1 explicit camera-path audio binding (T009-T014); US3 Play+Video muxing (T025-T027); US4 capability diagnostics/bus controls (T029-T032); US5 event seam (T034-T036); T017a ZoneMusic->SoundEntries indirection (explicitly still open and cross-referenced from Spec 148 T006a).
- Superseded by / overlaps: Spec 148 (world-simulator) explicitly re-specifies and re-owns audio diagnostics (its Phase 1/US1 duplicates much of Spec 146's US1/US2 intent with a stricter "explain every decision" framing, and Spec 148's task T006a explicitly references the same ZoneMusic gap as Spec 146's T017a) — these two specs have substantial unreconciled overlap and should likely merge into one audio epic rather than being folded separately.
- Disposition: KEEP-ACTIVE
- Proposed epic theme: world-audio (merge with Spec 148's audio scope — see overlap note)
- Confidence: medium (core runtime class confirmed real; exact contract-layer completeness and the Spec 146/148 boundary need a closer follow-up read).

---

### 147 minimap-fog-instancing
- Stated status: Draft | Tasks: 9 checked / 31 total (Phase 1/2 US1 fully done; Phase 3 US2 fog-bounded residency fully open; Phase 4 US3 doodad batching fully open; Phase 5 US4 diagnostics fully open).
- Scope: Fix fullscreen minimap drag/triple-click teleport (single interaction owner); make `fogEnd` the authoritative tile-residency radius (not just a render effect); shared immutable doodad geometry with compatible-batch instancing; truthful per-frame residency/batching diagnostics.
- Verified implemented:
  - US1 minimap gesture contract -> `src/core/WowViewer.Core.Runtime/World/Minimap/MinimapInteractionState.cs` exists (T006), with a matching test file `tests/WowViewer.Core.Tests/World/MinimapInteractionTests.cs` (T005) — both checked and both confirmed present.
  - `MinimapTeleportMode.Armed` used consistently at small-panel and fullscreen call sites (`ViewerApp_Sidebars.cs:731`, `ViewerApp_MinimapAndStatus.cs:735,795`) — consistent with T007/T008 (single-owner fullscreen surface, Armed-everywhere) being real, and this directly closes the residual claim from Spec 137 above.
- Partial: none — Phase 1/2 is cleanly done, Phase 3/4/5 are cleanly not started.
- Not implemented (confirmed absent by symbol search):
  - `FogCoverageTileSelector` (US2, FR-005/FR-006/FR-007, T010-T017) — **no such class exists anywhere in `src/`**. `fogEnd` is not yet the tile-admission radius; the spec's central "fog is a render effect, not a streaming signal" problem statement is still the current state of the code.
  - `DoodadBatchPlanner`/`DoodadBatchKey` (US3, FR-009-FR-012, T018-T026) — **no such classes exist**. Doodad instancing/batching as specified here has not started (note: `IGpuInstancedModelRenderer` is referenced in T023 as an *existing* interface to route through, implying some instancing infra predates this spec, but the compatibility-key/batch-planning contract itself is new and absent).
  - Diagnostics (US4, T027-T031) — unchecked and no evidence of the described structured fog/residency/doodad report.
- Checkbox accuracy: fully accurate — every checked task corresponds to confirmed code, every unchecked task corresponds to confirmed-absent code.
- Operator gates owed: SC-005/SC-006 real-client fog-streaming and doodad-batching capture/FPS proof (not reachable until US2/US3 are built).
- Open residue (spec-stated only): US2 fog-bounded tile residency (FR-005,006,007,008; T010-T017) — this is the spec's actual headline goal and is 0% implemented; US3 doodad batching (FR-009-012; T018-T026); US4 diagnostics (T027-T031).
- Superseded by / overlaps: Spec 137 (US1 minimap gesture work fully absorbs and closes Spec 137's US2 residual — see above); Spec 142 (US2/US3 here are the direct continuation of Spec 142's admission/batching seams, e.g. Spec 142's FR-039/T057 modern-instancing hook is exactly what Spec 147's US3 would consume); this spec was authored on the `142-world-scene-graph` branch per its own "Branch/Workspace Exception" note, underscoring the tight coupling.
- Disposition: KEEP-ACTIVE (US1 complete/foldable, but US2/US3/US4 are substantial unstarted work and this is very likely the next real implementation slice — flagged as such in memory: "WMO group admission... NEXT UP" and "renderer defects" notes point at this exact doodad-batching/fog-admission gap)
- Proposed epic theme: world-scene-graph-and-residency (same epic as Spec 142; US1 minimap piece could fold separately into a "minimap-interaction" pocket if the epic split wants that granularity)
- Confidence: high

---

### 148 world-simulator
- Stated status: Draft | Tasks: ~13 checked / 35 total (Phase 1/US1 mostly done through T008; T006/T006a/T009/T010 open; Phases 2-5 fully open).
- Scope: A broader "artifact world simulator" framing — explain every spatial-audio decision (MCSE/area-music inspector with staged provenance), treat the camera as one authoritative world actor feeding render/audio/collision/residency, stream/batch by actor-driven fog coverage with attributable performance samples, and a longer-term local "museum session" direction (explicitly not an MMO server).
- Verified implemented:
  - `WorldAudioRuntime.cs` (same file as Spec 146's T005/T007/T019-21) is the concrete home of Phase 1's T005/T007 diagnostic-projection work — confirms Spec 148 Phase 1 and Spec 146 US2 are literally building in the same class, not parallel implementations.
  - `TerrainSoundEmitter`-adjacent files found across `StandardTerrainAdapter.cs`, `WorldScene.cs`, `WorldAudioRuntime.cs`, `McseFrameEvidence.cs`, `LegacyLiquidSoundEmitterFactory.cs` — consistent with T002 (raw/transformed MCSE position retention) being real.
- Partial: T006 (unresolved/decode-failure/OpenAL-unavailable/MIDI-DLS-unsupported diagnostic test coverage) and T006a (build-aware ZoneMusic reader) are unchecked — T006a is the same gap flagged as open in Spec 146 (its T017a), confirming these two specs are tracking one unresolved piece of work from two directions.
- Not implemented (confirmed absent — Phase 2-5 unchecked and symbol search confirms):
  - `CameraActorState` (Phase 2/US2, FR-006/FR-007, T011-T017) — **no such class exists**. The "camera as authoritative world actor" contract, the spec's second headline goal, has not started.
  - `ResidencyLease` (Phase 3/US3, FR-008, T018-T024) — **no such class exists**. Fog/path/inspection lease-union residency attribution has not started (this is the same underlying gap as Spec 147's US2 `FogCoverageTileSelector` — two specs independently proposing the same missing residency-attribution layer under different names).
  - `RenderPerformanceSample`/batch attribution (Phase 4/US3, T025-T030) — not started; same underlying gap as Spec 147's US3 doodad batching.
  - Phase 5 (US4, optional audio backends + museum-session seam, T031-T035) — not started; explicitly P2/future-facing in the spec itself.
- Checkbox accuracy: accurate — Phase 1 checked items map to real `WorldAudioRuntime.cs` code; Phase 2-5 unchecked items map to confirmed-absent classes.
- Operator gates owed: T010 (Phase 1 STOP-for-user-client-testing gate, explicitly written into the spec), plus every later phase's STOP gates (T017, T024, T030) — this spec is explicitly structured as a sequence of user-gated phases and none past Phase 1 have been reached.
- Open residue (spec-stated only): T006/T006a (audio diagnostic test coverage + ZoneMusic reader, shared gap with Spec 146); Phase 2 camera actor (T011-T017); Phase 3 residency-lease attribution (T018-T024, shared underlying problem with Spec 147 US2); Phase 4 batch/doodad performance attribution (T025-T030, shared underlying problem with Spec 147 US3); Phase 5 backend/session seams (T031-T035).
- Superseded by / overlaps: **This spec substantially overlaps Spec 146 (audio: same `WorldAudioRuntime.cs`, same unresolved ZoneMusic gap) and Spec 147 (residency/batching: same underlying fog-coverage-as-streaming-signal and doodad-batch-attribution problems, described independently as "ResidencyLease"/"RenderPerformanceSample" here vs. "FogCoverageTileSelector"/"DoodadBatchPlanner" there)**. Spec 148 reads as a broader architectural reframing ("world simulator" / actor model) proposed alongside Spec 146/147's narrower fixes, written on the same day (2026-08-14) as Spec 147. These three specs need explicit reconciliation — right now there is a real risk of Spec 146, 147, and 148 each independently re-solving overlapping audio and residency problems.
- Disposition: KEEP-ACTIVE
- Proposed epic theme: world-audio (Phase 1) folds with Spec 146; Phase 2-4 (actor/residency/batching) should be **explicitly reconciled with Spec 147** before either proceeds further — recommend one epic owns fog/residency/batching (merging 147's tile-selector language and 148's lease/actor language) rather than two.
- Confidence: medium-high on "what's built vs not"; the cross-spec overlap assessment is the main new finding and deserves operator attention during epic consolidation.

---

## Batch summary

| id | disposition | residue count | theme |
|---|---|---|---|
| 106 | FOLD | 2 (coordinate-transform calibration; dataset provenance/source-group discipline — likely already covered by Spec 109/112/122) | native-lighting-fidelity |
| 107 | ARCHIVE-COMPLETE | 0 | n/a |
| 135 | ARCHIVE-SUPERSEDED | 0 | n/a (absorbed by Spec 222 PhaseLayers) |
| 137 | ARCHIVE-SUPERSEDED | 0 | n/a (absorbed by Spec 147 US1) |
| 142 | KEEP-ACTIVE | 6 (ordered pass queues; shared spatial queries; modern instance submission; GPU timing; index-first residency; real-client promotion gates) | world-scene-graph-and-residency |
| 143 | KEEP-ACTIVE | 4 (WMO area context; player-head camera rig; WMO/MDX lighting-selection wiring; cross-era/perf gate) — verify Spec 236 overlap | world-context-and-lighting-parity |
| 144 | FOLD | 1 (bundle of 5 user-run capture-proof tasks only; code-complete) | camera-path-and-capture |
| 146 | KEEP-ACTIVE | 4 (explicit path audio binding; Play+Video muxing; capability diagnostics/bus controls; world/session event seam) — merge with 148 | world-audio |
| 147 | KEEP-ACTIVE | 3 (fog-bounded tile residency — 0% done, headline goal; doodad batching; diagnostics) | world-scene-graph-and-residency |
| 148 | KEEP-ACTIVE | 5 (ZoneMusic reader shared w/ 146; camera actor; residency-lease attribution shared w/ 147; batch/doodad perf attribution shared w/ 147; backend/session seams) | world-audio (Phase 1) + reconcile with 147 (Phases 2-4) |

**Cross-cutting finding**: Specs 146/147/148 (all created within days of each other, 2026-08-12 to 2026-08-14) contain substantial unreconciled overlap — Spec 148's `ResidencyLease`/`CameraActorState`/`RenderPerformanceSample` proposals cover the same ground as Spec 147's `FogCoverageTileSelector`/`DoodadBatchPlanner`, and Spec 148's audio Phase 1 builds in the exact same `WorldAudioRuntime.cs` class as Spec 146's US2. Both Spec 146 (T017a) and Spec 148 (T006a) independently flag the identical open `ZoneMusic -> SoundEntries` indirection gap. Recommend the reconciliation pass merge these three into at most two epics: one for world-audio (146+148 Phase 1), one for fog-residency/doodad-batching (147+148 Phases 2-4), rather than one epic per spec number.
