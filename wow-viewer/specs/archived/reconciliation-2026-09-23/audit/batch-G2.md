# Batch G2 Audit — Portal/Game-Mode, WMO Doodad Chronology, Precise Selection, LIT Docs, Alpha Demo Restoration, WTF Inspection, 3D Cursor, WMO Interior Picking, MCP Harness, Physics, Weather, Cursor Light, Audio Lifecycle, Creature Staging

Specs audited: 151, 155, 156, 157, 158, 159, 210, 211, 213, 214, 215, 216, 217, 218.
Method: read each spec's `spec.md`/`plan.md`/`tasks.md` where present, then grepped/read the real
source tree (`src/core/*`, `src/viewer/WoWViewer/`, `tools/*`, `tests/*`, `docs/`) to verify every
claim. Checkboxes were treated as claims requiring code confirmation, not evidence.

---

### 151 Portal-Aware Rendering, Game Mode, and Simple Viewer Surface
- Stated status: Draft | Tasks: 12/30 checked (Phase 1 + Phase 4 docs only; Phases 2-3 all unchecked)
- Scope: WMO portal visibility (US1), simple viewer surface (US2), game-mode head camera (US3), bounded
  physics (US4), interactive/forensic diagnostic profiles (US5).
- Verified implemented: WMO portal visibility decision + bounded traversal ->
  `src/core/WowViewer.Core.Runtime/World/WmoPortalVisibilityDecision.cs`; diagnostic bridge ->
  `src/core/WowViewer.Core.Runtime/World/SceneGraph/WorldScenePortalVisibilityEvaluator.cs`; integrated
  into `src/viewer/WoWViewer/Rendering/WmoRenderer.cs` `UpdateRuntimeVisibility` (replaces
  center-distance admission); tests in `tests/WowViewer.Core.Tests/World/WmoPortalVisibilityDecisionTests.cs`
  and `WorldScenePortalVisibilityEvaluatorTests.cs` (16 passed per quickstart.md, 2026-08-14).
- Partial: none for Phase 1 — fully landed and matches checked tasks.
- Not implemented: Phase 2 game-mode core (`GameModeState.cs`, `GameModePhysics.cs`,
  `CharacterHeadAnchorProvider.cs`, `ViewerApp_GameMode.cs` — none exist); Phase 3 simple surface
  (`ViewerSurfaceProfile.cs`, `ViewerApp_SimpleSurface.cs`, diagnostic-profile gating in
  `ViewerLog.cs` — none exist). T009-T025 all unchecked and confirmed absent from source.
- Checkbox accuracy: accurate — checked tasks (T001-T008, T026-T028) all verified present; unchecked
  tasks (T009-T025, T029-T030) all verified absent.
- Operator gates owed: real-client WMO portal visual/submission-time/FPS comparison (T030, SC-005);
  game-mode/simple-surface stories never reached implementation so no gate is owed yet for them.
- Open residue (spec-stated only): US2 simple interactive surface (FR-006/007), US3 game-mode head
  camera (FR-008/009), US4 bounded physics (FR-010/011), US5 diagnostic profiles (FR-012/013/014).
- Superseded by / overlaps: none found.
- Disposition: FOLD
- Proposed epic theme: renderer-performance-correctness (portal visibility landed; game-mode/simple-surface residue folds into a UI/approachability or world-simulation epic)
- Confidence: high

---

### 155 Asset Reference Inventory — Expected vs Catalogued vs Present (dir: `155-wmo-doodad-chronology`)
- Stated status: Draft | Tasks: no tasks.md (plan.md phases 0-6 are the only breakdown)
- Scope: full-corpus sweep of world-object/model asset references (US1/Phase1), 3-set
  catalogued-vs-present classification (US2/Phase2), candidate matching (US3/Phase3), cross-build
  chronology (US4/Phase4), reversible repair (US5/Phase5), conversion-capability survey (US6/Phase6).
- Verified implemented: Phase 0-1 only. `src/core/WowViewer.Core.IO/AssetReferences/` —
  `AssetReferenceModel.cs` (3-state `AssetResolution`: Present/Absent/Unreadable, deliberately not
  collapsed), `AssetReferenceSweeper.cs` (285 lines), `WmoReferenceExtractor.cs`,
  `ModelReferenceExtractor.cs`, `ModelRouteClassifier.cs` (routes blocked models per spec 154 findings).
  CLI: `tools/inspect/WowViewer.Tool.Inspect/AssetReferenceCommandSupport.cs` exposes `assets refs` and
  `assets sweep` only. Tests: `tests/WowViewer.Core.Tests/AssetReferenceReportTests.cs`.
- Partial: sweep produces per-reference Present/Absent/Unreadable outcomes (FR-001 through FR-005,
  FR-009 satisfied) but does **not** yet compute the catalogued-set comparison — no
  working/catalogue-claims-but-absent/catalogue-gap/missing classification exists in code (FR-006/007
  unimplemented despite being named in every contract doc).
- Not implemented: three-set comparison + orphan detection (Phase 2, FR-006/007), candidate matching
  (Phase 3, FR-008), cross-build chronology (Phase 4, FR-010/011), repair + reversal (Phase 5,
  FR-012/013/014), conversion-capability survey (Phase 6, FR-015). No `compare`/`candidates`/`timeline`/
  `repair` CLI verbs exist; no `CatalogueGap`/`CandidateMatch`/`IntroductionWindow`/`RepairRecord`
  types exist anywhere in `src/`.
- Checkbox accuracy: n/a (no tasks.md) — plan.md phase claims verified directly against code as above.
- Operator gates owed: none claimed yet (feature has not reached a phase requiring real-client visual
  proof); sweep counts (492 world objects / 5,545 models for 0.5.3.3368) are asserted in quickstart.md
  as "verified 2026-08-16" but not independently re-run in this audit.
- Open residue (spec-stated only): US2 three-set comparison (FR-006, FR-007, SC-004/005), US3 candidate
  matching (FR-008, SC-006), US4 chronology (FR-010/011, SC-008), US5 repair (FR-012/013/014,
  SC-009/010), US6 conversion-capability survey (FR-015, SC-011).
- Superseded by / overlaps: none.
- Disposition: FOLD
- Proposed epic theme: formats-readers-writers / asset-corpus-inventory
- Confidence: high

---

### 156 Precise Object Selection, PM4 Match Confirmation, and a World-Space Cursor
- Stated status: Draft | Tasks: no tasks.md (research.md + contracts/ exist; speckit-tasks never run)
- Scope: triangle-precise picking for regular objects (US1), triangle-precise picking for PM4 overlay
  objects (US2), durable human-confirmed PM4<->placement match library (US3), world-space cursor
  marker (US4).
- Verified implemented: none. No `ConfirmedMatch`, `WorldCursorMarker`, `PrecisePick`, `MeshHitTest`, or
  `RayTriangle` symbols anywhere in `src/`. `src/viewer/WoWViewer/ViewerApp_ClickSelection.cs` (453
  lines) still does bounding-box-only picking (comment at line 116-131 references "WMO's bounding box"
  fall-through, not triangle testing).
- Partial: research.md (Phase 0) measured that pickable mesh data is parsed then discarded at load
  time (`WorldAssetManager.cs` summaries retain only counts/bounds, not full vertex/index arrays) —
  this is real research output but zero implementation followed it.
- Not implemented: everything — FR-001 through FR-016 all unbuilt. Spec 156's own note that Spec 210
  ("3D Scene Cursor & In-World Spatial Selection") independently shipped a different world-space
  cursor is worth flagging: 210's `SceneCursorRenderer`/`SceneSpatialPicker` covers 156's US4 cursor
  marker and part of US1/US2's "test against real geometry near the cursor" intent via a different
  route (procedural/authentic cursor models + direct 3D contact detection), but 156's specific
  triangle-precise-picking-with-bounding-box-prefilter design and its PM4 confirmed-match library are
  not covered by 210 and remain unbuilt.
- Checkbox accuracy: n/a (no tasks.md).
- Operator gates owed: none (nothing built to validate).
- Open residue (spec-stated only): all of FR-001 through FR-016; likely substantially superseded/
  absorbed by Spec 210's shipped cursor + Spec 211's shipped WMO-doodad/container-fallthrough picking,
  but the *triangle-precise mesh test* and *PM4 confirmed-match library* pieces are still open and
  spec-stated, not delivered by either 210 or 211.
- Superseded by / overlaps: 210 (world-space cursor marker), 211 (container fall-through / doodad
  picking) — both overlap this spec's problem space but neither implements 156's specific mechanisms
  (mesh-triangle test, PM4 match library).
- Disposition: FOLD
- Proposed epic theme: renderer-performance-correctness / precise-picking (residue: triangle-mesh
  picking + PM4 confirmed-match library only — the cursor-marker piece is superseded by 210)
- Confidence: high

---

### 157 LIT Documentation Update
- Stated status: (no explicit Status field in spec.md; FR/NFR-only format, no tasks.md/plan present as
  a phased breakdown — spec.md is the whole artifact) | Tasks: no tasks.md
- Scope: produce wowdev.wiki-ready documentation for LIT v2/v83/v84/v85, distinguishing implementation
  findings from original wiki content (FR-001 through FR-008).
- Verified implemented: `docs/wowdev-wiki/lit-draft.md` (276 lines) — covers header (8 bytes), light
  header (64 bytes) with the XZY 1/36-fixed-point note, v2/v83/v84/v85 version differences, color
  tracks (BGRX, 2880 time units/day, cyclic interpolation), float bands per version, light group kinds
  including `LegacyPartialAlternate`, validation rules, known files, and a "Differences from Original
  Wiki" summary table. Every non-obvious claim is tagged `[Implementation Note: ...]` per NFR-002.
  Traceable to code: `src/core/WowViewer.Core.IO/Lit/LitProfileReader.cs`,
  `LitProfileModels.cs`, `LitSummaryReader.cs`.
- Partial: none found — all 8 FRs and all 6 acceptance criteria are addressed in the doc.
- Not implemented: none.
- Checkbox accuracy: n/a (no tasks.md; doc-completeness verified by direct read).
- Operator gates owed: NFR-001 "ready for wowdev.wiki submission" — the doc exists and reads as
  submission-ready, but actual community submission is an operator action outside this repo's scope
  and unwitnessed here.
- Open residue (spec-stated only): none — all FRs satisfied by the existing draft.
- Superseded by / overlaps: none.
- Disposition: ARCHIVE-COMPLETE
- Proposed epic theme: formats-readers-writers (documentation)
- Confidence: high

---

### 158 Alpha Demo Restoration — WTF Commands, Camera Follow, and Torchlight
- Stated status: Draft, **self-marked SUPERSEDED (2026-08-16)** in its own scope note for the
  WTF-content-survey portion — spec explicitly says its filename-based WTF conclusion is retracted
  pending Spec 159 | Tasks: no tasks.md
- Scope: US1 WTF settings reader, US2 worldport/teleport command execution, US3 WTF browser/waypoint
  tab, US4 Alt+P perf-overlay toggle, US5 camera-follow-model, US6 equipped torch point light, US7
  replay a captured investor demo (explicitly blocked-on-data).
- Verified implemented: none of 158's own stories are built in the viewer. A WTF reader does exist
  (`src/core/WowViewer.Core.IO/Wtf/WtfModel.cs`, `WtfLineClassifier.cs`, `WtfSweeper.cs`) but per
  Spec 159's plan.md this was built *under Spec 159*, not 158 — it classifies SET/bind/port-command-
  shaped lines for inspection, it does not execute worldport/teleport commands or move the camera.
- Partial: US1 (WTF settings reader) is satisfied by 159's `WtfModel`/`WtfSweeper`, which already
  parses `SET name "value"` — but that capability lives under and is credited to Spec 159, not 158.
- Not implemented: US2 command execution (no `worldport`/`teleport` command runner reachable from
  `ViewerApp*`; the only "worldport"/"teleport" hits in the viewer are unrelated minimap-tile-click
  teleport code in `ViewerApp_MinimapAndStatus.cs`), US3 WTF browser tab (no WTF UI panel found), US4
  Alt+P binding (no `Alt` + `P` keybind anywhere in `src/viewer/WoWViewer`), US5 camera-follow-model (no
  `CameraFollow`/`FollowBone` symbols), US6 equipped torch point light (no `AttachedLightSource`/
  `TorchLight` symbols — also blocked on Spec 218's unbuilt attachment rendering), US7 explicitly
  blocked-on-data per spec's own text (no demo*.wtf file possessed).
- Checkbox accuracy: n/a (no tasks.md).
- Operator gates owed: none built yet to validate.
- Open residue (spec-stated only): US2 (FR-004/005/006/007), US3 (FR-017/018/019/020), US4
  (FR-008/009), US5 (FR-010/011), US6 (FR-012/013/014/015) — US1's reader is effectively delivered via
  159's WtfSweeper and US7 stays explicitly blocked-on-data per FR-016.
- Superseded by / overlaps: 159 (WTF reading/classification — 158's US1 scope note itself declares
  158's earlier WTF survey superseded by 159); 218 (torch-in-hand attachment, a precondition for US6);
  216 (effect-cast-light mechanism, also a precondition for US6).
- Disposition: FOLD
- Proposed epic theme: world-simulation-audio-environment (camera-follow + torch stories fold in
  alongside 216/218; WTF command execution folds in alongside 159's inspection findings)
- Confidence: high

---

### 159 WTF Command Inspection
- Stated status: Draft | Tasks: no tasks.md — plan.md is written *after* implementation and documents
  real progress/gaps directly (explicitly noted as a process deviation in plan.md)
- Scope: sweep every `.wtf` file (loose + archive-packed) across staged clients, classify every line
  (SET / bind / port-command-candidate / unrecognized), sweep every staged build, and probe candidate
  filenames directly against archive hash tables (bypassing listfiles).
- Verified implemented: `src/core/WowViewer.Core.IO/Wtf/WtfModel.cs` (`WtfLineKind`: Set, Bind,
  PortCommandCandidate, Unrecognized; `WtfFileSurvey`, `WtfBuildSurvey` with dedup'd
  `DistinctUnrecognizedLines`; `WtfCandidateProbeResult`), `WtfSweeper.cs` (`EnumerateCorpus` unions
  internal listfiles + `GetAllKnownFiles` + loose-disk walk; `Sweep`; `SweepFile`; `ProbeCandidate`
  bypasses listfiles via direct hash-table read). CLI:
  `tools/inspect/WowViewer.Tool.Inspect/WtfCommandSupport.cs`. Tests:
  `tests/WowViewer.Core.Tests/WtfLineClassifierTests.cs`.
- Partial: plan.md records real, honest results — Phase 3 (US3, sweep every staged build) ran against
  only 2 of ~10 builds (0.5.3.3368, 2.0.0.5610, both 100% recognized/zero unrecognized) and explicitly
  flags its `--listfile` wiring as "done in code, but never validated end-to-end" (silent-failure risk
  noted by the spec's own author). Phase 4 (US4, candidate probing) ran for real: 2,217 real zone-named
  candidates from the community listfile probed against 8 staged builds — zero zone-named WTF files
  resolved in any staged build (a genuine negative result, with one real find: `wtf\runonce.wtf`, an
  EULA/TOS flag file, not demo content).
- Not implemented: sweeping the remaining ~6-8 staged builds not yet covered (1.x line, 3.0.1.x,
  3.3.5.x); the `--listfile` path re-validated with an absolute path per plan.md's own "must be re-run"
  note.
- Checkbox accuracy: n/a (no tasks.md; plan.md's own status markers were spot-checked against code and
  are accurate — e.g. "SHIPPED, UNEXERCISED" for Phase 4 candidate probing matched by the later Phase 4
  results section showing it was, in fact, subsequently exercised).
- Operator gates owed: obtaining a build that actually contains zone-named WTF files is explicitly a
  data-acquisition problem, not a code problem, per plan.md's own "Outstanding Work" section.
- Open residue (spec-stated only): FR-005 (sweep every staged build — only 2/~10 done), the unvalidated
  `--listfile` absolute-path re-run, obtaining a build with the sought demo-adjacent content.
- Superseded by / overlaps: supersedes 158's earlier (retracted) WTF-content conclusion.
- Disposition: FOLD
- Proposed epic theme: world-simulation-audio-environment / formats-readers-writers (residue: finish
  the build sweep, validate `--listfile`, keep hunting for a zone-named-WTF-bearing build)
- Confidence: high

---

### 210 3D Scene Cursor & In-World Spatial Selection
- Stated status: (no explicit spec.md Status field) | Tasks: 27/30 checked — only Phase 5 (T401-T404,
  verification/operator gates) unchecked.
- Scope: authentic 3D in-scene cursor (US1), 3D spatial selection with dynamic cursor states (US2),
  in-scene 3D disambiguation for clustered picks (US3), camera-culling invariant + configurable cursor
  style (US4), OpenSCAD MCP interface for procedural mesh generation (US5).
- Verified implemented: `src/viewer/WoWViewer/Rendering/SceneCursorRenderer.cs`,
  `ProceduralMeshLoader.cs`, `src/core/WowViewer.Core/Geometry/OffGeometry.cs`,
  `src/viewer/WoWViewer/Rendering/SceneClusterSelector3D.cs`,
  `src/viewer/WoWViewer/Rendering/CameraHudRig.cs` (task text names it `CameraHudRig3D.cs`, actual
  filename is `CameraHudRig.cs` — same capability, minor filename drift, not a functional gap). Test:
  `tests/WowViewer.Core.Tests/OffGeometryTests.cs`.
- Partial: `tests/WowViewer.Core.Tests/SceneSpatialPickerTests.cs` named in T401 does **not exist** —
  T401 is correctly left unchecked.
- Not implemented: T402/T403 interactive real-client verification (Alpha 0.5.3 and Standard 1.12+),
  T404 STATUS.md/activeContext.md completion doc update — all correctly unchecked. STATUS.md line 64
  independently confirms: "Interactive visual gates still owed on shipped specs (210, 211, ...)".
- Checkbox accuracy: accurate, with one cosmetic filename mismatch (`CameraHudRig` vs
  `CameraHudRig3D` in task T106 — functionally present either way).
- Operator gates owed: T402/T403 interactive verification (cursor rendering, culling, cluster pop-up,
  cursor-style switching) — real-client visual proof not yet witnessed.
- Open residue (spec-stated only): T401 unit tests, T402/T403 operator interactive verification, T404
  doc updates.
- Superseded by / overlaps: 211 (WMO doodad/container-fallthrough picking extends the same click
  pipeline); 156 (156's unbuilt triangle-precise-picking and PM4 match-library remain distinct residue
  not covered by 210).
- Disposition: FOLD
- Proposed epic theme: renderer-performance-correctness (residue: T401 unit test + operator visual
  verification only — code is otherwise complete)
- Confidence: high

---

### 211 WMO Interior Ray Picking, Doodad Selection & Ghost Transparent Wireframes
- Stated status: (no explicit Status field) | Tasks: 27/29 checked — only T407 and T505 (both explicit
  "Operator interactive verification" tasks) unchecked.
- Scope: WMO container click fall-through to interior objects (US1), WMO doodad ray picking (US2),
  ghost transparent wireframe rendering at ~33% alpha for objects and terrain (US3), plus two later
  operator-follow-up phases (hover tooltip, doodad selection bounds/3D aids).
- Verified implemented: `WmoRenderer.TryPickDoodadsByRay`, `WmoRenderer.TryGetDoodadLocalBounds` ->
  `src/viewer/WoWViewer/Rendering/WmoRenderer.cs`; container fall-through rule extracted to
  `src/core/WowViewer.Core.Runtime/World/WmoContainerFallThroughFilter.cs` (reused by both hover and
  click paths per T402); `WorldScene.cs` wires `CollectSceneObjectPickHits`,
  `TryBuildHoveredSceneInfoByRay`, `TryBuildSelectedWmoDoodadInstance`; tests in
  `tests/WowViewer.Core.Tests/World/WmoContainerFallThroughFilterTests.cs` including the T405-added
  `ApplyFallThrough_PreservesWmoDoodadCandidateAndUnrelatedRayOrder` case.
- Partial: none — every checked implementation task has a matching source symbol.
- Not implemented: T407 (hover-a-doodad-inside-a-WMO interactive verification), T505 (select-a-barrel-
  in-Ironforge interactive verification) — both correctly unchecked, both are real-client visual gates.
- Checkbox accuracy: accurate.
- Operator gates owed: T407, T505 — both explicit real-client interactive checks, consistent with
  STATUS.md's "Interactive visual gates still owed on shipped specs (210, 211, ...)".
- Open residue (spec-stated only): T407, T505 operator verification only.
- Superseded by / overlaps: 210 (shares the click-selection pipeline).
- Disposition: FOLD
- Proposed epic theme: renderer-performance-correctness (residue: two operator interactive-verification
  checks only — code is complete)
- Confidence: high

---

### 213 MCP Tooling Harness
- Stated status: Draft | Tasks: no tasks.md, no plan.md, no research.md — only `spec.md` and
  `checklists/requirements.md` exist. Planning phase never started.
- Scope: expose the repo's 10 CLI tools (`inspect`, `harvest`, `capture`, `converter`, `enrich`,
  `mask-validate`, `validation-capture`, `wdl-read`, `wmo-minimap`) over a Model Context Protocol
  server, with schema/parser contracts derived from one shared definition, progress/cancellation for
  long-running invocations, resource addressing for produced artifacts, and an operator-controlled
  exposure boundary (read-only by default).
- Verified implemented: none. No `ModelContextProtocol`/`McpServer`/`ToolRegistry`/"Tool Contract"
  symbols anywhere in `src/`. No `tools/*mcp*` server project exists (the only `*mcp*` hit in `tools/`
  is unrelated bin/obj build artifacts). Note: an *unrelated* OpenSCAD MCP server was wired up for
  Spec 210 (`.mcp.json` config, per 210-T001) — that is a consumed third-party MCP client integration
  for procedural mesh generation, not this spec's server-exposing-the-CLI-tools deliverable; the two
  are easily confused by name only.
- Partial: none.
- Not implemented: all of FR-001 through FR-020 — the entire spec is unbuilt.
- Checkbox accuracy: n/a (no tasks.md).
- Operator gates owed: none (nothing built).
- Open residue (spec-stated only): the entire spec — FR-001 through FR-020, US1 through US6 (US6
  explicitly deferred within the spec itself).
- Superseded by / overlaps: none found; spec explicitly notes Spec 212 (spatial UI shell) is
  independent and covers "the other half of the same long-range direction."
- Disposition: FOLD
- Proposed epic theme: infrastructure-governance (whole-spec residue carries forward as-is; nothing to
  trim)
- Confidence: high

---

### 214 5.0.1 Physics — Decode the Data, Drive a Licensed Solver
- Stated status: Draft | Tasks: 13/31 checked (Phases 1-4 + T013-T018 of Phase 5 substantially done;
  T021-T031 correctly unchecked pending operator go-ahead)
- Scope: recover Domino/PhysData contract from the 5.0.1 binary as data+behavior only (no algorithm
  transcription) (US1), parse physicalised model sidecars (US2), integrate a licensed third-party
  solver for rigid bodies (US3), simulate cloth/flags (US4), budget simulation by distance cull (US5),
  support the joint family (US6), era-gate physics to 5.0.1+ only (US7).
- Verified implemented: era/provenance/admission policy ->
  `src/core/WowViewer.Core.Runtime/World/Physics/PhysicsRuntimePolicy.cs` +
  `tests/WowViewer.Core.Tests/PhysicsRuntimePolicyTests.cs` (16/16 per tasks.md T011). `.phys` sidecar
  reader -> `src/core/WowViewer.Core.IO/Phys/PhysSidecarPath.cs`,
  `src/core/WowViewer.Core.IO/Phys/PhysReader.cs`, `src/core/WowViewer.Core/Phys/PhysDocument.cs`
  (T013-T018, 24/24 focused tests per tasks.md). Extensive evidence trail in
  `specs/214-mop-physics-domino/evidence/`: `domino-caller-map.md`, `physics-adapter-contract.md`,
  `physics-contract.md`, `solver-selection.md` (Jitter2 2.8.10, MIT license, selected over
  BepuPhysics v2 for cloth/determinism support), `current-implementation-audit.md`.
- Partial: T008's own review is an honest partial — SC-002 (line-level attribution of ~90 Domino
  internal callers) is explicitly graded PARTIAL with a recorded rationale for not closing it (no
  downstream consumer, since algorithms are never transcribed) rather than silently marked done.
- Not implemented: T002 (real-client manifest/hash capture) still open and gates fidelity validation;
  T021 (operator approval to add the Jitter2 package reference — confirmed **not present** in any
  `.csproj`) blocks all of Phase 3 (T022-T027: solver facade, shape mapping, gravity/FP-state adapter
  obligations, deterministic solver mode, resting/energy fixtures, cull-distance binding); T028 (joint
  mapping — SHOJ/WELJ have no clean Jitter2 counterpart, flagged as open risk), T029 (cloth adapter),
  T030 (viewer binding), T031 (BOXS 0..47 field re-measurement) all explicitly deferred with reasons.
- Checkbox accuracy: accurate — every checked task has a verified file/evidence match; every unchecked
  task is genuinely blocked on an explicit, stated gate (operator go-ahead or a prior open task).
- Operator gates owed: T021 (package-reference approval, a permanent third-party dependency decision);
  T002 (real-client asset manifest); all real-client visual/motion comparison against reference footage
  (SC-006) once Phase 3 lands.
- Open residue (spec-stated only): T002, T021-T031 in full (solver integration, cloth, joints, viewer
  binding, budget wiring, BOXS field re-measurement).
- Superseded by / overlaps: feeds spec 215 (wind input) at one declared boundary only.
- Disposition: KEEP-ACTIVE
- Proposed epic theme: world-simulation-audio-environment (largest, most actively evidenced spec in
  this batch — too big/live to fold into a passive epic list entry)
- Confidence: high

---

### 215 5.0.1 Weather System — Decode and Implement
- Stated status: Draft | Tasks: no tasks.md, no plan.md, no research.md — only `spec.md` and
  `checklists/requirements.md` exist (checklist itself is fully checked, i.e. spec-quality-complete,
  not implementation-complete).
- Scope: recover a Weather.dbc/MapWeather contract from the 5.0.1 binary (US1), render precipitation
  (US2), blend weather transitions (US3), expose a wind field consumed by spec 214's cloth (US4),
  modulate lighting/fog only through their owning systems (specs 143/147/160) (US5), storm lightning
  (US6), era-gate to 5.0.1+ (US7).
- Verified implemented: none. No `WeatherContract`/`WeatherState`/`MapWeather`/`PrecipitationField`/
  `WindField` symbols anywhere in `src/`.
- Partial: none.
- Not implemented: all of FR-001 through FR-022 — entire spec unbuilt, planning phase not started.
- Checkbox accuracy: n/a (no tasks.md).
- Operator gates owed: none (nothing built).
- Open residue (spec-stated only): the entire spec — explicitly blocked in sequence behind spec 214
  (wind consumer) per spec's own Assumptions ("Spec 214 is the wind consumer, not a dependency" —
  technically unblocked to start, but practically low-value before 214 has cloth to drive).
- Superseded by / overlaps: none; declared boundary with specs 143 (lighting), 147 (fog), 160 (sky),
  214 (physics/wind consumer).
- Disposition: FOLD
- Proposed epic theme: world-simulation-audio-environment
- Confidence: high

---

### 216 Model Cursor as a Scene Light Source
- Stated status: Draft | Tasks: no tasks.md, no plan.md, no research.md — only `spec.md` and
  `checklists/requirements.md`.
- Scope: generalize the Spec-210 scene cursor to any MDX/M2 model (US1), play the cursor model's
  particles (US2), make the cursor model's particle *effects* (not just static LITE lights) cast real
  light onto surrounding scene geometry (US3 — the actual novel gap identified in the spec's own
  research), reproduce a specific 2001-era torch-lit screenshot as an acceptance test (US4), era-gate
  the mechanism (US5).
- Verified implemented: none of this spec's new work. The spec's own pre-work research (measured
  directly in this audit) is accurate: `LitSourcePathResolver`/`LitLoader` do already probe
  `areatest.lit` (confirmed same as Spec 157's documented reader); `ModelRenderer` does build
  `ParticleEmitter`s from `MdxParticleEmitter2` via `ParticleRenderer`; `UploadMdxLights` exists but
  (per spec's own accurate claim, not independently re-verified line-by-line in this audit) only
  uploads a model's own LITE lights into that model's own shader — no scene-wide light propagation
  exists. No `EffectDerivedLight`/`CursorModel`/`ScenePreset`/`CursorLightContribution` symbols found.
- Partial: none — this is a pure planning-stage spec building on Spec 210's shipped cursor and Spec
  143's LIT chain, neither of which this spec has extended yet.
- Not implemented: all of FR-001 through FR-021.
- Checkbox accuracy: n/a (no tasks.md).
- Operator gates owed: none (nothing built); SC-006 (screenshot match) is explicitly operator-judged
  once built.
- Open residue (spec-stated only): entire spec; explicitly coordinated-not-blocking with spec 218
  (torch-in-hand rig) and spec 143 (LIT chain ownership, FR-015a coordination point).
- Superseded by / overlaps: builds on 210 (cursor asset path) and 143 (LIT chain) without modifying
  either; pairs with 218 (attachment rig) for the full screenshot reconstruction.
- Disposition: FOLD
- Proposed epic theme: world-simulation-audio-environment
- Confidence: high

---

### 217 Audio Playback Lifecycle and Correctness
- Stated status: Draft | Tasks: no tasks.md, no plan.md, no research.md — only `spec.md` and
  `checklists/requirements.md`.
- Scope: explicit six-list sound lifecycle matching 5.0.1's `SoundKitObject` model (US1), distinguish
  one-shot/periodic/looping repeat modes (US2), play zone music/ambience on category buses (US3),
  bounded+prioritized playback channels (US4), weighted variation selection (US5), enable audio by
  default once the above are demonstrated (US6), era-gate 0.5.3 vs 5.0.1 audio chains (US7).
- Verified implemented: none of the described lifecycle/repeat-mode/budget machinery. No
  `SoundKitObject`/`SoundLifecycle`/`PlaySoundKitID`/`PeriodicSound`/`forceNoDuplicates` symbols in
  `src/`. Pre-existing (pre-217) audio system found at `src/viewer/WoWViewer/Audio/*` and
  `ViewerApp_Audio.cs` — this is the *broken* baseline the spec's Context section describes (audio
  disabled by default; the very defect the spec exists to fix), not new work product of this spec.
- Partial: none.
- Not implemented: all of FR-001 through FR-024.
- Checkbox accuracy: n/a (no tasks.md).
- Operator gates owed: none built yet; SC-007 "long unattended session produces no runaway sound" and
  all listening-based judgement are explicitly operator work per spec's own Assumptions.
- Open residue (spec-stated only): entire spec, all 7 user stories.
- Superseded by / overlaps: none.
- Disposition: FOLD
- Proposed epic theme: world-simulation-audio-environment
- Confidence: high

---

### 218 Creature Staging — Spawn, Equip, Pose, Reconstruct
- Stated status: Draft | Tasks: no tasks.md, no plan.md, no research.md — only `spec.md` and
  `checklists/requirements.md`.
- Scope: attach a model to another model's attachment point, following animation (US1); deliberate
  creature spawn/place/pose (US2, building on existing spawn infra); paper-doll equipment panel (US3);
  save/restore a full scene arrangement (subject+equipment+pose+LIT+time+camera) (US4); drive the
  existing capture automation unattended (US5); era-gate creature/display/item resolution (US6).
- Verified implemented: none of this spec's new work. The spec's own pre-work research is accurate as
  measured in this audit: `MdxAttachment`/`MdxAttachmentFile`/`MdxAttachmentSummary` types exist in
  Core but are referenced **nowhere** in `src/viewer/WoWViewer/Rendering/` (confirmed by direct grep —
  zero hits) — this is the exact gap the spec names as its reason for existing. Existing spawn
  (`WorldSpawnRecord`, `WorldScene.SetExternalSpawns`, `AlphaCoreDbReader` creature->display->model
  chain) and existing capture automation (`camera_shot_points.json`, capture queue) were not
  independently re-verified line-by-line in this audit but are consistent with prior batch findings
  for those systems elsewhere in the codebase.
- Partial: none — attachment rendering, equipment resolution, and scene save/restore are all unbuilt.
- Not implemented: all of FR-001 through FR-025.
- Checkbox accuracy: n/a (no tasks.md).
- Operator gates owed: none built yet; SC-009 (screenshot comparison) explicitly operator-judged.
- Open residue (spec-stated only): entire spec; explicitly the load-bearing prerequisite for spec 216's
  torch-in-hand reconstruction (216 can be tested on a bare model without 218, but the full screenshot
  reproduction needs both).
- Superseded by / overlaps: pairs with 216 (effect-casts-light) for the shared screenshot-reconstruction
  goal; depends on nothing else in this batch.
- Disposition: FOLD
- Proposed epic theme: world-simulation-audio-environment
- Confidence: high

---

## Batch summary

| id | disposition | residue count | theme |
|---|---|---|---|
| 151 | FOLD | 4 (US2-US5: simple surface, game mode, physics, diagnostic profiles) | renderer-performance-correctness |
| 155 | FOLD | 5 (US2-US6: 3-set comparison, candidates, chronology, repair, conversion survey) | formats-readers-writers |
| 156 | FOLD | 2 (triangle-mesh picking, PM4 confirmed-match library) | renderer-performance-correctness |
| 157 | ARCHIVE-COMPLETE | 0 | formats-readers-writers |
| 158 | FOLD | 5 (US2-US6: command execution, WTF browser, Alt+P, camera-follow, torch light) | world-simulation-audio-environment |
| 159 | FOLD | 3 (finish build sweep, validate --listfile, find zone-WTF build) | world-simulation-audio-environment |
| 210 | FOLD | 3 (T401 unit test + T402/T403 operator visual verification) | renderer-performance-correctness |
| 211 | FOLD | 2 (T407 + T505 operator visual verification) | renderer-performance-correctness |
| 213 | FOLD | 1 (entire spec, unbuilt) | infrastructure-governance |
| 214 | KEEP-ACTIVE | 6 (T002, T021-T031: solver integration, cloth, joints, viewer binding) | world-simulation-audio-environment |
| 215 | FOLD | 1 (entire spec, unbuilt) | world-simulation-audio-environment |
| 216 | FOLD | 1 (entire spec, unbuilt) | world-simulation-audio-environment |
| 217 | FOLD | 1 (entire spec, unbuilt) | world-simulation-audio-environment |
| 218 | FOLD | 1 (entire spec, unbuilt) | world-simulation-audio-environment |
