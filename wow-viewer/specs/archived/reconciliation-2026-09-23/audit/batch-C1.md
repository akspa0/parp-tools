# Batch C1 audit — renderer performance, frame stability, lighting, GPU/LOD

Specs: 056, 136, 138, 150, 152, 153, 226, 236. Read-only audit per `AUDIT-BRIEF.md`.

---

### 056 ViewerApp Refactor + GPU Acceleration + LOD Modernization
- Stated status: Draft | Tasks: 0/81 checked (9 phases, Phase 1/4 each split into a/b)
- Scope: Split `ViewerApp.cs` into a thin host + promote `WowViewer.Core.Renderer` into a real
  multi-tile, retained-mode, instanced, LOD-aware shared renderer (terrain/WDL/object/water/light
  LOD, BLP mip selection), then retire `wow-viewer/src/viewer/WoWViewer/Rendering/*`. Supersedes
  `specs/036-renderer-improvements`.
- Verified implemented: `WowViewer.Core.Renderer` exists with `Terrain/`, `Wmo/`, `Sky/`, `Liquid/`,
  `Texture/`, `Scene/`, `Headless/`, `Output/`, `Validation/`, `ObjectCapture/` directories (skeleton
  present) -> only used by `tools/capture`, `tools/harvest`, `tools/validation-capture`
  (`grep` of `.csproj` references), never by `WowViewer.App`/`WowViewer.CrossPlatform`.
  `WorldTerrainLodSelector` (`src/core/WowViewer.Core.Runtime/World/Terrain/WorldTerrainLodSelector.cs`)
  and `WorldObjectVisibilityCollector` (`.../World/Visibility/WorldObjectVisibilityCollector.cs`) do
  exist as static classes — but they are consumed directly by the **viewer-app** `Terrain/WorldScene.cs`,
  not by `WowViewer.Core.Renderer` (FR-005 as specified never happened; equivalent LOD/visibility logic
  exists via a different architecture than the spec planned).
- Partial: `WowViewer.Core.Renderer/Scene/SceneRenderer.cs` (105 lines), `Terrain/TerrainRenderer.cs`
  (102 lines), `Wmo/WmoRenderer.cs` (164 lines) — 371 lines total, single-tile, no `Tiles`/AOI concept,
  no instanced-draw calls anywhere in the library (`grep "Instanced"` hits only compiled `.dll`
  binaries, not source). This matches the spec's own Context description of the pre-spec state
  verbatim — the library has not moved past where it was when the spec was written.
- Not implemented: FR-001 (`RenderScene` multi-tile contract) — no such type exists. FR-002/003
  (retained VBO/IBO/UBO + instancing in the shared lib). FR-012 (`IRenderBackend`/`OpenGL/*` split) —
  no `OpenGL` directory, no `IRenderBackend` symbol anywhere in source. FR-013/014/015 (host cutover) —
  `wow-viewer/src/viewer/WoWViewer/Rendering/` still has 39 files; `ViewerApp.cs` is 16,746 lines plus
  27 `ViewerApp_*.cs` partials (~35k+ total per Spec 152's measurement) — the opposite of "substantially
  smaller." No `WowViewer.Core.Renderer.Tests` project exists (Phase 0 T001 never landed).
- Checkbox accuracy: accurate (0/81 genuinely reflects near-zero delivery against this spec's own FRs).
- Operator gates owed: all of them — no phase reached a checkpoint requiring real-client capture.
- Open residue (spec-stated only): the LOD/instancing/host-cutover intent is real and still wanted,
  but nothing in this spec's specific phase structure or file layout should carry forward verbatim —
  the equivalent capability has since been (partially) delivered through a completely different route:
  viewer-app-local GPU instancing (Spec 136/153/207), WMO shell instancing (Spec 138 slice), frame
  attribution (Spec 150/152), not through `WowViewer.Core.Renderer` promotion. FR-016/017 (Alpha MCAL
  + LK parity gates) and FR-021 (MCCV preservation, D9) are evergreen constraints worth restating in
  any successor epic, not FR-001..015's file-by-file plan.
- Superseded by / overlaps: 136, 138, 150, 152, 153, 226, 236 collectively deliver the *performance and
  correctness* intent piecemeal, in the viewer-app, without the library promotion. 207 (not in this
  batch) continues MDX GPU instancing in the same viewer-app location this spec wanted moved out of.
- Disposition: ARCHIVE-COLD
- Proposed epic theme: renderer-performance-and-lod (if the library-promotion goal is still wanted, it
  needs a fresh, much smaller spec — this one's 81-task file-by-file plan does not match where the
  code actually went)
- Confidence: high

---

### 136 M2 Doodad Rendering Performance Optimization
- Stated status: no header status field | Tasks: 8/11 checked, 3 unchecked (T008, T011, and implicitly
  the operator visual gate)
- Scope: Stop forcing M2/MDX doodads onto the unbatched draw path; batch opaque WMO-internal doodads by
  renderer; deduplicate per-frame `UpdateAnimation()` calls; bound deferred WMO-doodad and minimap
  client I/O to scene-wide budgets.
- Verified implemented: `IModelRenderer.RequiresUnbatchedWorldRender` narrowed to particle/ribbon
  presence (`M2Renderer.cs:173`, `ModelRenderer.cs:267`). `IGpuInstancedModelRenderer.cs` exists
  (`src/viewer/WoWViewer/Rendering/IGpuInstancedModelRenderer.cs`); `MdxRenderer` implements it with a
  real `BeginGpuInstanceBatch`/`QueueGpuInstance`/`EndGpuInstanceBatch` contract
  (`MdxRenderer.cs`). `WorldScene.cs` (~line 11653) actually **uses** the GPU-instanced path in
  production for opaque MDX submission when `MdxOpaqueBatchingEnabled` (default `true`,
  `WorldScene.cs:1425`) — contradicts this spec's own "Technical Approach" text claiming the GPU phase
  is "currently held out of production" (that note is stale; the gate was lifted by a later spec, see
  overlaps). `WorldAssetManager` (`Terrain/WorldAssetManager.cs`) exists and centralizes the deferred
  asset budget (T009). `WorldObjectPassCoordinator` (`src/core/WowViewer.Core.Runtime/World/Passes/`)
  groups opaque WMO doodads by `IModelRenderer` (T003/T004).
- Partial: T007 (GPU submission) is implemented and now live, not merely "added." T008 (prove visual
  parity + CPU/GPU/driver-wait measurement) remains unchecked and no receipt was found for it in this
  spec's own files.
- Not implemented: T011 (user-owned real-client comparison for the I/O containment slice) — unchecked,
  no evidence found.
- Checkbox accuracy: 1 unchecked-but-effectively-superseded (T008 — the GPU path this task gates was
  promoted to production by Spec 207's work, per code comments citing "Spec 207 US1" and "Spec 202
  research R1/R3" directly inside `WorldScene.cs`, so T008's proof gate was carried forward under a
  different spec number rather than closed here).
- Operator gates owed: T008 (visual/CPU/GPU parity proof), T011 (real-client I/O comparison) — both
  still open per the spec's own wording, though the underlying code has since moved past what T008 was
  gating.
- Open residue (spec-stated only): T008, T011 as stated.
- Superseded by / overlaps: 207 (object-draw-call-reduction, not in this batch) is the direct
  continuation of this spec's Phase 3 GPU submission work and appears to be the spec that actually
  promoted `MdxOpaqueBatchingEnabled` to default-on. 153 Phase 3 ("Restore MDX batching") also touches
  this exact code path and its research.md documents finding/fixing the MDX-batching-is-inert defect —
  likely the same fix landing under three spec numbers (136/153/207); reconcile which one owns the
  current code before folding.
- Disposition: FOLD
- Proposed epic theme: doodad-batching-and-instancing
- Confidence: medium (the T008/T011 gap is real; the cross-spec ownership of the now-live GPU path is
  inferred from code comments, not confirmed against 207's own spec text since 207 is outside this batch)

---

### 138 Cataclysm 4.x Renderer Evolution
- Stated status: Draft | Tasks: no tasks.md (uses `checklists/requirements.md` + a separate
  `wmo-doodad-batching-slice.md` sub-document with its own 5/6-item checklist)
- Scope: A broad, evidence-first epic — index a 19-module Ghidra dossier for Cataclysm build 11792,
  build a build-scoped capability-profile system spanning 0.5.3–11.x, preserve >4 MCLY layers, wire
  MCLV/MCTV/MCMT/height-blend signals, fix 4.x M2/WMO material handling, and land one bounded
  batching slice for dense 4.x scenes — all gated behind an evidence ledger (FR-001) before any
  renderer rewrite.
- Verified implemented: `research.md` (220 lines) and `data-model.md`/`contracts/source-profile.schema.json`
  exist as the evidence/profile-gate deliverable (Epic Phase Gate 1, partial). `MCLV`/`MCMT` chunk IDs
  are recognized in `AdtChunkIds.cs`, `Mcnk.cs`, `LkAdtReader.cs`/`LkAdtWriter.cs`,
  `AdtTensorPackBuilder.cs` (parsed/round-tripped at the chunk level). `MCTV` has **no** matches
  anywhere in `src/core` — unimplemented. No formal "4.x capability profile" class (FR-002's Key
  Entity) was found under that or an obviously equivalent name in `src/core/WowViewer.Core.IO`.
- Partial: the bounded first slice, `wmo-doodad-batching-slice.md` — its own status line says
  "implemented; real-client WMO-shell smoke proof complete; doodad/performance proof pending." Steps
  1-5 of 6 checked; step 6 ("capture a WMO placement with loaded internal doodads and compare
  visual/performance behavior") unchecked. The slice's own narrative log records a **real, unresolved
  defect**: a native access violation at `WmoRenderer.DrawBatch -> GL.DrawElements` reproduced across
  four consecutive user test runs (load-time crash, then post-camera-motion crash, then a regression
  back to immediate load-time crash) as of the slice document's last entry — "real-client stability is
  pending" is the final recorded state, not a resolved bug.
- Not implemented: FR-003 (>4 MCLY layer preservation, no cap removal found), FR-004 (MCTV/MCMT signal
  inventory-before-render), FR-006 (MD21/4.x WMO material/lava-effect variant routing), FR-013
  (point-light ownership model for terrain/WMO/M2) — the last of these is superseded in substance by
  Spec 236's `SceneLightManager` work (see overlaps). SC-001 (100% of selected 4.x claims proof-typed)
  not verifiable as met — no single ledger artifact enumerating claim-by-claim status was found beyond
  narrative `research.md` prose.
- Checkbox accuracy: n/a for the epic (no tasks.md); the sub-document's checklist is accurate — its own
  step 6 is correctly left unchecked given the recorded crash history.
- Operator gates owed: SC-003 (representative-scene real-render proof), SC-007 (witnessed 4.0.0
  baseline), and the WMO-doodad slice's own crash-free real-client run — all still open.
- Open residue (spec-stated only): the unresolved `WmoRenderer.DrawBatch` access-violation investigation
  documented in `wmo-doodad-batching-slice.md` step 6 is real, current, and load-bearing — it is an
  active correctness bug in the same instanced-WMO-shell code path this batch's other specs (136, 153)
  also touch, and should not be silently dropped when this spec is archived.
- Superseded by / overlaps: SceneLightManager / MOLT point-light work is now owned by Spec 236 (US2),
  which is more advanced than 138's FR-013. WMO opaque instancing overlaps 153's WMO-admission residue
  and 136's batching work — three specs converging on the same `WmoRenderer` code.
- Disposition: ARCHIVE-COLD for the broad 19-module evidence epic (large unexecuted scope, superseded
  in its lighting/batching sub-goals by later narrower specs); FOLD for the specific unresolved
  `WmoRenderer.DrawBatch` crash residue.
- Proposed epic theme: cataclysm-4x-terrain-fidelity (if resumed) / wmo-instancing-stability (for the
  crash residue, folds into the same theme as 136/153)
- Confidence: high on the crash-residue finding (directly quoted from the spec's own log); medium on
  the broader epic's implementation percentage (very large declared scope, evidence checked by sampling)

---

### 150 Alpha 0.5.3 Renderer Performance Evidence and Optimization
- Stated status: Draft | Tasks: 0/21 checked
- Scope: Build a repeatable frame-attribution report for Alpha 0.5.3 (CPU stage timing, GPU-unavailable
  labeling, workload counters), record 0.5.3-specific native (Ghidra) evidence anchors, then apply
  exactly one reversible, measured optimization.
- Verified implemented: `src/core/WowViewer.Core.Runtime/World/WorldRenderFrameStats.cs` (121 lines)
  and `WorldRenderDiagnostics.cs` (258 lines) exist; the diagnostics class genuinely emits a
  `dominant-cpu-stage` finding and a `gpu-timing-not-yet-attributed` finding
  (`WorldRenderDiagnostics.cs:138-160`) — i.e. FR-002/FR-003's shape is real code, not just a task
  description. `tests/WowViewer.Core.Tests/WorldRenderDiagnosticsTests.cs` exists (T009-equivalent
  coverage). `memory-bank/workstream-alpha053-renderer-performance.md` (the evidence ledger file named
  in T002) exists but its own content states plainly: "No native renderer anchor has been recorded
  yet," "No source optimization has been implemented under Spec 150," "No repeatable 0.5.3
  `profile-render` baseline has been run in this session."
- Partial: Phase 1/2 (attribution contract) is substantially real in code despite 0 checked boxes.
  Phase 2's counters (FR-002/FR-004 workload/pressure counters) appear present based on the same
  `WorldRenderFrameStats`/`WorldRenderDiagnostics` files, though this batch did not exhaustively map
  every named counter (visible/culled terrain chunks, WMO group/liquid/doodad submissions, etc.)
  against FR-004's list field-by-field.
- Not implemented: Phase 1 T002 (native Ghidra evidence ledger with anchors) — the ledger file exists
  as a template only, zero anchor rows filled in. Phase 3 (T010-T014, "one reversible optimization")
  — explicitly stated as not started in the workstream doc itself.
- Checkbox accuracy: accurate overall for Phase 1 (native evidence)/Phase 3 (optimization); **stale**
  for the attribution-contract shape, which exists in code as of this audit even though the workstream
  doc (dated 2026-08-14, the spec's creation date) predates it slightly — the attribution work likely
  landed later, under Spec 152 (see overlaps), not under 150's own unchecked T005-T009.
- Operator gates owed: all of Phase 3-5 (T010-T021) — no optimization has shipped, no native-vs-viewer
  comparison has been run.
- Open residue (spec-stated only): T002 (native 0.5.3 Ghidra evidence ledger, still a template) and the
  entire Phase 3 "apply one reversible optimization" if this lane is still wanted independently of
  Spec 153, which already delivered attributed, measured optimizations on Alpha 0.5.3 content (see
  overlaps) — likely duplicate intent.
- Superseded by / overlaps: this spec's Phase 1/2 goal (CPU stage attribution, GPU-unavailable
  labeling, one reversible measured optimization on a real 0.5.3 scene) is exactly what Spec 152 Phase 0
  and Spec 153 delivered, with more rigor (rolling frame history, hitch detection, before/after capture
  protocol, named defects A-D) and on the same client family (0.5.3 Kalimdor/Azeroth). 150 reads as an
  earlier, less-executed attempt at the same problem that 152/153 superseded one day after creation
  (150 created 2026-08-14; 152/153 created 2026-08-15).
- Disposition: ARCHIVE-SUPERSEDED
- Proposed epic theme: renderer-performance-attribution (absorbed into 152/153's theme)
- Confidence: high

---

### 152 Renderer Frame-Time Stability and Per-Era Terrain Lighting
- Stated status: Draft, but `plan.md` carries a live status banner: "STATUS 2026-08-15 — the Phase 1
  decision point fired, and it refuted this plan's premise... Phases 3, 4 and 5 are suspended." |
  Tasks: no tasks.md; phases tracked in `plan.md` narrative + `checklists/requirements.md`.
- Scope: Build an in-viewer rolling frame-history detector (proven via synthetic-stall injection)
  before any renderer change; attribute the "gallop"; kill per-frame churn (C1-C7); flatten the scene
  into retained draw lists (contingent on Phase 1 confirming an allocation cause); add focused view
  modes; fix per-era terrain lighting brightness; consolidate UI panel ownership.
- Verified implemented: `src/core/WowViewer.Core.Runtime/World/WorldRenderFrameHistory.cs` exists
  (Phase 0 ring-buffer deliverable). `src/viewer/WoWViewer/ViewerApp_Investigation.cs` exists (Phase 0
  in-viewer history view, US1b). Phase 1's own measured finding — reproduced in `plan.md`'s status
  banner — is that median world-render CPU is 0.33-8.58ms and traversal maxed at 0.22ms, **refuting**
  the allocation-churn hypothesis Phases 3-4 (scene-graph flattening) were built on.
- Partial: Phase 2 (churn kill, C1-C7) is stated "landed" per the status banner and "worth keeping on
  its own merits" even though it didn't fix the gallop — not independently re-verified line-by-line in
  this pass beyond the plan's own claim.
- Not implemented: Phase 6 (per-era terrain lighting, US3) — verified in code:
  `src/viewer/WoWViewer/Terrain/TerrainLighting.cs` has no era/build parameter anywhere (`grep` for
  "era"/"Era"/"Build" found no such gating), confirming the plan's own "not started" label for this
  phase is accurate and current. Phase 7 (UI single-ownership) not started — consistent with the
  God-Class Freeze note in AGENTS.md §10 still describing `ViewerApp`/`WorldScene` as unresolved.
  Phases 3, 4, 5 explicitly suspended by the plan itself, not merely unchecked.
- Checkbox accuracy: n/a (no tasks.md); the plan's self-reported phase statuses were spot-verified and
  found accurate (Phase 0 files exist; Phase 6 era param genuinely absent).
- Operator gates owed: SC-005 (1.0.0+ brightness vs native client, blocked — Phase 6 not started);
  SC-011/SC-012 (view modes, UI ownership — Phases 5/7 not started).
- Open residue (spec-stated only): Phase 6 (per-era terrain lighting, US3, FR-012..FR-016) — explicitly
  independent of the suspended phases and still fully open. Phase 7 (UI single-ownership, US7,
  FR-034..FR-037) — still fully open and explicitly tied to AGENTS.md §10's god-class freeze. Phases 3/4
  (scene flattening) are **not** residue — the plan itself says they must not proceed on the original
  premise and "may return later on its own evidence," i.e. this spec does not carry them forward as-is.
- Superseded by / overlaps: 153 is the direct successor that took over Phase 1's measured findings and
  owns the actual defect fixes; 153's own text says "Spec 152 keeps the measurement infrastructure...
  and the independent per-era terrain lighting work" — so Phase 6 here is the one piece of 152 that is
  still 152's to close, everything else moved to 153.
- Disposition: FOLD (Phase 6 era-lighting and Phase 7 UI-ownership residue only); the measurement
  infrastructure itself (Phase 0/2) is ARCHIVE-COMPLETE-in-substance and Phases 3-4 are ARCHIVE-COLD
  (refuted by measurement, explicitly not to be resumed on the old reasoning).
- Proposed epic theme: renderer-performance-attribution (measurement lineage with 150/153) +
  per-era-terrain-lighting (Phase 6, could also fold into a lighting-themed epic with 236)
- Confidence: high

---

### 153 Renderer Hitch Elimination and MDX Batching Restoration
- Stated status: "Phases 1, 3 and 5 implemented (source-proven, unmeasured). Phase 0 capture open;
  Phases 2 and 4 gated behind it." (spec.md header, dated 2026-08-15) | Tasks: no tasks.md; phase
  status tracked in `plan.md`/`research.md` narrative.
- Scope: Owns the four defects Spec 152's detector found: an unattributed ~212ms periodic stall inside
  `PrepareObjectPhase`, 100%-unbatched MDX opaque submission, a rare 454ms `SceneMaintenance` spike,
  and an unenforced deferred-asset-load budget.
- Verified implemented (cross-checked against `research.md`, which is far newer/more complete than the
  header's Aug-15 status line — later capture entries dated the same day carry it further):
  Phase 1 (per-pass stage timer for `PrepareObjectPhase`) — closes the instrumentation hole. Phase 0
  (name the stall) — found and fixed: `research.md`'s "Capture 2 — Stormwind" shows `PrepareObjectPhase`
  max collapsed 283.4ms -> 2.5ms after the audio-scoping fix (Phase 2b, MCSE frame scoped to camera
  tile) landed — **SC-001 is explicitly recorded as met** ("the ~212/283 ms periodic `PrepareObjectPhase`
  pattern is absent"). Phase 3 (MDX batching cause found and fixed) — confirmed in code: `MdxRenderer`
  implements `IGpuInstancedModelRenderer`, and `WorldScene.cs` (~line 11653) drives
  `BeginGpuInstanceBatch`/`QueueGpuInstance` for opaque MDX in production, matching `research.md`'s "the
  MDX batching cause, found and fixed" section. Phase 5 step 1 (budget checked before each load, not
  only between) — stated implemented; step 2 (decode off render thread) explicitly "not attempted."
- Partial: Phase 2's audio-scoping sub-fix landed and is measured-confirmed by the Stormwind capture
  collapse. `SceneMaintenance` (Phase 4) dropped to 3.9ms max in the same capture without dedicated
  Phase-4 work — `research.md` explicitly says "Phase 4 may need no work; re-measure before
  implementing it."
- Not implemented: the dominant remaining cost as of the last recorded capture is **not** any of this
  spec's five original defects — it is **WMO group admission** in dense interiors (Stormwind: 7512
  visible groups, 80,484 draw calls, 100% of the 592 recorded hitches at 153-157ms each, all reading
  `<- WmoSubmission`). `research.md` explicitly hands this off: "belongs to Spec 151... it needs its own
  spec/plan slice." Phase 5 step 2 (async decode) also explicitly not attempted, with `DeferredAssetLoads`
  max still 442.9ms in the Stormwind capture.
- Checkbox accuracy: n/a (no tasks.md); the header status line is stale relative to `research.md`'s later
  entries (Phase 0 is done, not "open"; Phase 2 is measured, not merely landed) — read `research.md`
  over `spec.md`'s header for current status.
- Operator gates owed: SC-007 (interactive smoothness judgement) still explicitly open per spec.md;
  the WMO-group-admission fix itself is unwitnessed since it's out of scope for this spec.
- Open residue (spec-stated only): Phase 5 step 2 (async MDX decode off the render thread, SC-005 not
  fully met — 442.9ms vs 3.5ms budget). WMO group admission is real, measured, and load-bearing residue
  but is explicitly **not** this spec's own scope — it's handed to Spec 151.
- Superseded by / overlaps: continues 152's Phase 1 finding directly. Its Phase 3 MDX-batching fix is
  the same code territory as Spec 136's Phase 3/T008 and Spec 207 (see 136's entry above) — three specs
  converged on one fix; 153's `research.md` is the most detailed measured account of it actually landing.
  Its handed-off WMO-admission residue is the direct subject of the `project_wmo_group_admission.md`
  memory note ("NEXT UP") and of Spec 151 (not in this batch).
- Disposition: ARCHIVE-COMPLETE for Phases 0/1/2/3 (measured, landed, source-proven); FOLD for Phase 5
  step 2 residue; the WMO-admission finding should be cross-referenced into whatever epic/spec-151
  successor owns that work, not re-opened here.
- Proposed epic theme: renderer-performance-attribution (with 150/152) — this is the spec where the
  lineage actually paid off with measured, landed fixes.
- Confidence: high

---

### 226 Renderer Polish — Wireframe Consistency & MDX Lighting
- Stated status: "Draft — authored verbatim from operator feedback; not diagnosed or implemented" |
  Tasks: no tasks.md (spec.md only, no plan/tasks artifacts at all).
- Scope: US1 — ghost-wireframe overlay (Spec 211) fails/disappears depending on camera angle, terrain
  texture-tile type, and entirely on MDX/M2/WMO; suspected cause is missing slope-scaled polygon-offset
  bias. US2 — MDX/M2 lighting has no specular term and per-vertex shading looks flat.
- Verified implemented: none of US1/US2 as scoped by this spec. `ModelRenderer.cs:1061` still calls
  `_gl.PolygonOffset(-1.0f, -1.0f)` — a **constant** offset with no slope-scale factor, exactly the
  cause this spec's US1 hypothesizes and asks to be fixed; this line has not changed to add
  slope-scaling. A specular term does exist in `ModelRenderer.cs:2099` (`pow(max(dot(N,H),0),32)*0.2`,
  gated to `SphereEnvMap` materials only) — but `git log -S` shows this line was added **2026-08-14**,
  three weeks **before** this spec was authored (2026-09-06), so it predates and does not answer US2's
  complaint; the spec's own claim that models "evaluate Lambert + ambient only" was already slightly
  stale when written, and the real gap (a general, material-gated, non-SphereEnvMap-only specular term)
  is still open.
- Partial: none.
- Not implemented: US1 (slope-independent wireframe line pass — FR is implicit in acceptance criteria,
  no FR-numbered list in this spec) — not started. US2 (material-gated specular, per-pixel lighting) —
  not started as scoped (the one specular path that exists is narrower and older than the ask).
- Checkbox accuracy: n/a (no tasks.md; spec's own "not diagnosed or implemented" status was verified
  accurate against current code).
- Operator gates owed: SC-1 (360-degree wireframe orbit capture), SC-2 (before/after specular capture)
  — both fully open, nothing to gate yet.
- Open residue (spec-stated only): US1 (wireframe slope-scaled bias / MDX-WMO wireframe pass entirely
  missing per the operator's second report) and US2 (material-gated specular + per-pixel lighting) as
  stated — both genuinely untouched.
- Superseded by / overlaps: none found; this is a standalone, still-live diagnostic note. Loosely
  related to Spec 236 (MDX lighting correctness) but 236 addresses diffuse/Half-Lambert/local-light
  casting, not specular or wireframe — no functional overlap, just thematic adjacency (both touch
  `ModelRenderer.cs` lighting math).
- Disposition: KEEP-ACTIVE (small, well-scoped, entirely unstarted — folding it would just relabel it;
  it's cheap to carry forward as-is into a lighting/polish epic)
- Proposed epic theme: renderer-visual-polish (wireframe + specular), could co-locate with 236's
  lighting theme given both touch the same shader file
- Confidence: high

---

### 236 Unified Scene Lighting, Doodad Performance & Client-Constrained World Pipeline
- Stated status: "Draft — authored verbatim from operator directive; ready for execution" (current
  active lane, branch `v0.5.4-dev`) | Tasks: 6/19 checked across 5 phases (counting only `[x]`/`[ ]`
  task and gate lines, not sub-bullets)
- Scope: Fix dark/inverted MDX normals and missing Half-Lambert diffuse (US1); build a `SceneLightManager`
  so doodad/WMO `MOLT` lights actually illuminate surrounding WMO/terrain/doodad surfaces (US2); unify
  WMO doodad GPU instancing for performance (US3); constrain the procedural map generator to the loaded
  client's own listfile instead of hardcoded Wrath assets (US4); fix GLB export path and implement the
  still-missing map-merge save backend (US5).
- Verified implemented: T001 — confirmed **not a blind removal but a real, scoped fix**:
  `!gl_FrontFacing` inversion in `ModelRenderer.cs`/`M2Renderer.cs` is now confined to the
  `SphereEnvMap` UV-lookup branch only (`ModelRenderer.cs:2027-2035`); the general lighting normal
  (`vNormal`/`surfaceNormal`) is unaffected by facing, matching the fix's intent. T002 — Half-Lambert
  confirmed verbatim: `float diff = nDotL * 0.5 + 0.5; diffuseStrength = diff * diff;`
  (`ModelRenderer.cs:2059-2061`). T010 — `src/viewer/WoWViewer/Rendering/SceneLightManager.cs` exists.
  T011 — `WmoRenderer.cs` has `uLocalLightPos[8]`/`uLocalLightColor[8]` uniforms and a per-light loop
  (`WmoRenderer.cs:1862-1890`). T012 — `TerrainRenderer.cs` has `MaxTerrainLocalLights`,
  `_chunkLocalLights`/`_tileLocalLights` (`LocalLightUniforms`) wired for both chunk and tile shader
  programs (`TerrainRenderer.cs:37-40, 349`). Four evidence receipts exist under
  `specs/236-scene-lighting-doodad-performance/evidence/`.
- Partial: T013 — tasks.md's own 2026-09-18 note is precise and matches code: WMO, terrain, and the
  doodad/model external-light consumer for unbatched/state-hoisted/transparent + WMO-internal doodads
  landed; the **GPU-instanced opaque doodad batch path cannot yet carry per-placement lights** (a real,
  named architectural gap — instancing and per-instance local lighting are in tension) and native
  (non-legacy) M2 remains base-lit. Gate 2 (operator visual proof that lights actually cast onto
  surrounding geometry) is correctly left unchecked.
- Not implemented: Phase 3 (T020-T022, WMO doodad GPU-instancing unification + spatial culling +
  Ironforge/Shattrath profiling) — none started. Phase 4 (T030-T032, client-constrained map generator)
  — none started; not independently re-verified against `BiomePalette.ForTheme` in this pass. Phase 5
  (T040-T043) — verified in code: `ViewerApp.cs:257` sets
  `ExportDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "output", "export")`, i.e. still
  resolves under the **build output directory** (`bin/Debug/...`), not the project workspace root —
  confirms T040 is genuinely unfixed and matches the operator's original complaint ("no files in the
  output folder... folder is created, but no glb is, ever" is explained by files landing under
  `bin/Debug/output/export`, not the workspace `output/export/` the operator is looking in).
  `MapSaveService.cs` does not exist anywhere in `src/` — confirms T041 (Spec 234's save backend) is
  still unimplemented.
- Checkbox accuracy: accurate — spot-verification of every checked task (T001, T002, T010, T011, T012)
  and every unchecked one sampled (T013, T040, T041) matched the checkbox state exactly. This is the
  cleanest checkbox record in the batch.
- Operator gates owed: T003 (standalone-model visual proof), Gate 2 (light-casting visual proof), Gate
  3/4/5 (all unstarted phases' proofs).
- Open residue (spec-stated only): T013's remaining boundary (GPU-instanced opaque doodad batch +
  native M2 external lighting — explicitly noted as "shared with Spec 242's per-placement instancing
  decision"), all of Phase 3 (US3, doodad GPU-instancing unification), all of Phase 4 (US4,
  client-constrained generator), all of Phase 5 (US5, export path + `MapSaveService` + save-button
  wiring + GLB fix).
- Superseded by / overlaps: Phase 3's doodad-instancing goal is the same territory as 136/138/153's
  batching work — by the time Phase 3 here is picked up, check whether 153/207's MDX-instancing fix
  already covers part of it. References Spec 242 (not in this batch) for the instancing/lighting
  boundary; references Spec 234 (not in this batch) for the save-pipeline it's implementing.
- Disposition: KEEP-ACTIVE (explicitly the current active lane per the audit brief; substantial real
  progress on Phases 1-2, three full phases genuinely still open)
- Proposed epic theme: unified-scene-lighting-and-doodad-performance (this spec IS the epic Phase 1/2
  work should anchor)
- Confidence: high

---

## Batch summary

| id | disposition | residue count | theme |
| --- | --- | --- | --- |
| 056 | ARCHIVE-COLD | 2 (evergreen constraints only: MCAL/LK parity gates, MCCV preservation) | renderer-performance-and-lod |
| 136 | FOLD | 2 (T008 visual/perf parity proof, T011 real-client I/O proof) | doodad-batching-and-instancing |
| 138 | ARCHIVE-COLD (epic) / FOLD (crash residue) | 1 (WmoRenderer.DrawBatch access-violation investigation) | wmo-instancing-stability |
| 150 | ARCHIVE-SUPERSEDED | 0 (superseded by 152/153, same day later) | renderer-performance-attribution |
| 152 | FOLD | 2 (Phase 6 era terrain lighting, Phase 7 UI single-ownership) | per-era-terrain-lighting / ui-ownership |
| 153 | ARCHIVE-COMPLETE (core) / FOLD (residue) | 1 (Phase 5 step 2 async decode) + 1 handoff note (WMO group admission -> Spec 151) | renderer-performance-attribution |
| 226 | KEEP-ACTIVE | 2 (US1 wireframe slope-bias, US2 material-gated specular) | renderer-visual-polish |
| 236 | KEEP-ACTIVE | 5 (T013 boundary, Phase 3 doodad instancing, Phase 4 generator, Phase 5 save/export) | unified-scene-lighting-and-doodad-performance |
