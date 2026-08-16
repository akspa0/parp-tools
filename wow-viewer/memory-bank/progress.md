# Progress — wow-viewer

Last updated: 2026-08-15

## 2026-08-15 — Spec 151 WMO admission instrumentation (source proof only, nothing measured)

- Built the counter that was missing. `WmoAdmissionTally` / `WmoAdmissionStats` in
  `Core.Runtime/World/Visibility` record **two layers**: which WMO placements entered the visible set
  and why, and which groups inside them were submitted **and on whose authority** — runtime-visibility
  disabled, placement-transform invalid, portal conservative fallback, portal only, frustum only,
  both, or GPU-instanced shell. Surfaced at Utilities > Perf > "WMO admission (this frame)".
- `CollectVisibleWmos` gained a `ref` overload; a test asserts it produces identical cull counts and
  identical visible sets to the original, so the accounting cannot be blamed for a behaviour change.
  It also counts the two rules the old `WmoCulledCount` never reported at all — hidden-by-uniqueId
  and asset-not-resident both `continue`d without incrementing it.
- **Four source findings, all with magnitudes unmeasured.** Portal culling **cannot reject** a group:
  `UpdateRuntimeVisibility` unions the portal decision with raw frustum visibility, and a conservative
  fallback admits everything by construction. WMO placements are **never rejected by the frustum**,
  because the call sites pass `IgnoreVisionConeCulling: true` and the only frustum-rejecting branch is
  guarded by `!IgnoreFrustumCulling && !IgnoreVisionConeCulling` — the same shape as the
  old-Ironforge-past-fog symptom. Group admission is evaluated **twice per placement per frame**
  (opaque + transparent both call `UpdateRuntimeVisibility`). And the recorded 7512 is a
  **submission** count spanning both passes, not distinct groups; the old counter is left untouched so
  the recorded capture stays comparable.
- Solution builds with 0 errors. Core suite failure set is **byte-identical to the pre-change
  baseline** (same 9 unrelated failures), +11 passing. Verified by stashing the change and diffing the
  failure names, after an allocation-free-recording regression showed up and was fixed: the new frame
  stats field had been initialised from a static property, putting a lazy class-constructor check on a
  path the frame history asserts allocates nothing.
- **Nothing is measured.** The user-owned Stormwind flight is what turns these counters into a
  diagnosis. No admission rule may change before that reading exists.

## 2026-08-15 — v0.5.2.1 released; renderer owner moves to WMO group admission

- **Shipped v0.5.2.1** (commit `975d0c79`, tag `v0.5.2.1`, branch `v0.5.3-dev`, release workflow
  green with all four self-contained builds attached). Out-of-band patch because v0.5.2 shipped with
  known, unresolved frame-pacing jank. Added `wow-viewer/CHANGELOG.md` and
  `docs/releases/v0.5.2.1.md`; bumped both csproj files and `ViewerApp.ViewerProductName`; updated
  both READMEs. The unfixed Stormwind WMO problem is documented as a **known issue** with real
  numbers rather than left for users to discover.
- **Phase 2 — the periodic stall was named by capture, then fixed.** `PrepareObjectPhase` peak
  283.47 ms of which `AudioRuntime.Update` was **283.46 ms** (PM4 overlay 0.12, remainder ~0), so no
  subdivision was needed. The cost was not audio: `RefreshEmitterDiagnosticsIfDue` rebuilt an
  `AudioTriggerDiagnostic` per resident emitter (5565) on a **wall-clock 250 ms** timer on the render
  thread — explaining why the "every 47–50 frames" interval drifted with framerate — and it ran
  whether or not anything displayed the result. A second, movement-triggered copy: `RemoveTile`
  rebuilt the list synchronously on streaming eviction. Gated the rebuild on
  `NoteEmitterDiagnosticsObserved()`; eviction now invalidates only.
- **Audio scoped to the camera tile.** `WorldAudioRuntime.Update` consulted no tile information at
  all and scanned every resident tile. It now takes `TerrainManager.CameraTileX/Y` (passed in, never
  re-derived) within `AudibleTileRadius`. Tile keying was investigated and **cleared** as a cause.
- **MCSE coordinate frame: measured, not guessed.** MCSE emitters read as permanently out of range
  while water-derived ones work. `ConvertSoundPosition` does `chunkCorner - local` on an unevidenced
  comment; the Ghidra work proved the 0x34 field layout, not the frame. Added `McseFrameEvidence`
  (raw min/max per axis, chunk/tile/beyond counts, explicit "inconclusive" verdict on a mixed sample)
  rather than switching frames on a hunch. Still open.
- **Confirmed by the Stranglethorn capture:** unaccounted median 0.05 / p99 0.16 ms and pass gap
  259–314 → 9.45 ms (Phase 1); **526 batched / 3 unbatched** from 0/312 with `MdxOpaqueSubmission`
  p99 30.75 → 14.12 ms (Phase 3). SC-002/003/006 met. Frame p99 barely moved (259.70 → 246.62), so
  Phase 3 was explicitly **not** credited with fixing the gallop.
- **Confirmed by the Stormwind capture (2048 frames):** `PrepareObjectPhase` max **283.4 → 2.5 ms**
  and gone from the hitch list; `SceneMaintenance` max **454.5 → 3.9 ms**; unaccounted median 0.02 /
  p99 0.11 ms; median frame 17.40 → 6.98 ms.
- **New measured owner: `WmoSubmission`** — p99 154.10 / max 161.3 ms against a 0.71 ms median, and
  **all 592 recent hitches** attribute to it at 153–157 ms. Stormwind submits **all districts at
  once**: 7512 visible groups, 80484 draw calls, 15852 doodad submissions. **An admission problem,
  not batching** — 80200 of 80484 calls are correctly batched. Belongs to Spec 151; first step is to
  instrument group admission (considered / admitted / rejected + which rule) before changing logic.
- **Spec 153 Phase 4 is likely moot** — written against `SceneMaintenance` max 454.8 ms, which no
  longer reproduces. Re-measure before implementing. **Phase 5 step 2 still owed:**
  `DeferredAssetLoads` max 442.9 ms in Stormwind against a 3.5 ms budget; the admission policy
  bounded the additive overshoot but the single-load residual needs decode off the render thread.

## 2026-08-15 — Spec 153 Phases 1/3/5 implemented (source proof only, nothing measured)

- **Phase 1 (FR-001, SC-006).** `PrepareObjectPhase` now has a stage timer and appears in the stage
  table, so its cost — including the ~212 ms stall — is no longer part of the unaccounted pass gap.
  `WorldRenderStage` gained the value, `StageCount` went 18 → 19, and the pass-gap subtraction in
  `RecordRenderRegionBreakdown` includes it so the region breakdown and stage table agree instead of
  double-counting. New `WorldFramePassInstrumentation` declares pass → stage ownership; a reflection
  test fails if any `WorldFramePasses` member has no timer, any stage is unowned, or a stage is
  claimed twice. The Perf panel gained an `other (unprobed remainder)` row, which is what decides
  Phase 0's "subdivide rather than guess" branch.
- **Phase 3 (FR-004/005/006).** The MDX batching cause was found and it was not a capability gap:
  `WorldScene.PlanVisibleMdxPasses` passed `PlanOpaqueMdxRoutes` a `requiresUnbatchedRender`
  predicate whose whole body was `return true`, so the planner routed 100% of opaque MDX to the
  fallback by construction while WMO — which reads the renderer's own declaration — batched 198/198.
  The predicate now consumes `IModelRenderer.RequiresUnbatchedWorldRender`, the same contract the
  WMO-internal doodad path already used. GPU instancing stays held out
  (`MdxRenderer.SupportsGpuInstancedOpaque` is still `false`); the win is `BeginBatch` once per
  renderer instead of a full state setup per draw, with submission order unchanged. Found and fixed
  the one real visual divergence between the two paths: `RenderInstance` ignores `_wireframe` while
  `RenderWithTransform` honours it, so `_wireframe` is now part of `RequiresUnbatchedWorldRender`
  (which also corrects the WMO doodad path). Live-revertible via `WorldScene.MdxOpaqueBatchingEnabled`,
  exposed as a checkbox so the before/after is one flight rather than two builds.
- **Phase 5 (FR-008).** New pure `DeferredLoadBudget` learns per-kind (MDX/WMO) load cost via EWMA
  plus a decaying high-water mark, and `WorldAssetManager.ProcessPendingLoads` consults it **before**
  each load rather than only between them — the old `elapsed < budget` condition would start a 55 ms
  load with 0.1 ms of a 3.5 ms budget left. **Deliberately partial:** the first load of a frame is
  always admitted so an oversized asset cannot starve, and that admission is counted in
  `OversizedAdmissionCount`. SC-005 is *not* claimed; a synchronous load larger than the whole budget
  still costs what it costs, and closing that needs decode off the render thread (plan Phase 5 step 2).
- Solution builds with 0 errors. Core suite: 9 failures before these changes and the same 9 after,
  all pre-existing and unrelated; net +15 passing tests.
- **Nothing is measured.** Every SC still needs the user-owned Stranglethorn before/after. Phase 0's
  capture protocol and an empty results table are in
  [Spec 153 research.md](../specs/153-renderer-hitch-and-batching/research.md); Phases 2 and 4 stay
  gated behind it.

## 2026-08-15 — Renderer gallop diagnosed; Spec 153 opened, flattening lane suspended

- Built the missing detector. `WorldRenderFrameStats` already produced `TotalCpuMs` plus 18 per-stage
  timers every frame and the viewer discarded all of it — `LastRenderFrameStats` held one frame and
  no history existed anywhere. Added `WorldRenderFrameHistory` (fixed-capacity ring, per-stage
  percentiles, hitch marking, unaccounted time, region peaks, allocation-free recording asserted by
  test) and an in-viewer panel under Utilities > Perf with an injected-stall self-check.
- Diagnosed against real clients across four zones. **Four measured defects**, all recorded in
  [Spec 153](../specs/153-renderer-hitch-and-batching/spec.md): a ~212 ms stall every ~47–50 frames
  living entirely in the pass gap; 100% of opaque MDX submitting unbatched while WMO batches 198/198;
  `SceneMaintenance` max 454.8 ms; and a deferred-load budget checked only between loads (58 ms vs
  3.5 ms nominal).
- Root structural cause of the invisibility: `WorldFramePasses` declares eleven passes and only
  `PrepareObjectPhase` assigns no stage timer, so its cost could not appear in the stage table at all.
- **The allocation-churn hypothesis was refuted by measurement** (median world-render CPU 0.33–8.58 ms;
  traversal max 0.22 ms). Per Spec 152's own Phase 1 decision point, Phases 3–5 (flatten the scene
  graph into retained draw lists, view modes) are suspended rather than continued on momentum. The
  churn fixes that landed — traversal now allocation-free in steady state, diagnostics off the hot
  path, opaque-pass buffers reused and pooled — are kept on their own merits.
- Ruled out with evidence so they are not re-proposed: decoded-asset caching and LRU thrash
  (`MaxMdxCached = 0`, unlimited; 554 models serve 18663 instances), scene-graph traversal, per-frame
  diagnostic logging.
- Stranglethorn Vale is now the standard benchmark scene; it exposed both defects where quieter zones
  hid them. Baseline table recorded in the Spec 153 plan.
- No renderer defect is fixed yet. Next bounded action is Spec 153 Phase 0: capture the
  `PrepareObjectPhase` sub-probes to name the stall before proposing a fix.

## 2026-08-15 — Repair Utilities minimap routing after sidebar consolidation

- Isolated the Utilities page selector from the shared Inspect/Scene/Experimental index. Legacy
  menu, keyboard, and persisted-settings paths remain synchronized, while opening Utilities defaults
  to Minimap again.
- The minimap renderer and tile data path were not changed. The focused cross-platform viewer build
  passes with 0 errors; live loaded-world and interaction proof remains user-owned.

## 2026-08-15 — Spec 080 Phase 2D placement ownership cleanup

- Implemented the ownership contract: Scene Placements is a list-only WMO/MDX route, Inspect owns
  selected-placement editing, scene investigation, world context, and full MCNK/ADT analysis,
  Phase Map selection is under left World Maps, and SQL population has a named Experimental page.
- The historical composite world-object body remains compatibility-only. Focused route checks,
  diff checks, and the isolated cross-platform viewer build pass with 0 errors; compact-window and
  runtime visual proof remain open.
- Renderer, fog, ADT admission, object submission, and streaming behavior remain out of scope.

## 2026-08-15 — Spec 080 Phase 2E Inspect and terrain page consolidation

- Scene now keeps Placements/LOD only, Terrain Lab owns tiles with chunk clipboard/save, and Inspect
  exposes Archeology as a direct dropdown page alongside MCNK/ADT, world context, investigation,
  animation, and action pages.
- Compatibility callers map legacy Tiles to Experimental Terrain Lab and legacy Archeology to
  Inspect. Focused route checks and the isolated viewer build pass with 0 errors; compact-window
  and runtime visual proof remain open. Renderer and streaming behavior remain out of scope.

## 2026-08-15 — Sidebar entry-point convergence; renderer hitch deferred

- Main Panels entries now land directly on their named Utilities page: Log Viewer,
  Perf, Asset Catalog, and Taxi. Capture and Camera Path continue to land on the shared Capture page.
- Recorded the reported camera-movement hitch as a deferred Spec 150 observation. No renderer, fog,
  ADT admission, object submission, or streaming code was changed during this sidebar pass.
- Sidebar-only source checks and the full Debug build pass with 0 errors; compact-window manual proof
  remains open.

## 2026-08-15 — Utilities ownership and animation restoration

- Promoted Utilities to the canonical fourth right-sidebar destination. Audio now exists only as
  Utilities -> Audio; it is no longer a duplicate top-level destination or an Experimental page.
- Restored the existing MDX/M2 animation controls inside Inspect for standalone models and selected
  world MDX instances without adding another model-information route.
- The WMO-beyond-fog symptom remains deferred to Spec 150; no renderer or streaming code changed.

## 2026-08-15 — Keep source navigation in the left sidebar

- Removed the right-sidebar Scene `Source` page and its dead source-body route. The right Scene
  selector now contains only Placements, Tiles, and LOD.
- Kept compatibility mapping for placements, tiles, selection, and LOD; source/file/map loading is
  owned exclusively by the left Navigator sidebar.

## 2026-08-15 — Spec 080 Phase 2A sidebar IA and unified inspector

- Replaced the visible tabbed `Model / World / Tools` top row with five deliberate destinations:
  `Quick`, `Inspect`, `Scene`, `Utilities`, and `Experimental`. Quick, Inspect, and Scene now render
  their bodies directly without a second page strip; Utilities and Experimental retain only their
  purpose-specific page selectors.
- Added one inline unified inspector for selected models/objects, PM4 context, and current ADT/MCNK
  facts. Area-trigger and WL-liquid loading actions are inline rather than nested popup-only controls.
- Combined tile targeting, chunk selection, and MCNK/chunk clipboard operations in Experimental >
  Terrain Lab. Existing legacy callers map into the new destinations, while legacy route retirement
  remains a separate inventory/manual-proof gate.
- Focused source checks and the full Debug build pass with 0 errors. The full solution test command
  timed out after two minutes; the focused core suite completed with 1,019 passing, 9 unrelated
  baseline failures, and 1 skipped test. User-owned visual proof remains open.

## 2026-08-14 — Mute unproven automatic ZoneMusic playback

- Added an explicit world-audio policy that keeps automatic ZoneMusic playback disabled.
- Area/ZoneMusic resolution and status diagnostics remain visible, while MCNK/MCSE water and
  environmental emitter behavior remains independent and testable.
- Added a focused policy regression test; live water/audio proof remains user-owned.

## 2026-08-14 — Resident audio speaker-marker overlay

- Added a residency-change-only normalized emitter snapshot to WorldAudioRuntime and forwarded it
  through WorldScene.
- Added an off-by-default Audio-panel toggle that renders one source-colored 3D pin per finite
  resident MCSE/MCNK emitter through the existing batched overlay: amber MCSE, cyan MCNK water, and
  purple MCNK environment.
- Marker rendering does not probe audio files, enable world triggers, or create OpenAL sources.
  Focused diagnostics/build proof and user-owned visual/audible proof remain pending.

## 2026-08-14 — Spatial audio emitter coordinate correction

- Added the shared `TerrainCoordinateTransform` contract. Alpha and standard MCSE records retain
  raw/local positions but now anchor renderer positions to their owning chunk before range checks,
  OpenAL placement, and diagnostics.
- Corrected legacy MCNK liquid marker placement from `chunk corner + halfChunk` to the terrain
  convention `chunk corner - halfChunk`; diagnostics now identify the MCSE or MCNK coordinate path.
- Focused audio-contract tests pass (11/11). The Windows viewer project builds with 0 errors in an
  isolated output directory because the live viewer owns the normal Debug binaries. Live visual and
  audible proof against the configured 0.5.3 client remains user-owned.

## 2026-08-14 — Alpha 0.5.3 time-of-day cycle checkpoint

- Added the existing 2,880-unit / 24-real-minute world clock as a pure tested contract.
- Interactive WorldScene lighting now advances it from a monotonic frame clock by default; manual
  slider input freezes it until resumed, and Light DBC/LIT/sky/audio receive one same-frame value.
- Synthetic minimap generation remains fixed at its requested time and records
  timeOfDayMode=frozen in its manifest. Focused clock tests (10), viewer build, and harvest build
  pass with 0 errors; live early-client timing and authored-minimap tint history remain user-owned.

This is a short newest-first implementation ledger. It is not a changelog or archive. Older detail
belongs to the owning spec, linked workstream, or `memory-bank/archive/`.

## 2026-08-14 — MDX material shader parity checkpoint

- Restored the live GLSL MDX material inputs that the CPU path was already uploading: UV0/UV1
  selection, animated UV transforms, and view-normal sphere-environment mapping.
- Added a bounded `SphereEnvMap` reflective highlight, finite/clamped model-local lighting and
  emissive values, and a shared/tested material policy. The implementation translates the native
  BLS material contract; it does not load or port BLS bytecode.
- Viewer build passes with 0 errors. Real shader compilation, translucent/reflective model appearance,
  and comparison against the configured client/build remain user-owned proof gates.

## 2026-08-14 — MCNK liquid audio and camera residency stability checkpoint

- Projected resident MCNK flags/MCLQ/MH2O liquid state into typed environmental audio candidates.
  Exact-build `SoundWaterType` rows resolve `(liquid family, subtype)` to `SoundEntries` IDs without
  inventing IDs; world-trigger playback remains default-off and unresolved mappings stay visible.
- Corrected the Alpha MCLQ handoff to preserve 81 packed vertex records and 64 tile flags instead of
  flattening the surface. Standard MH2O liquid assignment replaces the provisional MCNK candidate for
  the same chunk rather than duplicating it.
- Removed camera-heading-driven residency churn, added one capped unload-hysteresis ring, and kept WMO
  group rendering fail-open for transformed frustum-visible groups after portal evaluation. These are
  source-level smoothness/correctness changes; user real-client visual/FPS/audio proof is still open.
- Active render tiles now follow mouse-look without reopening the streaming lease; focused catalog,
  flag, and audio contract tests pass 53/53. Viewer project build passes with 0 errors; the full
  solution build also passes with 0 errors. Full-suite test completion, focused PM4/area coverage,
  per-trigger toggles, ZoneMusic indirection, and runtime proof remain.

## 2026-08-14 — Dual-era AreaTable identity routing

- Restored an explicit build-selected area identity contract: Alpha 0.5.x uses packed MCNK
  `AreaNumber` high/low Zone/SubZone words with `ParentAreaNum`, while 3.3.5+ uses direct
  `AreaTable.ID` and `ParentAreaID` values.
- Applied the same layout to the status-bar/overlay `AreaTableService`, the area-audio catalog reader,
  and `WorldAudioRuntime`. Modern direct IDs now resolve through primary IDs only, so compatibility
  aliases cannot steal a row that numerically resembles a packed Alpha AreaNumber.
- Added collision and parent-path regression tests. Focused area/audio tests pass 13/13; full Debug
  build and real-client dual-era visual/audio proof remain pending.

## 2026-08-14 — Spec 151 portal/game-mode/simple-surface planning

## 2026-08-14 — Spec 151 Phase 1 bounded WMO portal visibility (`c70e1945`)

- Implemented the shared fail-open portal decision from the 0.5.3 Ghidra evidence: exterior seeds,
  source-side plane admission, transformed portal polygons, recursively narrowed clip volumes, and
  bounded depth/visit traversal. Invalid/missing geometry, singular placement transforms, boundary
  cameras, and capacity overflow remain conservative rather than dropping WMO surfaces.
- Replaced the WMO renderer's old center-distance/queue heuristic with one decision reused for group
  geometry, doodad admission, and liquids. `WmoRenderStats` now carries portal-tested, fallback, and
  admitted-group counters. The graph evaluator remains explicitly diagnostic and no longer marks a
  destination visited before its portal volume is accepted.
- Focused portal/graph tests pass 16/16 and the full solution Debug build passes with 0 errors.
  Real-client WMO visual/submission/FPS comparison remains user-owned. Next bounded slice is Phase 2
  pure game-mode state/physics and character-head anchoring; simple surface/logging work waits.

- Checkpointed the accumulated prior work as `3bfbbba4` before opening branch
  `151-portal-game-mode-surface`.
- Created the Spec Kit specification, Ghidra-backed research, data model, contracts, quickstart,
  plan, checklist, and dependency-ordered tasks for WMO portal-aware visibility, an opt-in character
  head/game-mode physics path, a low-information interactive surface, and interactive/forensic
  diagnostic profiles.
- Queried the live 0.5.3 Ghidra REST bridge directly (no GUI driving) and recorded native anchors for
  `RRenderThruPortals`, `RTransformPortal`, `ClipPortal`, `StabPortals`, and portal intersection.
  The current renderer's center-distance/depth heuristic and inside-root full-visibility fallback are
  documented as the first implementation gap.
- No source implementation or runtime/FPS claim has been made. Next bounded action is Spec 151 Phase
  1 portal decision/tests; game-mode and UI work wait for that checkpoint.

## 2026-08-14 — Spec 149 resident Zone/SubZone overlay slice

- Queried the restarted Ghidra bridge's open 0.5.3 program and confirmed `CMapChunk::Create @
  0x00698e10` stores the MCNK area value, while `AreaTableRec::Read @ 0x00585e20` exposes
  `AreaNumber`/`ContinentID`/`ParentAreaNum` and the native script getters keep ZoneText and
  SubZoneText separate.
- Added the Spec 149 resident-area story and handoff. `TerrainRenderer` now exposes a revisioned
  resident chunk snapshot; `AreaOverlayRegionBuilder` groups map-aware AreaTable results into finite
  Zone/Subzone footprint regions with deterministic colors and unresolved counts; `WorldScene` renders
  opt-in footprint outlines/pins; `ViewerApp` projects one label per group; the investigation panel
  exposes `Show Area Boundaries`, default off.
- The viewer project compiles with 0 errors using an alternate output directory because the normal
  running viewer executable is locked by PID 46216. Focused aggregation tests remain open; live visual
  streaming proof remains user-owned.

## 2026-08-14 — Spec 150 Alpha 0.5.3 renderer performance planning

- Created an evidence-first Spec Kit pack for learning from the 0.5.3 OpenGL renderer without
  porting original code: `specs/150-alpha-renderer-performance/`.
- Reused the existing production `profile-render` path and current WorldScene/TerrainRenderer/object
  counters as the baseline seam. The new lane requires native Ghidra anchors, explicit CPU versus
  GPU/driver timing classification, one reversible optimization at a time, and user-owned real
  client visual/FPS proof.
- Current retained tile VAOs/texture arrays, bounded tile admission, object visibility collectors,
  opaque batching, and GPU-instancing seams are foundations to measure, not proof of performance.
- No renderer source optimization or native performance claim was made. Next step: record 0.5.3
  renderer anchors and run two unchanged-source profiles before selecting the first owner.

## 2026-08-14 — Spec 149 PM4 region navigation and audio trigger controls

- Created the Spec Kit specification, research, data model, contracts, quickstart, and dependency-
  ordered tasks for replacing PM4 correlation UI with decoded resident-region navigation and double-
  click camera focus.
- Amended the audio scope after live viewer evidence: decoded MCNK flags/liquid data are first-class
  legacy environmental/water trigger inputs even when Alpha 0.5.3 has no MCSE, and later MCSE data is
  additive. MCSE raw/local positions must be normalized with the owning tile/chunk before range checks
  or OpenAL placement; diagnostics retain both forms.
- Audited the music path against the 0.5.3 client contract: AreaMIDIAmbiences pairing is represented
  correctly as day/night MIDI plus shared DLS metadata, but AreaTable.ZoneMusic is still incorrectly
  treated as a direct SoundEntries ID. Spec 149 now requires ZoneMusic row -> Sounds[day/night] ->
  SoundEntries indirection and explicit underwater ambience selection before playback claims.
- Defined a default-off master/per-trigger world-audio policy covering MCNK, MCSE, and current-area
  ZoneMusic inspection, while preserving explicit preview, provenance diagnostics, and MIDI/DLS gates.
- Player-height/game-mode movement is explicitly deferred. Implementation, focused tests, build proof,
  and user-run visual/streaming/audible proof remain open.

## 2026-08-14 — Shared Alpha AreaNumber high/low resolution

- Fixed the Alpha area contract across status and terrain audio: raw `AreaNumber` values are
  decoded as `high16=zone` and `low16=subzone`, with unsigned bit preservation and continent/map
  qualification. `AreaNumber`/`ParentAreaNum` are preferred for Alpha rows.
- Removed the unsafe legacy aliases that treated the two component words as standalone area IDs.
  Audio receives the status-bar's resolved ZoneText/SubzoneText context, so display and music row
  selection cannot drift to different Zone/SubZone rows.
- Added packed-word, high-bit, parent/inheritance, ambiguity, and reader regression coverage;
  focused tests pass. Real-client audio/archive/audible proof remains user-owned.

## 2026-08-14 — Alpha 0.5.3 Ghidra audio contract correction

- Read the open 0.5.3 client in Ghidra and recorded the evidence in
  `memory-bank/workstream-audio-client-053-ghidra.md`.
- Confirmed `AreaMIDIAmbiences` row-level MIDI/DLS pairing and DirectMusic hand-off; confirmed that
  `ZoneMusic` selects ordinary SoundEntries IDs rather than mapping a missing SoundEntries ID to MIDI.
- Confirmed Alpha MCSE’s 0x34-byte on-disk record versus the client’s 0x4c-byte in-memory
  `CWSoundEmitter`; corrected the reader and preserved its scheduler fields through the Alpha terrain
  hand-off. Focused decoder coverage is the next validation target.
- Confirmed the client’s map-emitter callback slots are cleared at initialization with no in-process
  registration xref. Native callback equivalence and audible proof remain open.

## 2026-08-14 — Alpha packed AreaNumber ZoneMusic resolution

- Corrected the active area-audio lookup to match the repository's DBCTool contract: Alpha
  `MCNK.Unknown3` is packed `(zone << 16) | subzone`, rows are matched by continent and
  `AreaNumber`, and parent audio inheritance follows `ParentAreaNum` before modern ID fallback.
- Preserved `AreaNumber`/`ParentAreaNum` in the shared DBC audio catalog and exposed both values in
  ZoneMusic status, including when the OpenAL backend is unavailable.
- Added packed-parent, continent-ambiguity, and reader-preservation regression coverage. Focused
  tests and the cross-platform Debug build pass; real-client ZoneMusic playback remains user-owned.

## 2026-08-14 — Spec 148 world-simulator plan and audio diagnostic slice

- Created the provenance-first `148-world-simulator` Spec Kit pack: audio truth, explicit camera
  actor, lease-based residency/batching, and local artifact-museum boundaries.
- Preserved raw MCSE coordinates alongside the existing renderer-world transform and exposed
  current resident emitter diagnostics for SoundEntries resolution, path/source, optional read/decode,
  distance, mute, and backend states in the audio panel. Diagnostics refresh on a bounded cadence
  instead of doing file-existence work on every render frame; explicit probing still reads/decodes
  only when requested.
- Cross-platform viewer Debug build passes with 0 errors; focused AudioRuntimeContractTests pass
  (9/9). Real MPQ provenance, audible playback, coordinate correctness, and performance remain
  user-owned gates.

## 2026-08-14 — Spec 146 visible audio mute control

- Added a clearly labeled, color-coded `AUDIO: ON` / `AUDIO: MUTED` button to the bottom status bar.
- Routed mute through the runtime master bus so resident emitters, preview audio, and ZoneMusic
  all silence together while preserving the configured master gain.
- Viewer cross-platform Debug build passes with 0 errors; audible mute/unmute remains user-owned.

## 2026-08-14 — Spec 147 minimap interaction, LIT coverage, and WMO doodad batching

- Removed duplicate fullscreen minimap ownership and routed docked/fullscreen input through a pure
  gesture state. Focused proof covers drag classification, target changes, timeout, invalid tiles,
  and exactly-once third-click teleport; real-client input proof remains user-owned.
- Added a low-alpha fog-colored LIT radius fill and outline to minimap markers. The color comes
  from the active LIT fog track.
- Added a cross-WMO opaque internal-doodad collection path that groups shared renderers across
  visible WMO placements, using GPU instance batches where supported and renderer-level CPU batches
  otherwise. Transparent/effect-sensitive fallbacks remain unchanged.
- Cross-platform viewer build passes with 0 errors; focused minimap and area-catalog tests pass
  (6/6). Dense Stormwind draw/CPU/FPS comparison is still user-owned.

## 2026-08-14 — DBC-driven area music resolution

- The active-build DBC provider now feeds AreaTable and optional AreaMIDIAmbiences into the viewer
  audio runtime. A resident terrain chunk's AreaID selects the most specific area or parent area.
- ZoneMusic IDs resolve through the existing SoundEntries catalog and active client file paths, then
  loop through the existing OpenAL source path. MIDI/DLS choices are reported explicitly as
  unsupported instead of being converted or guessed.
- Focused catalog inheritance tests pass; audible client proof and camera/capture synchronization
  remain open.

## 2026-08-14 — Spec 143 pre-alpha v2 LIT compatibility

- Added an evidence-bounded parser profile for the observed 0.5.3 `areatest.lit` payload:
  embedded 64-byte Global Light header, 60-byte legacy prefix, and two `0xA24` nine-track data
  sets inside a `0x1484` group payload.
- Retained the secondary data set as `LegacyPartialAlternate`; only the primary `Partial` group
  drives lighting selection. Modern v8.3–v8.5 track lengths remain strict `0..32`.
- Focused LIT tests pass (8/8), inspect tool builds, and archive-backed `lit profile` proof decodes
  `H:\\053-client` Azeroth `areatest.lit`. Viewer visual/runtime proof remains user-owned.
- Next: validate the active viewer's v2 LIT lighting/fog path, then resume evidence-backed WMO/M2
  lighting work.

## 2026-08-14 — Spec 147 minimap, fog residency, and doodad instancing plan

- Authored the bounded Spec Kit feature, research, data model, interaction/fog/batching contracts,
  quickstart, requirements checklist, and dependency-ordered tasks.
- The read-only audit found duplicate fullscreen minimap draw ownership, `TerrainManager` explicitly
  discarding `fogEnd` for streaming targets, and WMO-internal doodad batching remaining
  placement-local.
- No production code changed. Branch creation was blocked by the shared workspace's
  `.git/index.lock` permission; the user-owned `wow-viewer/imgui.ini` change remains untouched.
- Next: implement and validate Spec 147 Phase 1 only.

## 2026-08-14 — Spec 143 LIT source switching and fallback

- LIT discovery now enumerates every `.lit` file directly in the active map folder while retaining
  conventional archive probes; `.lit` is included in loose-file indexing.
- The Lighting and investigation panels can switch variants even when the selected profile failed
  to parse. When no usable map-scoped Light DBC profile exists, LIT loads automatically and its
  lighting/fog override starts enabled; marker overlays remain opt-in.
- Focused source proof passes (25 tests); the isolated viewer build passes with 0 errors and the
  existing warning set. Real-client variant/fallback appearance remains user-owned.

## 2026-08-13 — Spec 143 LIT spatial coordinate correction

- LIT list-header positions now decode client fixed-point XZY values by dividing by 36, swapping
  file Y/Z into semantic WoW XYZ, and applying the map-origin transform for renderer consumers.
- Viewer LIT diagnostics now distinguish raw XZY, decoded WoW, and renderer coordinates; minimap
  markers and camera focus use the same shared conversion.
- Focused source proof passes (23 tests); real-client marker/focus placement is user-owned.

## 2026-08-13 — Spec 142 near-field detail selection correction

- The focused regression reproduced the reported failure: a 25-tile budget kept only the immediate
  3×3 ring, then spent the remaining slots on forward tiles, allowing nearby side/rear ADTs to be
  removed while distant terrain remained visible.
- `DirectionalTileSelector` now protects the largest complete camera-centered square supported by
  the budget: 3×3 for 9–24 tiles and 5×5 at 25, before bounded forward-cone expansion.
- Focused selector tests pass (9/9). Viewer build and real-client movement/camera-path proof remain
  open and user-owned.

## 2026-08-13 — Spec 146 SoundEntries preview and diagnostics slice

- Added a reachable Tools > Utilities > Audio page with resident SoundEntries ID discovery,
  camera-local preview/stop, master/emitter gain controls, backend status, and last diagnostics.
- Hardened OpenAL source updates so listener/emitter/preview failures disable audio cleanly instead of
  escaping into the render loop; active source gain now tracks current attenuation and bus changes.
- Focused audio tests pass (10/10) and the cross-platform viewer build passes with 0 errors. User-run
  audible proof against a configured client remains open; MIDI/DLS, camera transport, and capture
  muxing remain out of scope for this slice.

## 2026-08-13 — Spec 146 packaged OpenAL Soft MCSE runtime

- Added the OpenAL Soft native package to both viewer targets and copy the selected desktop DLL
  beside Debug and publish executables as `soft_oal.dll` plus Silk.NET's `openal32.dll` name.
- The production native probe retains the loaded module before Silk.NET constructs `AudioContext`,
  preventing the prior missing-library and premature-unload failures; cleanup remains guarded.
- Focused audio tests pass (3/3), both viewer targets build with 0 errors, and a process-level
  `AudioContext` create/dispose smoke test passes against the packaged Windows output.
- Proven scope is resident MCSE positional PCM-WAV playback. User-run audible client proof remains
  required; MIDI/DLS/MP3/OGG playback and Play + Video audio muxing remain explicitly unsupported.

## 2026-08-13 — Alpha audio catalog documentation

- Added a plain-language guide for the Alpha area-audio catalog, including the `AreaTable` to
  `AreaMIDIAmbiences` join, day/night/underwater semantics, loose/archive asset resolution, exact
  `audio alpha-area` inspect commands, and the boundary between metadata proof and playback.
- Linked the guide from the viewer README, CLI guide, Spec 146 quickstart, and audio-engine plan;
  corrected the plan/audit wording that incorrectly described the existing catalog proof as absent.
- No playback backend or runtime audio claim was added; user-run audible proof remains out of scope.

## 2026-08-12 — Spec 104 MDX material/effect artifact repair

- Added classic `LITE` parsing to `MdxFile`, including static Omni/Ambient values and deferred
  `PIVT` resolution; the MDX shader now receives up to eight model-local light records.
- Added a focused synthetic parser test for the light entry/pivot contract. Source validation
  passes; the full Windows solution build remains blocked in this sandbox by denied access to
  `C:\Users\akspa\AppData\Local\Microsoft SDKs`, while the core I/O build passes.
- User-run viewer proof remains required to confirm visible MDX lamp/effect illumination.

- Implemented the missing premultiplied-alpha shader output that the transparent MDX blend state
  already requested; the compatibility fragment path now follows the same alpha contract.
- Preserved classic MTLS static emissive gain in the runtime MDX material layer and applied it as
  self-illumination only, without introducing dynamic scene lighting.
- Removed white 1x1 fallbacks from transparent MDX geosets and unresolved particle emitters; missing
  effect textures now fail closed instead of drawing invented white squares/webs. Alpha-key particles
  still use an explicit discard threshold.
- Focused parser/build proof and real-client visual proof remain separate; the user owns the latter.

## 2026-08-12 — Spec 142 shared WMO placement transform

- Routed global and tile-local MDX/WMO placement creation plus translation-only editing through one
  renderer-space transform, correcting the prior WMO-only positive-axis rotation path.
- Bounds use the same transform as mesh submission; focused transform tests pass. Real-client camera
  movement remains user-owned proof for WMO flash-in behavior.

## 2026-08-12 — Spec 142 retained-window object admission

- Resident neighbor tiles now remain eligible for MDX/WMO collection even when they are outside the
  directional detailed-terrain list; object bounds and frustum tests remain the submission gate.
- This closes the path where a 25-tile resident window still made nearby buildings disappear on camera
  turns because WorldScene admitted objects from selected tiles only. Real-client movement proof remains
  user-owned.

## 2026-08-12 — Spec 142 resident WMO camera-turn stability

- Kept camera heading as a pending WMO-load priority signal, but removed rear-cone draw-distance
  culling for resident WMOs already admitted by active ADT tiles.
- WMO visibility remains bounded by active tile admission, bounds/frustum checks, and distance;
  focused collector proof and a viewer build are required before user movement validation.

## 2026-08-12 — Spec 142 fog admission and detailed/WDL ownership correction

- Terrain tile and legacy chunk distance admission now measures the nearest point on the
  geometry bounds, not the tile/chunk center. A camera near a tile edge can no longer lose
  that nearby terrain merely because the center lies beyond the fog cutoff.
- WDL suppression now follows the selected-and-GPU-resident detailed ADT set each frame.
  Retained-only neighbors remain streamable and keep their WDL underlay until detailed terrain
  is actually submitted. Three focused bounds-distance tests pass; viewer runtime proof remains
  user-owned.

## 2026-08-12 — Spec 142 near-field WMO readiness ordering

- Pending GPU tile uploads are now ranked by selected active tiles before retained and stale
  completions, preventing background parse completion order from delaying neighboring ADTs.
- WMO assets for the camera tile and immediate retained neighbors are prioritized before WMO
  visibility collection; inactive retained tiles remain admission-gated and are not submitted.
- Source build passes in an isolated output directory; runtime movement/WMO flash-in proof remains
  user-owned.

## 2026-08-12 — Spec 142 restored bounded detail and selected-tile residency

- Preserved the renderer's established ADT coordinate span (`WoWConstants.ChunkSize`, 533.333
  yards) while widening directional selection; `WoWConstants.TileSize` is a legacy aggregate and
  is not interchangeable with the camera's ADT span.
- Selected detail tiles now participate in both desired residency and unload protection, while the
  separate retained window remains the camera-centered streaming policy.
- The selector now fills the active tile's immediate 3×3 safety ring before spending remaining
  budget forward, preventing close side/rear ADTs from popping out when the slider is 12 or lower.
- Added focused proof for the 1–25 selector and the established ADT coordinate span; real-client
  movement/FPS validation remains user-owned.

## 2026-08-12 — Spec 144 swept camera-path residency correction

- Corrected camera-path tile conversion to use ADT `TileSize` rather than terrain `ChunkSize`.
- Added a core swept-footprint selector that connects path samples in tile space and applies the
  configured tile radius, preventing fast/spline paths from skipping ADTs that then unload.
- Enabled ordinary Play to wait on the existing bounded preload lease when enabled; playback and
  capture completion/stop release it. Active directional rendering remains separate.
- Focused core proof and real-client playback/unload proof remain separate; the latter is user-owned.

## 2026-08-12 — Spec 144 cross-era client camera import repair

- Routed loose and loaded-client `.m2` camera imports through `M2ModelReaderDispatcher` instead of
  the later-era reader directly.
- Added explicit MD20 `0x109+` modern camera records with strict `0x74` span validation; removed
  the prior unconditional camera suppression that caused `cameraIndex` failures.
- Added the documented MD20 `0x100` early camera layout: `0x7c` records, old `0x1c` tracks, range
  slicing, and normalization into the shared sampler without changing later M2 track semantics.
- Focused source proof is 35 passing tests. Archived Cata `FlybyUndead` and `FlybyDwarf` both inspect
  as MD20 `0x109` with `cameras=1`; viewer playback/origin placement remains user-owned proof.

## 2026-08-12 — Spec 142 bounded camera-centered residency

- Added a pure `CameraTileWindowSelector` with deterministic bounded retention; radius two is the
  default and radius three is the explicit maximum.
- `TerrainManager` now uses the retained window for streaming/unload protection while preserving
  the directional active list for detailed terrain, liquids, scene graphs, and WMO/MDX objects.
- Added retained count/radius diagnostics and runtime controls. Focused tests and the full solution
  build are the source-level proof; radius 2/3 production capture remains user-owned.

## 2026-08-12 — Spec 142 active-tile object admission

- Scene-graph traversal and portal preparation now enumerate only the camera-selected ADT graphs
  plus external content; flat WMO/MDX collection and deferred bounds promotion use the same gate.
- Full-load retains residency for stress work without turning every resident tile into an object
  visibility candidate. Explicit capture-preload tiles remain admitted.
- The viewer builds with 0 errors and the focused directional-selector proof passes 4/4. A user-run
  production capture is still required to prove frame-time improvement and visual parity.

## 2026-08-12 — Spec 142 camera-inside WMO group admission

- WMO runtime group visibility now treats containment in any local group bounds as an inside-WMO
  state, even when the root MOHD bounds miss the camera; this keeps interior groups visible instead
  of entering portal traversal with no valid starting group.
- Focused inside/outside policy tests pass. Camera-track playback/video remains the real-client
  proof owner because the recorder is the benchmark for path stability.

## 2026-08-12 — Spec 142 strict directional tile baseline

- Added the pure `DirectionalTileSelector` contract and four focused geometry tests.
- Replaced normal fog/radial ADT admission with the active tile plus at most three immediately
  forward-facing neighbors; normal detailed/manual budgets are capped at four.
- Added render-boundary active-tile/detailed-draw diagnostics. Capture preloads and `--full-load`
  remain explicit exceptions and are not normal camera admission.
- Focused selector proof passes 4/4 and the viewer dependency graph builds with existing warnings;
  user-run movement/FPS proof remains open before any FOV-radiation work.

## 2026-08-12 — Documentation continuity cleanup

- Replaced the oversized root and viewer agent guides with short operational guides.
- Added `specs/STATUS.md` as the single current-spec router.
- Condensed this ledger and `activeContext.md`; removed duplicated historical narrative.
- Updated the documentation and plans indexes to point at the new handoff path.
- No source code, project files, client data, generated output, or active spec requirements were
  changed by this cleanup.

## 2026-08-12 — Spec 146 planning package

- Added the audio/camera playback spec, plan, tasks, and single-player roadmap.
- Scope includes capability-gated MP3/OGG/WAV/MIDI playback, emitters, camera-track audio, and
  future client/server seams. It does not select or implement a backend yet.
- Next: Phase 1 contracts and capability tests only.

## 2026-08-13 — Spec 146 resident MCSE playback slice

- Added build-aware standard/Alpha 0.5.3 MCSE emitter extraction to tile load results, including raw
  Alpha 76-byte identity, position, range, timing, mode, and preservation bytes.
- Added `SoundEntries` catalog loading from the active DBC provider, dependency-free PCM WAV decoding,
  and a viewer-owned OpenAL runtime that admits only resident tile emitters and releases them on unload.
- Added lower status-bar `Audio active/resident` diagnostics and focused Alpha MCSE/WAV contract tests.
- Source proof: focused audio tests pass (2/2), focused MCSE tests pass (2/2), and the cross-platform
  viewer Debug build passes with existing warnings. User-owned proof remains audible playback in the
  configured client; MIDI/DLS, MP3/OGG/FLAC, camera-track audio, and capture muxing remain open.

## 2026-08-13 — Sparse MCCV terrain preservation correction

- Fixed the 3.x–4.x sparse-MCNK loss where a short MCNR declaration caused the padded subchunk walk to
  skip a following 580-byte MCCV payload, even when MCLY and MCAL were absent.
- Split root/texture/object sources now retain whichever valid MCCV payload exists, and the live terrain
  adapter selects MCCV independently from the layer/alpha source. MCCV guide-image and tensor extraction
  retry using declared subchunk sizes for the same sparse layout.
- Focused source proof passes 8/8 across parser, guide-image, tensor-adjacent, and split-ADT tests. User
  still owns real 3.x–4.x client visual validation on a tile with MCCV but no MCLY/MCAL.

## 2026-08-12 — Spec 144 capture path slice

- Camera path authoring, JSON camera state, roll/time controls, contextual keybinds, path preload,
  client FlyBy import, collision hooks, and capture controls are present in the current viewer
  surface.
- Focused source/build proof exists; user-run real-client and capture proof remains open.

## 2026-08-12 — Spec 145 UI first slice

- Contextual help/keybind surface, bounded sidebar navigation, wrapped log output, and v0.5.2 UI
  metadata were landed.
- Remaining work is the explicitly listed persistent-window and placeholder/control audit; do not
  infer a complete UI overhaul from the first slice.

## 2026-08-11 — Specs 142/143 world slices

- Scene-graph/performance and world-context work have source slices, but FPS, runtime stability,
  WMO-area decoding, and lighting still require their owning evidence gates.
- Treat crash logs and user screenshots as validation inputs, not as implementation proof.

## Handoff rule

When a task completes, update the owning spec first, then adjust this ledger only if the next-agent
routing changed. Move superseded detail to the owning archive rather than appending more history.
## 2026-08-13 — OpenAL absence must fail closed

- Added a platform-aware native OpenAL probe in `WowViewer.Core.Audio`.
- `WorldAudioRuntime` now avoids Silk.NET `AudioContext` construction when the
  optional native library is absent and guards cleanup after backend failure.
- Added a missing-library contract test and documented the no-OpenAL acceptance
  path in Spec 146.
- Remaining proof: run the viewer without OpenAL installed and verify it stays
  alive; then validate actual emitter playback with an OpenAL-enabled client.
## 2026-08-13 — Workbench tab rails replace unreachable overflow arrows

- Replaced the workbench's horizontal primary and nested sub-tab strips with
  directly clickable vertical rails.
- Capture Automation and Camera Path now use the same reachable rail when
  opened inside the Utilities surface.
- Updated Spec 080 with the reachable-navigation requirement and task proof
  row. Remaining proof is compact-window UI validation by the user.
