# Active Context — wow-viewer

Last updated: 2026-08-15

This file is the interchange for the next agent. It records only the current routing state. Read
the owning spec for requirements and proof; read a workstream only when the spec links it.

## Current handoff

**START HERE: Spec 153 — the stall is NAMED and Phase 2 is written. Re-capture Stranglethorn to confirm.**

- **Defect A is `AudioRuntime.Update`, and it was never audio.** Captured 2026-08-15 in Stranglethorn:
  `PrepareObjectPhase` peak 283.47 ms of which `AudioRuntime.Update` was **283.46 ms**; PM4 overlay
  window 0.12 ms; unprobed remainder ~0. The cost is `RefreshEmitterDiagnosticsIfDue` rebuilding an
  `AudioTriggerDiagnostic` per resident emitter (5565 of them) four times a second on the render
  thread — a wall-clock 250 ms period, which is why the frame interval drifted with framerate.
  **It is a diagnostics surface, and it ran whether or not anything displayed it.** A second,
  movement-triggered copy: `RemoveTile` rebuilt the whole list *synchronously* on streaming eviction.
- **Next action is a re-capture, not more code.** Both stalls are fixed (Phase 2 below) but
  **unmeasured**. Re-capture checklist is in
  [Spec 153 research.md](../specs/153-renderer-hitch-and-batching/research.md): with the audio panel
  closed, `PrepareObjectPhase` max should collapse from 283.4 to ~0 and p99 from 221.59 to ~0; the
  "Recent hitches" list should stop reading `<- PrepareObjectPhase`; cross tile boundaries
  deliberately because the `RemoveTile` stall was movement-triggered; and confirm Utilities > Audio
  still populates. SC-001 needs the periodic pattern *absent*, not smaller.
- **Landed this session (source proof only, no measurement):**
  - *Phase 1 (FR-001, SC-006)* — `PrepareObjectPhase` now has a stage timer and appears in the stage
    table; its cost is out of the unaccounted pass gap. `WorldFramePassInstrumentation` declares
    pass → stage ownership and a reflection test fails the build if any pass has no timer, any stage
    is unowned, or a stage is double-counted. `StageCount` 18 → 19.
  - *Phase 3 (FR-004/005/006)* — the MDX batching cause was **not** a capability gap: the planner
    predicate in `PlanVisibleMdxPasses` was a literal `return true`, routing 100% unbatched by
    construction. It now consumes `IModelRenderer.RequiresUnbatchedWorldRender`, the same contract
    the WMO doodad path already used. GPU instancing stays off (`SupportsGpuInstancedOpaque` is
    still `false`); the win is begin-once/submit-many state. `_wireframe` was added to
    `RequiresUnbatchedWorldRender` — it was the one genuine visual divergence between the batched and
    unbatched paths. Revertible live via `WorldScene.MdxOpaqueBatchingEnabled`.
  - *Phase 5 (FR-008)* — `DeferredLoadBudget` (core, pure, 9 tests) learns per-kind load cost and is
    consulted **before** each load instead of only between them. **Partial by design:** it removes
    the additive overshoot but a single synchronous load bigger than the whole budget still costs
    what it costs, counted as `OversizedAdmissionCount`. SC-005 is not claimed; closing it needs
    decode off the render thread (plan Phase 5 step 2).
  - *Phase 2 (FR-003, SC-001) — written after the capture named its target, unmeasured.* The
    diagnostics rebuild is now gated on `NoteEmitterDiagnosticsObserved()`, which only the audio
    panel calls, so `Update` does no diagnostics work when nothing displays it; and the eager
    synchronous rebuild in `RemoveTile` is removed (invalidate alone is sufficient). Audio playback
    never reads that list, so gating it cannot change what is audible. Deliberately **not** done:
    making `BuildEmitterDiagnostic` cheaper — with the gate in place that cost is only paid by
    someone who opened the panel, and it is a separate change with its own before/after.
- **Confirmed by the same capture:** Phase 1 works (unaccounted median 0.05 / p99 0.16 ms; pass gap
  peak 259–314 → 9.45 ms; hitches now name a stage). Phase 3 works (**526 batched / 3 unbatched**,
  from 0/312; `MdxOpaqueSubmission` p99 30.75 → 14.12 ms, median 2.03 → 0.01). **SC-002/003/006 met.**
- **Do not credit Phase 3 with fixing the gallop.** Frame p99 barely moved (259.70 → 246.62) because
  the periodic stall was never the MDX cost — the exact confusion the plan's risk table warned about.
- **Audio emitters: scoped to the camera tile; the MCSE coordinate frame is now MEASURED, not fixed.**
  User report: only water-triggered emitters behave, every MCSE emitter reads out of range wherever
  the camera goes. `Update` consulted **no tile information at all** — it scanned every resident tile.
  It now takes `TerrainManager.CameraTileX/Y` (passed in, never re-derived) and considers only tiles
  within `WorldAudioRuntime.AudibleTileRadius` (1); the diagnostics panel uses the same window.
  **Tile keying was checked and cleared** — `AddTile`/`RemoveTile`/`EmitterKey` all agree with
  `OnTileLoaded`; scanning every tile just made it look like a keying fault.
  **The out-of-range cause is NOT settled.** `AlphaTerrainAdapter.ConvertSoundPosition` does
  `chunkCorner - local` on the strength of an unevidenced comment ("Alpha MCSE stores a chunk-local
  C3Vector"); the Ghidra work proved the 0x34 *field layout*, not the frame. If it is not chunk-local,
  every MCSE emitter lands tens of thousands of units off-map — which also explains why MCNK liquid
  rows work, since they derive from the renderer's own `chunk.WorldPosition`. New `McseFrameEvidence`
  reports raw min/max per axis and chunk/tile/beyond counts with an explicit verdict at the top of
  Utilities > Audio. **Read that line before touching the transform**; it says "inconclusive" rather
  than picking a winner on a mixed sample. No viewer test assembly exists, so this is source proof only.
- **Named remaining owners of movement jank, in measured priority order:** (1) re-capture Phase 2;
  (2) **Phase 4 — `SceneMaintenance` max 454.5 ms**, now the largest single stage observation, gate
  is open; (3) **Phase 5 step 2 — decode off the render thread**, since `DeferredAssetLoads` p99
  14.31 / max 103.1 ms against a 3.5 ms budget proves admission policy alone cannot meet SC-005.
  Spec 152 Phase 6 (per-era terrain lighting, fixes 1.0.0+ darkness) remains independent, not started.
- **The measurement tool exists and works.** Utilities > Perf > Frame history: rolling per-frame
  history, hitch detection with dominant-cause attribution, unaccounted time, region peaks,
  submission batching counts, and an injected-stall self-check. Recording is allocation-free.
  Use it for every before/after. **Benchmark scene: Stranglethorn Vale.**
- **The hitching observation is no longer deferred and is no longer a mystery.** Four measured
  defects, recorded in Spec 153 with numbers: (A) ~212 ms stall every ~47–50 frames, all of it in the
  pass gap; (B) 100% of opaque MDX submit unbatched while WMO batches 198/198; (C) `SceneMaintenance`
  max 454.8 ms; (D) load budget checked only between loads, 58 ms against 3.5 ms.
- **Refuted, do not revive without new evidence:** the allocation-churn hypothesis. Median
  world-render CPU is 0.33–8.58 ms and traversal maxes at 0.22 ms. Spec 152 Phases 3–5 (flatten the
  scene graph into retained draw lists, view modes) are **suspended** because they rested on that
  premise. Also ruled out with evidence: decoded-asset caching / LRU thrash (`MaxMdxCached = 0`,
  unlimited; 554 models serve 18663 instances — nothing is re-decoded, the cost is submission).
- **Detector lessons that made this possible:** p99 hides rare hitches (they land at p100 — use max
  and over-threshold count); ranking stages by p99 buries a rare-but-huge stage (sort by max);
  always report unaccounted time, or attribution names a 0.2 ms stage for a 350 ms frame.
- **Lower-priority target:** User-run visual/compact-window proof for the Spec 080 Phase 2E IA.
  Check Scene for only Placements/LOD, Experimental > Terrain Lab for tiles plus chunk clipboard,
  Inspect's dropdown for Archeology, MCNK/ADT, scene investigation, world context, animations, and
  actions, Utilities > Minimap for the restored route, and Navigator > World Maps for the Phase Map
  selector.
- **Deferred WMO/fog observation:** The supplied screenshot shows distant WMO content, including old
  Ironforge, still visible beyond the effective fog end while terrain has already been culled. Treat
  this as a concrete symptom, not a proven owner; reopen with WMO visibility/submission counters and
  a trace of camera-to-bounds distance against the fog plane before changing admission logic.
- **Proof owner:** Focused PM4/audio contract tests and cross-platform viewer build pass; the user owns
  real-client region-camera, streaming, archive-provenance, and audible proof. The current camera
  slice updates active tiles on mouse-look without reopening the residency lease.
- **Time-of-day checkpoint:** The interactive lighting path has a pure 2,880-unit/24-minute Alpha
  clock enabled by default, with manual slider freeze/resume; Light DBC and LIT consume the same frame
  time, while synthetic minimap manifests record a frozen time-of-day mode.
- **Completed slice:** Checkpoint commits `3bfbbba4` (accumulated audio, AreaNumber, Ghidra, and
  Zone/SubZone overlay work), `de41b183` (Spec Kit design pack), and `c70e1945` (portal phase)
  contain the work completed on this lane. Spec 151 Phase 1 now has a pure, fail-open WMO portal
  decision using transformed portal polygons/clip volumes, source-side admission, bounded
  depth/visit limits, renderer integration, and portal counters in `WmoRenderStats`; the old
  center-distance/queue traversal scaffolding is removed. Focused portal/graph tests pass 16/16 and
  the full solution Debug build passes with 0 errors. The graph evaluator is explicitly diagnostic;
  the shared runtime decision owns final renderer admission. Spec 149 now has an opt-in resident
  Zone/SubZone overlay slice: Ghidra-backed
  MCNK AreaNumber evidence, revisioned resident chunk enumeration, AreaTable-grouped footprint regions,
  distinct Zone/Subzone styling, projected labels, and unresolved-count diagnostics. Spec 148 now has a
  provenance-first world-simulator spec/plan/tasks pack;
  MCSE emitters preserve raw/transformed positions and the proven Alpha 0.5.3 0x34-byte scheduler
  fields; shared Alpha AreaNumber resolution splits high/low `ushort` zone/subzone words and follows
  `ParentAreaNum` without half-word aliases; the area contract now branches explicitly so 3.3.5+
  direct AreaTable IDs cannot be captured by Alpha AreaNumber aliases; status-bar and terrain audio now
  consume the same resolved Zone/SubZone result; the runtime exposes non-playing diagnostic rows; the audio panel
  shows IDs, coordinates, path/source, decode/backend state, terminal reason, and coordinate provenance.
  The audio runtime also exposes a residency-change-only normalized emitter snapshot and the audio
  panel can opt into source-colored 3D speaker pins without starting playback.
  Automatic ZoneMusic playback is now hard-muted behind a tested policy; its area assignment remains
  diagnostic-only so the working MCNK/MCSE water path is not affected.
  The current Spec 080 Phase 2A/2C sidebar slice replaces the visible Model/World/Tools top row with
  Quick/Inspect/Scene/Utilities/Experimental, gives selection/model/ADT/MCNK/PM4 facts one inline
  inspector route, restores MDX/M2 animation controls in Inspect, and combines terrain targeting
  with MCNK/chunk clipboard actions in Experimental Terrain Lab. Audio is owned only by Utilities.
  The viewer build and source checks pass; the full test command timed out and the
  focused core suite reports nine unrelated baseline failures.
  Main Panels utility entries now select their exact Utilities page; compact-window
  manual proof remains open. Source/file/map loading is explicitly left-sidebar-only; the right
  Scene selector now contains only Placements and LOD, while Utilities keeps an isolated page index
  so Inspect/Scene page selection cannot hide or misroute the Minimap page.
- **Main unproven gap:** The sidebar slice still needs user-owned visual proof at normal and compact
  window sizes, including selected-context transitions and legacy caller reachability. The time-of-day
  slice still needs live early-client visual proof and a
  comparison of authored minimap tint behavior; the theory that shipped minimaps captured a moving
  clock remains unproven. Spec 104's restored MDX material shader inputs still need real model/shader
  compilation and visual proof. Full BLS bytecode parity remains out of scope. Spec 151's game-mode head anchor/physics, simple-surface policy, and
  diagnostic budget remain unimplemented. Portal admission is source-tested but still needs the
  user-owned real-client visual/submission/FPS comparison. Spec 149's PM4 region
  bounds/focus, correlation UI retirement, focused area aggregation tests and default-off per-trigger
  audio controls remain open. MCSE tile/chunk
  normalization and MCNK liquid-center placement now have focused source/test proof, pending live
  runtime/audible proof. Speaker-marker placement is source-tested through the normalized snapshot
  path, pending live visual proof. The area overlay is resident chunk coverage, not a
  proven complete polygon. Automatic ZoneMusic playback is intentionally muted until its handoff is
  proven; area resolution remains diagnostic-only.
  ZoneMusic table indirection, exact `sounds.mpq` provenance, MIDI/DLS
  playback, and native MCSE callback installation remain separate proof gates. Spec 150 still lacks
  native renderer anchors, repeatable 0.5.3 baseline capture, and CPU/GPU attribution.
- **Explicitly out of scope for the next slice:** Simple-surface UI, logging-policy retirement,
  whole renderer rewrite, `.bls` bytecode loading/porting, fake audio conversion, and claims of
  visual/FPS/audible gains. Game-mode input/UI follows the pure Phase 2 runtime-core checkpoint.

## Active spec lanes

| Spec | State | Next handoff |
|---|---|---|
| **153 Renderer hitch and MDX batching** | **Phases 1/3/5 implemented, source-proven, UNMEASURED; Phases 0/2/4 open** | **Run the Stranglethorn capture protocol in `research.md`: name the ~212 ms stall, and take the MDX-batching before/after. No fix may be credited before it.** |
| 152 Renderer frame-time stability / per-era lighting | Detector landed and used; its Phase 1 gate refuted the allocation hypothesis so Phases 3–5 are suspended | Owns the measurement infrastructure (done) and Phase 6 per-era terrain lighting (independent, not started, fixes 1.0.0+ darkness). |
| 151 Portal-aware rendering/game mode/simple surface | Phase 1 portal checkpoint implemented; Phase 2 open | Add pure game-mode state/physics and character-head anchor; preserve editor camera state and stop at the focused physics checkpoint. |
| 104 Legacy M2/MDX rendering | 1.0.0 route complete; MDX material/effect shader checkpoint implemented with visual proof open | Validate shader compilation and translucent/reflective models against the configured client/build; keep full BLS parity separate. |
| 149 PM4 region navigation/audio trigger controls | Draft pack; resident area overlay, MCNK liquid producer, coordinate normalization, and opt-in speaker-marker slices implemented | Add area aggregation/audio-control tests, then complete per-trigger toggles and ZoneMusic indirection; retire correlation UI only after the region checkpoint; keep world triggers default-off. |
| 150 Alpha 0.5.3 renderer performance | Draft evidence/planning pack complete; no source optimization started | Recover native world/terrain/object/resource/LOD anchors and run two repeated production `profile-render` baselines before choosing one owner. |
| 148 Artifact world simulator runtime | Phase 1 diagnostics in progress; client contract correction landed | Add ZoneMusic indirection, then finish read/decode/source-stage coverage and user real-client inspection. |
| 147 Minimap/fog/doodad instancing | Phase 2 implemented; Phase 3/4 open | User-run minimap proof, then implement fog coverage and structured batching diagnostics. |
| 146 Audio/camera playback | AreaNumber-aware area selection and master mute control implemented; client audio contracts recovered | Add ZoneMusic row resolution; MIDI/DLS and native MCSE callback proof remain gated. |
| 144 Camera capture paths | Implemented with user gates | Validate swept path residency during playback and confirm lease release before extensions. |
| 145 WoW UI overhaul | First slice implemented | Continue only the remaining persistent-window/placeholder audit tasks. |
| 080 WoW UI consolidation | Phase 2A tabbed sidebar IA and unified inspector source slice implemented; manual proof open | Run the five-tab visual/compact-window check, then resume the legacy route inventory before deleting old methods. |
| 143 World context and lighting | LIT source/fallback, pre-alpha v2 parser, and default-on 0.5.3 time cycle implemented with user gate | Validate live clock/manual freeze and authored-minimap tint boundaries, then continue WMO area and lighting evidence. |
| 142 World scene graph | In progress | User-run dense-WMO capture to compare internal-doodad batching against the prior placement-local path. |
| 139–141 Terrain/minimap reconstruction | Active/parked ML lanes | Reopen only for the named spec and user-run training/validation. |
| 138 Cross-era renderer research | Evidence/planning | Do not generalize one client build to every era. |
| 128–131 PM4 | Established research lane | Use the PM4 spec pack and `workstream-pm4-decode.md`. |

## Stable boundaries

- New code, tests, tools, and viewer docs go in `wow-viewer/`; the legacy tree is read-only
  reference unless a bounded compatibility fix is explicitly requested.
- Keep format readers library-first and tools thin. Do not duplicate or rewrite working client-file
  readers. Keep the Alpha/standard terrain split. `AlphaWdtWriter.cs` is frozen unless explicitly
  reopened with focused proof.
- Client roots are runtime configuration. `H:\CLIENTS` is approved; never hardcode a local client
  path. Record root, build identity, and fingerprint for client-backed proof.
- Training, GPU work, broad harvests, long captures, and real-client/runtime testing are user-run.
- The default source proof is:
  `dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`
  and
  `dotnet test I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`.

## Continuity rules

- Update this dashboard and `progress.md` only when the implementation handoff changes.
- Put durable technical findings in the owning workstream or architecture note, not here.
- Preserve negative results and open gates, but remove superseded narrative from the default path.
- End every handoff with: current target, proof owner, completed slice, unproven gap, next bounded
  action, and explicit out-of-scope items.
