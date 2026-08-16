# Active Context — wow-viewer

Last updated: 2026-08-15

This file is the interchange for the next agent. It records only the current routing state. Read
the owning spec for requirements and proof; read a workstream only when the spec links it.

## Current handoff

**START HERE: read the new WMO admission counters on a real Stormwind flight, then pick the rule to
fix. The instrumentation landed; the measurement has not been taken.**

- **Next bounded action, user-owned:** fly Stormwind, open Utilities > Perf > **"WMO admission (this
  frame)"**, and record the numbers into
  [Spec 151 research.md](../specs/151-portal-game-mode-surface/research.md). The panel names the
  dominant rule. **Do not change an admission rule before that reading exists.**
- **Confirmed by the Stormwind capture (2048 frames), taken after the Spec 153 fixes:**
  `PrepareObjectPhase` max **283.4 → 2.5 ms** and gone from the hitch list; `SceneMaintenance` max
  **454.5 → 3.9 ms**; unaccounted median 0.02 / p99 0.11 ms; median frame 17.40 → 6.98 ms.
- **Owner: `WmoSubmission`** — p99 154.10 / max 161.3 ms with a median of 0.71 ms, and **all 592
  recent hitches** read `<- WmoSubmission` at 153–157 ms. Stormwind submits **all districts at once**:
  7512 visible groups, 80484 draw calls, 15852 doodad submissions. **This is an admission problem,
  not batching** — 80200 of 80484 calls are correctly batched.
- **Instrumentation shipped (source proof + 11 core tests, nothing measured).** `WmoAdmissionTally` /
  `WmoAdmissionStats` in `Core.Runtime/World/Visibility` count placements and groups by admitting
  rule; `CollectVisibleWmos` has a `ref` overload proven identical to the old one; `WmoRenderer`
  records the per-group rule; the Perf panel displays both layers. Four **source** findings drove the
  counter shape, all with magnitudes still unmeasured: portal culling **cannot reject** a group
  (the decision is unioned with raw frustum visibility); WMO placements are **never rejected by the
  frustum** (`IgnoreVisionConeCulling: true` also disables the only frustum branch), which is the
  shape of the old-Ironforge-past-fog symptom; group admission is evaluated **twice per placement per
  frame** (opaque + transparent); and the recorded 7512 is a **submission** count spanning both
  passes, not distinct groups. Detail in Spec 151 `research.md`.
- Still suspect group-to-group visibility data the client uses and this renderer does not read.
- **Spec 153 Phase 4 may be moot** — it was written against `SceneMaintenance` max 454.8 ms, which no
  longer reproduces (3.9 ms). Re-measure on a route that forces `_instancesDirty` before implementing.
- **Spec 153 Phase 5 step 2 still owed:** `DeferredAssetLoads` max 442.9 ms in Stormwind against a
  3.5 ms budget. The admission policy bounded the additive overshoot; the single-load residual needs
  decode off the render thread.
- **Released:** v0.5.2.1 (commit `975d0c79`, tag `v0.5.2.1`, branch `v0.5.3-dev`). Version bumped in
  both csproj files and `ViewerApp.ViewerProductName`; `wow-viewer/CHANGELOG.md` added;
  `docs/releases/v0.5.2.1.md` is what the release workflow publishes; both READMEs updated and the
  Stormwind WMO issue is documented as a known issue rather than left for users to discover.

---

**Spec 153 detail (shipped in v0.5.2.1, confirmed by capture).** Full numbers and both capture
tables are in [Spec 153 research.md](../specs/153-renderer-hitch-and-batching/research.md).

- **Defect A was `AudioRuntime.Update`, and it was never audio.** `RefreshEmitterDiagnosticsIfDue`
  rebuilt an `AudioTriggerDiagnostic` per resident emitter (5565) on a **wall-clock 250 ms** timer on
  the render thread — which is why the "every 47–50 frames" interval drifted with framerate — and it
  ran **whether or not anything displayed the result**. A second, movement-triggered copy: `RemoveTile`
  rebuilt the list synchronously on streaming eviction. Fixed by gating the rebuild on
  `NoteEmitterDiagnosticsObserved()` (only the audio panel calls it) and making eviction invalidate
  only. **Measured 283.4 → 2.5 ms max.** General lesson: *a diagnostics surface nothing is reading
  still pays full cost unless something gates it.*
- **Defect B was a hardcoded `return true`.** `PlanVisibleMdxPasses` gave the route planner a
  `requiresUnbatchedRender` predicate whose whole body was `return true`, so 100% of opaque MDX took
  the per-instance fallback while the batching machinery sat inert. Now consumes
  `IModelRenderer.RequiresUnbatchedWorldRender` — the contract the WMO doodad path already used.
  **0/312 → 526 batched / 3 unbatched**; `MdxOpaqueSubmission` p99 30.75 → 14.12 ms. GPU instancing
  stays off (`SupportsGpuInstancedOpaque` is still `false`); the win is begin-once/submit-many state.
  `_wireframe` folded into `RequiresUnbatchedWorldRender` — the one real visual divergence between
  the paths. Live-revertible via `WorldScene.MdxOpaqueBatchingEnabled`.
- **Do not credit Defect B's fix with fixing the gallop.** Frame p99 barely moved on that capture
  (259.70 → 246.62) because the periodic stall was never the MDX cost.
- **Instrumentation is now self-defending.** `PrepareObjectPhase` has a stage timer (`StageCount`
  18 → 19) and `WorldFramePassInstrumentation` + a reflection test fail the build if a pass has no
  timer, a stage is recorded by nothing, or a stage is double-counted. Unaccounted time is now
  median 0.02 / p99 0.11 ms, so hitch attribution names a stage instead of a void.
- **Audio is scoped to the camera tile.** `Update` consulted **no tile information at all** — it
  scanned every resident tile. Now takes `TerrainManager.CameraTileX/Y` (passed in, never re-derived)
  within `WorldAudioRuntime.AudibleTileRadius` (1); the diagnostics panel uses the same window.
  **Tile keying was checked and cleared** — `AddTile`/`RemoveTile`/`EmitterKey`/`OnTileLoaded` all
  agree; scanning every tile just made it look like a keying fault.
- **OPEN: MCSE emitters read as permanently out of range; only water works.**
  `AlphaTerrainAdapter.ConvertSoundPosition` does `chunkCorner - local` on the strength of an
  unevidenced comment ("Alpha MCSE stores a chunk-local C3Vector"); the Ghidra work proved the 0x34
  **field layout**, not the frame. If it is not chunk-local, every MCSE emitter lands tens of
  thousands of units off-map — which also explains why MCNK liquid rows work, since they derive from
  the renderer's own `chunk.WorldPosition` and never touch the transform. **`McseFrameEvidence`
  measures it** (raw min/max per axis, chunk/tile/beyond counts, explicit verdict) at the top of
  Utilities > Audio. **Read that line before touching the transform** — it deliberately reports
  "inconclusive" rather than picking a winner on a mixed sample.
- **Refuted, do not revive without new evidence:** the allocation-churn hypothesis. Median
  world-render CPU is 0.33–8.58 ms and traversal maxes at 0.22 ms. Spec 152 Phases 3–5 (flatten the
  scene graph into retained draw lists, view modes) are **suspended** because they rested on that
  premise. Also ruled out with evidence: decoded-asset caching / LRU thrash (`MaxMdxCached = 0`,
  unlimited; 554 models serve 18663 instances — nothing is re-decoded, the cost is submission).
- **The measurement tool exists and works.** Utilities > Perf > Frame history: rolling per-frame
  history, hitch detection with dominant-cause attribution, unaccounted time, region peaks,
  submission batching counts, and an injected-stall self-check. Recording is allocation-free.
  Use it for every before/after. **Benchmarks: Stranglethorn Vale** (dense doodads),
  **Stormwind** (dense WMO groups).
- **No viewer test assembly exists** (`tests/` has Core, Core.Anim, Core.Curation, Core.PM4 only), so
  viewer-side changes — `McseFrameEvidence`, the audio tile window, the batching predicate — carry
  source proof plus capture, not unit tests. Moving those types into core is an open follow-up.
- **Detector lessons that made this possible:** p99 hides rare hitches (they land at p100 — use max
  and over-threshold count); ranking stages by p99 buries a rare-but-huge stage (sort by max);
  always report unaccounted time, or attribution names a 0.2 ms stage for a 350 ms frame.
- **Lower-priority target:** User-run visual/compact-window proof for the Spec 080 Phase 2E IA.
  Check Scene for only Placements/LOD, Experimental > Terrain Lab for tiles plus chunk clipboard,
  Inspect's dropdown for Archeology, MCNK/ADT, scene investigation, world context, animations, and
  actions, Utilities > Minimap for the restored route, and Navigator > World Maps for the Phase Map
  selector.
- **Related WMO/fog observation — fold this into the group-admission work above.** A screenshot showed
  distant WMO content, including old Ironforge, still visible beyond the effective fog end while
  terrain had already been culled. Same shape as the Stormwind finding: WMO geometry is being admitted
  that should not be. Treat as a concrete symptom, not a proven owner; it wants the same
  visibility/submission counters plus a trace of camera-to-bounds distance against the fog plane
  before any admission logic changes.
- **Proof owner:** Focused PM4/audio contract tests and cross-platform viewer build pass; the user owns
  real-client region-camera, streaming, archive-provenance, and audible proof. The current camera
  slice updates active tiles on mouse-look without reopening the residency lease.
  **For renderer performance specifically, the proof is now the in-viewer frame history** — the user
  flies the route and reads Utilities > Perf. Two captures (Stranglethorn, Stormwind) are recorded
  with full numbers in Spec 153 `research.md`; every renderer claim must cite one.
- **Time-of-day checkpoint:** The interactive lighting path has a pure 2,880-unit/24-minute Alpha
  clock enabled by default, with manual slider freeze/resume; Light DBC and LIT consume the same frame
  time, while synthetic minimap manifests record a frozen time-of-day mode.
- **Completed slice (latest):** `975d0c79` — Spec 153 Phases 1/2/3/5, audio tile scoping,
  `McseFrameEvidence`, and the v0.5.2.1 release (version bump, CHANGELOG, release notes, READMEs).
  `bda47bdb` — handoff repointed at WMO group admission. Both pushed to `v0.5.3-dev`; tag `v0.5.2.1`
  published with all four platform builds.
- **Completed slice (earlier):** Checkpoint commits `3bfbbba4` (accumulated audio, AreaNumber, Ghidra, and
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
- **Main unproven gap:** **WMO group admission is measured as the renderer's dominant remaining cost
  but its cause is still not diagnosed** — the counters that name the admitting rule now exist and
  have never been read on a real flight. Reading them is the next bounded action.
  The MCSE coordinate frame is also open: measured, verdict not yet read on real data.
  The sidebar slice still needs user-owned visual proof at normal and compact
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
| **151 Portal-aware rendering / WMO group admission** | **PRIORITY 1 — instrumentation shipped, measurement owed** | **Fly Stormwind, read Utilities > Perf > "WMO admission (this frame)", record it in Spec 151 `research.md`. Only then pick the rule to change.** |
| 153 Renderer hitch and MDX batching | Phases 1/2/3/5 shipped as v0.5.2.1 and confirmed by capture; Phase 4 likely moot | Re-measure `SceneMaintenance` before implementing Phase 4; Phase 5 step 2 (decode off the render thread) still owed |
| 152 Renderer frame-time stability / per-era lighting | Detector landed and used; its Phase 1 gate refuted the allocation hypothesis so Phases 3–5 are suspended | Owns the measurement infrastructure (done) and Phase 6 per-era terrain lighting (independent, not started, fixes 1.0.0+ darkness). |
| 151 Portal-aware rendering/game mode/simple surface | Phase 1 portal checkpoint implemented; Phase 2 open | Add pure game-mode state/physics and character-head anchor; preserve editor camera state and stop at the focused physics checkpoint. |
| **154 M2 reader era parity (1.x–3.0.1)** | **NEW — spec drafted, not planned** | Run US1 first: survey every staged build before touching a reader. Three measured defects; the "4.0.0 works" premise is contradicted by measurement and must be resolved, not assumed. |
| 104 Legacy M2/MDX rendering | 1.0.0 route complete; MDX material/effect shader checkpoint implemented with visual proof open | Validate shader compilation and translucent/reflective models against the configured client/build; keep full BLS parity separate. **Blocked in part by Spec 154** — M2 bone reading is broken outside the Alpha and late-3.x routes. |
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
