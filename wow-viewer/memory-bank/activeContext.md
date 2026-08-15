# Active Context — wow-viewer

Last updated: 2026-08-15

This file is the interchange for the next agent. It records only the current routing state. Read
the owning spec for requirements and proof; read a workstream only when the spec links it.

## Current handoff

- **Next implementation target:** User-run visual/compact-window proof for the Spec 080 Phase 2E
  IA. Check Scene for only Placements/LOD, Experimental > Terrain Lab for tiles plus chunk clipboard,
  Inspect's dropdown for Archeology, MCNK/ADT, scene investigation, world context, animations, and
  actions, Utilities > Minimap for the restored route, and Navigator > World Maps for the Phase Map
  selector.
- **Performance planning target:** Spec 150 remains the broad Alpha 0.5.3 renderer evidence lane. Spec
  151 owns the first concrete portal-specific optimization; use its counters and fallback evidence to
  inform later repeated `profile-render` baselines. Do not infer a win from an interactive screenshot.
- **Deferred renderer observation:** The user reports camera movement hitching at nearly every move,
  independent of fog settings, with suspected ADT/object rendering or residency regression. Do not
  touch renderer, fog, ADT admission, object submission, or streaming during the sidebar pass; reopen
  Spec 150 with repeated stage-attributed profiles.
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
