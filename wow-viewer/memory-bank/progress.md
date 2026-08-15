# Progress — wow-viewer

Last updated: 2026-08-14

This is a short newest-first implementation ledger. It is not a changelog or archive. Older detail
belongs to the owning spec, linked workstream, or `memory-bank/archive/`.

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
