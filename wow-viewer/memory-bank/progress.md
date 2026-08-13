# Progress — wow-viewer

Last updated: 2026-08-13

This is a short newest-first implementation ledger. It is not a changelog or archive. Older detail
belongs to the owning spec, linked workstream, or `memory-bank/archive/`.

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
