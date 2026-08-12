# Progress — wow-viewer

Last updated: 2026-08-12

This is a short newest-first implementation ledger. It is not a changelog or archive. Older detail
belongs to the owning spec, linked workstream, or `memory-bank/archive/`.

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
