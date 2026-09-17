# Changelog — parp-tools WoW Viewer

Release notes for each tagged version live in [`docs/releases/`](docs/releases/) and are what the
GitHub Release publishes. This file is the index and the short version.

## v0.5.4-alpha — 2026-09-16

Alpha pre-release: modern CASC install support for **WoW: Forever** (`wow_classic_beta` 1.60.1.69876,
12.0-based client), FileDataID-era maps, M2 and WMO, DB2 tables for that build, and a first reader and
renderer for the new DAT v26 terrain project format.

Full notes: [`docs/releases/v0.5.4-alpha.md`](docs/releases/v0.5.4-alpha.md)

### Added
- **CASC installs**: File → Open CASC Install (local / local + CDN fill) with a game version picker, built on TACTSharp. Parallel reads, background prefetch, and a download cache.
- **FileDataID assets**: WDT `MAID`, tex0 `MDID`, FileDataID placement flags, `MD21` M2 (`SFID`/`TXID`), WMO `GFID`/`MODI`/`MOMT` texture IDs, DB2 tables via WoWDBDefs.
- **DAT v26 terrain**: File → Open DAT v26 Terrain Folder, `ALOC` tile placement, height scale option (default ÷36).
- **CLI**: `inspect casc …` and `inspect adt-ahdr check`.

### Fixed
- Newer WMO root/group chunk layouts no longer crash the world (unknown chunks skipped; unparseable WMOs skipped with a log line).
- Listfile resolution prefers the newest listfile instead of a stale bundled copy.

## v0.5.3.1 — 2026-09-15

Hotfix release: Mjollna community credit, authentic asset mappings in New Map Creator, GLB exports for active terrain & M2 models, cartography phase tile alignment, console window suppression, and startup resilience.

Full notes: [`docs/releases/v0.5.3.1.md`](docs/releases/v0.5.3.1.md)

### Added
- **About Box Credits**: Added Mjollna to the list of community research, exploration, and development acknowledgments.

### Fixed
- **Authentic New Map Creator Assets**: Mapped all biome themes strictly to verified community listfile textures and models, eliminating fictitious fallback paths.
- **Editor Data I/O GLB Exports**: Added GLB scene and collision mesh export support for loaded terrain tiles; enabled M2 doodad model conversion to MDX mesh for GLB inclusion.
- **Phase Map Alignment**: Fixed donor tile world coordinate leaking and co-location tearing in `PhaseChunkMerger` and terrain adapters; corrected donor placement translation offsets.
- **Windows Console Suppression**: Switched executable output to `WinExe` to prevent spawning an empty command prompt behind the GUI window.
- **Startup Crash Logging & Resilience**: Global exception handlers logging to `crash.log` and bounded non-blocking listfile fetching.

## v0.5.3 — 2026-09-15

Feature release: Legacy MDX/M2 model rendering & bone animation, Cartography composition with rigid cell shifting, in-viewer New Map Creator, Editor IA overhaul, marketing capture automation, and 5.0.1 era groundwork.

Full notes: [`docs/releases/v0.5.3.md`](docs/releases/v0.5.3.md)

### Added
- **1.x M2 Bone Animation Playback**: Full-body skeletal movement evaluation for 1.x M2 models with zero start timestamps and proper duration/speed decoding.
- **In-Viewer New Map Creator (`NewMapCreatorService`)**: Interactive map creation dialog in Editor > Data I/O for generating LK ADT/WDT terrain maps with selectable themes, grid dimensions, and elevation.
- **Cartography Composition & Rigid Cell Shift**: Intact donor map cell alignment, project file persistence (JSON transforms/visibility/locks), and WDL edge-snapping.
- **Editor IA Overhaul (Spec 231)**: 4-page workspace (Terrain, Placement, Cartography, Data I/O), persistent top toolbar with world doodad animation toggle and WMO doodad-set selector.
- **Marketing Capture Automation (Spec 233)**: Automated sequence camera recording and video encoding support.
- **Cross-Platform Release Builds**: CI/CD matrix build for Windows x64, Linux x64, macOS arm64, and macOS x64.

### Fixed
- **Tree Foliage Alpha Cutout**: Adjusted cutout threshold to 0.15f to preserve antialiased pine needles and leaf geometry across distance.
- **MDX Texture Wrapping**: Restored standard bitwise decoding for `WrapWidth` (0x1) and `WrapHeight` (0x2).
- **Two-Sided Surface Illumination**: Surface normal flipped on back-faces in `M2Renderer` fragment shader to properly illuminate leaves and foliage.
- **Character Naked Geosets**: Pruned default 3D armor attachment geosets and added Goblin race ID mapping.
- **Wireframe Rendering**: Untextured flat-color line passes across terrain and models.

## v0.5.2.2 — 2026-08-27

Maintenance, Calibration, and Spec Kit release.

### Added
- **Rosetta Calibration Corpus Generator (`rosetta-generate`)**:
  - High-resolution $1024 \times 1024$ MCAL/MCLY terrain text rasterization (`RosettaAlphaPainter`) with antialiased quincunx sampling and 4-bit nibble slicing into 2-layer chunk blocks.
  - Museum pedestal heightfield generation (`MCVT`): 4m elevated plinth with 12.5m linear bevel ramp under each model cell to eliminate base clipping.
  - Full-continent Alpha WDT support: removed artificial 512-tile limit, unlocking full $64 \times 64$ (4096) tile single-map generation.
  - Added CLI options `--ink-texture`, `--pedestal-height`, `--pedestal-bevel`, and automated era auto-detection.
- **Documentation Overhaul**:
  - Complete rewrite of root `README.md`, `wow-viewer/README.md`, `USERGUIDE.md`, and `CLI-TOOLS.md`.

### Fixed
- **`SampleCoverage` in `RosettaTextPainter`**: Normalized overlap area by arbitrary sample bounding box area `(su1 - su0) * (sv1 - sv0)` for sub-meter texel accuracy.
- **`AdtPlacementEditor` Chronology**: Maintained high-water marks on ID allocations across delete-and-substitute operations.

## v0.5.2.1 — 2026-08-15

Out-of-band patch. **v0.5.2 shipped with known, unresolved rendering jank**; this fixes the causes
that were found, and names the one that was not.

### Fixed

- **Periodic ~283 ms render-thread stall, four times a second.** The audio *emitter diagnostics*
  panel was rebuilding a record for every resident sound emitter (5,565 in a dense zone) on a 250 ms
  timer, **whether or not anything was displaying it**, plus a second synchronous rebuild on every
  streaming tile eviction — i.e. while moving. Now gated on being observed, and the eviction path
  only invalidates. `PrepareObjectPhase` max **283.4 → 2.5 ms**, and it no longer appears in the
  hitch list.
- **Opaque MDX drew one call per instance.** The route planner's predicate body was a hardcoded
  `return true`, so 100% of instances took the per-instance fallback while the batching path sat
  unused. It now consumes the renderer's own `RequiresUnbatchedWorldRender` declaration — the same
  contract the WMO doodad path already used. **0 batched / 312 unbatched → 526 batched / 3
  unbatched**; `MdxOpaqueSubmission` p99 **30.75 → 14.12 ms**, median **2.03 → 0.01 ms**.
- **Batched and unbatched MDX output could diverge.** `RenderInstance` ignored the per-renderer
  wireframe flag that `RenderWithTransform` honours, so a wireframe-flagged model would have drawn
  filled once batched. Now declared as requiring the unbatched path.
- **Deferred asset loads checked their budget only *between* loads**, so a started load ran to
  completion — 58 ms against a 3.5 ms budget. Loads are now admitted against a learned per-asset-kind
  cost estimate before starting.
- **Audio range ignored which tile the camera was on**, testing distance against every streamed tile.
  Now scoped to the camera tile and its ring, using the terrain manager's own camera tile rather than
  re-deriving it.

### Added

- **`PrepareObjectPhase` has a stage timer.** It was the only one of eleven frame passes without one,
  which is why a 283 ms stall stayed invisible for an entire investigation. A test now fails the
  build if a pass is added without a timer, a stage is recorded by nothing, or a stage is
  double-counted. Unaccounted frame time: **259–314 ms pass gap → median 0.02 / p99 0.11 ms**.
- **`McseFrameEvidence`** — measures what coordinate frame decoded MCSE sound-emitter positions are
  actually in, reported in Utilities > Audio. Added instead of guessing at a fix (see Known issues).
- **Live on/off switch for opaque MDX batching** in Utilities > Perf, so a before/after comparison is
  one flight rather than two builds.
- Submission counters, scanned/in-range emitter counts, and deferred-load budget counters, so each of
  the above is verifiable rather than asserted.

### Known issues

- **Dense WMO interiors are slow; Stormwind is the worst case.** The city submits all districts at
  once — 7,512 visible groups, 80,484 draw calls — instead of the district the camera occupies. This
  is a group *admission* problem, not batching (80,200 of those calls are correctly batched). **Not
  fixed here**; it is the next work item.
- **Deferred loading can still spike ~443 ms on one load** in a dense zone. The new policy bounds the
  additive overshoot, but a single synchronous decode costs what it costs. Moving decode off the
  render thread is the real fix and is not done.
- **`MCSE` sound emitters read as permanently out of range**; only water-triggered emitters behave.
  The position transform assumes a chunk-local frame on the strength of an unevidenced code comment.
  This release ships the measurement, not a guessed fix.

### Notes

The leading theory going in — per-frame allocation churn in the scene graph — was **refuted by
measurement** (median world-render CPU 0.33–8.58 ms; traversal max 0.22 ms) and the planned scene
flattening work was suspended rather than continued on momentum. Restoring MDX batching did **not**
remove the hitching and is not credited with having done so.

## v0.5.2 — 2026-08-15

Portal-aware WMO visibility, per-ADT scene graph, bounded camera-centered streaming, audio runtime,
Alpha 0.5.3 time-of-day clock, LIT lighting decode, PM4 coordinate solve, and the five-destination
workbench UI. 238 commits since `v0.5.1-build1`.

Full notes: [`docs/releases/v0.5.2.md`](docs/releases/v0.5.2.md)

## v0.5.0

Full notes: [`docs/releases/v0.5.0.md`](docs/releases/v0.5.0.md)
