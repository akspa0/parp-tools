# Progress — wow-viewer

Last updated: 2026-07-16

## Spec 110 — viewer stabilization

### Completed code proof

- Created the Spec Kit spec, plan, research, contract, quickstart, and dependency-ordered tasks.
- Restored active fog ownership: user Fog Start/Fog End now apply after LIT/DBC recommendation
  selection instead of being overwritten each frame.
- Added Core fallback normalization for invalid, reversed, or collapsed ranges; all render/culling
  consumers receive the normalized active range.
- Moved active range control/status/reset to Lighting and made Settings load-default-only. Legacy
  sliders now activate the same override.
- Added a diagnostic-only LIT map inspector: opt-in markers appear through the shared normal and
  full-screen minimap renderer; a virtualized Lighting list shares selection and double-clicks
  frame a safe camera view. Default/non-finite entries are explicitly non-mappable. This path does
  not change LIT/DBC selection, fog, terrain loading, or renderer routing.
- Replaced the obsolete Harvest `synthetic-minimap` stub with direct terrain synthesis. It composes
  existing MCLY/MCAL textures with MCNR terrain lighting while preserving MCSH as a separate signal,
  writes per-tile PNGs and/or a
  sparse stitched whole-map PNG, and records the exact client, selected time, and LIT-or-authored
  lighting fallback in `synthesis-manifest.json`. It does not require an authored minimap asset.
- Corrected the initial compositor after a real export reported static-like terrain. Ordered MCAL
  overlays now compose over the base in file order rather than normalizing their weights. A first
  renderer-UV/mip correction emitted a real 0.5.3 Kalimdor tile but visibly retained diffuse-repeat
  phase as moire/interpolation, so synthesis now caches each decoded BLP's phase-independent
  material average instead. Spatial variation remains from MCLY, MCAL, and MCNR—not a
  low-resolution diffuse-texture capture.
- Corrected Alpha MCLY/MCAL alignment: the raw Alpha reader retains `[chunkX,chunkY,layer]` for
  legacy consumers, while `TerrainTileTensorPack` is row-major. Alpha tensor construction now
  transposes texture IDs and presence masks, and the compositor honors layer presence so absent
  slots cannot ghost-blend texture ID zero. Focused Alpha/tensor/minimap suite: 11 passed; Debug
  Harvest build: 0 errors.
- Corrected Alpha MCNR lighting interpolation: the new time-of-day compositor was sampling the
  sparse dense-grid gaps as `UnitZ`, producing a false checkerboard that looked like texture or
  shadow interpolation. It now evaluates vertex Lambert values over the native staggered terrain
  lattice and interpolates them across the terrain triangles. `McshShadowMask256` remains an
  unchanged independent target for model/shadow work. Focused lighting/minimap/Alpha suite: 19
  passed; Debug Harvest build: 0 errors.
- Corrected minimap lighting semantics and dataset provenance: normal synthesized RGB now omits
  MCSH, with `--bake-mcsh` reserved for an explicitly labeled exceptional-history preview.
  `MinimapLightingProvenance` v1 compares authored minimap RGB with a neutral terrain-material
  baseline, records tint fit and MCSH residual correlation, and adds only a clearly labeled
  global-clear LIT chroma bucket—not a claimed historical capture time. Full/V22 raw streams carry
  this sidecar. Texture payloads are now all-or-nothing name-aligned with MTEX, and V22 model
  payload arrays/metadata use deterministic keys so downstream consumers can reproduce the
  baseline without shifted identities.
- Corrected whole-map resilience for readable base-only terrain: if an Alpha tile has MCLY material
  but no MCAL payload, the compositor now exports its layer-zero material with normal/LIT evaluation
  instead of aborting the tile or inventing overlay alpha.
- Corrected partial Alpha MCNR-mask resilience: a mask shape that differs from the normal grid now
  yields a neutral normal outside the available mask rather than an out-of-bounds exception.
- Added deterministic missing-texture recovery for derived minimaps/datasets and the live terrain
  renderer: try the original, then a successfully decoded same-stem `_s.blp` companion, then at
  most sixteen decoded ordinary `.blp` candidates scanned from the archive/listfile catalog. Exact
  or strongly similar basenames rank before directory-theme tokens, permitting moved historical
  assets to repair stale ADT links. Metadata preserves original MTEX identity plus
  `specular_companion_rgb_proxy` or `related_diffuse_rgb_proxy`; the viewer logs the selected proxy.
  No path claims native specular/alpha/material behavior.
- Corrected bounded export selection after a user-run 0.5.3 Kalimdor command reported
  `written=0, skipped=1`: the first ordered WDT coordinate `(12,19)` could not decode and consumed
  `--limit 1`. The command now limits emitted PNGs, not attempted coordinates, while preserving
  skipped/failed diagnostics in the synthesis manifest. Debug Harvest build passes.
- Added `Tools > Export > Synthesized Terrain Minimap...`. Its user-driven background invocation
  resolves the in-repository Harvest executable, DLL, or project and reports an actionable error
  if none is available; it does not depend on a separately installed converter binary.
- Added minute-precise synthesized-minimap time selection. CLI `--time-hours` accepts `HHmm`
  (`1215`), `HH:mm`, and compatible decimal hours; it normalizes to an exact minute and the
  v3 manifest writes canonical `HH:mm` plus decimal hours. The in-app export surface now has
  exact Hour/Minute inputs and launches Harvest with compact clock input.
- Corrected Alpha WDT occupancy enumeration after a user-run full Kalimdor manifest wrote only
  `361/951` tiles: 16-byte Alpha MAIN cells are row-major like the Alpha reader, not transposed.
  Failed tile records now include the decode/texture/composition/write stage for actionable
  follow-up.
- Replaced the fixed-bias authored sun vector with a shared terrain/raster-axis solar path: it is
  vertical at noon and projects from top-left after noon. LIT global-clear colours remain client
  data, while this direction remains explicitly authored because LIT does not carry a sun vector.
- Added paired liquid minimap output. Every successful synthetic tile now writes the existing
  liquid-free terrain PNG and an aligned `_liquid.png`; `--whole-map` writes matching terrain and
  liquid stitched maps. The v4 manifest records liquid paths, pixel count, and the
  `viewer_flat_liquid_overlay_v1` profile. It is an analytic flat overlay from decoded unified
  coverage/basic types, not a claim of native water texture/animation/reflection parity. Alpha MCLQ
  257² surface data with a 16×16 type grid is normalized before unified-liquid composition, removing
  the mismatched-array failure mode that can affect liquid tiles.
- Corrected liquid coverage geometry after a 0.5.3 map showed water above dry terrain-cell
  boundaries: Alpha MCLQ now respects its 8×8 cell flags instead of treating an entire chunk as
  wet, and the minimap compositor requires all four source-cell coverage corners before emitting a
  liquid pixel. Added focused regression proof plus `synthetic-minimap --tile-x/--tile-y` and a
  first relevant source frame in failure diagnostics. Latest focused liquid/Alpha/minimap suite:
  22 passed; Debug Harvest and Viewer builds: 0 errors (existing warnings only).
- Resolved the residual Alpha `IndexOutOfRangeException` from a user-run Kalimdor export. The
  emitted frame was `AlphaTensorPackBuilder.PaintCircle`: cross-tile WMO roof footprints target
  256² minimap buffers but were bounds-checked against the 257² terrain grid. All Alpha footprint
  painters now check their actual destination dimensions; a WMO at tile edge is regression-covered.
  Separately, 220 `no referenced BLP texture could be decoded` skips now enter a provenance-labeled
  catalog RGB last-resort tier (same folder, terrain family, then prior verified decode) rather than
  discard otherwise readable terrain; missing MCLY/MTEX uses that material as an explicit base.
  Latest focused Alpha/minimap/texture-policy suite: 27 passed; Debug Harvest and Viewer builds:
  0 errors (existing warnings only).
- Replaced drag-only fog fields with visible slider controls. Moved UniqueId ranges/layers out of
  World into Tools > Archeology, gave Archeology a dedicated nested-tab index, and kept pause/stop
  visible while playback is active. Playback now stops safely if its world or scoped range vanishes;
  the legacy menu opens the dedicated Archaeology window with the same transport.
- Validation: pre-correction fog/minimap suite → 10 passed; material-average correction
  `dotnet test wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~TerrainMinimapCompositorTests"` → 6 passed;
  Debug `WowViewer.Tool.Harvest.csproj` build → 0 errors. The running viewer locked its normal
  Debug output, so the active Viewer was rebuilt after this correction with
  `OutputPath=C:\tmp\wowviewer-minimap-validation` → 0 errors. Existing package/nullability
  warnings remain outside this correction. Latest focused minimap/provenance/NPZ/raw-stream suite
  → 49 passed; latest texture-policy/minimap/serializer suite → 39 passed; Debug
  `WowViewer.Tool.Harvest.csproj` and `WoWViewer.csproj` builds → 0 errors. Latest liquid/Alpha/
  minimap focused suite → 21 passed; Debug Harvest and Viewer builds → 0 errors.

### Required user proof before next code phase

- Re-export one bounded terrain tile first with the corrected compositor; confirm stable material
  regions, MCAL blends, normal unshadowed RGB, smoothly interpolated terrain lighting, and an
  exactly aligned `_liquid` companion with plausible liquid coverage/type coloring and no narrow
  strips over dry terrain-cell boundaries—without repeated texture moire, dense-MCNR checkerboard,
  unexpected MCSH bake, or blurred interpolation—before a whole-map job. First run the remaining
  failing `Kalimdor` coordinates `(36,44)` and `(37,44)` through `--tile-x/--tile-y` first; they
  should now pass the clipped cross-tile WMO roof path. Then rerun a small sequence that includes
  one former texture-only skip and verify its manifest records `catalog_rgb_last_resort_proxy`
  rather than `no referenced BLP texture could be decoded`.
- Inspect one bounded V22/full stream's `minimap_lighting` metadata. It must either carry explicit
  tint/shadow evidence plus a non-capture-proof time bucket or an explicit not-evaluated reason;
  it must never silently infer an exact historical time.
- Then run Spec 110 quickstart against a configured LIT map and no-LIT map at dawn/noon/dusk/night.
- Confirm active source/range changes and no terrain disappearance; on the LIT map, enable markers,
  compare both minimap views, select an entry, and double-click it to frame the 3D camera. Export a
  small per-tile set first, then a whole map; inspect `synthesis-manifest.json` and capture client
  root/build/fingerprint. Confirm visible fog grabs and that Archeology playback is reachable,
  pausable, and stoppable through every nested tab in both UI modes.

### Next phase (blocked on above proof)

- Make M2 runtime native-only and remove all M2→MDX / adapter-backed MdxRenderer fallback branches.
- Then clean Tools entries and publish conversion capability levels. WMO v14↔v17 is fixture-covered
  in both directions; M2→MDX remains synthetic-test-only until real-client export proof.

## Separate continuity

- Spec 109 v50 clean-room dataset work is unchanged and separate from Spec 110.
