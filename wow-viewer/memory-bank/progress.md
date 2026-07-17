# Progress — wow-viewer

Last updated: 2026-07-17

## Spec 111 — minimap lighting calibration

- Implemented all code phases (T001–T018, T020–T021) up to the explicitly gated T019 training run.
  C# side: six additive shading-match fields on `MinimapLightingProvenance`;
  `Core.IO/Maps/MinimapShadingMatch.cs` sweeps 24 hourly `TerrainMinimapCompositor` candidates and
  scores tint-invariant luma Pearson correlation (gradient-direction cosine was tried and discarded:
  with the fixed azimuth it cannot distinguish hours); chained onto the existing Full/V22
  `AnalyzeAuthoredMinimapLighting` streaming pathway with an internal 0.5.3.3368 fingerprint gate —
  no new command, zero cost for other builds. Python side: `harvester/spec111/`
  (`lighting_buckets.py` reconciled report, `rebalance_lighting_variants.py` bare-float
  largest-remainder `lighting_times`, `checkpoint_comparison.py` where regressed and inconclusive
  both keep the deployed checkpoint) plus three thin CLIs; `train_spec111_reconstruction.py`
  validates and refuses to train without `--confirm-run` (smoke-proven). The drifted
  `terrain_lighting.py` direction formula became a documented port of the corrected C#
  `TerrainSolarDirection` with regression coverage.
- Proof: focused C# sweep 42/42; Debug Harvest build 0 errors; `tests/spec111/` 16/16;
  `tests/spec103/test_terrain_lighting.py` 10/10. Full data-harvester suite 548 passed with 3
  pre-existing unrelated failures (v24 export-map fixture, v25 h1_coarse neighbor-context API).
- Remaining user-run proof: bounded real-0.5.3.3368 `harvest-stream --stream-profile v22` bucketing
  pass, the quickstart side-by-side eyeball check on `matched` tiles, whole-build report, then the
  separately authorized T019 retrain/evaluate.

### Planning record

- Created the Spec Kit spec, plan, research, data-model, contract, quickstart, and dependency-ordered
  tasks (`specs/111-minimap-lighting-calibration/`).
- Three user stories: US1 shading-based lighting-bucket inference for the real 0.5.3.3368 dataset
  (MVP), US2 rebalance synthetic-lighting-variant training sampling to match the real distribution,
  US3 retrain-and-evaluate the existing reconstruction model with an explicit go/no-go gate.
- Confirmed with the user before writing the spec: build scope is 0.5.3.3368 only; training scope
  includes the full retrain-and-evaluate loop (not just data prep), with its GPU/cloud execution step
  explicitly gated on separate authorization at run time.
- Research surfaced that `data-harvester/src/harvester/spec103/terrain_lighting.py` independently
  reimplements the solar-direction model and has drifted to `v1` while the corrected C# path is now
  `v3` — the plan retires that duplication rather than syncing constants by hand.

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
  but no MCAL payload, the compositor now exports its layer-zero material with normal/white-top-edge evaluation
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
- Corrected the minimap-light consumer boundary after visual proof showed south-side hillshade that
  inverted basins into mountains. `synthetic-minimap` now excludes LIT color/fog tracks and the
  recovered 0.5.3 native ray, using instead pure-white direct light, achromatic ambient, and a
  negative terrain-X north/top-edge source. Native-ray recovery remains diagnostic research only.
  Focused minimap coverage: 22 passed; Debug Harvest build: 0 errors.
- Corrected that fix's sign: negative terrain-X was mislabeled "north." The traced 1.0.0 native
  `SetDirection` ray (`docs/architecture/wow-1.0.0-world-lighting-shadow-model-2026-07-15.md` §2.1),
  cross-checked against `AdtTensorPackBuilder.AssembleNormals` (MCNR decoded with no axis swap) and
  `TerrainMeshBuilder` (vertex world-X built from row/tileY-indexed quantities that decrease
  southward), confirms this codebase's MCNR/MCVT convention is +X = North, +Y = West, +Z = Up — so
  negative X is south. `TerrainSolarDirection` now locks the horizontal bias to positive X; updated
  the compositor test assertion and the spec/contract/task wording that encoded the inverted claim.
- Corrected a second, more consequential defect in the same function: a user-run side-by-side of a
  synthesized tile against the real 0.5.3 client minimap (same crater/lake feature) showed the client
  keeps a persistent bright-north/dark-south hillshade at every sampled time, while ours looked washed
  out. The horizontal bearing was swept with `cos(sunAngle - pi/2)`, which is exactly zero at solar
  noon/midnight, pointing the sun straight up with no shadow direction at those instants — matching
  the traced native ray's constant azimuth (theta = 225 degrees across all four sampled table
  entries) rather than a sweeping one. `TerrainSolarDirection` now locks the horizontal bearing to a
  fixed north-west share all day; only elevation varies. Added `TerrainSolarDirectionTests` (28 total
  focused lighting/minimap tests passed) and corrected `AuthoredTerrainDayNightProfileTests`'
  now-inverted "vertical at noon" assertion. Debug Harvest build: 0 errors. The pre-existing,
  unrelated `LkToAlphaRoundTripTests`/reader/coordinator failures (9 tests) reproduce identically on
  unmodified `HEAD` and are outside this correction's scope.
- Corrected WL* liquid synthesis beyond checkerboards: archive and loose-file paths share actual
  world-geometry triangle rasterization of all nine surface quads, reject samples below the aligned
  terrain height, and resolve per-pixel type into `LiquidBasicType257`. WLW/WLQ use parsed header
  classes; WLM is magma and WLL lava flows through the canonical magma class. Focused liquid/minimap
  suite: 32 C# tests passed; Debug Harvest build succeeds with only existing package/nullability
  warnings.
- Elevated the WL* defect to complete dataset provenance: corrected shards require
  `wl_liquid_surface_quads_v1`, `wl_liquid_above_terrain_v1`, and
  `wl_liquid_basic_type_header_v1`; V16/V18/V50 reject any incomplete WL fallback rather than
  converting sparse, through-terrain, or default-water pixels into training facts. Existing earlier
  WL datasets are invalid for liquid-aware work and require client-backed re-harvest; their stored
  masks cannot reconstruct missing visibility or type semantics. Focused Python provenance tests:
  5 passed.
- Added paired liquid minimap output. Every successful synthetic tile now writes the existing
  liquid-free terrain PNG and an aligned `_liquid.png`; `--whole-map` writes matching terrain and
  liquid stitched maps. The v4 manifest records liquid paths, pixel count, and the
  `viewer_flat_liquid_overlay_v1` profile. It is an analytic flat overlay from decoded unified
  coverage/basic types, not a claim of native water texture/animation/reflection parity. Alpha MCLQ
  257² surface data with a 16×16 type grid is normalized before unified-liquid composition, removing
  the mismatched-array failure mode that can affect liquid tiles.
- Corrected Alpha liquid classification granularity. A visible MCLQ cell now uses its own raw type
  nibble before falling back to the containing MCNK's flags: `0x01=Ocean`, `0x03=Slime`,
  `0x04=River/Water`, and `0x06=Magma`. The former ordinal mapping turned all `0x04` rivers green;
  they now select the blue water palette. `LiquidBasicType257` remains the separate supervision
  signal. Focused decoder/tensor/Alpha-round-trip coverage: 52 passed.
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
  discard otherwise readable terrain. That recovery now applies only to named materials; a tile
  with no non-empty MTEX name composes as an unlit solid-white empty baseline. Latest focused
  Alpha/minimap/texture-policy suite: 27 passed; Debug Harvest and Viewer builds: 0 errors
  (existing warnings only).
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

- Spec 109 v50 clean-room dataset work is separate from Spec 110. V50 does not yet have a canonical
  per-build writer, so `v50_build_dataset.py` now refuses to delegate to the legacy mixed-copy
  builder. Its initial liquid contract keeps `liquid_mask`/`liquid_height` as fresh-only targets:
  historic payloads are rejected; fresh WL sources require contiguous, above-terrain, and typed
  provenance; non-WL sources must retain their reader identity in row lineage. Focused V50/WL
  contract coverage: 5 Python tests passed.
