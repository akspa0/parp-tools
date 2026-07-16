# Research: Viewer Stabilization

## Decision: Normalize the active fog range at the Core boundary

- **Decision**: Add a reusable normalizer in `TerrainLightingMath` and call it after lighting
  source selection, before shader uniforms, terrain culling, object visibility, and WDL selection.
- **Rationale**: `ComputeFogRange` already safely handles malformed DBC values, but the scene assigns
  LIT `FogStart`/`FogEnd` directly. A same-value pair reaches GLSL as a zero denominator and makes
  terrain visibility undefined.
- **Alternatives considered**: Clamp only inside each shader (rejected: culling and objects would
  still use bad distances); accept every LIT range as authored (rejected: a missing/degenerate
  profile must not hide the map).

## Decision: Fog overrides belong to WorldScene, not settings defaults

- **Decision**: Keep a user override state and current lighting recommendation in `WorldScene`.
  Settings remain the fallback default applied at load; the Lighting panel owns active controls.
- **Rationale**: `ViewerApp_Sidebars` writes the lighting object, then `WorldScene` overwrites it
  each frame from DBC/LIT. This is why a visible slider does not actually alter the live range.
- **Alternatives considered**: Disable lighting whenever a user adjusts fog (rejected: color and
  other lighting data should continue to update); persist every map-specific override now
  (deferred: global/session behavior is enough for stabilization).

## Decision: LIT minimap inspection reuses one scene selection and the shared minimap renderer

- **Decision**: Add a diagnostic-only `ShowLitMinimapMarkers` state to `WorldScene`, render loaded
  positional LIT entries through `MinimapHelpers` (already shared by regular and full-screen maps),
  and keep `SelectedLitLightIndex` as the single selection authority. The Lighting surface owns the
  virtualized list and double-click camera focus.
- **Rationale**: Both minimap modes already call the same helper and the scene already retains LIT
  entries plus selected state for the 3D diagnostic overlay. Reusing those seams prevents a second
  map coordinate transform or a UI-only LIT copy that can drift from the active source.
- **Alternatives considered**: Draw markers independently in each minimap view (rejected: duplicate
  transforms/selection behavior); make marker selection change the lighting profile (rejected:
  inspection must not mutate rendering); navigate directly to the raw entry position (rejected:
  placing the camera inside terrain is not a usable inspection view).

## Decision: Terrain minimap export is direct composition, not VLM/MK recovery

- **Decision**: Put `TerrainMinimapCompositor` and `TerrainMinimapStitcher` in `WowViewer.Core.IO`.
  The compositor consumes the existing `TerrainTileTensorPack` texture IDs, MCAL alpha, MCNR normals,
  and decoded BLP pixels. MCSH remains attached to the pack as independent terrain evidence and is
  excluded from normal minimap RGB. The Harvest command supplies archive-backed tiles and the viewer
  invokes that command through a self-resolving in-repository launch contract.
- **Rationale**: `MinimapBakeService` requires a previously-exported VLM dataset, while the current
  `synthetic-minimap` command is a stub. Neither can make a minimap for an arbitrary on-disk map
  that lacks authored minimap assets. Core.IO already owns the decoded terrain inputs and ImageSharp.
- **Alternatives considered**: Render top-down screenshots (rejected: renderer output is neither
  deterministic nor terrain-only); revive MK/VLM menus (rejected: data-authoring is not the user
  workflow); use an existing client minimap as an input (rejected: the target maps may have none).

## Decision: Time of day is provenance-first

- **Decision**: At the requested normalized clock time, use
  `LitTerrainDayNightProfile.EvaluateGlobalClear` only when its global clear-weather tracks can be
  evaluated. Otherwise use `AuthoredTerrainDayNightProfile` and label the export accordingly.
  Keep local LIT zones out of the exporter until their spatial transform is proven.
- **Rationale**: Global LIT colors are decoded and tested, but the current local-zone coordinate
  transform remains explicitly diagnostic-only. A visible fallback makes a missing-LIT map useful
  without representing an authored approximation as client-exact.
- **Alternatives considered**: Make export fail without LIT (rejected: defeats the missing-minimap
  use case); apply local zones speculatively (rejected: unproven spatial semantics); use a static
  color grade (rejected: ignores the requested time of day).

## Decision: Export time uses an exact clock minute

- **Decision**: Accept compact `HHmm` and `HH:mm` as exact clock input, retain decimal hours for
  compatibility, and normalize every accepted value to a single minute before LIT evaluation.
  Persist both canonical `HH:mm` and decimal hours in the manifest.
- **Rationale**: `12` is only a convenient default, not necessarily solar noon for a map's authored
  light track. Exact clock input permits a reproducible 12:15 sample without ambiguous floating
  slider positions.
- **Alternatives considered**: Decimal-only input (rejected: users cannot plainly express a clock
  minute); second-precision input (rejected: no downstream LIT/minimap requirement currently needs
  it).

## Decision: Authored direction follows the terrain raster axes

- **Decision**: Keep the LIT global-clear colour tracks, but derive the required direction through
  one shared authored solar profile. It is vertical at 12:00 and follows the terrain world-to-raster
  diagonal after noon so the source projects from top-left rather than retaining a fixed horizontal
  bias.
- **Rationale**: Early LIT profiles provide timed colours but not a sun vector. The previous
  profile held world Y positive throughout the day, producing directional noon shading and the
  wrong visible origin regardless of selected minute.
- **Alternatives considered**: Treat LIT colour as a light direction (rejected: it is not vector
  data); leave the axis arbitrary (rejected: produces stable but visibly wrong terrain relief).

## Decision: distinguish ordinary minimap RGB from rare baked-lighting evidence

- **Decision**: Synthetic minimaps default to the conventional unshadowed material/MCAL/MCNR RGB
  baseline and never bake MCSH. For an authored minimap tile, derive a robust global RGB-ratio tint
  against that neutral terrain baseline, correlate residual darkening with the separately decoded
  MCSH mask, and compare tint chroma with build-local global-clear LIT candidates. Serialize the
  result as `minimap_lighting` evidence, with an optional time bucket only when the chroma match is
  sufficiently close.
- **Rationale**: MCSH is normally omitted from minimap imagery. A small exceptional population has
  baked static shadow or zone/light coloration, and treating either population as the universal
  renderer contract corrupts both output and supervision. A conservative sidecar lets downstream
  consumers bucket likely lighting conditions while retaining MCSH as its own recoverable target.
  A chroma match can suggest a time-of-day family; it cannot prove the historical capture time.
- **Alternatives considered**: Always bake MCSH (rejected: wrong for normal minimaps); discard MCSH
  (rejected: loses a useful independent target); label a closest LIT sample as exact capture time
  (rejected: no capture-time proof); serialize partial texture swatches after a failed decode
  (rejected: shifts MTEX identity and poisons the baseline).

## Decision: use a declared `_s` companion, then catalog-wide related diffuse assets, as missing-diffuse RGB proxies

- **Decision**: If a terrain BLP cannot decode, try its same-stem `_s.blp` companion first. If that
  also fails, scan the loaded archive/listfile catalog and try at most sixteen decodable ordinary
  `.blp` candidates. Exact and strongly similar basenames rank first; shared directory-theme tokens
  break ties, allowing historical moved assets to recover. Preserve the original MTEX table entry;
  serialize `specular_companion_rgb_proxy` or
  `related_diffuse_rgb_proxy` for every selected derived-artifact substitute. The terrain viewer
  uses the same candidate policy and logs the selected path.
- **Rationale**: Some Alpha terrain references point at missing/unsupported diffuse assets while a
  a same-stem companion or a nearby named color asset survives. The current minimap path consumes
  decoded RGB averages only, while the terrain viewer does not reproduce native specular, alpha,
  texture-flag, or effect routing. Strong-name matching plus directory-theme ranking recovers moved
  material color without treating an arbitrary texture in the same zone as a valid replacement.
- **Alternatives considered**: Arbitrary nearest-name replacement (rejected: no provenance basis);
  same-folder-only matching (rejected: historical asset moves leave stale ADT links); treat `_s` as
  equivalent diffuse engine material (rejected: unsupported renderer claim); drop the tile despite
  a decodable related asset (rejected: loses recoverable terrain evidence).

## Decision: CPU minimap composition is material-derived, not a terrain-renderer UV capture

- **Decision**: Treat layer zero as the opaque base, then apply layers 1–3 with ordered `mix`/lerp
  composition. For each decoded BLP, cache the phase-independent full-texture average colour and apply it
  through MCLY chunk selection and MCAL alpha masks; do not project or bilinear/trilinear sample the
  terrain renderer's repeated diffuse UVs.
- **Rationale**: A user-run 0.5.3 tile showed that a projected CPU mip sampler still preserves the
  two-by-two mip's repeat phase, producing visible moire and interpolation bands. A minimap pixel
  covers many diffuse repetitions, so its stable visual contract is material colour plus real
  terrain layer/mask structure, not a low-resolution screen capture of diffuse detail.
- **Alternatives considered**: Keep renderer UVs plus point/mip sampling (rejected: static noise or
  phase-dependent moire); use normalized alpha weights (rejected: it disagrees with the terrain
  fragment shader whenever overlays overlap); use renderer screenshots (rejected: non-deterministic
  and not terrain-only); fail a readable base-layer tile without MCAL (rejected: MCAL is overlay
  evidence, so absent MCAL means base-only rather than a reason to lose the map tile).

## Decision: normalize Alpha MCLY at the tensor-pack boundary

- **Decision**: Preserve Alpha reader/legacy-adapter storage as native `[chunkX, chunkY, layer]`,
  then transpose MCLY texture IDs and layer presence into row-major `[chunkY, chunkX, layer]` when
  building `TerrainTileTensorPack`. Compose only layers marked present by `MclyLayerMask`.
- **Rationale**: Alpha MCAL is row-major, but the raw MCLY grid is column-major. Passing its native
  layout straight into the row-major minimap path pairs a real alpha mask with a different chunk's
  texture. The resulting hard chunk boundaries and apparently interpolated texture detail match the
  user-run 0.5.3 output.
- **Alternatives considered**: Teach each consumer Alpha's special layout (rejected: duplicated
  era-specific indexing); transpose the raw reader data (rejected: breaks the legacy adapter and
  writer-facing raw contract); ignore `MclyLayerMask` (rejected: missing layers default to texture
  ID zero and can become ghost overlays).

## Decision: interpolate Alpha MCNR lighting on terrain triangles

- **Decision**: Keep the sparse 257×257 MCNR compatibility array as an extraction contract, but
  evaluate `N·L` only at its real staggered lattice vertices and barycentrically interpolate the
  scalar over the four terrain triangles around each inner vertex. Preserve the 256² MCSH occupancy
  array unchanged as a distinct shadow/model signal.
- **Rationale**: The dense array deliberately has alternating non-vertex positions. Nearest sampling
  of that array substituted `UnitZ` in each gap, producing a pixel-scale checkerboard in the new
  time-of-day lighting pass that can be mistaken for texture interpolation or MCSH corruption.
  Native terrain lighting is vertex-based, so its interpolated Lambert result is the correct visual
  quantity; it does not modify the decoded MCSH target.
- **Alternatives considered**: Fill sparse MCNR gaps with `UnitZ` (rejected: creates false lighting);
  bilinearly interpolate/re-normalize normals per pixel (rejected: differs from the fixed-function
  vertex-light path); discard MCSH from the tensor pack (rejected: static-shadow evidence remains a
  valid independent training signal); assume every MCNR validity mask has the normal-grid shape
  (rejected: partial Alpha data exists, and unknown normals should fall back safely rather than
  aborting a whole-map export).

## Decision: emit paired liquid-bearing minimap targets

- **Decision**: Every synthesized terrain PNG has an aligned `_liquid` companion. The baseline
  remains terrain-only; the companion overlays decoded `UnifiedLiquidMask` coverage using the
  resolved basic liquid type and the current viewer's flat type palette/opacity. Stitch both sets
  independently and record paths, liquid-pixel count, and render-profile identity in the manifest.
  Alpha's 16×16 MCLQ type grid is normalized to the 257² MCLQ surface before unified-liquid output.
  Its native 8×8 MCLQ tile flags are the authoritative cell-presence signal; the output compositor
  requires all four source-cell corners to be covered before painting a liquid pixel. For a visible
  Alpha cell, its MCLQ low-nibble liquid type is the primary class; MCNK liquid flags classify only
  cells that omit a type nibble. The output uses deterministic distinct flat colors for water,
  ocean, magma, and slime while retaining the resolved basic type array separately from RGB.
- **Rationale**: A terrain-only image cannot supervise or classify liquid evidence in historical
  input minimaps. Coupled, coordinate-identical baseline/liquid outputs retain both signals without
  confusing the terrain model or pretending to have recreated water materials. Sampling a single
  covered vertex caused false liquid strips over adjacent dry terrain cells, which is a geometry
  error rather than a water-material characteristic. Collapsing a whole chunk to its MCNK type
  discarded valid local ocean/magma/slime evidence and made the companion color misleading.
- **Alternatives considered**: Replace the terrain baseline with liquid RGB (rejected: loses the
  no-liquid signal); emit only a mask (rejected: omits the visual input target and liquid type);
  reuse client minimap imagery (rejected: violates direct synthesis); call the analytic overlay
  native water parity (rejected: no texture, animation, reflection, or shader-effect proof).

## Decision: never discard readable terrain solely for stale texture linkage

- **Decision**: Preserve narrow recovery first (original BLP, same-stem `_s`, then related diffuse
  candidates), but when all of those fail, choose a successfully decoded ordinary catalog BLP
  deterministically: same folder, then the same terrain family, then any terrain-likely catalog
  path. Cache verified decoded candidates during an export for incomplete early-client listfiles.
  Record `catalog_rgb_last_resort_proxy`; do not rewrite the original MTEX identity. If MCLY/MTEX
  is absent, compose using that proxy as the explicit base material.
- **Rationale**: The 0.5.3 Kalimdor manifest had 220 otherwise-readable tiles skipped only because
  no referenced BLP resolved. A visible, provenance-labeled terrain tile is more useful than a
  transparent hole, while the retained source/resolved identities keep the artifact honest.
- **Related defect**: The remaining `IndexOutOfRangeException` tiles were unrelated to texture
  lookup. The source frame proved that cross-tile WMO roof-mask footprints were checked as 257² in
  Alpha `PaintCircle` even though their destination buffers are 256²; all footprint painters now
  use their own buffer dimensions.

## Decision: M2 runtime has no MDX fallback

- **Decision**: Remove M2-to-MDX and adapter-backed MdxRenderer fallback routes from
  `WorldAssetManager`, `WmoRenderer`, and the runtime bridge. Use native `M2Renderer` or return a
  precise capability diagnostic.
- **Rationale**: The current loader still converts an M2 when direct adaptation fails and can choose
  an MDX drawing backend. That conflates export compatibility with runtime ownership and violates
  Spec 104 FR-011.
- **Alternatives considered**: Preserve fallback behind an environment switch (rejected: it makes
  rendered behavior non-deterministic and leaves the incorrect architecture intact).

## Decision: Conversion capability is direction-specific

- **Decision**: Publish WMO v14→v17 and v17→v14 separately, and keep M2→MDX export provisional
  until profile-level real-client proof exists.
- **Rationale**: Both WMO directions already have Core.IO converter classes and dedicated fixture
  tests. M2→MDX has comprehensive synthetic artifact tests, not client-profile reliability proof.
- **Alternatives considered**: Describe all three paths as generically reliable (rejected: it would
  overstate fixture coverage and invite renderer reuse).

## Decision: Remove dead UI instead of wrapping absent binaries

- **Decision**: The Tools audit removes obsolete ML/dataset entries and keeps only actions with a
  current in-repo owner; unavailable dependencies get a diagnostic, not a broken process launch.
- **Rationale**: Spec 073b is a tasks-only legacy integration artifact, while the active UI owner is
  Spec 080. Its proposed executable-relative launcher is not a reliable modern contract.
- **Alternatives considered**: Restore every old card first (rejected: preserves a stale tool surface
  and ignores the missing-binary failure the user reported).
