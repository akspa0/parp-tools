# Active Context — wow-viewer

Last updated: 2026-07-16

## Active work: Spec 110 viewer stabilization

- Current phase: **Phase 1d terrain-minimap fidelity correction** after fog, terrain visibility,
  LIT inspection, export, and control reachability. A real 0.5.3 Kalimdor export now emits a tile;
  the remaining gate is a bounded visual re-export with phase-independent materials, unshadowed
  default RGB, and its accompanying lighting-provenance record before M2/tool/conversion work.
- `TerrainLightingMath.NormalizeFogRange` now guarantees a finite non-zero range. Collapsed or
  invalid LIT/lighting values resolve to a visible fallback rather than entering GLSL/culling.
- `WorldScene` now resolves DBC/LIT/fallback recommendation first, then applies an independent user
  override. Lighting can update color/recommendations but cannot overwrite the active override.
- Lighting panel exposes active Fog Start/Fog End, source, normalization status, and reset. Legacy
  sliders route through the same override. Active and default Fog Start/Fog End are true visible
  slider controls rather than drag-only fields. Settings are load defaults only.
- Lighting now also exposes an opt-in `Show LIT minimap markers` diagnostic layer and a virtualized
  LIT entry list. Positional entries render through the same helper in normal and full-screen
  minimaps; selection is shared with the existing 3D LIT diagnostic overlay. Default/non-finite
  entries stay listable but cannot create a marker or camera jump. Double-clicking a positional
  list row (or marker) frames a safe downward-looking 3D camera point without changing fog or
  lighting selection.
- `TerrainMinimapCompositor` directly synthesizes a terrain-only tile from existing MCLY/MCAL,
  MCNR, and decoded BLP inputs. `TerrainMinimapLiquidCompositor` then produces an aligned `_liquid`
  companion using decoded unified coverage plus the current flat viewer liquid type palette/opacity.
  The companion is explicitly analytic—not water texture/animation/reflection parity—and its path,
  render profile, and liquid-pixel count are in the v4 manifest. Both terrain and liquid sets stitch
  independently while preserving transparent holes. MCSH remains separate static-shadow evidence
  and is omitted from normal RGB unless an explicit `--bake-mcsh` exceptional-history preview is chosen.
  MCAL overlays are applied over the base in file order; each BLP is cached as a phase-independent
  material average, preserving real chunk/mask structure without sampling renderer diffuse UVs or
  adding interpolation. The baseline never reads or substitutes a pre-authored client minimap.
- A user-run full-map Kalimdor export exposed a readable tile with no `McalAlphaPack256`. Missing
  MCAL now means base-layer-only composition (no fabricated overlay weights), not a tile failure;
  normal/white-top-edge evaluation still applies. The next encountered tile had a partial MCNR mask shape;
  bounds-safe normal reads now treat out-of-mask samples as neutral rather than indexing beyond the
  mask. Focused regression proof covers both fallbacks.
- Missing terrain diffuse BLPs first use a successfully decoded same-stem `_s.blp` companion as
  `specular_companion_rgb_proxy`, then (only if needed) up to sixteen decoded ordinary `.blp`
  candidates scanned from the loaded archive/listfile catalog as `related_diffuse_rgb_proxy`.
  Exact/strong basename similarity ranks before shared directory-theme tokens so moved historical
  assets can repair stale ADT links. Original MTEX identity remains unchanged and raw/NPZ metadata
  records the substitute. `TerrainRenderer` applies the identical order and logs its selected proxy.
  Neither path implements native `_s` specular, alpha, blend-mode, or other material semantics.
- Alpha 0.5.3 proof exposed a second compositing defect: raw Alpha MCLY is `[chunkX,chunkY,layer]`,
  while MCAL/tensor consumers are row-major. `AlphaTensorPackBuilder` now normalizes the layer IDs
  and masks to `[chunkY,chunkX,layer]`, and the compositor refuses layers absent from
  `MclyLayerMask`; focused Alpha/tensor/compositor proof is 11/11.
- Alpha liquid data can similarly carry a 257² MCLQ surface with only a 16×16 chunk-type grid.
  `AlphaTensorPackBuilder` now normalizes that type grid before building presence/unified liquid and
  resolves Alpha basic types directly. This prevents the mismatched-array bounds failure and makes
  liquid-bearing companion outputs available for Alpha tiles.
- The latest 0.5.3 Kalimdor v4 manifest reports `951` occupied, `687` written, `220` skipped, and
  `44` decode failures. Twenty-six failures were already present before the Alpha WDT coordinate
  correction; the other eighteen became visible because it now reaches the actual occupied cell.
  Do not label the residual bounds fault fixed until a targeted rerun reports its first code frame.
  `synthetic-minimap --tile-x <x> --tile-y <y>` selects that one occupied coordinate and failed
  diagnostics now preserve the first relevant WowViewer source frame.
- That source frame resolved the 44 decode faults: cross-tile WMO roof footprints reach a 256²
  roof-mask buffer, but Alpha `PaintCircle` and sibling painters were checking against the 257²
  terrain grid. Every painter now clips to its destination buffer, so an edge/cross-tile WMO cannot
  abort terrain decoding. The 220 skips are separate stale/missing material-linkage cases: after
  original, `_s`, and related diffuse candidates, Harvester now uses a successfully decoded catalog
  RGB fallback (same directory, then terrain family, then a verified prior decode) and records
  `catalog_rgb_last_resort_proxy` for a declared material whose BLP cannot resolve. A tile with no
  non-empty MTEX name is an unlit solid-white empty baseline rather than a catalog proxy.
- Liquid pixels are now rasterized from complete source cells: Alpha honors the 8×8 MCLQ tile flags,
  and the companion compositor requires all four coverage corners before overlaying a liquid pixel.
  This replaces the single-vertex/all-chunk behavior that produced liquid strips on dry cell edges.
- Alpha liquid class is now resolved at the same visible-cell granularity: the MCLQ cell type nibble
  takes precedence over MCNK's containing-chunk type flags, with MCNK retained as the fallback.
  Raw MCLQ values are `0x01=Ocean`, `0x03=Slime`, `0x04=River/Water`, and `0x06=Magma`; the former
  ordinal mapping called `0x04` slime, turning rivers green. The companion palette therefore keeps
  rivers blue while `LiquidBasicType257` remains the independent data signal. Focused decoder,
  tensor, and Alpha round-trip proof: 52 passed.
- The same time-of-day path had a distinct lighting defect: Alpha MCNR's 257² compatibility grid
  intentionally has alternating non-vertex positions, but minimap synthesis sampled those gaps as
  `UnitZ`. `TerrainMinimapCompositor` now evaluates Lambert at the five real staggered vertices and
  interpolates across terrain triangles. It leaves decoded `McshShadowMask256` unchanged as the
  separate MCSH/model target. Focused compositor/lighting/Alpha tests: 19 passed; Harvest Debug
  build passes (existing package/nullability warnings only).
- `MinimapLightingProvenance` v1 now records authored-minimap analysis separately from synthesis:
  tint vector/fit, MCSH darkening correlation, and only a conservative LIT-chroma time bucket with
  explicit non-capture-proof evidence. Missing RGB, incomplete terrain textures, or no LIT
  candidate produce explicit not-evaluated states. Full and V22 streams perform the baseline decode;
  texture/model sidecars are emitted only with stable, name-aligned identities, never shifted after
  a missing decode. Focused compositor/provenance/NPZ/raw-stream suite: 49 passed; Harvest Debug
  build: 0 errors (existing warnings only).
- `WowViewer.Tool.Harvest synthetic-minimap` accepts a configured client root, map, time of day,
  resolution, and per-tile/whole-map outputs. Time accepts minute-precise `HHmm`/`HH:mm` clock
  input as well as legacy decimal hours; the manifest records canonical clock plus decimal hours.
  The in-app export dialog exposes exact Hour and Minute inputs rather than a fractional slider.
  It records client/build and LIT-or-authored fallback lighting provenance in
  `synthesis-manifest.json`. The in-app `Tools > Export > Synthesized Terrain
  Minimap...` dialog resolves the in-repository Harvest executable, DLL, or project rather than
  assuming an external binary.
- The original direct compositor had arbitrary UV projection, normalized MCAL overlays, and point
  sampling. A first correction copied the renderer UV/mip approach, but the real 0.5.3 output
  proved that phase still creates moire/interpolation at minimap scale. The current correction
  intentionally uses per-BLP material averages instead. Current proof: `TerrainMinimapCompositorTests`
  7/7 (material averaging, ordered overlays, high-frequency stability, MCSH exclusion, MCNR triangle
  lighting, stitching) and
  Debug Harvest build pass. User must first re-export one bounded tile from the known client root,
  then complete the Spec 110 LIT/no-LIT time, overlay, whole-map, fog-slider, and Archeology proof
  with client build/fingerprint recorded.
- Real-client bounded-export feedback exposed a separate command defect: Kalimdor's first ordered
  WDT slot `(12,19)` could not decode, and `--limit 1` counted that attempt, leaving no PNG.
  `synthetic-minimap --limit` now counts emitted tiles only while retaining skipped/failed entries
  in `synthesis-manifest.json`; rerun the same one-tile command before assessing visual fidelity.
- A later full Kalimdor manifest (`951` occupied, `361` written, `543` false decode skips) proved
  Alpha WDT MAIN enumeration had transposed `(tileX,tileY)` despite Alpha tile offsets being
  row-major. `WdtTileIndexReader` now matches `AlphaWdtReader`'s `tileY * 64 + tileX` mapping.
  Tile failures now record the exact pipeline stage in the manifest.
- A user visual review proved that applying LIT colors and the recovered 0.5.3 world-light ray to
  minimaps was the wrong consumer contract: it tinted terrain orange/pink and put hillshade on the
  south side, reversing basin/hill perception. `synthetic-minimap` now excludes LIT/native-ray
  inputs entirely. It uses pure-white direct light, achromatic ambient, and a negative terrain-X
  north/top-edge source; native `SetDirection` recovery remains diagnostic research only.
- That negative-terrain-X "north" lock was itself backwards. `wow-1.0.0-world-lighting-shadow-model-2026-07-15.md`
  §2.1 traces the native `SetDirection` ray in raw WoW world axes; `AdtTensorPackBuilder.AssembleNormals`
  (no axis swap) and `TerrainMeshBuilder`'s row/tileY-driven vertex world-X confirm this codebase's
  MCNR/MCVT convention is +X = North, +Y = West, +Z = Up. So negative X is south, not north, and the
  prior fix was still sourcing the sun from the south. `TerrainSolarDirection` now locks the horizontal
  bias to positive X; the compositor test and spec/contract wording were corrected to match.
- A user side-by-side of a synthesized tile against the real 0.5.3 client minimap (same crater/lake
  feature) exposed a second, more consequential defect: the horizontal bearing was swept with
  `cos(sunAngle - pi/2)`, which is exactly zero at solar noon/midnight, making the sun point straight
  up with no shadow direction at those instants and washing out the ring-shaped relief. The real
  minimap keeps a persistent bright-north/dark-south hillshade at every sampled time, matching the
  traced native ray's constant azimuth (theta = 225 degrees across all four sampled table entries).
  `TerrainSolarDirection` now locks the horizontal bearing to a fixed north-west share all day and
  varies only elevation. Added `TerrainSolarDirectionTests` and corrected
  `AuthoredTerrainDayNightProfileTests`' now-inverted "vertical at noon" assertion.
- WL* liquid output was reduced to sparse origin/vertex stamps, causing a checkerboard. Both loose
  and archive-backed paths now share a geometry rasterizer that fills each block's nine 4x4-grid
  surface quads with interpolated heights, then drops every sample below the aligned terrain height.
  The same path resolves `LiquidBasicType257`: WLW/WLQ use their parsed header class, WLM is magma,
  and WLL lava uses the canonical magma class. Corrected shards carry all three markers:
  `wl_liquid_surface_quads_v1`, `wl_liquid_above_terrain_v1`, and
  `wl_liquid_basic_type_header_v1`; V16/V18/V50 fail closed on any incomplete WL provenance.
  All earlier WL liquid-aware datasets are invalid and must be re-harvested, not patched from their
  stored masks. Focused C# liquid/minimap tests: 32 passed; V16/V18/V50 provenance tests: 5 passed;
  Harvest Debug build succeeds with only existing package/nullability warnings.
- UniqueId controls are now owned only by Tools > Archeology. Archeology has a separate nested
  Range/Layers/Playback/Capture selection, so opening Playback cannot switch the parent Tools tab
  to PM4. Active playback exposes pause/stop on every Archeology subtab; manual range edits and a
  missing world/range stop it safely. Legacy Tools > UniqueId Archeology opens its dedicated window
  with the same transport instead of exposing controls in World.

## Next implementation slice after the visual gate

- Native M2 recovery only: world-object and WMO-doodad code still has adapter-backed MdxRenderer and
  M2→MDX runtime fallback branches, which are forbidden. Route 1.0.0 through
  `BuildEra100StaticRenderModel`/native `M2Renderer`; then remove all MDX conversion runtime paths.
- Do not use converted MDX as renderer proof. M2→MDX is explicit Alpha export only.

## Audited follow-ons

- WMO v14→v17 and v17→v14 Core.IO converters and fixture tests exist. Their real-client fidelity is
  not yet signed off.
- Core M2→MDX has synthetic conversion tests only; do not call it reliable for a client profile yet.
- Main Tools menu audit, removal of MK Dataset/VLM Dataset, and Inspect/Converter launch repair are
  planned in Spec 110 Phase 3 after the fog and native-M2 phases.

## Separate active lane

- Spec 109 v50 clean-room dataset work remains separate. `H:\CLIENTS` is the approved configured
  client library; legacy workspace output was cleared. Do not recreate pre-v50 outputs. V50 does
  not yet have a per-build store writer: its former Spec 108 mixed-copy wrapper now fails closed.
  Its frozen liquid policy preserves `liquid_mask`/`liquid_height` as useful targets but makes them
  fresh-only; a WL source requires all three contiguous/above-terrain/typed markers, while non-WL
  sources retain reader identity in row lineage.

## Durable boundaries

- `gillijimproject_refactor` is read-only. New code lives in `wow-viewer`.
- User runs client-backed visual proof, training, capture, and heavy work. Report client root,
  build identity, and fingerprint with any real-data conclusion.
- `AlphaWdtWriter.cs` is frozen. Renderer reader ownership is native M2; export conversion is separate.
