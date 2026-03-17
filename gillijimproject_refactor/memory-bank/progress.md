# Progress

## ✅ Working

### MdxViewer (3D World Viewer) — Primary Project
- **Alpha 0.5.3 WDT terrain**: ✅ Monolithic format, 256 MCNK per tile, async streaming
- **0.6.0 split ADT terrain**: ✅ StandardTerrainAdapter, MCNK with header offsets (Feb 11)
- **0.6.0 WMO-only maps**: ✅ MWMO+MODF parsed from WDT (Feb 11)
- **Terrain liquid (MCLQ)**: ✅ Per-vertex sloped heights, absolute world Z, waterfall support (Feb 11)
- **WMO v14 rendering**: ✅ 4-pass: opaque → doodads → liquids → transparent
- **WMO liquid (MLIQ)**: ✅ matId-based type detection, correct positioning (Feb 11)
- **WMO doodad culling**: ✅ Distance (500u) + cap (64) + nearest-first sort + fog passthrough
- **WMO doodad loading**: ✅ FindInFileSet case-insensitive + mdx/mdl swap → 100% load rate
- **MDX rendering**: ✅ Two-pass opaque/transparent, alpha cutout, specular highlights, sphere env map
- **MDX GEOS version compatibility**: ✅ Ported version-routed GEOS parser behavior from `wow-mdx-viewer` (v1300/v1400 strict path + v1500 strict path + guarded fallback)
- **MDX SEQS name compatibility**: ✅ Counted 0x8C named-record detection broadened to reduce fallback `Seq_{animId}` names on playable models
- **MDX PRE2/RIBB parsing parity**: ✅ Expanded parser coverage for PRE2 and RIBB payload/tail animation chunks (runtime visual verification pending)
- **MDX animation engine**: ✅ BONE/PIVT/HELP parsing, keyframe interpolation, bone hierarchy (Feb 12)
- **Full-load mode**: ✅ `--full-load` (default) loads all tiles at startup with progress (Feb 11)
- **MCSH shadow maps**: ✅ 64×64 bitmask applied to all terrain layers
- **AOI streaming**: ✅ 9×9 tiles, directional lookahead, persistent tile cache, MPQ throttling (Feb 11)
- **Frustum culling**: ✅ View-frustum + distance + fade
- **AreaID lookup**: ✅ Low 16-bit extraction + low byte fallback for MapID mismatch
- **DBC Lighting**: ✅ LightService loads Light.dbc + LightData.dbc, zone-based ambient/fog/sky colors
- **Replaceable Textures**: ✅ DBC CDI variant validation against MPQ + model dir scan fallback
- **Minimap overlay**: ✅ From minimap tile images
- **WDL preview map spawn selection (Alpha 0.5.3)**: ✅ Runtime-confirmed. Preview orientation, clicked tile selection, and resulting world spawn match the terrain grid.
- **Later-client map loading after WDL preview UI change**: ✅ Direct WDT load restored. Non-0.5.x clients no longer depend on unsupported WDL preview parsing just to open a map.

### Model Parsers & Tools
- **MDX-L_Tool**: ✅ Core parsing and Archaeology logic complete.
- **GEOS Chunk (Alpha)**: ✅ Robust scanner for Version 1300 validated.
- **Texture Export**: ✅ DBC-driven `ReplaceableId` resolution working.
- **OBJ Splitter**: ✅ Geoset-keyed export verified on complex creatures.
- **0.5.3 Alpha WDT/ADT**: ✅ Monolithic format, sequential MCNK.
- **WMO v14/v17 converter**: ✅ Both directions implemented.
- **BLP**: ✅ BlpResizer complete.

### Data Generation
- **VLM Datasets (Alpha)**: ✅ Azeroth v10 (685 tiles).

## ⚠️ Partial / In Progress

### MdxViewer — Rendering Quality & Performance
- **3.3.5 ADT loading freeze**: Needs investigation
- **Terrain alpha-mask regressions (post-343dadf baseline)**: still unresolved for active 3.x runtime rendering. Some recent MCAL-path assumptions/tests were ahead of reality; later-client terrain texturing is still visibly broken in the viewer and should not be described as signed off.
- **Terrain debug UX regression**: the alpha-mask debug checkbox currently prevents chunk/tile overlays from being visible in the same view because the shader exits early in alpha-debug mode. This is now part of the terrain-debugging problem, not just a UI annoyance.
- **Batched missing-texture terrain parity**: The tile-array renderer now invalidates diffuse indices for missing BLP slices before draw, so later-client overlay layers no longer blend against synthetic white fallback textures when the per-chunk path would have skipped them.
- **Current terrain handoff**: return to direct 3.x layer decode/sourcing investigation. Runtime screenshots still show broken later-client terrain, and the debug overlay path needs to be fixed so alpha plus chunk/tile boundaries can be inspected together.
- **WMO stream reload regression**: Cached failed WMO/MDX loads no longer block reloads after stream-out/stream-in. Runtime verification against Ironforge re-entry is still needed.
- **WMO culling too aggressive**: Objects outside WMO not visible from inside
- **MDX GPU skinning**: Bone matrices computed per-frame but not yet applied in vertex shader (needs BIDX/BWGT vertex attributes)
- **MDX animation UI**: Sequence selection combo box in ImGui panel not yet wired
- **MDX per-geoset color/alpha**: Only static alpha used; animated GeosetAnims not wired
- **MDX particles/ribbons**: Parser coverage expanded; runtime behavior verification still pending on effect-heavy assets
- **MDX texture UV animation**: Not implemented
- **MDX billboard bones**: Not implemented
- **WMO lighting**: v14-16 grayscale lightmap + v17 MOCV vertex colors not implemented
- **Vulkan RenderManager**: Research phase — `IRenderBackend` abstraction for Silk.NET Vulkan

### Build & Release Infrastructure
- **GitHub Actions**: ✅ `.github/workflows/release-mdxviewer.yml` — tag push or manual dispatch
- **WoWDBDefs bundling**: ✅ 1315 `.dbd` files copied to output via csproj Content items
- **Self-contained publish**: ✅ `dotnet publish -c Release -r win-x64 --self-contained` verified

### MDX-L_Tool Enhancements
- **M2 Export (v264)**: 🔧 Implementing binary writer.

## ❌ Known Issues

### MdxViewer Rendering Bugs (Feb 12, 2026)

#### Terrain Alpha / Shadow Batched Upload Regression (Mar 16, 2026)
- **Symptom**: Batched terrain path can diverge from per-chunk alpha/shadow rendering, especially on LK big-alpha and already-fixed 64x64 data.
- **Root cause candidate**: `TerrainTileMeshBuilder.FillAlphaShadowSlice` duplicated row/column 63 from 62 for all alpha/shadow slices, while `Mcal` and the per-chunk upload path already preserve final decode output.
- **Real-data validation**: Confirmed on Alpha `Azeroth` tile `(0,0)` and WoWMuseum 3.3.5 `development` tile `(0,0)`. The old remap would have changed 17,367 explicit alpha/shadow bytes on the Alpha tile and 5,166 on the LK tile.
- **Status**: 🔧 Tile-array packing and `TerrainImageIo` atlas roundtrips now preserve decoded data. Re-run audit shows atlas roundtrip diffs at zero on both validation tiles. First-party regression tests now cover the fixed semantics in `src/MdxViewer.Tests`.

#### MDX Sphere Env / Specular Orientation (Feb 14, 2026)
- **Symptom**: Reflective/specular surfaces (e.g., dome-like geometry) appeared inward-facing on some two-sided materials.
- **Fix Applied**: Fragment shader now flips normals/view-space normals on backfaces before env UV generation and lighting/specular.
- **Status**: 🔧 Patched in code, pending visual confirmation on Dalaran dome repro.

#### WMO Semi-Transparent Window Materials
- **Symptom**: Stormwind WMO maps blue/gold stained glass textures to white marble columns instead of window frames
- **Hypothesis 1**: Secondary MOTV chunk not skipped → MOBA batch parsing misalignment
- **Fix Attempt 1**: Added `reader.BaseStream.Position += chunkSize;` when secondary MOTV encountered in `WmoV14ToV17Converter.ParseMogp` (line 922)
- **Result**: ❌ FAILED — window materials still map to wrong geometry
- **Status**: Root cause still unknown. May not be MOTV-related. Need to check console logs to verify if secondary MOTV is even present in Stormwind groups.

#### MDX Cylindrical Texture Stretching
- **Symptom**: Barrels, tree trunks show single wood plank stretched around entire circumference instead of tiled texture
- **Hypothesis 1**: Texture wrap mode incorrectly clamping both S and T axes when only one should clamp
- **Fix Attempt 1**: Changed `ModelRenderer.LoadTextures` to use per-axis clamp flags (clampS/clampT) based on `tex.Flags & 0x1` and `tex.Flags & 0x2` (lines 778-779)
- **Result**: ❌ FAILED — textures still stretched on cylindrical objects
- **Status**: Root cause still unknown. May not be wrap mode related. Need to check console logs to verify texture flags and investigate UV coordinates.

### AdtModfInjector
- **Problem**: Appends MWMO/MODF chunks to end of file; result is Noggit-incompatible.

## Key Technical Insights

### WDL Preview Spawn Path (Mar 16, 2026)
- The WDL preview failure was not a late camera reset in the world load path. `LoadFileFromDataSource` applies the world default camera first, and the preview dialog then overwrites it with the selected spawn.
- The actual bug was preview-to-world conversion: preview pixels were being interpreted on a transposed tile grid, and an earlier regression had also pushed preview spawns onto the wrong world scale.
- The current path is runtime-confirmed working for Alpha 0.5.3: preview tiles are drawn in screen-major order, converted back into terrain tile coordinates for spawning, and WDL hide/show uses the same `tileX*64 + tileY` indexing as WDL load.
- Follow-up correction: later clients must not use that preview path yet. Their WDL files are not handled by the Alpha-only parser, so the map browser now loads WDTs directly and only exposes preview for supported Alpha 0.5.x data.

### Terrain Alpha Pipeline (Mar 16, 2026)
- Per-chunk path uploads decoded alpha/shadow data as-is; it does not add a second edge fix.
- Batched tile path had been re-sampling texel row/column 63 from 62 inside `TerrainTileMeshBuilder`, which made tile-array rendering semantically different from per-chunk rendering.
- Real-data audit against Alpha `Azeroth` and WoWMuseum 3.3.5 `development` confirmed that those edge texels do differ in shipped data, so the divergence was not theoretical.
- `TerrainImageIo` atlas export/import has now been aligned with the fixed batched renderer path; the follow-up audit reports zero atlas roundtrip diffs on the validation tiles.
- `StandardTerrainAdapter` now routes split-ADT alpha decode by explicit profile family: 0.x profiles stay legacy sequential, 3.x profiles use strict LK `Mcal` decode. `AlphaMapService` still has not been adopted as the viewer-side source of truth.
- There is now minimal first-party automated regression coverage for the active terrain alpha path under `gillijimproject_refactor/src`, focused on batched packing parity and `TerrainImageIo` atlas roundtrips. Decode-path coverage is still missing.
- The active viewer terrain alpha tests now also cover explicit legacy-vs-LK decode gating, the fixed-path WoWMuseum 3.3.5 tile load, and the batched missing-diffuse remap behavior. Coverage is still minimal, but it is no longer limited to packing/export parity alone.
- The strict 3.x test seam now also guards the narrower missing-metadata behavior: `StandardTerrainAdapter_LichKingProfiles_DoNotInferAlpha_WhenUseAlphaFlagIsMissing()` ensures the viewer does not invent alpha maps when later-client flags and offsets are both absent.
- Additional strict-path coverage now keeps relaxed decode fenced to the explicit offset-present edge case: `StandardTerrainAdapter_LichKingProfiles_RelaxedFallbackOnlyRuns_WhenOffsetExists()`.
- Additional legacy split-ADT coverage now guards the row-end nibble behavior again: `StandardTerrainAdapter_LegacyProfiles_PreserveFinalHighNibble()` locks the viewer's sequential 4-bit decode back to baseline semantics.
- Viewer-side MH2O handling no longer stops at the first instance in a chunk. `StandardTerrainAdapter.ParseMh2o` now uses the shared `Mh2oChunk` parser and composes visible tiles from all instances, with unit coverage for exists-bitmaps and overlap precedence.
- The fixed-path WoWMuseum `development_0_0.adt` investigation showed raw A2/A3 bytes are present on disk and preserved through adapter decode. Remaining visual failures on that loose sample can still come from missing overlay diffuse assets, not just alpha decode.

### MCLQ Liquid Heights (Feb 11, 2026)
- MCLQ per-vertex heights (81 entries × 8 bytes) are absolute world Z values
- Heights can slope for waterfalls — adjacent water planes at different Z levels
- MH2O (3.3.5) was overwriting valid MCLQ data with garbage on 0.6.0 ADTs
- Fix: Skip MH2O when MCLQ liquid already found; never overwrite existing MCLQ
- WMO MLIQ liquid type: use `matId & 0x03` from MLIQ header, NOT tile flag bits

### Performance Tuning (Feb 11, 2026)
- AOI: 9×9 tiles (radius 4), forward lookahead 3, GPU uploads 8/frame
- MPQ read throttling: `SemaphoreSlim(4)` prevents I/O saturation
- Persistent tile cache: `TileLoadResult` stays in memory, re-entry is instant
- Dedup sets removed: objects always reload correctly after tile unload/reload

### WMO/MDX Coordinate System (Feb 9, 2026)
- WoW: right-handed (X=North, Y=West, Z=Up), Direct3D CW winding
- OpenGL: CCW winding for front faces
- **Fix**: Reverse winding at GPU upload + 180° Z rotation in placement
- MDX rotations: `rx = Rotation.X`, `ry = Rotation.Y` — NO axis swap
- WMO-only maps: raw WoW world coords (no MapOrigin conversion)

### WMO MLIQ Liquid Positioning (Feb 9, 2026)
- MLIQ data has inherent 90° CW misrotation (wowdev wiki)
- Fix: `axis0 = cornerX - j * tileSize`, `axis1 = cornerY + i * tileSize`
- Tile visibility: bit 3 (0x08) = hidden
- GroupLiquid=15 always → magma (old WMO "green lava" type)

### Replaceable Texture Resolution (Feb 10, 2026)
- Try ALL CDI variants, validate each resolved texture exists in MPQ
- If no DBC variant validates, fall through to model directory scan
