# Progress

## Working (0.5.3 through 0.12 — fully usable)

- Terrain: Alpha WDT monolithic + 0.6.0 split ADT + WMO-only maps + MCSH shadows + alpha debug
- AOI streaming: 9×9, directional lookahead, persistent cache, MPQ throttling
- Terrain liquid (MCLQ): per-vertex heights, waterfall slopes
- MDX: two-pass rendering, skeletal animation (GPU skinning, compressed quats), PRE2 particles, geoset anim, specular, sphere env map
- WMO v14: 4-pass rendering, doodads, MLIQ liquid, shared static shaders
- DBC: lighting, area names, replaceable textures, taxi paths, area POIs
- WDL preview + spawn selection (Alpha 0.5.3 only)
- VLM: dataset load/generate/minimap
- Asset catalog: SQL dump → browse/filter → JSON+GLB+screenshot export
- Build/release: `dotnet publish -r win-x64 --self-contained`, GitHub Actions CI

## 3.3.5 WotLK: IN PROGRESS — NOT USABLE

- **Terrain texturing: BROKEN** — runtime layer decode/blending still visually wrong despite MCAL/MCCV fixes
- **MH2O liquid: code-level fix applied, NOT runtime-verified**
- Split ADT loading + MPHD flags parsed, M2→MdxFile adapter works
- Patch MPQ priority + BZip2 decompression working

## In Progress / Partial

- **Terrain recovery**: Wave 1 topology rollback pending (fused alpha+shadow tile pass)
- **Alpha debug overlay**: blocks chunk/tile overlays — terrain diagnosis regression
- WMO stream reload: cached-null fix applied, needs runtime verify
- MDX: no GPU skinning vertex attributes yet, no anim UI, no UV animation, no billboards
- WMO: culling too aggressive from inside, no v14-16 lightmap/MOCV
- Skybox: backdrop pass works, no DBC/WMO-driven metadata yet
- Lava type mapping still broken (green)

## Known Rendering Bugs

| Bug | Status |
|-----|--------|
| 3.x terrain texturing broken at runtime | 🔧 Active investigation |
| Alpha debug hides chunk/tile overlays | 🔧 Needs fix before diagnosis |
| WMO stained glass → wrong geometry | ❌ Root cause unknown |
| MDX cylindrical texture stretching | ❌ Root cause unknown |
| MDX sphere env orientation (dome) | 🔧 Patched, needs visual confirm |

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
