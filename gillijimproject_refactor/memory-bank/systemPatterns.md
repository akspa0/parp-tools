# System Patterns

## FourCC Handling (CRITICAL)

### The Rule
```
READ:  Reverse on disk → Forward in memory (XETM → MTEX)
WRITE: Forward in memory → Reverse on disk (MTEX → XETM)
```

## Terrain Adapter Pattern

### ITerrainAdapter Interface
Two implementations for different WDT/ADT formats:
- **AlphaTerrainAdapter** — Alpha 0.5.3 monolithic WDT (all tiles in one file)
- **StandardTerrainAdapter** — 0.6.0 / 3.3.5 split ADTs (per-tile files from MPQ)

Both produce `TileLoadResult` with `TerrainChunkData` + `MddfPlacement` + `ModfPlacement`.

### WMO-Only Maps
- Detected when `MAIN` chunk has 0 tiles with flags set
- WDT contains `MWMO` (name) + `MODF` (placement) chunks
- Both adapters parse these from WDT and use raw WoW world coords (no MapOrigin conversion)

### Liquid Pipeline
- **MCLQ** (per-chunk, 0.5.3/0.6.0): Extracted in terrain adapter, per-vertex heights are absolute world Z
- **MH2O** (per-tile, 3.3.5): Parsed via MHDR offset, only when no MCLQ found
- **MLIQ** (WMO groups): Parsed in WmoRenderer, type from `matId & 0x03`
- MCLQ heights can slope for waterfalls — always use per-vertex data when available

### AOI Streaming
- `TerrainManager` handles tile load/unload based on camera position
- Background thread pool for MPQ reads, throttled by `SemaphoreSlim(4)`
- Persistent `_tileCache` keeps parsed data forever — re-entry is instant (GPU meshes only)
- Priority-sorted load queue: tiles ahead of camera heading load first

## MDX Alpha 0.5.3 Patterns

### GEOS Sub-Chunk Parsing
- **Padding Aware**: Always scan for 4-byte UTF-8 tags.
- **Null Safety**: Avoid fixed-offset jumps between sub-chunks. Padding can be 0-12 bytes.
- **UVAS (v1300)**: Count=1 contains raw UV data directly. No `UVBS` tag.
- **Version-Routed GEOS Strategy**: Prefer strict parser routes by MDX version (`v1300/v1400` classic tagged layout, `v1500` packed two-pass layout), then fall back to adaptive parser only on strict-parse failure.

### PRE2 / RIBB Parsing Pattern
- Read full fixed scalar payload blocks first (including known extended fields) to keep stream alignment stable.
- Parse/skip known animation-vector tails by keyword (`KP2*`, `KVIS`, `KR*`) instead of jumping blindly to emitter end.
- Preserve compatibility by safely skipping unknown tail chunks at emitter boundary.

### Two-Sided Reflective Shading Pattern
- For sphere env map + specular on two-sided geometry, flip normals on backfaces in fragment shader (`!gl_FrontFacing`) before env UV generation and lighting.
- Use face-corrected view-space normal for env UV derivation to avoid inward-facing reflections on dome-like meshes.

### Texture Resolution (DBC + Fallback)
- **Primary Source**: `DbcService` (Resolves `ModelID` → `TextureVariation` via `CreatureDisplayInfo`).
- **Baked Skins**: Queries `CreatureDisplayInfoExtra` for `BakeName` when `ExtraId > 0`.
- **Legacy Fallback**: If DBC lookup fails, default to `<ModelName>Skin.blp` or local directory scan.

## ADT Structure

### Alpha 0.5.3 (Monolithic)
```
<map>.wdt — Monolithic tileset:
  MPHD, MAIN, MDNM, MONM, then per-tile MHDR+MCIN+MCNKs
```

### 0.6.0 (Split, reversed FourCC)
```
<map>.wdt              — MPHD, MAIN, [MWMO+MODF for WMO-only]
<map>_XX_YY.adt        — MHDR, MCIN, MTEX, MDDF, MODF, MCNKs
MCNK sub-chunks: header offsets to MCVT, MCNR, MCLY, MCAL, MCSH, MCLQ
MCLQ: packed instances (0x2D4 bytes each), one per liquid flag bit
```

### LK 3.3.5 (Split)
```
<map>_XX_YY.adt      — Root (terrain + MH2O)
<map>_XX_YY_obj0.adt — Objects (MDDF, MODF)
<map>_XX_YY_tex0.adt — Textures (MCAL, MCSH)
```

## Coordinate System

### World Coordinates
- WoW: right-handed, X=North, Y=West, Z=Up, Direct3D CW winding
- Renderer: `rendererX = MapOrigin - wowY`, `rendererY = MapOrigin - wowX`, `rendererZ = wowZ`
- WMO-only maps: raw WoW world coords (no MapOrigin conversion)

### GPU Fix
- Reverse triangle winding at upload (CW→CCW for OpenGL)
- 180° Z rotation in all placement transforms
