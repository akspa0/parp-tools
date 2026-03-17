# System Patterns

## FourCC: Reverse on read (disk→memory), reverse on write (memory→disk). Always readable in code/logs.

## Terrain Adapters
- `AlphaTerrainAdapter` — 0.5.3 monolithic WDT
- `StandardTerrainAdapter` — 0.6.0/3.3.5 split ADTs (version-gated alpha decode)
- Both produce `TileLoadResult` → `TerrainChunkData` + placements
- WMO-only maps: `MAIN` has 0 tile flags → use WDT `MWMO`+`MODF` (raw world coords, no MapOrigin)

## Liquid Pipeline
- MCLQ (per-chunk, 0.5.3/0.6.0): absolute world Z, can slope for waterfalls
- MH2O (per-tile, 3.3.5): via MHDR offset, only when no MCLQ — **currently broken**
- MLIQ (WMO groups): matId-based type

## AOI Streaming
- `TerrainManager`: camera-driven tile load/unload, background thread pool, `SemaphoreSlim(4)`
- Persistent `_tileCache` (parsed data forever, GPU meshes only), heading-priority queue

## MDX Patterns
- GEOS: version-routed (v1300/v1400 strict, v1500 strict, guarded fallback)
- PRE2/RIBB: read fixed scalars first, parse anim tails by keyword, skip unknown at boundary
- Two-sided reflective: flip normals on `!gl_FrontFacing` before env UV + lighting
- Texture resolution: DBC primary → `CreatureDisplayInfoExtra` baked skins → `<Model>Skin.blp` fallback

## ADT Formats
- **0.5.3**: monolithic `<map>.wdt` with MPHD, MAIN, MDNM, MONM, per-tile MHDR+MCIN+MCNKs
- **0.6.0**: split `<map>.wdt` + `<map>_XX_YY.adt` with reversed FourCCs, MCNK sub-chunks via header offsets
- **3.3.5**: split root + `_obj0.adt` + `_tex0.adt` (but viewer stays on root ADT for 3.x)

## Coordinates
- WoW: RH, X=North, Y=West, Z=Up. Renderer: `rX = MapOrigin - wowY`, `rY = MapOrigin - wowX`, `rZ = wowZ`
- WMO-only maps: raw world coords (no MapOrigin). GPU: reverse winding CW→CCW, 180° Z rotation.
