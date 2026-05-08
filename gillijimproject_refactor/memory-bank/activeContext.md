# ACTIVE CONTEXT — V14 Branch (V11 Reset)

## BRANCH
`v0.4.9-strict-guards` forked from `971fff2` on 2026-05-06.

## wow-viewer Alpha WDT Reading — Phase B Validation DONE

Phase B (Alpha WDT validation) is effectively complete. Key fixes landed 2026-05-07:

### CRITICAL BUG FIXED: MCLY Chunk Header Not Stripped
**File**: `AlphaWdtReader.cs:421-425`
The MCLY subchunk in Alpha format stores an 8-byte chunk header (FourCC + size) before the layer entry array. The reader was not skipping this header, causing:
- `texIds[0]` = bytes of "MCLY" FourCC → garbage texture ID
- `mclyFlags` = chunk size field → wrong alpha format detection
- `layerAlphaOff` = wrong MCAL offset → all alpha data read from wrong positions → "scrambled TV signal" noise

**Fix**: Skip 8 bytes (`mclyOffset += 8`) before reading layer entries, matching the reference implementation (`McnkAlpha.cs` uses `new Chunk()` which auto-strips the header).

### METADATA FIX: Missing AvailableSignals
**File**: `AlphaTensorPackBuilder.cs:51-58`
`mcly_texture_ids`, `mcly_layer_mask`, `mcal_alpha_pack_256` signals were not being added to `AvailableSignals` even though the arrays were correctly serialized into the NPZ.

### VERIFIED: NPZ Pipeline Produces Correct Data
Tested against `0_5_3_3368` staged client, map `Azeroth`, tile (32,32):
- `extract-unified` reads WDT from per-map MPQ (`Azeroth.wdt.MPQ`)
- NPZ contains all 8 signals: height_257, height_65, height_17, mcly_texture_ids, mcly_layer_mask, mcal_alpha_pack_256, mcnr_normal_xyz, mcsh_shadow_mask_256, minimap_rgb_256, hole_mask_16
- Alpha data shows real blend weights (layers 1-3 with varying means)
- 10 unique texture IDs, all 256 chunks have 4 layers
- Minimap orientation matches expected Alpha 0.5.3 coordinate convention

### FIXED: Tile Assembly Coordinate Convention (cx/cy Swap)
The flat tile-level arrays (heightmap 257×257, alpha 1024×1024→256×256, shadow 1024×1024, etc.) in the NPZ pipeline use:
- `cx` (= IndexX from MCNK header offset 0x04) → **column** / X direction / horizontal
- `cy` (= IndexY from MCNK header offset 0x08) → **row** / Y direction / vertical

Storage: `heightmap[cy * 16 + sampleY, cx * 16 + sampleX]` (note: row first, col second).
Slice: same convention for reading back chunks.
Alpha: `alphaPack[cy * 64 + ay, cx * 64 + ax, l]` (row first, col second).
Texture IDs: `texIds[cy, cx, l]` (IndexY→row first, IndexX→col second).

The MCNK header field naming is misleading: `IndexX` is the column (X) index in the tile's 16×16 chunk grid, and `IndexY` is the row (Y) index. The reference MdxViewer uses `chunkX = IndexX` and `chunkY = IndexY` in its local variables, but the world position formula (`worldX` depends on `chunkY`, `worldY` depends on `chunkX`) applies a coordinate swizzle for the WoW world system — this swizzle happens at render time, NOT at data storage time. For flat NPZ arrays, the natural storage convention (IndexX→col, IndexY→row) produces correct orientation, matching the game minimap.

## WHAT WORKS
- `extract-unified` for Alpha 0.5.3 monolithic WDT: full MCVT, MCNR, MCLY, MCAL, MCSH, MCLQ extraction
- NPZ shard generation with all terrain signals
- Metadata JSON with correct AvailableSignals set
- Per-map MPQ archive loading (0.5.3 `*.wdt.MPQ` files in `World/Maps/`)

## WHAT BROKE (archive path, DONT USE)
- `--client-root` mode for old dataset-build (pre-harvest tool)
- `build_v10_2_dataset.py`, `train_v10_2_terrain_synth.py` — dead code
- Shadow masks — never exist on minimap tiles, removed from channel list

## KEY FILES — wow-viewer Library
- Domain types: `wow-viewer/src/core/WowViewer.Core/Maps/`
- IO readers: `wow-viewer/src/core/WowViewer.Core.IO/Maps/`
  - `AlphaWdtReader.cs` — main Alpha WDT parser (MCVT/MCNR/MCLY/MCAL/MCSH/MCLQ)
  - `AlphaTensorPackBuilder.cs` — AlphaTileData → TerrainTileTensorPack bridge
  - `AlphaTerrainAdapter.cs` — AlphaTileData → TerrainChunkData bridge
  - `NpzTileSerializer.cs` — TerrainTileTensorPack → NPZ serialization
- NativeMpqService: `wow-viewer/src/core/WowViewer.Core.IO/Files/NativeMpqService.cs`
- Harvest tool: `wow-viewer/tools/harvest/WowViewer.Tool.Harvest/Program.cs`
- Library completeness plan: `wow-viewer/docs/architecture/wow-viewer-library-completeness-plan-2026-05-06.md`
- Format spec: `gillijimproject_refactor/docs/ADT_WDT_Format_Specification.md`

## NEXT
1. Wire AlphaTileData.ToPlacementCatalog into harvest output (`--export-placements`)
2. Test Alpha 0.5.5 prototype ADT (8-byte padding after MCNK header)
3. Test Alpha 0.6.0 split ADT through AdtTensorPackBuilder
4. Extract training shards via harvest tool on staged clients (0.5.3 Azeroth confirmed working)
5. Consider adding tileset tile BLP → synthetic minimap for data-harvester pipeline
