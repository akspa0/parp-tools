# Alpha WDT Unified Tensor Extraction — Plan

## Problem

`extract-unified` produces 1051 tiles from 0.5.3 Azeroth but tile coordinates are incorrect — the stitched heightmap shows the continent rotated 90° CCW and duplicated onto itself. The V11 cache has 518 tiles with known-good coordinates but was built from VLM JSONs (indirect pipeline), not from the game client directly.

## Goal

One command that reads any game client's map files directly and produces a complete NPZ shard per tile: `extract-unified --client-root <dir> --map <name> --output-dir <dir>`. Works for both Alpha monolithic WDT (0.5.3-0.7.x) and retail split ADT (3.x+). No intermediate datasets, no VLM JSONs, no old caches.

## Source of Truth

MdxViewer's `AlphaTerrainAdapter` renders 0.5.3 Azeroth correctly — tile positions, heightmaps, normals, placements all match the game. Port the reading logic from there into `WowViewer.Core.IO`, which is the canonical home for format readers.

## Implementation

### 1. Port Alpha WDT tile reading to `WowViewer.Core.IO`

Existing code to reference (read-only, do not add references):
- `gillijimproject_refactor/src/MdxViewer/Terrain/AlphaTerrainAdapter.cs` — `LoadTileWithPlacements()`, coordinate system, chunk assembly
- `gillijimproject_refactor/src/gillijimproject-csharp/WowFiles/Alpha/` — `WdtAlpha`, `AdtAlpha`, `McnkAlpha` byte-level parsing
- `wow-viewer/src/viewer/WowViewer.App/AlphaEmbeddedAdtReader.cs` — already-linked reference implementation (currently used by converter but has the coordinate bug)

Create `wow-viewer/src/core/WowViewer.Core.IO/Maps/AlphaWdtReader.cs`:
- `public static bool TryReadTile(string wdtFilePath, int tileX, int tileY, out AlphaTileData? data)`
- Returns per-tile: 257×257 heights, 256×256×4 MCAL alpha, 16×16×4 MCLY texture IDs, MDDF/MODF placements with model paths and bounds, liquid data, hole mask, texture name table
- Coordinate system: match `AlphaTerrainAdapter` exactly — tileX maps to column in MAIN table (index = tileX * 64 + tileY), world coord projection verified against viewer rendering
- No archive dependency — takes a filesystem WDT path (caller extracts from MPQ if needed)

### 2. Fix coordinate verification

After porting:
- Extract 3 known tiles from Elwynn, Stormwind, and Westfall regions
- Compare against MdxViewer's rendered view of the same tiles
- Confirm height values, object placement positions match pixel-for-pixel
- Stitch 64×64 heightmap quilt and verify continent shape is correct (no rotation or duplication)

### 3. Update `extract-unified` command

- For Alpha: use `AlphaWdtReader.TryReadTile` (local to `WowViewer.Core.IO`)
- For retail: delegate to `dataset-build-v10-stage1` (already works)
- Auto-detect client type from WDT chunk signatures (MDNM/MONM present = Alpha)
- Extract WDT from MPQ using `StormLibPatchArchiveReader` (already in `WowViewer.Core.IO`) for staged clients
- Same output format for both: `{tile}_v10.npz` + `{tile}_v10_placements.json`
- Include `weak_signal_audit.txt` richness classification per tile

### 4. Add MCNR normal synthesis

Alpha 0.5.3 has MCNR data at known offset in MCNK header, but `AlphaEmbeddedAdtReader` doesn't read it. Synthesize from MCVT heights using central differences — more reliable and works for all clients.

### 5. Remove legacy references

- Remove `AlphaEmbeddedAdtReader` link from converter csproj after `AlphaWdtReader` is operational
- Remove `generate_alpha_placements.py` and `prepare_object_detection_data.py` references to VLM JSONs
- All Python scripts use unified NPZ format only

## Files to Create

| File | Purpose |
|------|---------|
| `wow-viewer/src/core/WowViewer.Core.IO/Maps/AlphaWdtReader.cs` | Alpha monolithic WDT tile reader |
| `wow-viewer/src/core/WowViewer.Core/Maps/AlphaTileData.cs` | Per-tile data model |

## Files to Modify

| File | Change |
|------|--------|
| `wow-viewer/tools/converter/WowViewer.Tool.Converter/Program.cs` | `RunExtractUnified` → use `AlphaWdtReader` instead of `AlphaEmbeddedAdtReader` |
| `wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj` | Remove `AlphaEmbeddedAdtReader` link after migration verified |

## Validation Criteria

- Stitched 64×64 heightmap quilt matches Azeroth continent shape exactly (no rotation/mirroring/duplication)
- Tile `Azeroth_30_32` has same heights, MCAL, MCLY as V11 cache `Azeroth_30_32` (V11 is the known-good reference)
- Unified shard for a retail 3.3.5 tile matches `dataset-build-v10-stage1` output byte-for-byte
- 0.5.3 Azeroth produces 518+ tiles (matching V11 count) plus additional weak-signal ocean tiles
