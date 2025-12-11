# Active Context

## Current Focus: Split ADT → Monolithic 3.3.5 Conversion (Dec 10, 2025)

### Problem Statement
The development map source files are in **Cataclysm split format** (root + `_obj0.adt` + `_tex0.adt`), but we need **monolithic 3.3.5 ADTs** for the WotLK client. Our current `AdtPatcher.MergeSplitAdt()` produces files that are ~500KB smaller than reference files.

### Key Discovery
- **Source files**: 2010 official development map files (split format)
- **Reference files**: `test_data/WoWMuseum/335-dev/` contains monolithic 3.3.5 ADTs
- **333 tiles have `_tex0.adt`** files; some tiles are missing texture data entirely
- **MTEX chunk is empty (0 bytes)** in our merged output for tiles without `_tex0.adt`

### Comparison: `development_1_1.adt`
| Chunk | Reference | Ours | Issue |
|-------|-----------|------|-------|
| **MTEX** | 367 bytes | **0 bytes** | ❌ No texture paths |
| **MCNK** | 1,253,060 | 467,472 | -785KB (missing MCLY/MCAL) |
| MDDF | 3,096 | 2,376 | -720 bytes |
| MODF | 448 | 128 | -320 bytes |

### Root Cause Analysis
1. Source `development_1_1.adt` is split format (no MTEX at root level)
2. No `development_1_1_tex0.adt` exists → no texture data to merge
3. Our merger correctly handles tiles WITH `_tex0`, but can't create data that doesn't exist

### Reference Libraries for Solution
These libraries in `lib/` provide proven ADT parsing/writing:

1. **MapUpconverter** (`lib/MapUpconverter/`)
   - Converts WotLK monolithic → Legion/BfA split format
   - Uses `Warcraft.NET.Files.ADT.Terrain.Wotlk.Terrain` for parsing
   - Key files: `ADT/Tex0.cs`, `ADT/Root.cs`, `ADT/Obj0.cs`

2. **Warcraft.NET** (`lib/Warcraft.NET/`)
   - Full ADT chunk definitions for all versions
   - `Files/ADT/Terrain/Wotlk/Terrain.cs` — monolithic 3.3.5 structure
   - `Files/ADT/Terrain/Wotlk/MCNK.cs` — chunk with all subchunks

3. **WoWFormatLib** (`lib/wow.tools.local/WoWFormatLib/`)
   - Additional format handling utilities

### Strategy for Next Session
1. **Use Warcraft.NET** to parse split files into structured objects
2. **Combine** root + obj0 + tex0 data into `Wotlk.Terrain` structure
3. **Serialize** using Warcraft.NET's writer to produce correct monolithic ADT
4. **Validate** against WoWMuseum reference files

## Completed This Session
- ✅ MCCV painting from minimap images (`MccvPainter.cs`)
- ✅ Wired MCCV into WDL→ADT generation pipeline
- ✅ Generated 1496 ADTs with MCCV from development minimaps
- ✅ Identified texture merger issue (missing `_tex0` files)
- ✅ Located reference libraries for proper ADT handling

## Key Insight
> The source development files are incomplete — some tiles lack `_tex0.adt`. For complete texture support, we may need to use Warcraft.NET to properly parse and merge the split format, or accept that some tiles will have no textures.
