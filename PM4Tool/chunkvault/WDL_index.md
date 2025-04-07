# World Data Low-resolution (WDL) Format Documentation

## Related Documentation
- [WDT Format](WDT_index.md) - Parent map format
- [ADT Format](ADT_index.md) - High detail terrain format
- [Common Types](common/types.md) - Shared data structures
- [Format Relationships](relationships.md) - Dependencies and connections

## Overview
The WDL (World Data Low-resolution) format contains simplified, low-resolution height map data used for rendering distant terrain in World of Warcraft. This format plays a crucial role in the game's rendering pipeline by providing efficient terrain data for areas that are far from the player, improving performance while maintaining visual quality.

Unlike the full-resolution ADT files, which provide detailed terrain for nearby areas, WDL files contain only the essential height data needed for distant terrain rendering. This optimization allows the game to draw terrain far into the distance without the memory and processing overhead of loading full-detail terrain data.

## File Structure
WDL files follow the standard chunk format used in World of Warcraft files, with each chunk identified by a 4-byte identifier and a 4-byte size value. The file represents a 64×64 grid of map areas, similar to the structure of WDT files but with simplified terrain data.

The WDL file structure supports both efficient access to specific map areas (through the MAOF chunk) and direct access to height data (through the MAHO chunk). This dual access pattern allows the client to quickly load and render appropriate terrain data based on the player's position and view distance.

## File Versions
| Version | Description |
|---------|-------------|
| 18 | Original WDL format |
| 22+ | Updated versions with potential format modifications |

## WDL Chunks
| Chunk ID | Name | Description | Status | Documentation |
|----------|------|-------------|--------|---------------|
| MVER | Version | Identifies the version of the WDL file format | ✅ Documented | [chunks/WDL/MVER.md](chunks/WDL/MVER.md) |
| MWMO | Map WMO | Contains filenames of global WMO models | ✅ Documented | [chunks/WDL/MWMO.md](chunks/WDL/MWMO.md) |
| MWID | Map WMO Index | Contains indices referencing WMO filenames | ✅ Documented | [chunks/WDL/MWID.md](chunks/WDL/MWID.md) |
| MODF | Map Object Definition | Contains WMO placement information | ✅ Documented | [chunks/WDL/MODF.md](chunks/WDL/MODF.md) |
| MAOF | Map Area Offset | Contains offsets to map area data (MARE chunks) | ✅ Documented | [chunks/WDL/MAOF.md](chunks/WDL/MAOF.md) |
| MARE | Map Area | Contains information about a specific map area | ✅ Documented | [chunks/WDL/MARE.md](chunks/WDL/MARE.md) |
| MAOC | Map Area Compression | Contains compression information for a map area | ✅ Documented | [chunks/WDL/MAOC.md](chunks/WDL/MAOC.md) |
| MAHE | Map Height | Contains heightmap data for a map area | ✅ Documented | [chunks/WDL/MAHE.md](chunks/WDL/MAHE.md) |
| MAHO | Map Height Offset | Contains offsets to heightmap data (MAHE chunks) | ✅ Documented | [chunks/WDL/MAHO.md](chunks/WDL/MAHO.md) |
| MAIN | Main tile table | Contains main tile table information | ✅ Documented | [chunks/WDL/MAIN.md](chunks/WDL/MAIN.md) |

## Relationship to WDT
The WDL format is closely related to the WDT format, sharing many structural similarities:

- Both use a 64×64 grid to represent the world map
- Both can contain global object definitions (WMO models)
- WDL provides low-resolution terrain data for the same geographic areas defined in WDT
- Every area in WDT that has terrain (ADT files) typically has corresponding low-resolution data in WDL

The main difference is that WDL contains simplified height data optimized for distant rendering, while WDT primarily acts as a reference table to more detailed ADT files.

## Relationship to ADT
The WDL format provides a simplified version of the terrain data found in ADT files:

- ADT: High-resolution terrain (145×145 height points per map area)
- WDL: Low-resolution terrain (typically 17×17 height points per map area)

This significant reduction in detail is appropriate for distant terrain rendering, where the high resolution of ADT files is not necessary and would be too memory-intensive.

## Low-Resolution Rendering
The WDL format enables efficient rendering of distant terrain through:

1. **Simplified geometry**: Terrain is represented with far fewer vertices than full ADT data
2. **Direct height access**: Height data can be accessed directly via the MAHO chunk
3. **Metadata-driven rendering**: MARE chunks provide additional information to aid in rendering decisions
4. **Strategic object placement**: Only the most significant WMO objects are included for distant visibility

This approach allows the game client to render terrain all the way to the horizon without overwhelming system resources, creating a seamless visual experience as players traverse the world.

## Implementation Status
✅ **Fully Implemented** - Core parsing system and all chunks are implemented

### Core Components
- ✅ `WdlFile` - Main WDL file parser
- ✅ `WdlTile` - Low detail tile management
- ✅ `WdlHeightmap` - Height data handling
- ✅ `WdlAssetLoader` - Asset reference system

### Features
- ✅ Low detail terrain
- ✅ Height data processing
- ✅ Asset reference tracking
- ✅ Validation reporting
- ⏳ Terrain visualization (Planned)
- ⏳ Editing tools (Planned)

## Implemented Chunks

### Main Chunks
| Chunk | Status | Description | Documentation |
|-------|--------|-------------|---------------|
| MVER | ✅ | Version information | [chunks/WDL/MVER.md](chunks/WDL/MVER.md) |
| MWMO | ✅ | WMO filenames | [chunks/WDL/MWMO.md](chunks/WDL/MWMO.md) |
| MWID | ✅ | WMO indices | [chunks/WDL/MWID.md](chunks/WDL/MWID.md) |
| MAOF | ✅ | Map area offsets | [chunks/WDL/MAOF.md](chunks/WDL/MAOF.md) |
| MARE | ✅ | Map area data | [chunks/WDL/MARE.md](chunks/WDL/MARE.md) |
| MAOC | ✅ | Map area compression | [chunks/WDL/MAOC.md](chunks/WDL/MAOC.md) |
| MAHO | ✅ | Map hole information | [chunks/WDL/MAHO.md](chunks/WDL/MAHO.md) |
| MAIN | ✅ | Main tile table | [chunks/WDL/MAIN.md](chunks/WDL/MAIN.md) |

Total Progress: 8/8 chunks implemented (100%)

## Implementation Notes
- Used for distant terrain rendering
- Lower resolution than ADT files
- Shares structure with WDT format
- Optimized for performance

## File Structure
```
<MapName>.wdl        - Low detail terrain data
<MapName>.wdt        - Parent map definition
<MapName>_xx_yy.adt  - High detail terrain tiles
```

## Next Steps
1. Implement terrain visualization
2. Add editing tools
3. Create format conversion utilities
4. Add validation tools

## References
- [WDL Format Specification](../docs/WDL.md)
- [Map Format Overview](../docs/Map_Formats.md) 