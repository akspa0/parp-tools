# World Data Low-resolution (WDL) Format Documentation

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
| Chunk ID | Name | Description | Status |
|----------|------|-------------|--------|
| MVER | Version | Identifies the version of the WDL file format | ✅ Documented |
| MWMO | Map WMO | Contains filenames of global WMO models | ✅ Documented |
| MWID | Map WMO Index | Contains indices referencing WMO filenames | ✅ Documented |
| MODF | Map Object Definition | Contains WMO placement information | ✅ Documented |
| MAOF | Map Area Offset | Contains offsets to map area data (MARE chunks) | ✅ Documented |
| MARE | Map Area | Contains information about a specific map area | ✅ Documented |
| MAHE | Map Height | Contains heightmap data for a map area | ✅ Documented |
| MAHO | Map Height Offset | Contains offsets to heightmap data (MAHE chunks) | ✅ Documented |

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
All 8 out of 8 WDL chunks have been documented (100% complete):

- ✅ MVER (L001) - Version identifier
- ✅ MWMO (L002) - Global WMO filenames
- ✅ MWID (L003) - Global WMO indices
- ✅ MODF (L004) - Global WMO placement
- ✅ MAOF (L005) - Map area offset table
- ✅ MARE (L006) - Map area information
- ✅ MAHE (L007) - Heightmap data
- ✅ MAHO (L008) - Heightmap offset table

## Next Steps
1. Update the ChunkVault main index to reflect complete WDL documentation
2. Implement parsers for the WDL format
3. Create sample visualization tools for WDL height data
4. Begin documentation of the next format according to the expansion plan (WDB/DBC formats)

## Key Documentation Notes
- The WDL format exemplifies efficient memory usage through its use of low-resolution data for distant rendering
- The dual access patterns (MAOF→MARE→MAHE and direct MAHO→MAHE) provide flexibility in data access
- The relationship between WDL and ADT formats shows how level-of-detail techniques are implemented in World of Warcraft
- The WDL format maintains the same coordinate system and geographical representation as WDT and ADT formats 