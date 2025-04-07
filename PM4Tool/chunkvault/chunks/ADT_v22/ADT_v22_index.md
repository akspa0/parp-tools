# ADT v22 Format Documentation

## Overview
The ADT v22 format is used in World of Warcraft from Cataclysm (4.x) through Mists of Pandaria (5.x). It introduces a split file system where each ADT tile is divided into multiple files:
- Base file: Contains header information and references to other chunks
- Texture file: Contains texture data for the tile
- Objects file: Contains object placement data

## Main Chunks
The following main chunks are defined for ADT v22 format:

| ID | Chunk | Documentation Status | Description |
|----|-------|---------------------|-------------|
| C001 | AHDR | ✅ Documented | Terrain file header |
| C002 | AVTX | ✅ Documented | Vertex data |
| C003 | ANRM | ✅ Documented | Normal data |
| C004 | ATEX | ✅ Documented | Texture definitions |
| C005 | ADOO | ✅ Documented | Doodad (M2) object placement data |
| C006 | ACNK | ✅ Documented | Chunk data (container for subchunks) |

## Subchunks
The following subchunks are contained within ACNK chunks:

| ID | Subchunk | Parent | Documentation Status | Description |
|----|----------|--------|---------------------|-------------|
| S001 | ALYR | ACNK | ✅ Documented | Texture layer information |
| S002 | AMAP | ALYR | ✅ Documented | Alpha map data for texture blending |
| S003 | ASHD | ACNK | ✅ Documented | Shadow map data |
| S004 | ACDO | ACNK | ✅ Documented | Doodad references for chunk |

## File Structure
ADT v22 introduced the split file system, with separate files for base data, texture, and objects:

1. `{mapname}_{x}_{y}.adt` - Base file containing header and references
2. `{mapname}_{x}_{y}_tex0.adt` - Texture file containing texture information
3. `{mapname}_{x}_{y}_obj0.adt` - Object file containing object placement data
4. `{mapname}_{x}_{y}_obj1.adt` - Additional object file (if needed)

## Implementation Status
- Documentation: 100% Complete (10/10 chunks)
- Parser Implementation: Not Started

## References
- Original Format Specification: [WoWDev Wiki](http://www.wowdev.wiki)
- Implementation Example: [WoWFormatLib](https://github.com/WoWFormatLib)
- Related Formats: [ADT v18](../ADT_v18/ADT_v18_index.md), [ADT v23](../ADT_v23/ADT_v23_index.md)

## Notes
- Compared to ADT v18, the v22 format uses "A" prefixes for chunk names instead of "M" prefixes
- The v22 format introduces a more modular approach with separate files for different data types
- The ACNK chunk system replaces the MCNK system from v18, with a similar structure but different organization 