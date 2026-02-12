# ADT v23 Format Documentation

## Overview
The ADT v23 format is used in World of Warcraft from Warlords of Draenor (6.x) through the current expansions. It builds upon the split file system introduced in v22, adding support for improved lighting, atmospheric effects, and other graphical enhancements.

## Main Chunks
The following main chunks are defined for ADT v23 format:

| ID | Chunk | Documentation Status | Description |
|----|-------|---------------------|-------------|
| C001 | AFBO | ✅ Documented | Fog Block Object data |
| C002 | ACVT | ✅ Documented | Color Vertex data for atmospheric effects |
| C003 | ABSH | ✅ Documented | Blend Shadow data for terrain shading |
| C004 | AHDR | ✅ Documented | Terrain file header (updated from v22) |
| C005 | AVTX | ✅ Documented | Vertex data (updated from v22) |
| C006 | ANRM | ✅ Documented | Normal data (updated from v22) |
| C007 | ATEX | ✅ Documented | Texture definitions (updated from v22) |
| C008 | ADOO | ✅ Documented | Doodad (M2) object placement data (updated from v22) |
| C009 | ACNK | ✅ Documented | Chunk data - container for subchunks (updated from v22) |

## Subchunks
The subchunks are the same as ADT v22, with the following contained within ACNK chunks:

| ID | Subchunk | Parent | Documentation Status | Description |
|----|----------|--------|---------------------|-------------|
| S001 | ALYR | ACNK | ✅ Documented | Texture layer information |
| S002 | AMAP | ALYR | ✅ Documented | Alpha map data for texture blending |
| S003 | ASHD | ACNK | ✅ Documented | Shadow map data |
| S004 | ACDO | ACNK | ✅ Documented | Doodad references for chunk |

## File Structure
ADT v23 maintains the split file system from v22, with separate files for base data, texture, and objects:

1. `{mapname}_{x}_{y}.adt` - Base file containing header and references
2. `{mapname}_{x}_{y}_tex0.adt` - Texture file containing texture information
3. `{mapname}_{x}_{y}_obj0.adt` - Object file containing object placement data
4. `{mapname}_{x}_{y}_obj1.adt` - Additional object file (if needed)

## Implementation Status
- Documentation: 
  - Main Chunks: 100% Complete (9/9 chunks)
  - Subchunks: 100% Complete (4/4 subchunks)
  - Overall: 100% Complete (13/13 total chunks)
- Parser Implementation: Not Started

## Key Differences from ADT v22
- Added AFBO chunk for fog boundary objects
- Added ACVT chunk for enhanced vertex coloring and atmospheric effects
- Added ABSH chunk for enhanced shadow blending
- Updated AHDR chunk with additional fields for WoD features
- Enhanced AVTX chunk with improved height precision
- Enhanced ANRM chunk with additional reserved byte and improved precision
- Enhanced ATEX chunk with material type, specular multiplier, and new flags
- Enhanced ADOO chunk with uniqueId, phaseId, and new flags
- Enhanced ACNK chunk with heightTextureId, groundEffectId, windAnimId, and detailLayerMask
- Improved support for atmospheric effects and distant terrain rendering
- Better integration with the lighting system
- Added support for particle effects and wind animation
- Enhanced material properties and displacement mapping

## References
- Original Format Specification: [WoWDev Wiki](http://www.wowdev.wiki)
- Implementation Example: [WoWFormatLib](https://github.com/WoWFormatLib)
- Related Formats: [ADT v18](../ADT_v18/ADT_v18_index.md), [ADT v22](../ADT_v22/ADT_v22_index.md)

## Notes
- The v23 format maintains compatibility with v22 while adding features specific to newer expansions
- Many chunks are identical or very similar to v22, but with additional fields or modified structures
- Subchunks (ALYR, AMAP, ASHD, ACDO) have the same structure as v22, but with different interpretations in v23
- Subchunks (ALYR, AMAP, ASHD, ACDO) have the same structure as v22, but may have different interpretations in v23
- WoD significantly enhanced the rendering capabilities with a focus on atmospheric effects, material properties, and environmental dynamics 