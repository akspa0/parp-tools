# ADT Format Documentation

## Overview
ADT (Areadata Tile) files contain terrain data for specific 64×64 yard map tiles in World of Warcraft. The format has evolved through multiple versions across expansions.

## Documentation Status

### ADT v18 Format (Classic - Current)
- Status: **100% Documented** (30/30 main chunks, 11/11 subchunks, 14/14 Legion+ LOD chunks)
- Implementation: Not started

### ADT v22/v23 Format (Cataclysm Beta only)
- Status: **100% Documented** (10/10 chunks)
- Implementation: Not started
- Note: These versions appeared only in Cataclysm beta and were not used in the final release

## Chunk Documentation

### Main Chunks

| ID | Name | Status | Description |
|----|------|--------|-------------|
| C000 | MVER | ✅ Documented | Version information |
| C001 | MHDR | ✅ Documented | Header containing flags and offsets |
| C002 | MCIN | ✅ Documented | Chunk index array |
| C003 | MTEX | ✅ Documented | Texture filename list |
| C004 | MMDX | ✅ Documented | Model filename list |
| C005 | MMID | ✅ Documented | Model filename offsets |
| C006 | MWMO | ✅ Documented | WMO filename list |
| C007 | MWID | ✅ Documented | WMO filename offsets |
| C008 | MDDF | ✅ Documented | Doodad placement information |
| C009 | MODF | ✅ Documented | Object placement information |
| C010 | MH2O | ✅ Documented | Water data (added in WotLK) |
| C011 | MFBO | ✅ Documented | Flight boundaries (added in BC) |
| C012 | MTXF | ✅ Documented | Texture flags (added in WotLK) |
| C013 | MTXP | ✅ Documented | Texture parameters (added in MoP) |
| C014 | MAMP | ✅ Documented | Texture coordinate amplification (added in Cata) |
| C015 | MTCG | ✅ Documented | Texture color gradients (added in Shadowlands) |
| C016 | MDID | ✅ Documented | Diffuse texture FileDataIDs (added in Legion) |
| C017 | MCNK | ✅ Documented | Map chunk data (container for subchunks) |
| C018 | MHID | ✅ Documented | Heightmap FileDataIDs (added in MoP) |
| C019 | MBMH | ✅ Documented | Blend mesh headers (added in MoP) |
| C020 | MBBB | ✅ Documented | Blend mesh bounding boxes (added in MoP) |
| C021 | MBNV | ✅ Documented | Blend mesh vertices (added in MoP) |
| C022 | MBMI | ✅ Documented | Blend mesh indices (added in MoP) |
| C023 | MNID | ✅ Documented | Normal texture FileDataIDs (added in Legion) |
| C024 | MSID | ✅ Documented | Specular texture FileDataIDs (added in Legion) |
| C025 | MLID | ✅ Documented | Height texture FileDataIDs (added in Legion) |
| C026 | MLDB | ✅ Documented | Low detail blend distances (added in BfA) |
| C027 | MWDR | ✅ Documented | Doodad references (added in Shadowlands) |
| C028 | MWDS | ✅ Documented | Doodad sets (added in Shadowlands) |
| C029 | MCBB | ✅ Documented | Chunk bounding boxes (added in Shadowlands) |

### Subchunks (MCNK)

| ID | Name | Status | Description |
|----|------|--------|-------------|
| S001 | MCVT | ✅ Documented | Height map vertices |
| S002 | MCCV | ✅ Documented | Vertex colors (added in WotLK) |
| S003 | MCNR | ✅ Documented | Normal vectors |
| S004 | MCLY | ✅ Documented | Texture layer definitions |
| S005 | MCRF | ✅ Documented | Doodad and object references (<Cata) |
| S006 | MCSH | ✅ Documented | Shadow map |
| S007 | MCAL | ✅ Documented | Alpha maps |
| S008 | MCLQ | ✅ Documented | Liquid data (legacy) |
| S009 | MCSE | ✅ Documented | Sound emitters |
| S010 | MCBB | ✅ Documented | Bounding box (added in MoP) |
| S011 | MCLV | ✅ Documented | Light values (added in Cata) |
| S012 | MCRD | ✅ Documented | Doodad references (added in Cata) |
| S013 | MCRW | ✅ Documented | WMO references (added in Cata) |
| S014 | MCMT | ✅ Documented | Material IDs (added in Cata) |
| S015 | MCDD | ✅ Documented | Detail doodad disable bitmap (added in Cata) |

### LOD Chunks (Legion+)

| ID | Name | Status | Description |
|----|------|--------|-------------|
| C030 | MLHD | ✅ Documented | LOD header |
| C031 | MLVH | ✅ Documented | LOD vertex heights |
| C032 | MLVI | ✅ Documented | LOD vertex indices |
| C033 | MLLL | ✅ Documented | LOD level list |
| C034 | MLND | ✅ Documented | LOD node data (quad tree) |
| C035 | MLSI | ✅ Documented | LOD skirt indices |
| C036 | MLLD | ✅ Documented | LOD liquid data |
| C037 | MLLN | ✅ Documented | LOD liquid nodes |
| C038 | MLLV | ✅ Documented | LOD liquid vertices |
| C039 | MLLI | ✅ Documented | LOD liquid indices |
| C040 | MLMD | ✅ Documented | LOD model definitions |
| C041 | MLMX | ✅ Documented | LOD model extents |
| C042 | MBMB | ✅ Documented | Unknown Legion+ blend mesh data |
| C043 | MLMB | ✅ Documented | Unknown BfA+ data |

## Version Differences

The ADT format has evolved across expansions:

- **Classic (1.x)**: Original format with basic terrain features
- **The Burning Crusade (2.x)**: Added MFBO for flight boundaries
- **Wrath of the Lich King (3.x)**: Added MH2O for improved water, MTXF for texture flags, MCCV for vertex coloring
- **Cataclysm (4.x)**: Introduced split files, added MAMP, MCLV, MCRD, MCRW, MCMT, MCDD
- **Mists of Pandaria (5.x)**: Added MTXP for texture parameters, MCBB, and blend mesh chunks (MBMH, MBBB, MBNV, MBMI) for improved terrain/object transitions
- **Warlords of Draenor (6.x)**: Further refinements to existing structures
- **Legion (7.x)**: Added FileDataID chunks (MDID, MNID, MSID, MLID) replacing filenames, extensive LOD system (ML* chunks)
- **Battle for Azeroth (8.x)**: Added MLDB for LOD blending
- **Shadowlands (9.x)**: Added MTCG for texture color gradients, MWDR/MWDS for improved doodad referencing

## Format Relationships

- All retail ADT files use version 18 (MVER=18), regardless of expansion
- ADT files can be either monolithic or split:
  - **Monolithic format** (used primarily pre-Cataclysm):
    - Single `mapname_xx_yy.adt` containing all data
  - **Split format** (used from Cataclysm onwards, still v18):
    - `mapname_xx_yy.adt`: Main file with headers
    - `mapname_xx_yy_tex0.adt`: Texture data
    - `mapname_xx_yy_obj0.adt`: Object data
    - `mapname_xx_yy_lod.adt`: Level of detail data
- The v22/v23 formats appeared only in Cataclysm beta and were not used in the final release

## Implementation Notes

- Many chunks are optional, especially newer ones
- Features dependent on expansion-specific chunks should have fallback behavior
- Water rendering has changed significantly across versions
- Texture handling differs between early v18 (filenames) and modern v18 (FileDataIDs)
- Implementation should handle both monolithic and split files with common interfaces
- Unit tests should verify parsing against real game files for all versions
- Blend mesh chunks (MBMH, MBBB, MBNV, MBMI) work together to create smooth transitions between terrain and objects
- FileDataID chunks (MDID, MNID, MSID, MLID) form a physically-based rendering material system
- Doodad management improved in Shadowlands with MWDR/MWDS for more efficient object placement
- LOD system in Legion+ provides multiple levels of detail for efficient distant terrain rendering

## Next Steps

1. Begin implementation of ADT parsers
2. Add serialization support for editing and saving
3. Develop terrain rendering utilities
4. Create diagram showing chunk relationships
5. Document LOD file formats 