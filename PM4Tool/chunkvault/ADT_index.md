# ADT Format Documentation

## Related Documentation
- [ADT v22 Format](ADT_v22_index.md) - Version 22 changes and additions
- [ADT v23 Format](ADT_v23_index.md) - Version 23 changes and additions
- [MCNK Structure](mcnk_structure.md) - Detailed terrain chunk documentation
- [WDT Format](WDT_index.md) - Parent world template format
- [WDL Format](WDL_index.md) - Low detail terrain companion format
- [Common Types](common/types.md) - Shared data structures
- [Format Relationships](relationships.md) - Dependencies and connections

## Implementation Status
✅ **Fully Implemented** - Core parsing system and all chunks are implemented

### Core Components
- ✅ `AdtFileBase` - Base class for all ADT file types
- ✅ `AdtFile` - Main ADT file parser
- ✅ `AdtTextureFile` - Texture data parser (split format)
- ✅ `AdtObjectFile` - Object data parser (split format)
- ✅ `AdtLodFile` - LOD data parser (Legion+)
- ✅ `AdtFileSet` - Manager for related ADT files

### Features
- ✅ Split format support (Cataclysm+)
- ✅ FileDataID support (8.1+)
- ✅ Asset reference tracking
- ✅ Validation reporting
- ⏳ Visualization (Planned)
- ⏳ Terrain rendering (Planned)
- ⏳ Collision detection (Planned)
- ⏳ Editing tools (Planned)

## Implemented Chunks

### Main Chunks (30/30 Complete)
| Chunk ID | Status | Description | Documentation |
|----------|--------|-------------|---------------|
| MVER | ✅ | Version information | [chunks/ADT/MVER.md](chunks/ADT/MVER.md) |
| MHDR | ✅ | Header containing flags and offsets | [chunks/ADT/MHDR.md](chunks/ADT/MHDR.md) |
| MCIN | ✅ | Chunk index array | [chunks/ADT/MCIN.md](chunks/ADT/MCIN.md) |
| MTEX | ✅ | Texture filename list | [chunks/ADT/MTEX.md](chunks/ADT/MTEX.md) |
| MMDX | ✅ | Model filename list | [chunks/ADT/MMDX.md](chunks/ADT/MMDX.md) |
| MMID | ✅ | Model filename offsets | [chunks/ADT/MMID.md](chunks/ADT/MMID.md) |
| MWMO | ✅ | WMO filename list | [chunks/ADT/MWMO.md](chunks/ADT/MWMO.md) |
| MWID | ✅ | WMO filename offsets | [chunks/ADT/MWID.md](chunks/ADT/MWID.md) |
| MDDF | ✅ | Doodad placement information | [chunks/ADT/MDDF.md](chunks/ADT/MDDF.md) |
| MODF | ✅ | Object placement information | [chunks/ADT/MODF.md](chunks/ADT/MODF.md) |
| MH2O | ✅ | Water data | [chunks/ADT/MH2O.md](chunks/ADT/MH2O.md) |
| MFBO | ✅ | Flight boundaries | [chunks/ADT/MFBO.md](chunks/ADT/MFBO.md) |
| MTXF | ✅ | Texture flags | [chunks/ADT/MTXF.md](chunks/ADT/MTXF.md) |
| MTXP | ✅ | Texture parameters | [chunks/ADT/MTXP.md](chunks/ADT/MTXP.md) |
| MAMP | ✅ | Texture coordinate amplification | [mcnk_structure.md](mcnk_structure.md#mamp-texture-coordinate-amplification) |
| MTCG | ✅ | Texture color gradients | [mcnk_structure.md](mcnk_structure.md#mtcg-texture-color-gradients) |
| MCNK | ✅ | Map chunk data | [mcnk_structure.md](mcnk_structure.md#mcnk-header) |
| MDID | ✅ | Diffuse texture FileDataIDs | [mcnk_structure.md](mcnk_structure.md#mdid-diffuse-texture-filedataids) |
| MHID | ✅ | Heightmap FileDataIDs | [mcnk_structure.md](mcnk_structure.md#mhid-heightmap-filedataids) |
| MBMH | ✅ | Blend mesh headers | [mcnk_structure.md](mcnk_structure.md#mbmh-blend-mesh-headers) |
| MBBB | ✅ | Blend mesh bounding boxes | [mcnk_structure.md](mcnk_structure.md#mbbb-blend-mesh-bounding-boxes) |
| MBNV | ✅ | Blend mesh vertices | [mcnk_structure.md](mcnk_structure.md#mbnv-blend-mesh-vertices) |
| MBMI | ✅ | Blend mesh indices | [mcnk_structure.md](mcnk_structure.md#mbmi-blend-mesh-indices) |
| MNID | ✅ | Normal texture FileDataIDs | [mcnk_structure.md](mcnk_structure.md#mhid-heightmap-filedataids) |
| MSID | ✅ | Specular texture FileDataIDs | [mcnk_structure.md](mcnk_structure.md#msid-specular-texture-filedataids) |
| MLID | ✅ | Height texture FileDataIDs | [mcnk_structure.md](mcnk_structure.md#mlid-height-texture-filedataids) |
| MLDB | ✅ | Low detail blend distances | [mcnk_structure.md](mcnk_structure.md#mldb-low-detail-blend-distances) |
| MWDR | ✅ | Doodad references | [mcnk_structure.md](mcnk_structure.md#mwdr-doodad-references) |
| MWDS | ✅ | Doodad sets | [mcnk_structure.md](mcnk_structure.md#mwds-doodad-sets) |
| MCBN | ✅ | Collision data | [mcnk_structure.md](mcnk_structure.md#mcbn-collision-data) |

### Subchunks (15/15 Complete)
| Chunk ID | Status | Description | Documentation |
|----------|--------|-------------|---------------|
| MCVT | ✅ | Height map vertices | [mcnk_structure.md](mcnk_structure.md#mcvt-vertex-heights) |
| MCNR | ✅ | Normal vectors | [mcnk_structure.md](mcnk_structure.md#mcnr-normals) |
| MCLY | ✅ | Texture layer definitions | [mcnk_structure.md](mcnk_structure.md#mcly-texture-layers) |
| MCRF | ✅ | Doodad and object references | [mcnk_structure.md](mcnk_structure.md#mcrf-references) |
| MCSH | ✅ | Shadow map | [mcnk_structure.md](mcnk_structure.md#mcsh-shadow-map) |
| MCAL | ✅ | Alpha maps | [mcnk_structure.md](mcnk_structure.md#mcal-alpha-maps) |
| MCLQ | ✅ | Liquid data (legacy) | [mcnk_structure.md](mcnk_structure.md#mclq-legacy-liquid) |
| MCSE | ✅ | Sound emitters | [mcnk_structure.md](mcnk_structure.md#mcse-sound-emitters) |
| MCCV | ✅ | Vertex colors | [mcnk_structure.md](mcnk_structure.md#mccv-vertex-colors) |
| MCLV | ✅ | Light values | [mcnk_structure.md](mcnk_structure.md#mclv-light-values) |
| MCBB | ✅ | Bounding box | [mcnk_structure.md](mcnk_structure.md#mcbb-bounding-box) |
| MCRD | ✅ | Doodad references | [mcnk_structure.md](mcnk_structure.md#mcrf-references) |
| MCRW | ✅ | WMO references | [mcnk_structure.md](mcnk_structure.md#mcrw-wmo-references) |
| MCMT | ✅ | Material IDs | [mcnk_structure.md](mcnk_structure.md#mcmt-material-ids) |
| MCDD | ✅ | Detail doodad disable bitmap | [mcnk_structure.md](mcnk_structure.md#mcdd-detail-doodad-disable-bitmap) |

### LOD Chunks (14/14 Complete)
| Chunk ID | Status | Description | Documentation |
|----------|--------|-------------|---------------|
| MLHD | ✅ | LOD header | [mcnk_structure.md](mcnk_structure.md#mlhd-lod-header) |
| MLVH | ✅ | LOD vertex heights | [mcnk_structure.md](mcnk_structure.md#mlvh-lod-vertex-heights) |
| MLVI | ✅ | LOD vertex indices | [mcnk_structure.md](mcnk_structure.md#mlvi-lod-vertex-indices) |
| MLLL | ✅ | LOD level list | [mcnk_structure.md](mcnk_structure.md#mlll-lod-level-list) |
| MLND | ✅ | LOD node data | [mcnk_structure.md](mcnk_structure.md#mlnd-lod-node-data) |
| MLSI | ✅ | LOD skirt indices | [mcnk_structure.md](mcnk_structure.md#mlsi-lod-skirt-indices) |
| MLLD | ✅ | LOD liquid data | [mcnk_structure.md](mcnk_structure.md#mll-lod-liquid-data) |
| MLLN | ✅ | LOD liquid nodes | [mcnk_structure.md](mcnk_structure.md#mlln-lod-liquid-nodes) |
| MLLV | ✅ | LOD liquid vertices | [mcnk_structure.md](mcnk_structure.md#mllv-lod-liquid-vertices) |
| MLLI | ✅ | LOD liquid indices | [mcnk_structure.md](mcnk_structure.md#mlli-lod-liquid-indices) |
| MLMD | ✅ | LOD model definitions | [mcnk_structure.md](mcnk_structure.md#mlmd-lod-model-definitions) |
| MLMX | ✅ | LOD model extents | [mcnk_structure.md](mcnk_structure.md#mlmx-lod-model-extents) |
| MBMB | ✅ | Legion+ blend mesh data | [mcnk_structure.md](mcnk_structure.md#mbmb-legion-blend-mesh-data) |
| MLMB | ✅ | BfA+ data | [mcnk_structure.md](mcnk_structure.md#mlmb-bfa-data) |

Total Progress: 45/45 chunks implemented (100%)

## Version History
- v18: Original chunked format (Classic through Shadowlands)
- v22: [ADT v22](ADT_v22_index.md) - Battle for Azeroth changes
- v23: [ADT v23](ADT_v23_index.md) - Dragonflight changes

## Next Steps
1. Implement visualization system
2. Add terrain rendering support
3. Develop collision detection
4. Create editing tools
5. Add format conversion utilities

## References
- [ADT v18 Documentation](../docs/ADT_v18.md)
- [ADT v22 Documentation](../docs/ADT_v22.md)
- [ADT v23 Documentation](../docs/ADT_v23.md) 