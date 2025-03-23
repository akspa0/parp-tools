# World Format Documentation Index

## Format Overview

| Format | Status | Main Chunks | Subchunks | Notes |
|--------|--------|-------------|-----------|-------|
| ADT v18 | ✅ Complete | 30/30 (100%) | 15/15 (100%) | Complete documentation including all expansion-specific chunks |
| ADT v22 | ✅ Complete | 16/16 (100%) | 0/0 (N/A) | Complete documentation |
| ADT v23 | ✅ Complete | 8/8 (100%) | 4/4 (100%) | Complete documentation |
| WDT | ✅ Complete | 15/15 (100%) | 0/0 (N/A) | Includes both modern and alpha formats |
| WDL | ✅ Complete | 8/8 (100%) | 0/0 (N/A) | Complete documentation |
| WMO | ✅ Complete | 18/18 (100%) | 14/14 (100%) | Complete documentation |
| M2 | ✅ Complete | 27/27 (100%) | 25/24 (100%) | All chunks and structures documented |
| MDX | ✅ Complete | 24/24 (100%) | 2/2 (100%) | All chunks fully documented |
| DBC/DB2/ADB | ✅ Complete | 3/3 (100%) | 0/0 (N/A) | Complete documentation |

## Documentation Status

| Format | Status | Progress | Last Updated |
|--------|--------|----------|--------------|
| M2     | Complete | 27/27 chunks and 25/24 structures documented (100%) | 2025-04-13 |
| MDX    | Complete | 26/26 chunks documented (100%) | 2025-04-05 |
| WMO    | Complete | 18/18 chunks documented (100%) | 2025-04-01 |
| ADT    | Complete | 30/30 main chunks, 15/15 subchunks, 14/14 LOD chunks documented (100%) | 2025-04-15 |
| WDT    | Complete | 9/9 chunks documented (100%) | 2025-03-20 |
| WDL    | Complete | 4/4 chunks documented (100%) | 2025-03-15 |
| DBC    | Planned | 0/? records documented (0%) | - |
| WDB    | Planned | 0/? records documented (0%) | - |

## Recent Updates

- **2025-04-15**: Completed documentation for all Legion+ LOD chunks and Cataclysm-specific terrain chunks
- **2025-04-14**: Added comprehensive documentation for the pre-Legion M2 format structure
- **2025-04-13**: Added documentation for 5 additional Shadowlands M2 chunks (EDGF, NERF, DETL, DBOC, AFRA)
- **2025-04-12**: M2 structure documentation progress (added M2Sequence, M2AnimTrack, and M2Vertex structures)
- **2025-04-12**: Completed M2 structure documentation (all 24 structures documented)
- **2025-04-11**: Completed M2 chunk documentation (all 22 chunks documented)
- **2025-04-10**: M2 documentation in progress (17/22 chunks, 77% complete)
- **2025-04-05**: Completed MDX documentation (all 26 chunks documented)
- **2025-04-01**: Completed WMO documentation (all 18 chunks documented)
- **2025-03-25**: Initial ADT documentation (23/30 chunks documented)

## Implementation Progress

| Format | Status | Notes |
|--------|--------|-------|
| M2     | Ready for implementation | All chunks (27) and structures (25) fully documented (100%) |
| MDX    | Ready for implementation | All chunks fully documented |
| WMO    | Implemented | Core rendering system complete |
| ADT    | Ready for implementation | All chunks fully documented (100%), implementation planned next |
| WDT    | Implemented | Full support |
| WDL    | Implemented | World map rendering ready |
| DBC    | Planned for database phase | Not started |
| WDB    | Planned for database phase | Not started |

## Documentation Guidelines

Each chunk should include:

### Identification
- **Chunk ID**: Four-character identifier
- **Type**: Main chunk, subchunk, or special role
- **Source**: Original format documentation reference

### Structure Definition
- C# structure definition with field offsets and types
- Complete field definitions including arrays, nested types, and sizes
- Comments for each field explaining its purpose

### Properties Table
- Complete list of fields with types and detailed descriptions
- Flag definitions where applicable
- Enumerations of possible values

### Dependencies
- List of related chunks that this chunk depends on
- Description of how the dependencies work (references, counts, etc.)

### Implementation Notes
- Special handling requirements
- Version differences
- Endianness considerations
- Alignment requirements
- Size constraints

### Usage Context
- How the chunk is used in the game client
- Purpose and significance of the data
- Relationship to game mechanics or visuals

## Next Steps

1. **PRIORITY: Begin implementation of documented formats**:
   - ✓ Complete MDX documentation (all 26 chunks and structures)
   - ✓ Complete M2 chunk documentation (all 27 chunks)
   - ✓ Complete M2 structure documentation (all 25 structures)
   - ✓ Complete ADT documentation (all main chunks, subchunks, and LOD chunks)
   - Begin ADT parser implementation
   - Begin MDX parser implementation
   - Begin M2 parser implementation
   - Verify and refine existing documentation for other formats

2. ADT implementation plan:
   - ✓ Complete documentation of all ADT chunks, subchunks, and expansion-specific features
   - Develop core parser structure for ADT formats
   - Implement support for both monolithic and split file formats
   - Implement terrain height map and texture layer handling
   - Develop object placement and reference systems
   - Add Legion+ LOD system support for efficient distant rendering

3. M2 implementation plan:
   - ✓ Complete documentation of all M2 chunks and structures
   - Develop core parser structure for M2 format
   - Implement parsing for both chunked and non-chunked formats
   - Implement skeletal animation system
   - Develop material and texture parsing
   - Create particle system implementation
   - Add support for supplementary files (.skin, .anim, etc.)
   - Develop model rendering system

4. Begin implementation only after documentation is complete
5. Create rendering system for WMO visualization
6. Develop collision detection system using the BSP tree structure
7. Create tools for WMO editing and conversion
8. Complete implementation of ADT parsers
9. Create visualizers for parsed formats
10. Implement unified asset management system

## Contributors

- **Arcane Preservation Team** - Documentation and implementation 

## Chunk Documentation Overview

This index catalogs all documented chunks from various Blizzard file formats. Use this as your primary reference when looking for specific chunk documentation.

### M2 Chunks
- [Overview & Index](M2_index.md)
- [Pre-Legion Format](chunks/M2/PreLegionFormat.md) - Complete pre-Legion non-chunked format structure
- [MD21](chunks/M2/MD21.md) - Main data chunk (Legion+)
- [AFID](chunks/M2/AFID.md) - Animation file IDs
- [BFID](chunks/M2/BFID.md) - Bone file IDs
- [SFID](chunks/M2/SFID.md) - Skin file IDs
- [TXAC](chunks/M2/TXAC.md) - Texture/alt data
- [EXPT](chunks/M2/EXPT.md) - Extended particle data (original format)
- [EXP2](chunks/M2/EXP2.md) - Extended particle data (new format)
- [PABC](chunks/M2/PABC.md) - Particle bone combo
- [PADC](chunks/M2/PADC.md) - Particle data combo
- [PSBC](chunks/M2/PSBC.md) - Particle SB combo
- [PEDC](chunks/M2/PEDC.md) - Particle edge combo
- [SKID](chunks/M2/SKID.md) - Skeleton file IDs
- [TXID](chunks/M2/TXID.md) - Additional texture file IDs
- [LDV1](chunks/M2/LDV1.md) - LOD data version 1
- [RPID](chunks/M2/RPID.md) - Replacement ID
- [GPID](chunks/M2/GPID.md) - Geometry particle IDs
- [PFID](chunks/M2/PFID.md) - Physics file IDs
- [WFV1](chunks/M2/WFV1.md) - Warp field version 1
- [WFV2](chunks/M2/WFV2.md) - Warp field version 2
- [WFV3](chunks/M2/WFV3.md) - Warp field version 3
- [PGD1](chunks/M2/PGD1.md) - Particle geometry data
- [PFDC](chunks/M2/PFDC.md) - Physics fabric data
- [EDGF](chunks/M2/EDGF.md) - Edge fade data
- [NERF](chunks/M2/NERF.md) - Alpha attenuation data
- [DETL](chunks/M2/DETL.md) - Detail lighting data
- [DBOC](chunks/M2/DBOC.md) - Dynamic bone data
- [AFRA](chunks/M2/AFRA.md) - Animation framerate data

### M2 Supplementary Files
- [M2.BONE](chunks/M2Supplementary/BONE.md) - Enhanced bone transformations for facial animations
- [M2.PHYS](chunks/M2Supplementary/PHYS.md) - Physics data for the Domino engine
- [M2.SKEL](chunks/M2Supplementary/SKEL.md) - Skeletal animation data
- [M2.SKIN](chunks/M2Supplementary/SKIN.md) - Rendering optimization data

### WDT Chunks
- [Overview & Index](WDT_index.md)
- [MAIN](chunks/WDT/MAIN.md) - Main data array
- [MPHD](chunks/WDT/MPHD.md) - Map header
- [MPLT](chunks/WDT/MPLT.md) - Map palette

### WMO Chunks
- [Overview & Index](WMO_index.md)
- [MVER](chunks/WMO/MVER.md) - Version
- [MOHD](chunks/WMO/MOHD.md) - Header

### ADT Chunks
- [Overview & Index](ADT_index.md)
- [MVER](chunks/ADT/C001_MVER.md) - Version information
- [MHDR](chunks/ADT/C002_MHDR.md) - Header containing flags and offsets
- [MCIN](chunks/ADT/C003_MCIN.md) - Chunk index array
- [MTEX](chunks/ADT/C004_MTEX.md) - Texture filename list
- [MMDX](chunks/ADT/C005_MMDX.md) - Model filename list
- [MMID](chunks/ADT/C006_MMID.md) - Model filename offsets
- [MWMO](chunks/ADT/C007_MWMO.md) - WMO filename list
- [MWID](chunks/ADT/C008_MWID.md) - WMO filename offsets
- [MDDF](chunks/ADT/C009_MDDF.md) - Doodad placement information
- [MODF](chunks/ADT/C011_MODF.md) - Object placement information
- [MH2O](chunks/ADT/C012_MH2O.md) - Water data (added in WotLK)
- [MFBO](chunks/ADT/C013_MFBO.md) - Flight boundaries (added in BC)
- [MTXF](chunks/ADT/C014_MTXF.md) - Texture flags (added in WotLK)
- [MTXP](chunks/ADT/C015_MTXP.md) - Texture parameters (added in MoP)
- [MAMP](chunks/ADT/C016_MAMP.md) - Texture coordinate amplification (added in Cata)
- [MTCG](chunks/ADT/C017_MTCG.md) - Texture color gradients (added in Shadowlands)
- [MCNK](chunks/ADT/C018_MCNK.md) - Map chunk data (container for subchunks)
- [MDID](chunks/ADT/C023_MDID.md) - Diffuse texture FileDataIDs (added in Legion)
- [MHID](chunks/ADT/C018_MHID.md) - Heightmap FileDataIDs (added in MoP)
- [MBMH](chunks/ADT/C019_MBMH.md) - Blend mesh headers (added in MoP)
- [MBBB](chunks/ADT/C020_MBBB.md) - Blend mesh bounding boxes (added in MoP)
- [MBNV](chunks/ADT/C021_MBNV.md) - Blend mesh vertices (added in MoP)
- [MBMI](chunks/ADT/C022_MBMI.md) - Blend mesh indices (added in MoP)
- [MNID](chunks/ADT/C024_MNID.md) - Normal texture FileDataIDs (added in Legion)
- [MSID](chunks/ADT/C025_MSID.md) - Specular texture FileDataIDs (added in Legion)
- [MLID](chunks/ADT/C026_MLID.md) - Height texture FileDataIDs (added in Legion)
- [MLDB](chunks/ADT/C027_MLDB.md) - Low detail blend distances (added in BfA)
- [MWDR](chunks/ADT/C028_MWDR.md) - Doodad references (added in Shadowlands)
- [MWDS](chunks/ADT/C029_MWDS.md) - Doodad sets (added in Shadowlands)

### ADT Subchunks (MCNK)
- [MCVT](chunks/ADT/S001_MCVT.md) - Height map vertices
- [MCNR](chunks/ADT/S002_MCNR.md) - Normal vectors
- [MCLY](chunks/ADT/S003_MCLY.md) - Texture layer definitions
- [MCRF](chunks/ADT/S004_MCRF.md) - Doodad and object references (<Cata)
- [MCSH](chunks/ADT/S005_MCSH.md) - Shadow map
- [MCAL](chunks/ADT/S006_MCAL.md) - Alpha maps
- [MCLQ](chunks/ADT/S007_MCLQ.md) - Liquid data (legacy)
- [MCSE](chunks/ADT/S008_MCSE.md) - Sound emitters
- [MCCV](chunks/ADT/S009_MCCV.md) - Vertex colors (added in WotLK)
- [MCLV](chunks/ADT/S010_MCLV.md) - Light values (added in Cata)
- [MCBB](chunks/ADT/S011_MCBB.md) - Bounding box (added in MoP)
- [MCRD](chunks/ADT/S012_MCRD.md) - Doodad references (added in Cata)
- [MCRW](chunks/ADT/S013_MCRW.md) - WMO references (added in Cata)
- [MCMT](chunks/ADT/S014_MCMT.md) - Material IDs (added in Cata)
- [MCDD](chunks/ADT/S015_MCDD.md) - Detail doodad disable bitmap (added in Cata)

### ADT LOD Chunks (Legion+)
- [MLHD](chunks/ADT/C030_MLHD.md) - LOD header
- [MLVH](chunks/ADT/C031_MLVH.md) - LOD vertex heights
- [MLVI](chunks/ADT/C032_MLVI.md) - LOD vertex indices
- [MLLL](chunks/ADT/C033_MLLL.md) - LOD level list
- [MLND](chunks/ADT/C034_MLND.md) - LOD node data (quad tree)
- [MLSI](chunks/ADT/C035_MLSI.md) - LOD skirt indices
- [MLLD](chunks/ADT/C036_MLLD.md) - LOD liquid data
- [MLLN](chunks/ADT/C037_MLLN.md) - LOD liquid nodes
- [MLLV](chunks/ADT/C038_MLLV.md) - LOD liquid vertices
- [MLLI](chunks/ADT/C039_MLLI.md) - LOD liquid indices
- [MLMD](chunks/ADT/C040_MLMD.md) - LOD model definitions
- [MLMX](chunks/ADT/C041_MLMX.md) - LOD model extents
- [MBMB](chunks/ADT/C042_MBMB.md) - Unknown Legion+ blend mesh data
- [MLMB](chunks/ADT/C043_MLMB.md) - Unknown BfA+ data

## Structure Types
- [M2Vertex](chunks/M2/M2Vertex.md) - Vertex data structure
- [M2AnimTrack](chunks/M2/M2AnimTrack.md) - Animation track structure
- [M2Sequence](chunks/M2/M2Sequence.md) - Animation sequence structure
- [M2CompBone](chunks/M2/M2CompBone.md) - Compressed bone structure

# Chunk Documentation Index

## Common Chunks

### MVER - Version Chunk
- **Description**: Common version chunk used in multiple file formats
- **Structure**: 
  - Signature: "MVER" (0x5245564D)
  - Size: 4 bytes
  - Data: uint32 version

## WDT Format Chunks

### MAIN - Main Data Chunk
- **Description**: Contains map tile flags and data offsets
- **Structure**:
  - Signature: "MAIN" (0x4E49414D)
  - Size: 4096 * 64 * 8 bytes
  - Data: Array[4096] of MapTileEntry

### MPHD - Map Header Chunk
- **Description**: Contains global map information
- **Structure**:
  - Signature: "MPHD" (0x4448504D)
  - Size: Variable
  - Version: Required
  - Data: MapHeaderData

### MODF - Model Placement Chunk
- **Description**: Defines WMO placement in the world
- **Structure**:
  - Signature: "MODF" (0x46444F4D)
  - Size: Variable
  - Data: Array of ModelPlacement

## WMO Format Chunks

### MOHD - WMO Header
- **Description**: Contains WMO file header information
- **Structure**:
  - Signature: "MOHD" (0x44484F4D)
  - Size: Variable
  - Version: Required
  - Data: WMOHeader

### MOTX - Texture Names
- **Description**: Contains texture filenames
- **Structure**:
  - Signature: "MOTX" (0x58544F4D)
  - Size: Variable
  - Data: Array of strings

### MOMT - Material Information
- **Description**: Material definitions for WMO
- **Structure**:
  - Signature: "MOMT" (0x544D4F4D)
  - Size: Variable
  - Data: Array of MaterialInfo

## M2 Format Chunks

### MD20/MD21 - Model Data
- **Description**: Main model data chunk
- **Structure**:
  - Signature: "MD20" (0x3032444D) or "MD21" (0x3132444D)
  - Size: Variable
  - Version: Required for MD21
  - Data: ModelData

### SFID - Skin File IDs
- **Description**: Defines skin file references
- **Structure**:
  - Signature: "SFID" (0x44494653)
  - Size: Variable
  - Data: Array of uint32

### Implementation Status

#### Completed
- [x] Core chunk system
  - [x] IChunk interface
  - [x] ChunkBase class
  - [x] VersionedChunkBase class
  - [x] ChunkSignature utilities

#### In Progress
- [ ] WDT Format
  - [ ] MVER implementation
  - [ ] MAIN implementation
  - [ ] MPHD implementation
  - [ ] MODF implementation

#### Pending
- [ ] WMO Format chunks
- [ ] M2 Format chunks
- [ ] Additional format support 