# SFID: Skin File IDs

## Identification
- **Chunk ID**: SFID
- **Parent Format**: M2
- **Source**: M2 file format documentation

## Description
The SFID chunk contains FileDataIDs for skin files that are associated with the M2 model. These skin files contain optimized rendering data for the model. Prior to Legion, skin files were referenced by filename with patterns like `${basename}${view}.skin` and `${basename}_lod${lodband}.skin`. This chunk was introduced in Legion (expansion level 7) and provides direct FileDataID references instead.

## Structure
```cpp
struct SFIDChunk {
    uint32_t skinFileDataIDs[header.nViews];  // FileDataIDs for main skin views
    uint32_t lod_skinFileDataIDs[lodBands];   // FileDataIDs for LOD skin files (typically 2)
}
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| skinFileDataIDs | uint32_t[] | Array of FileDataIDs for the model's skin files, one per view |
| lod_skinFileDataIDs | uint32_t[] | Array of FileDataIDs for the model's LOD skin files (typically 2) |

## Dependencies
- Depends on the MD21 chunk to determine the number of views (header.nViews).
- The skin files referenced contain rendering data that supplements the main M2 geometry.

## Implementation Notes
1. The number of entries in `skinFileDataIDs` corresponds to `header.nViews` from the M2 header.
2. Some models may have 4 skin files and 2 LOD files, but only 20 bytes (5 entries) in the chunk.
3. LOD skins are selected based on distance using CVar settings:
   - `entityLodDist` for entities
   - `doodadLodDist` for doodads
4. LOD selection can be disabled by setting the `M2UseLOD` CVar to 0.
5. The length of the chunk may not match the expected length based on the header values, so be prepared to handle different sizes.

## Usage Context
Skin files (.skin) are used to optimize model rendering by:
- Organizing vertices into batches for efficient GPU processing
- Providing texture unit mappings
- Defining submeshes and their render properties
- Storing material and texture lookup information

These files are essential for rendering the model correctly and need to be loaded alongside the main M2 file. The SFID chunk provides direct FileDataID references to these skin files, eliminating the need to construct filenames based on patterns. 