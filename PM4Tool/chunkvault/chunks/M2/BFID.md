# BFID: Bone File IDs

## Identification
- **Chunk ID**: BFID
- **Parent Format**: M2
- **Source**: M2 file format documentation

## Description
The BFID chunk contains FileDataIDs for external bone files associated with the M2 model. Prior to Legion, bone files were referenced by filename with the pattern `${basename}_${i}.bone`. This chunk was introduced in Legion (expansion level 7) and provides direct FileDataID references to these bone files.

## Structure
```cpp
struct BFIDChunk {
    uint32_t boneFileDataIDs[];  // Array of FileDataIDs for bone files
}
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| boneFileDataIDs | uint32_t[] | Array of FileDataIDs for the model's bone files |

## Dependencies
- Depends on the MD21 chunk which contains the skeletal structure.
- The bone files referenced contain additional bone lookup data.

## Implementation Notes
1. The bone files were added in Shadowlands and provide additional bone lookup information.
2. The number of entries in the array is not explicitly specified in the chunk header, so it must be determined from the chunk size.
3. The bone files are used to optimize skeleton loading and animation.
4. The bone files contain data that works in conjunction with .skel files (referenced in the SKID chunk).

## Usage Context
Bone files (.bone) were introduced in Shadowlands to provide:
- Optimized bone lookup data
- Additional skeleton information
- Performance improvements for animated models

The BFID chunk provides direct FileDataID references to these bone files, eliminating the need to construct filenames based on patterns. This system allows the client to efficiently load the necessary bone data for model animation and rendering. 