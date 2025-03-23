# SKID: Skeleton File ID

## Identification
- **Chunk ID**: SKID
- **Parent Format**: M2
- **Source**: M2 file format documentation

## Description
The SKID chunk contains a single FileDataID that references an external skeleton file associated with the M2 model. Prior to this chunk's introduction, skeleton files were referenced by filename with the pattern `${basename}.skel`. This chunk was added in Legion (patch 7.3+) and provides a direct FileDataID reference to the skeleton file.

## Structure
```cpp
struct SKIDChunk {
    uint32_t SKeletonfileID;  // FileDataID for the skeleton file
}
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| SKeletonfileID | uint32_t | FileDataID for the skeleton file associated with the model |

## Dependencies
- Depends on the MD21 chunk which contains the basic bone structure.
- The skeleton file referenced contains optimized skeleton data.
- Works in conjunction with bone files referenced in the BFID chunk.

## Implementation Notes
1. The skeleton file format (.skel) contains shared skeleton information that can be reused across multiple models.
2. When a model has a .skel file, its animation files (.anim) are split into AFM2, AFSA, and AFSB chunks.
3. The AFSA chunk contains skeleton animation data for attachments, while the AFSB chunk contains data for bones.
4. The skeleton system was introduced to optimize animation data storage and reduce duplication across similar models.
5. This chunk is typically found in models created or updated in patch 7.3 or later.

## Usage Context
Skeleton files (.skel) are used to:
- Share common skeleton structures between related models
- Optimize animation data storage
- Reduce memory usage by eliminating duplicate skeleton data
- Improve loading times for models with similar skeletons

The SKID chunk provides a direct FileDataID reference to the skeleton file, eliminating the need to construct filenames based on patterns. This system allows the client to efficiently load and share skeleton data between multiple models. 