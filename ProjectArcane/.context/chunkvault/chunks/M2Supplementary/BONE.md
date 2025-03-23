# M2.BONE File Format

## Overview
The .bone files are supplementary files for M2 models, introduced in Shadowlands (expansion level 6). They use a chunked format with an extra header and are named using the pattern "%s_%02d.bone" where %s is the base model name and %02d is the variant number. These files are created for models that have sequence 808 (FacePose) present, with one file per variant of that sequence.

## File Header
```cpp
struct BoneFileHeader {
    uint32_t version;    // Shall be 1, possibly a version number
};
```

## Chunks

### BIDA (Bone ID Array)
```cpp
struct BIDA_Chunk {
    uint16_t bone_id[];    // Array of bone IDs
    // The count should be equivalent to number of FacePose (808) sequences - 1
};
```

#### Fields
- **bone_id**: An array of 16-bit bone identifiers that correspond to bones in the M2 model's bone array

### BOMT (Bone Matrix Transforms)
```cpp
struct BOMT_Chunk {
    C44Matrix boneOffsetMatrices[];    // Array of 4x4 transformation matrices
    // Same count as BIDA
};
```

#### Fields
- **boneOffsetMatrices**: An array of 4x4 transformation matrices (C44Matrix), one for each bone ID in the BIDA chunk

## Dependencies
- Requires the parent M2 model for reference
- Associated with FacePose (808) animation sequences
- Each .bone file corresponds to a specific variant of the model

## Usage
The .bone files are used to:
- Provide specialized bone transformation data for facial poses
- Support variant-specific bone offsets and positions
- Enable more complex facial animations in newer model versions
- Supplement the standard skeletal animation system with specialized bone data

## Legacy Support
- Not present in pre-Shadowlands models
- Earlier versions handled facial animations through standard animation sequences
- Only applies to models with sequence 808 (FacePose) present

## Implementation Notes
- The number of entries in the BIDA and BOMT chunks matches the number of FacePose (808) sequences minus one
- The bone matrices provide transformation offsets for specific bones during facial animations
- Implementation should apply these transformations when the corresponding FacePose animation is active
- The association between variants and specific appearances may be defined elsewhere in DB2 files

## Version History
- Introduced in Shadowlands (expansion level 6)
- Part of Blizzard's enhancements to the facial animation system
- Represents a specialized extension to the M2 animation capabilities for more detailed character expressions 