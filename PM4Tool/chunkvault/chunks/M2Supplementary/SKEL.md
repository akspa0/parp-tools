# M2.SKEL File Format

## Overview
The .skel files replace specific blocks from the M2 MD20 data. They use a chunked format where each chunk has a fixed header followed by raw data. The headers contain offsets into the respective chunk, meaning that there are no fixed sizes for most chunks, and the chunks may include alignment and padding bytes.

## Chunks

### SKL1 (Skeleton Data)
```cpp
struct SKL1_Header {
    uint32_t flags;                     // Assumed flags; always 0x100 in version 7.3.2.25079
    M2Array<char> name;                 // Name string array
    uint8_t padding[4];                 // Always 0 in version 7.3.2.25079
};
uint8_t skeleton_l_raw_data[];          // Raw data following the header
```

#### Fields
- **flags**: Control flags for the skeleton data, observed as 0x100
- **name**: Array of characters forming the skeleton name
- **padding**: Alignment padding, always 0

### SKA1 (Skeleton Attachment)
```cpp
struct SKA1_Header {
    M2Array<M2Attachment> attachments;           // Array of attachment points
    M2Array<uint16_t> attachment_lookup_table;   // Lookup table for attachments
};
uint8_t skeleton_attachment_raw_data[];          // Raw data following the header
```

#### Fields
- **attachments**: Array of M2Attachment structures defining attachment points
- **attachment_lookup_table**: Lookup table for quick reference to attachments

### SKB1 (Skeleton Bone)
```cpp
struct SKB1_Header {
    M2Array<M2CompBone> bones;            // Array of bones
    M2Array<uint16_t> key_bone_lookup;    // Lookup table for key bones
};
uint8_t skeleton_bone_raw_data[];         // Raw data following the header
```

#### Fields
- **bones**: Array of M2CompBone structures defining the skeleton
- **key_bone_lookup**: Lookup table for important bones in the skeleton

### SKS1 (Skeleton Sequence)
```cpp
struct SKS1_Header {
    M2Array<M2Loop> global_loops;          // Array of animation loops
    M2Array<M2Sequence> sequences;         // Array of animation sequences
    M2Array<uint16_t> sequence_lookups;    // Lookup table for sequences
    uint8_t padding[8];                    // Always 0 in version 7.3.2.25079
};
uint8_t skeleton_sequence_raw_data[];      // Raw data following the header
```

#### Fields
- **global_loops**: Array of animation loops
- **sequences**: Array of animation sequence definitions
- **sequence_lookups**: Lookup table for quick reference to sequences
- **padding**: Alignment padding, always 0

### SKPD (Skeleton Parent Data)
```cpp
struct SKPD_Chunk {
    uint8_t padding1[8];                  // Always 0 in version 7.3.2.25079
    uint32_t parent_skel_file_id;         // File ID of the parent skeleton
    uint8_t padding2[4];                  // Always 0 in version 7.3.2.25079
};
```

#### Fields
- **padding1**: Alignment padding, always 0
- **parent_skel_file_id**: File ID reference to the parent skeleton file
- **padding2**: Alignment padding, always 0

### AFID (Animation File IDs)
Same structure and semantics as in the main M2 format's AFID chunk.

### BFID (Bone File IDs)
Same structure and semantics as in the main M2 format's BFID chunk.

## Dependencies
- References the parent M2 model
- May reference parent skeleton files for inheritance
- Animation files (.anim) referenced through AFID
- Bone files (.bone) referenced through BFID

## Usage
The .skel files are used to:
- Store skeleton data separately from the main model geometry
- Enable skeleton sharing between related models
- Implement a parent-child relationship between skeletons for de-duplication
- Provide animation and bone lookup information for models
- Share animation data across multiple models

## Legacy Support
- Part of Blizzard's modularization of the M2 format
- Helps reduce redundancy in model data
- Enables more efficient model loading and memory usage

## Implementation Notes
- The parent skeleton file data ID is used for de-duplication (e.g., lightforgeddraeneimale references draeneimale_hd)
- Child models can reference animations from parent models via the parent-link
- Child .skel files still have their own SK*1 chunks, but share *FID chunks with parents
- Implementation should handle this hierarchy correctly, inheriting data from parent skeletons when needed

## Version History
- Part of Legion and later expansions' modular approach to model data
- Represents Blizzard's ongoing efforts to optimize game assets and reduce redundancy 