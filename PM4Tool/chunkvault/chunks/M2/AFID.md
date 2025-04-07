# AFID: Animation File IDs

## Identification
- **Chunk ID**: AFID
- **Parent Format**: M2
- **Source**: M2 file format documentation

## Description
The AFID chunk contains mappings between animation IDs and their corresponding FileDataIDs for external animation files. Prior to Legion, animation files were referenced by filename with the pattern `${basename}${anim_id}-${sub_anim_id}.anim`. This chunk was introduced in Legion (expansion level 7) and provides direct FileDataID references to these animation files.

## Structure
```cpp
struct AnimFileEntry {
    uint16_t anim_id;     // Animation ID (from AnimationData.dbc)
    uint16_t sub_anim_id; // Sub-animation ID
    uint32_t file_id;     // FileDataID for the animation file
};

struct AFIDChunk {
    AnimFileEntry anim_file_ids[];  // Array of animation file entries
}
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| anim_id | uint16_t | Animation ID matching an entry in AnimationData.dbc |
| sub_anim_id | uint16_t | Sub-animation ID, typically for variations of the same animation |
| file_id | uint32_t | FileDataID for the corresponding .anim file, can be 0 for "none" |

## Dependencies
- Depends on the MD21 chunk which contains the animation sequence definitions.
- References animations defined in AnimationData.dbc.
- The animation files referenced contain keyframe data for skeletal animation.

## Implementation Notes
1. The animation files are loaded on demand when needed, allowing for lazy loading of animations.
2. The `file_id` might be 0 to indicate "none", meaning the animation data is not stored externally.
3. Animation files are only loaded for animations where `(M2Sequence.flags & 0x130) == 0`.
4. In Legion and later, animation files can be chunked themselves, containing AFM2, AFSA, and AFSB chunks.
5. The structure is not sparse, so all animation-subanim combinations are typically present even if not used.

## Usage Context
Animation files (.anim) are used to store keyframe data for animations that aren't commonly used, to reduce memory usage until needed. These files contain:
- Timestamps for keyframes
- Translation, rotation, and scaling values for bones
- Other animation-related data

The AFID chunk provides a direct mapping between animation IDs and their file references, eliminating the need to construct filenames based on patterns. This system allows the client to efficiently load only the animations that are currently needed. 