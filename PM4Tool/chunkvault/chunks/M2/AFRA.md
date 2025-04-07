# AFRA Chunk (Alpha Frame)

## Overview
The AFRA (Alpha Frame) chunk was introduced in Shadowlands (9.0.1.33978) and appears to control alpha frames or alpha-related animations in M2 models. It provides specialized alpha animation capabilities beyond what was available in earlier versions of the format.

## Structure
```cpp
struct AFRA_Entry {
  uint16_t animationID;     // Animation ID reference
  uint16_t subAnimationID;  // Sub-animation ID reference
  uint32_t frameNumber;     // Frame number in the animation sequence
};

struct AFRA_Chunk {
  AFRA_Entry entries[];     // Array of AFRA entries
}
```

## Fields
- **animationID**: Identifies the animation sequence this alpha frame is associated with
- **subAnimationID**: Identifies the sub-animation or subset of animation this entry applies to
- **frameNumber**: Specifies the exact frame number within the animation sequence where the alpha effect should be applied

## Dependencies
- Requires the MD21 chunk for basic model data
- References animation data defined in M2Sequence structures
- May interact with transparency and alpha-related animations

## Usage
The AFRA chunk is used for:
- Defining key frames for alpha-based effects in animations
- Controlling transparency transitions at specific points in animations
- Enhancing animation control by providing frame-exact alpha modifications
- Creating sophisticated fade effects in model animations

## Legacy Support
- Not present in pre-Shadowlands M2 files
- Earlier versions relied solely on standard animation tracks for alpha control
- The functionality provided by AFRA may have been approximated with standard animation techniques in pre-Shadowlands models

## Implementation Notes
- Each entry in the AFRA chunk references a specific animation by ID
- The subAnimationID field likely refers to a specific part or layer of the animation
- Frame numbers should be interpreted within the context of the animation's total frame count and playback speed
- Implementation should handle the possibility of multiple AFRA entries affecting the same animation
- The entries are likely processed in sequence when the animation is played

## Version History
- Introduced in Shadowlands (9.0.1.33978)
- Part of the expanded animation control features added to the M2 format
- Represents specialized animation capability development in modern versions of the format 