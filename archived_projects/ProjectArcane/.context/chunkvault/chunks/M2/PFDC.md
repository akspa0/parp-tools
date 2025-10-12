# PFDC Chunk (Physics Force Data Content)

## Overview
The PFDC (Physics Force Data Content) chunk contains inline physics data for an M2 model. It was introduced in Shadowlands (9.0.1.33978) and embeds the same physics information that would typically be found in a separate .phys file. This allows models to have physics properties without requiring an external file.

## Structure
```cpp
struct PFDC_Chunk {
  PHYS physics;        // Physics data structure (same as .phys file)
  char PADDING[6];     // Alignment padding, possibly variable length to align to 8 or 16 bytes
};
```

## Fields
- **physics**: The complete physics data structure, identical to the content of a .phys file, containing bodies, joints, and shapes information.
- **PADDING**: Alignment padding that follows the physics data, typically filled with zeros. The exact size may vary to ensure proper alignment.

## Dependencies
- Relates to the PFID chunk, which provides a reference to an external .phys file
- A model should have either PFDC or PFID, but not both
- May reference bone indices defined in the MD21 chunk for physical body attachments

## Usage
The PFDC chunk is used to:
- Embed physics data directly within the M2 file instead of requiring a separate .phys file
- Define physical bodies, joints, and collision shapes for the model
- Support physics simulation for the model, such as cloth, ragdoll effects, or collision detection
- Allow models to interact with the game's physics engine (Domino)

## Legacy Support
- Not present in pre-Shadowlands M2 files
- Prior to this chunk, physics data was only stored in separate .phys files referenced by the PFID chunk
- Allows for simpler distribution of models with physics as they don't require separate files

## Implementation Notes
- The physics data structure matches the format of the .phys file
- The physics data is chunked, containing various sub-chunks:
  - PHYS: Main header with version information
  - BODY/BDY2/BDY3/BDY4: Physical body definitions that attach to bones
  - SHAP/SHP2: Shape definitions (boxes, capsules, spheres, polytopes)
  - BOXS/CAPS/SPHS/PLYT: Specific shape data
  - JOIN: Joint definitions connecting bodies
  - WELJ/SPHJ/SHOJ/PRSJ/REVJ/DSTJ: Specific joint data
  - PHYT/PHYV: Additional physics tuning parameters
- The padding after the physics data is likely variable length to ensure proper memory alignment
- When implementing, treat this chunk as if loading a .phys file, but from the M2 file's data stream
- Physics version in Shadowlands is typically 5 or 6, which determines the exact structure of the sub-chunks

## Version History
- Introduced in Shadowlands (9.0.1.33978)
- Streamlines the model loading process by embedding physics data directly
- Supports all the same physics features as external .phys files 