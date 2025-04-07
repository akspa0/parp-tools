# EDGF Chunk (Edge Fade)

## Overview
The EDGF (Edge Fade) chunk was introduced in Shadowlands (9.0.1.33978) and contains edge fade data for model meshes. This data is applied only to meshes that have the 0x8 flag set in their M2Batch.flags2 value, allowing for selective application of edge fading effects.

## Structure
```cpp
struct EDGF_Entry {
  /*0x00*/ float values[2];  // Two floating-point values
  /*0x08*/ float coefficient;  // Edge fade coefficient
  /*0x0C*/ char padding[0xC];  // Padding or additional data
};

struct EDGF_Chunk {
  EDGF_Entry edgf[];  // Array of edge fade entries
}
```

## Fields
- **values**: Two floating-point values related to edge fade calculations
- **coefficient**: A coefficient used in edge fade calculations
- **padding**: Either padding or additional data not fully documented

## Dependencies
- Requires the MD21 chunk for basic model data
- Applied selectively based on M2Batch.flags2 values in the skin (.skin) file

## Usage
The EDGF chunk is used to:
- Define edge fading effects for specific model meshes
- Create smoother transitions at mesh boundaries
- Enhance visual quality of models with complex edges
- Apply special rendering treatments to specific parts of a model

## Legacy Support
- Not present in pre-Shadowlands M2 files
- Earlier versions may have used different techniques to achieve similar visual effects

## Implementation Notes
- This chunk should only be applied to meshes that have the 0x8 flag set in their M2Batch.flags2 value
- Edge fading is a visual effect that typically softens the edges of geometry
- The exact algorithm for applying edge fading using these values is not fully documented
- The implementation may require custom shader support to properly utilize the edge fade data

## Version History
- Introduced in Shadowlands (9.0.1.33978)
- Part of Blizzard's ongoing enhancements to M2 rendering capabilities
- Represents an advanced visual technique added to the M2 format 