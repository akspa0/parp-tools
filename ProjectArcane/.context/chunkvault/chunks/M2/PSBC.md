# PSBC Chunk (Parent Sequence Bounds)

## Overview
The PSBC chunk defines parent sequence bounds for animations. It was introduced in Legion (7.x) and contains an array of M2Bounds structures that define spatial bounds for parent animation sequences.

## Structure
```cpp
struct M2Bounds {
  CAaBox extent;  // Axis-aligned bounding box
  float radius;   // Bounding sphere radius
};

struct PSBC_Chunk {
  M2Array<M2Bounds> parentSequenceBounds;  // Bounds for parent sequences
};
```

## Fields
- **parentSequenceBounds**: An array of M2Bounds structures that define the spatial boundaries of parent animation sequences.

## Dependencies
- Requires the MD21 chunk for basic model data
- Related to animation sequences defined in the MD21 chunk

## Usage
The PSBC chunk is used in conjunction with animation data to:
- Define spatial boundaries for parent animation sequences
- Optimize animation culling based on visibility
- Aid in animation blending between parent sequences

## Legacy Support
- Not present in pre-Legion M2 files
- In older versions, this information was likely calculated at runtime or not used

## Implementation Notes
- This chunk should be loaded after the MD21 chunk
- The bounds are used during animation playback to determine visibility and animation blending
- Bounds are defined as both an axis-aligned box and a sphere for efficient culling

## Version History
- Introduced in Legion (7.x) build
- The exact build number is currently unknown but was part of the Legion chunked format evolution 