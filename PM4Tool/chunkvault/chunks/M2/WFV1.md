# WFV1 Chunk (Waterfall Version 1)

## Overview
The WFV1 (Waterfall Version 1) chunk marks a model as using a specialized PBR (Physically Based Rendering) rendering path with normal mapping. It was introduced in Battle for Azeroth (8.2.0.30080) and was first used for waterfall models but was later expanded to other model types.

## Structure
```cpp
struct WFV1_Chunk {
  // No documented fields - presence alone indicates PBR rendering usage
};
```

## Fields
- Currently, no specific fields have been documented for this chunk. The presence of the chunk itself signals to the client to use the specialized rendering path.

## Dependencies
- Requires the MD21 chunk for basic model data
- May interact with texture definitions in the model to determine which textures serve as normal maps
- Works with TXID chunk to reference normal map textures

## Usage
The WFV1 chunk is used to:
- Mark a model for rendering with PBR-like techniques
- Enable normal map usage in the model's materials
- Trigger a separate rendering pipeline optimized for water and similar effects
- Provide enhanced visual fidelity compared to standard M2 rendering

## Legacy Support
- Not present in pre-BfA (8.2) M2 files
- Prior to this chunk, M2 models used a more basic rendering approach without advanced PBR features

## Implementation Notes
- Despite originally being designed for waterfalls (hence the "WF" prefix), this rendering approach was later used for various models
- The chunk's presence alone is enough to trigger the specialized rendering path
- Later replaced by WFV2 and WFV3 chunks which extended the functionality
- FileDataID 2445860 in patch 8.2.0 is cited as the first example model using this chunk
- Even when used in non-waterfall models, the rendering technique kept the "waterfall" name internally
- The implementation should check for the presence of this chunk before attempting to use PBR features

## Version History
- Introduced in Battle for Azeroth (8.2.0.30080)
- Represented the first step in Blizzard's move toward more advanced rendering for specific M2 models
- Later superseded by WFV2 and WFV3 chunks with expanded capabilities 