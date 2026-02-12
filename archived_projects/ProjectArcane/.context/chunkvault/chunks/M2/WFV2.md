# WFV2 Chunk (Waterfall Version 2)

## Overview
The WFV2 (Waterfall Version 2) chunk is an evolution of the WFV1 chunk, continuing the specialized PBR (Physically Based Rendering) rendering path for M2 models. It was introduced in Battle for Azeroth (8.2.0.30080) alongside WFV1, representing an intermediate stage between the basic WFV1 and more advanced WFV3 implementations.

## Structure
```cpp
struct WFV2_Chunk {
  // Internal structure not fully documented
  // Existence triggers specialized rendering path
};
```

## Fields
- The specific fields of this chunk have not been fully documented. Similar to WFV1, the presence of the chunk itself signals to the client to use an updated version of the specialized rendering path.

## Dependencies
- Requires the MD21 chunk for basic model data
- May have similar texture dependencies as WFV1
- Likely interacts with the TXID chunk for texture references

## Usage
The WFV2 chunk is used to:
- Signal an intermediate version of the specialized PBR rendering system
- Possibly add enhanced features compared to WFV1 models
- Trigger shader variations specific to this version
- Serve as a bridge between the initial WFV1 implementation and the more robust WFV3 system

## Legacy Support
- Not present in pre-BfA (8.2) M2 files
- Represents an evolution of the WFV1 rendering approach
- Was later superseded by WFV3 in Shadowlands

## Implementation Notes
- Despite limited documentation, the implementation should check for the presence of this chunk to determine the correct rendering path
- Models might contain either WFV1, WFV2, or WFV3 chunks, but not multiple versions
- When encountered, the client likely switches to a specific shader variation designed for WFV2 models
- Like WFV1, the "waterfall" terminology was retained even when used for non-waterfall models
- May require different texture channels or formats compared to standard M2 rendering

## Version History
- Introduced in Battle for Azeroth (8.2.0.30080) alongside WFV1
- Represented an intermediate step in the evolution of Blizzard's advanced M2 rendering techniques
- Later replaced by WFV3 in Shadowlands (9.0.1.33978) which added more specific rendering parameters 