# MLVH Chunk

## Overview
**Chunk ID**: MLVH  
**Related Expansion**: Legion and later  
**Used in Format**: ADT  
**Implementation Status**: Not implemented

## Description
The MLVH (Map Lod Vertex Heights) chunk was introduced in Legion. It contains height data for the Level of Detail (LOD) terrain mesh used for distant terrain rendering. This chunk is found in the split file system, specifically in the lod files.

## Chunk Structure

### C++ Structure
```cpp
struct MLVH {
    float heightData[]; // Variable length array of height values
    // Typically contains 129*129 + 128*128 + additional height values
};
```

### C# Structure
```csharp
public struct MLVH
{
    public float[] HeightData; // Array of height values for LOD terrain
}
```

## Fields

| Field | Type | Description |
|-------|------|-------------|
| heightData | float[] | Array of height values for LOD terrain vertices. The array is of variable length, typically containing 129*129 + 128*128 + additional height values |

## Related Chunks
- MLHD - Header for LOD terrain data
- MLVI - Contains vertex indices for LOD terrain
- Other ML* chunks - Various components of the LOD terrain system

## Notes
- This chunk was introduced in Legion as part of the expanded LOD system for terrain.
- The chunk is found in the lod files in the split file system.
- The height values are global, meaning they don't need to be adjusted with z-offset as in MCNK heights.
- The data is organized in a specific order:
  1. First 129*129 heights, arranged in the same order as in MCNK but without interleaved rows
  2. Next 128*128 heights with a slightly offset starting point
- The starting point is the same as in MCNK (max_x, max_y of ADT)
- One step size is (-1600/3/128) units
- For the second set of 128*128 heights, the starting point becomes (max_x - 0.5*(1600/3/128), max_y - 0.5*(1600/3/128))
- These height values are used to construct a simplified terrain mesh for distant viewing

## Version History
- **Legion**: Introduced as part of the enhanced LOD terrain system

## References
- [ADT_v18.md documentation](../../docs/ADT_v18.md) 