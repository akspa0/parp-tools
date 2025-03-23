# MLHD Chunk

## Overview
**Chunk ID**: MLHD  
**Related Expansion**: Legion and later  
**Used in Format**: ADT  
**Implementation Status**: Not implemented

## Description
The MLHD (Map Lod Header) chunk was introduced in Legion. It serves as a header for the Level of Detail (LOD) data used for terrain rendering at a distance. This chunk is found in the split file system, specifically in the lod files.

## Chunk Structure

### C++ Structure
```cpp
struct MLHD {
    uint32_t unknown;
    float boundingBox[6]; // Likely min/max values for the bounding box
};
```

### C# Structure
```csharp
public struct MLHD
{
    public uint Unknown;
    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 6)]
    public float[] BoundingBox; // Likely [minX, minY, minZ, maxX, maxY, maxZ]
}
```

## Fields

| Field | Type | Description |
|-------|------|-------------|
| unknown | uint32_t | Unknown value |
| boundingBox | float[6] | Six floating point values that likely define a bounding box for the LOD terrain (minX, minY, minZ, maxX, maxY, maxZ) |

## Related Chunks
- MLVH - Contains height data for LOD terrain
- MLVI - Contains vertex indices for LOD terrain
- Other ML* chunks - Various components of the LOD terrain system

## Notes
- This chunk was introduced in Legion as part of the expanded LOD system for terrain.
- The chunk is found in the lod files in the split file system.
- The bounding box values are likely used for efficient culling of LOD terrain at runtime.
- LOD (Level of Detail) terrain is used for distant views to improve performance while maintaining visual quality.

## Version History
- **Legion**: Introduced as part of the enhanced LOD terrain system

## References
- [ADT_v18.md documentation](../../docs/ADT_v18.md) 