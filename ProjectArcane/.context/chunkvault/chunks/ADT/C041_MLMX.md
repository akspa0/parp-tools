# MLMX Chunk

## Overview
**Chunk ID**: MLMX  
**Related Expansion**: Legion and later  
**Used in Format**: ADT  
**Implementation Status**: Not implemented

## Description
The MLMX (Map Lod Model Extents) chunk was introduced in Legion. It contains bounding box and radius information for objects placed in Level of Detail (LOD) terrain. This chunk complements the MLMD chunk, providing spatial information for efficient culling of objects in distant terrain rendering. This chunk is found in the split file system, specifically in the obj1 files.

## Chunk Structure

### C++ Structure
```cpp
struct MLMX {
    CAaBox bounding;  // Axis-aligned bounding box (min/max points)
    float radius;     // Radius of the object for culling purposes
};
```

### C# Structure
```csharp
public struct CAaBox
{
    public Vector3 Min; // Minimum point of the bounding box
    public Vector3 Max; // Maximum point of the bounding box
}

public struct MLMX
{
    public CAaBox Bounding; // Axis-aligned bounding box
    public float Radius;    // Radius for culling
}
```

## Fields

| Field | Type | Description |
|-------|------|-------------|
| bounding | CAaBox | Axis-aligned bounding box (min/max points) of the transformed model |
| radius | float | Radius of the object for culling purposes |

## Related Chunks
- MLMD - Contains object placement data for LOD terrain
- Other ML* chunks - Various components of the LOD terrain system

## Notes
- This chunk was introduced in Legion as part of the expanded LOD system for terrain.
- The chunk is found in the obj1 files in the split file system.
- The bounding box is for the transformed model, i.e., the bounding box from inside the file, rotated and scaled, and then the bounding box of that.
- The radius is generally approximately around 50 units (Radius is typically calculated as bounding.Max.X - bounding.Min.X).
- The visibility of objects depends both on this radius and the view distance setting (possibly using a factor like radius * viewDistanceFactor).
- The CAaBox is defined with a max point and a min point. The point coordinates are server coordinates, so you should take the object position in MODF (for WMO) or MDDF (for M2) and convert it from client coordinates to server coordinates.
- The entries in MLMD appear to be sorted based on the radius in MLMX, from largest to smallest, likely for optimization.
- This spatial information is used for efficient culling of objects in distant terrain rendering, allowing the game to determine which objects should be visible from a specific viewing distance.

## Version History
- **Legion**: Introduced as part of the enhanced LOD terrain system

## References
- [ADT_v18.md documentation](../../docs/ADT_v18.md) 