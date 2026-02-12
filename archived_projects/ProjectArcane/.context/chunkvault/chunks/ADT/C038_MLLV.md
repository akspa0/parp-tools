# MLLV Chunk

## Overview
**Chunk ID**: MLLV  
**Related Expansion**: Legion and later  
**Used in Format**: ADT  
**Implementation Status**: Not implemented

## Description
The MLLV (Map Lod Liquid Vertices) chunk was introduced in Legion. It contains vertex data for liquid surfaces in Level of Detail (LOD) terrain rendering. This chunk works together with MLLN and MLLI chunks in a specific order to define liquid surfaces for distant terrain rendering. This chunk is found in the split file system, specifically in the lod files.

## Chunk Structure

### C++ Structure
```cpp
struct MLLV {
    C3Vector liquidVertices[]; // Variable length array of 3D vector positions
};
```

### C# Structure
```csharp
public struct MLLV
{
    public Vector3[] LiquidVertices; // Array of 3D vector positions for liquid surfaces
}
```

## Fields

| Field | Type | Description |
|-------|------|-------------|
| liquidVertices | C3Vector[] | Array of 3D vector positions defining the vertices of liquid surfaces |

## Related Chunks
- MLLN - Contains header information for liquid surfaces in LOD terrain
- MLLI - Contains index data for liquid surfaces in LOD terrain
- MLLD - Contains additional liquid data for LOD terrain
- Other ML* chunks - Various components of the LOD terrain system

## Notes
- This chunk was introduced in Legion as part of the expanded LOD system for terrain.
- The chunk is found in the lod files in the split file system.
- MLLN, MLLI, and MLLV are order-dependent. A MLLN introduces a new liquid, the following MLLI defines the indices, and the next MLLV defines the vertices.
- The C3Vector type represents a 3D vector with X, Y, and Z coordinates as float values.
- These vertices define the 3D geometry of liquid surfaces (water, lava, etc.) for distant terrain rendering.
- The vertices are typically organized into triangles using the indices defined in the MLLI chunk.

## Version History
- **Legion**: Introduced as part of the enhanced LOD terrain system

## References
- [ADT_v18.md documentation](../../docs/ADT_v18.md) 