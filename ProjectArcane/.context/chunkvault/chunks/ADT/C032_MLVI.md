# MLVI Chunk

## Overview
**Chunk ID**: MLVI  
**Related Expansion**: Legion and later  
**Used in Format**: ADT  
**Implementation Status**: Not implemented

## Description
The MLVI (Map Lod Vertex Indices) chunk was introduced in Legion. It contains index data for the Level of Detail (LOD) terrain mesh used for distant terrain rendering. These indices define how the vertices from the MLVH chunk should be connected to form triangles. This chunk is found in the split file system, specifically in the lod files.

## Chunk Structure

### C++ Structure
```cpp
struct MLVI {
    uint16_t indices[]; // Variable length array of vertex indices
};
```

### C# Structure
```csharp
public struct MLVI
{
    public ushort[] Indices; // Array of vertex indices for LOD terrain
}
```

## Fields

| Field | Type | Description |
|-------|------|-------------|
| indices | uint16_t[] | Array of vertex indices that define triangles for the LOD terrain mesh |

## Related Chunks
- MLHD - Header for LOD terrain data
- MLVH - Contains height data for LOD terrain vertices
- Other ML* chunks - Various components of the LOD terrain system

## Notes
- This chunk was introduced in Legion as part of the expanded LOD system for terrain.
- The chunk is found in the lod files in the split file system.
- The indices are to be used with the GL_TRIANGLES primitive type, meaning every three consecutive indices define one triangle.
- These indices reference vertices defined in the MLVH chunk.
- The index data helps optimize the LOD terrain mesh by allowing vertex reuse and enabling efficient rendering.

## Version History
- **Legion**: Introduced as part of the enhanced LOD terrain system

## References
- [ADT_v18.md documentation](../../docs/ADT_v18.md) 