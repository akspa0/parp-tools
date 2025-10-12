# MLSI Chunk

## Overview
**Chunk ID**: MLSI  
**Related Expansion**: Legion and later  
**Used in Format**: ADT  
**Implementation Status**: Not implemented

## Description
The MLSI (Map Lod Skirt Indices) chunk was introduced in Legion. It contains index data for creating "skirts" around the edges of terrain LOD meshes. Skirts are vertical polygons that hang down from the edges of terrain meshes to prevent gaps when transitioning between different LOD levels. This chunk is found in the split file system, specifically in the lod files.

## Chunk Structure

### C++ Structure
```cpp
struct MLSI {
    uint16_t skirtIndices[]; // Variable length array of vertex indices into MLVH
};
```

### C# Structure
```csharp
public struct MLSI
{
    public ushort[] SkirtIndices; // Array of vertex indices for terrain skirts
}
```

## Fields

| Field | Type | Description |
|-------|------|-------------|
| skirtIndices | uint16_t[] | Array of vertex indices into MLVH for creating skirt geometry around terrain edges |

## Related Chunks
- MLHD - Header for LOD terrain data
- MLVH - Contains height data for LOD terrain vertices that MLSI references
- MLVI - Contains vertex indices for main LOD terrain
- Other ML* chunks - Various components of the LOD terrain system

## Notes
- This chunk was introduced in Legion as part of the expanded LOD system for terrain.
- The chunk is found in the lod files in the split file system.
- The indices in this chunk typically reference only the 129*129 part of the MLVH chunk.
- Skirts are used to prevent visible gaps or "cracks" between different LOD levels or adjacent terrain tiles.
- These vertical polygons extend down from the edges of the terrain tiles to ensure visual continuity.
- Without skirts, gaps might be visible when viewing terrain from certain angles, especially when different LOD levels meet.

## Version History
- **Legion**: Introduced as part of the enhanced LOD terrain system

## References
- [ADT_v18.md documentation](../../docs/ADT_v18.md) 