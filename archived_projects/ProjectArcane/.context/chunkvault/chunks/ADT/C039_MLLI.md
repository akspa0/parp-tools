# MLLI Chunk

## Overview
**Chunk ID**: MLLI  
**Related Expansion**: Legion and later  
**Used in Format**: ADT  
**Implementation Status**: Not implemented

## Description
The MLLI (Map Lod Liquid Indices) chunk was introduced in Legion. It contains index data for liquid surfaces in Level of Detail (LOD) terrain rendering. This chunk works together with MLLN and MLLV chunks in a specific order to define liquid surfaces for distant terrain rendering. This chunk is found in the split file system, specifically in the lod files.

## Chunk Structure

### C++ Structure
```cpp
struct MLLI {
    C3sVector liquidIndices[]; // Variable length array of index triplets (3 shorts per entry)
};
```

### C# Structure
```csharp
public struct C3sVector
{
    public short X;
    public short Y;
    public short Z;
}

public struct MLLI
{
    public C3sVector[] LiquidIndices; // Array of index triplets for liquid surfaces
}
```

## Fields

| Field | Type | Description |
|-------|------|-------------|
| liquidIndices | C3sVector[] | Array of index triplets (3 shorts per entry) that reference vertices in the MLLV chunk |

## Related Chunks
- MLLN - Contains header information for liquid surfaces in LOD terrain
- MLLV - Contains vertex data for liquid surfaces in LOD terrain
- MLLD - Contains additional liquid data for LOD terrain
- Other ML* chunks - Various components of the LOD terrain system

## Notes
- This chunk was introduced in Legion as part of the expanded LOD system for terrain.
- The chunk is found in the lod files in the split file system.
- MLLN, MLLI, and MLLV are order-dependent. A MLLN introduces a new liquid, the following MLLI defines the indices, and the next MLLV defines the vertices.
- Each C3sVector contains three short integers, which are indices into the MLLV chunk's vertex array.
- These index triplets define triangles for the liquid surface mesh.
- The number of indices is specified in the `numIndices` field of the preceding MLLN chunk.

## Version History
- **Legion**: Introduced as part of the enhanced LOD terrain system

## References
- [ADT_v18.md documentation](../../docs/ADT_v18.md) 