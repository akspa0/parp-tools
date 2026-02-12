# MLLN Chunk

## Overview
**Chunk ID**: MLLN  
**Related Expansion**: Legion and later  
**Used in Format**: ADT  
**Implementation Status**: Not implemented

## Description
The MLLN (Map Lod Liquid Node) chunk was introduced in Legion. It contains header information for liquid rendering in Level of Detail (LOD) terrain. This chunk works together with MLLI and MLLV chunks in a specific order to define liquid surfaces for distant terrain rendering. This chunk is found in the split file system, specifically in the lod files.

## Chunk Structure

### C++ Structure
```cpp
struct MLLN {
    uint32_t unknown1;
    uint32_t numIndices;  // Number of indices in the following MLLI chunk
    uint32_t unknown2;
    uint16_t unknown3a;
    uint16_t unknown3b;
    uint32_t unknown4;
    uint32_t unknown5;
};
```

### C# Structure
```csharp
public struct MLLN
{
    public uint Unknown1;
    public uint NumIndices;  // Number of indices in the following MLLI chunk
    public uint Unknown2;
    public ushort Unknown3a;
    public ushort Unknown3b;
    public uint Unknown4;
    public uint Unknown5;
}
```

## Fields

| Field | Type | Description |
|-------|------|-------------|
| unknown1 | uint32_t | Unknown value |
| numIndices | uint32_t | Number of indices in the following MLLI chunk |
| unknown2 | uint32_t | Unknown value |
| unknown3a | uint16_t | Unknown value |
| unknown3b | uint16_t | Unknown value |
| unknown4 | uint32_t | Unknown value |
| unknown5 | uint32_t | Unknown value |

## Related Chunks
- MLLI - Contains index data for liquid surfaces in LOD terrain
- MLLV - Contains vertex data for liquid surfaces in LOD terrain
- MLLD - Contains additional liquid data for LOD terrain
- Other ML* chunks - Various components of the LOD terrain system

## Notes
- This chunk was introduced in Legion as part of the expanded LOD system for terrain.
- The chunk is found in the lod files in the split file system.
- MLLN, MLLI, and MLLV are order-dependent. A MLLN introduces a new liquid, the following MLLI defines the indices, and the next MLLV defines the vertices.
- This system allows for efficient rendering of liquid surfaces (water, lava, etc.) at a distance.
- The exact purpose of most fields is not fully documented in the original source.

## Version History
- **Legion**: Introduced as part of the enhanced LOD terrain system

## References
- [ADT_v18.md documentation](../../docs/ADT_v18.md) 