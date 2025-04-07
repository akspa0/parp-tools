# MLND Chunk

## Overview
**Chunk ID**: MLND  
**Related Expansion**: Legion and later  
**Used in Format**: ADT  
**Implementation Status**: Not implemented

## Description
The MLND (Map Lod Node Data) chunk was introduced in Legion. It defines a quad tree structure for managing Level of Detail (LOD) in terrain rendering. This chunk is found in the split file system, specifically in the lod files.

## Chunk Structure

### C++ Structure
```cpp
struct MLNDEntry {
    uint32_t index;       // Index into MLVI
    uint32_t length;      // Number of elements in MLVI used
    uint32_t unknown1;    // Unknown value
    uint32_t unknown2;    // Unknown value
    uint16_t indices[4];  // Indices into MLND for child leaves (quad tree structure)
};

struct MLND {
    MLNDEntry entries[]; // Variable length array of quad tree nodes
};
```

### C# Structure
```csharp
public struct MLNDEntry
{
    public uint Index;      // Index into MLVI
    public uint Length;     // Number of elements in MLVI used
    public uint Unknown1;   // Unknown value
    public uint Unknown2;   // Unknown value
    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 4)]
    public ushort[] ChildIndices; // Indices into MLND for child nodes
}

public struct MLND
{
    public MLNDEntry[] Entries; // Array of quad tree nodes
}
```

## Fields

| Field | Type | Description |
|-------|------|-------------|
| index | uint32_t | Index into the MLVI chunk for this node's vertex data |
| length | uint32_t | Number of elements in MLVI to use for this node |
| unknown1 | uint32_t | Unknown value |
| unknown2 | uint32_t | Unknown value |
| indices | uint16_t[4] | Array of indices into MLND for the four child nodes of this quad tree node |

## Related Chunks
- MLHD - Header for LOD terrain data
- MLVH - Contains height data for LOD terrain vertices
- MLVI - Contains vertex indices for LOD terrain
- MLLL - Defines LOD bands and corresponding index ranges
- Other ML* chunks - Various components of the LOD terrain system

## Notes
- This chunk was introduced in Legion as part of the expanded LOD system for terrain.
- The chunk is found in the lod files in the split file system.
- MLND defines a quad tree structure, where each node is twice as narrow as its parent.
- Each node references a section of the MLVI data to use for rendering at that LOD level.
- The quad tree structure allows for efficient LOD selection based on viewing distance.
- MLND works in conjunction with MLLL - MLLL defines ranges of LOD in MLVI, and MLLL is used to determine LOD level of nodes in MLND by comparison of referenced ranges in MLVI.
- Since this is a quad tree, each non-leaf node has exactly four children (hence the array of four indices).
- The quad tree nature of this structure naturally corresponds to how terrain detail increases at closer view distances.

## Version History
- **Legion**: Introduced as part of the enhanced LOD terrain system

## References
- [ADT_v18.md documentation](../../docs/ADT_v18.md) 