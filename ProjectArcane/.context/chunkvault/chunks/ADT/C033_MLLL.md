# MLLL Chunk

## Overview
**Chunk ID**: MLLL  
**Related Expansion**: Legion and later  
**Used in Format**: ADT  
**Implementation Status**: Not implemented

## Description
The MLLL (Map Lod Level List) chunk was introduced in Legion. It contains Level of Detail (LOD) information for terrain rendering at different distances. This chunk defines different LOD bands and their corresponding vertex index ranges. This chunk is found in the split file system, specifically in the lod files.

## Chunk Structure

### C++ Structure
```cpp
struct MLLLEntry {
    float lod;                  // LOD band value (32, 16, 8, 4, 2)
    uint32_t heightIndex;       // Index into MLVI for height data
    uint32_t heightLength;      // Number of indices to use from MLVI for height data
    uint32_t mapAreaLowLength;  // Number of indices to use from MLVI for low-detail map area
    uint32_t mapAreaLowIndex;   // Index into MLVI for low-detail map area
};

struct MLLL {
    MLLLEntry entries[]; // Variable length array of LOD level entries
};
```

### C# Structure
```csharp
public struct MLLLEntry
{
    public float Lod;                // LOD band value
    public uint HeightIndex;         // Index into MLVI
    public uint HeightLength;        // Number of indices to use
    public uint MapAreaLowLength;    // Number of indices for low-detail area
    public uint MapAreaLowIndex;     // Index into MLVI for low-detail area
}

public struct MLLL
{
    public MLLLEntry[] Entries;      // Array of LOD level entries
}
```

## Fields

| Field | Type | Description |
|-------|------|-------------|
| lod | float | LOD band value (typically 32, 16, 8, 4, 2) where lower values indicate more detailed surfaces |
| heightIndex | uint32_t | Index into the MLVI chunk for the height data at this LOD level |
| heightLength | uint32_t | Number of indices to use from MLVI for height data |
| mapAreaLowLength | uint32_t | Number of indices to use from MLVI for low-detail map area |
| mapAreaLowIndex | uint32_t | Index into MLVI for low-detail map area data |

## Related Chunks
- MLHD - Header for LOD terrain data
- MLVH - Contains height data for LOD terrain vertices
- MLVI - Contains vertex indices for LOD terrain
- MLND - Defines a quad tree for LOD management
- Other ML* chunks - Various components of the LOD terrain system

## Notes
- This chunk was introduced in Legion as part of the expanded LOD system for terrain.
- The chunk is found in the lod files in the split file system.
- LOD values work in reverse - lower LOD values indicate more detailed terrain surfaces.
- The LOD levels correspond to different view distances:
  - LOD 32 = 16*8 chunks (least detailed, furthest distance)
  - LOD 16 = 8*8 chunks
  - LOD 8 = 4*8 chunks
  - LOD 4 = 2*8 chunks
  - LOD 2 = 8 chunks (most detailed, closest distance)
- LOD 2 is essentially the same as rendering terrain from the MCNK of the main ADT.
- The `mapAreaLow` data corresponds to the same data contained in WDL files, and is 0 for the most detailed layer.
- MLLL defines ranges of LOD in MLVI, and is used to determine LOD levels of nodes in MLND by comparison of referenced ranges in MLVI.

## Version History
- **Legion**: Introduced as part of the enhanced LOD terrain system

## References
- [ADT_v18.md documentation](../../docs/ADT_v18.md) 