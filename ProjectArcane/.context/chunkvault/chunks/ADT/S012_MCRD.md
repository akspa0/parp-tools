# MCRD Chunk

## Overview
**Chunk ID**: MCRD  
**Related Expansion**: Cataclysm and later  
**Used in Format**: ADT (as MCNK subchunk)  
**Implementation Status**: Not implemented

## Description
The MCRD (Map Chunk Reference Doodads) chunk is a Cataclysm+ replacement for part of the MCRF chunk. It contains indices into the file's MDDF chunk, specifying which doodads are drawn within the current MCNK chunk. This chunk is found in the split file system, specifically in the obj0 files.

## Chunk Structure

### C++ Structure
```cpp
struct MCRD {
    uint32_t mddf_entries[]; // Doodad references into MDDF chunk - variable length array
};
```

### C# Structure
```csharp
public struct MCRD
{
    public uint[] MddfEntries; // Doodad references into MDDF chunk
}
```

## Fields

| Field | Type | Description |
|-------|------|-------------|
| mddf_entries | uint32[] | Array of indices into the MDDF chunk, specifying which doodads are drawn within this MCNK chunk |

## Related Chunks
- MCRF - Original chunk in pre-Cataclysm that combined both doodad and WMO object references
- MCRW - Counterpart to MCRD, contains WMO object references
- MDDF - Contains doodad placement information that MCRD references
- MCNK - Parent chunk that contains this subchunk

## Notes
- This chunk replaced the doodad references portion of the MCRF chunk in Cataclysm and later versions.
- The chunk is found in the obj0 files in the split file system.
- The references in this chunk need to be sorted by size category if WDT's flag 8 is set, which is an optimization to speed up culling.
- Size categories affect rendering distance - models fade when close to their maximum rendering distance and disappear when hitting it.
- If a doodad entry from MDDF is never referenced in a chunk's MCRD, it won't be drawn at all.
- The client uses these entries to calculate collision. Only objects referenced in the current chunk get checked for collision (this applies only to MDX files; WMO appear to have different collision logic).

## Size Categories and Render Distances
Approximate values for WotLK:

| Size Category Limit | Maximum Render Distance |
|---------------------|-------------------------|
| 1.0 | 30 |
| 4.0 | 150 |
| 25.0 | 300 |

The size category limits per default are 1.0, 4.0, 25.0, 100.0, 100000.0. The relevant size is the longest side of an axis-aligned bounding box (AABB) of the transformed model's bounding box from the M2 header.

## Version History
- **Cataclysm**: Introduced as part of the split file system, replacing half of the MCRF chunk functionality

## References
- [ADT_v18.md documentation](../../docs/ADT_v18.md) 