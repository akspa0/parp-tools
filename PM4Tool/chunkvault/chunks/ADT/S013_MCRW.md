# MCRW Chunk

## Overview
**Chunk ID**: MCRW  
**Related Expansion**: Cataclysm and later  
**Used in Format**: ADT (as MCNK subchunk)  
**Implementation Status**: Not implemented

## Description
The MCRW (Map Chunk Reference WMOs) chunk is a Cataclysm+ replacement for part of the MCRF chunk. It contains indices into the file's MODF chunk, specifying which WMO objects are drawn within the current MCNK chunk. This chunk is found in the split file system, specifically in the obj0 files.

## Chunk Structure

### C++ Structure
```cpp
struct MCRW {
    uint32_t modf_entries[]; // WMO object references into MODF chunk - variable length array
};
```

### C# Structure
```csharp
public struct MCRW
{
    public uint[] ModfEntries; // WMO object references into MODF chunk
}
```

## Fields

| Field | Type | Description |
|-------|------|-------------|
| modf_entries | uint32[] | Array of indices into the MODF chunk, specifying which WMO objects are drawn within this MCNK chunk |

## Related Chunks
- MCRF - Original chunk in pre-Cataclysm that combined both doodad and WMO object references
- MCRD - Counterpart to MCRW, contains doodad references
- MODF - Contains WMO object placement information that MCRW references
- MCNK - Parent chunk that contains this subchunk

## Notes
- This chunk replaced the WMO object references portion of the MCRF chunk in Cataclysm and later versions.
- The chunk is found in the obj0 files in the split file system.
- If a WMO object entry from MODF is never referenced in a chunk's MCRW, it won't be drawn at all.
- The client uses the MCRF/MCRW/MCRD structures to determine which objects should be considered for collision within a terrain chunk.
- WMO objects appear to have different collision logic compared to doodads (M2/MDX models).

## Version History
- **Cataclysm**: Introduced as part of the split file system, replacing half of the MCRF chunk functionality

## References
- [ADT_v18.md documentation](../../docs/ADT_v18.md) 