# MCDD Chunk

## Overview
**Chunk ID**: MCDD  
**Related Expansion**: Cataclysm and later  
**Used in Format**: ADT (as MCNK subchunk)  
**Implementation Status**: Not implemented

## Description
The MCDD (Map Chunk Detail Doodads) chunk was introduced in Cataclysm. It contains a bitmap that specifies where detail doodads (small objects like grass, flowers, etc.) should be disabled in the current MCNK chunk. This is generally found in the root ADT file in the split file system.

## Chunk Structure

### C++ Structure
```cpp
struct MCDD {
    uint8_t disable[8]; // Bitmap of 8x8 cells where detail doodads are disabled
    // Note: There appears to be a high-resolution mode (16x16) which uses 32 bytes
    // instead of 8 bytes, but this doesn't seem to be used in live clients
};
```

### C# Structure
```csharp
public struct MCDD
{
    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 8)]
    public byte[] DisableMask; // Bitmap for disabling detail doodads
    
    // Helper method to check if detail doodads are disabled at a specific position
    public bool IsDisabled(int x, int y)
    {
        if (x < 0 || x >= 8 || y < 0 || y >= 8)
            return false;
            
        // Each bit in the byte represents a cell
        int byteIndex = y;
        int bitMask = 1 << x;
        
        return (DisableMask[byteIndex] & bitMask) != 0;
    }
}
```

## Fields

| Field | Type | Description |
|-------|------|-------------|
| disable | uint8_t[8] | Bitmap where each bit represents whether detail doodads are disabled in an 8x8 grid cell |

## Related Chunks
- MCNK - Parent chunk that contains this subchunk

## Notes
- This chunk was introduced in Cataclysm as part of the split file system.
- The bitmap allows for fine control over where small detail objects like grass are rendered within a chunk.
- Each bit in the bitmap represents a specific area within the MCNK chunk.
- There appears to be an unused high-resolution mode that would use a 16x16 grid (32 bytes instead of 8), but this doesn't seem to be used in live clients.
- When this chunk is inlined directly in MCNK, it uses the low-resolution 8x8 format.
- This chunk has been observed in Warlords of Draenor and later versions.

## Version History
- **Cataclysm**: Likely introduced as part of the split file system
- **Warlords of Draenor**: Confirmed to be present

## References
- [ADT_v18.md documentation](../../docs/ADT_v18.md) 