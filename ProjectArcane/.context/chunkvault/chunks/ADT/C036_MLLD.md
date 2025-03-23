# MLLD Chunk

## Overview
**Chunk ID**: MLLD  
**Related Expansion**: Legion and later  
**Used in Format**: ADT  
**Implementation Status**: Not implemented

## Description
The MLLD (Map Lod Liquid Data) chunk was introduced in Legion. It contains liquid data for Level of Detail (LOD) terrain rendering, allowing for efficient rendering of water and other liquid surfaces at a distance. This chunk is found in the split file system, specifically in the lod files.

## Chunk Structure

### C++ Structure
```cpp
struct MLLD {
    enum Flags {
        HasTileData = 0x1,     // Indicates presence of tile data
        CompressedDepth = 0x2, // If set, depth texture is compressed; otherwise it is uncompressed
        CompressedAlpha = 0x4  // If set, alpha texture is compressed
    };
    
    uint32_t flags;            // Bitfield of Flags
    // Additional fields would follow, likely variable in size based on flags
    // The exact structure isn't fully documented in the original source
};
```

### C# Structure
```csharp
public struct MLLD
{
    [Flags]
    public enum LiquidFlags : uint
    {
        HasTileData = 0x1,
        CompressedDepth = 0x2,
        CompressedAlpha = 0x4
    }
    
    public LiquidFlags Flags;
    
    // Additional fields would be included here based on flags
    // These would likely include data for liquid heights, alpha values, etc.
    
    public bool HasTileData => (Flags & LiquidFlags.HasTileData) != 0;
    public bool HasCompressedDepth => (Flags & LiquidFlags.CompressedDepth) != 0;
    public bool HasCompressedAlpha => (Flags & LiquidFlags.CompressedAlpha) != 0;
}
```

## Fields

| Field | Type | Description |
|-------|------|-------------|
| flags | uint32_t | Bitfield indicating various attributes of the liquid data |

## Flag Values

| Flag | Value | Description |
|------|-------|-------------|
| HasTileData | 0x1 | Indicates the presence of tile data |
| CompressedDepth | 0x2 | If set, depth texture is compressed; otherwise it is uncompressed data |
| CompressedAlpha | 0x4 | If set, alpha texture is compressed |

## Related Chunks
- MLHD - Header for LOD terrain data
- Other ML* chunks - Various components of the LOD terrain system
- MH2O - Regular (non-LOD) liquid data in ADT files

## Notes
- This chunk was introduced in Legion as part of the expanded LOD system for terrain.
- The chunk is found in the lod files in the split file system.
- The full structure following the flags field is not completely documented in the original source.
- The chunk likely contains simplified liquid data for distant viewing of water and other liquid surfaces.
- The compression flags indicate that some texture data may be stored in a compressed format to save space.
- This chunk works in conjunction with other LOD chunks to provide a complete distant terrain rendering solution.

## Version History
- **Legion**: Introduced as part of the enhanced LOD terrain system

## References
- [ADT_v18.md documentation](../../docs/ADT_v18.md) 