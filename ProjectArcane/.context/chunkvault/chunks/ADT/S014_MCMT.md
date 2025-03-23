# MCMT Chunk

## Overview
**Chunk ID**: MCMT  
**Related Expansion**: Cataclysm and later  
**Used in Format**: ADT (as MCNK subchunk)  
**Implementation Status**: Not implemented

## Description
The MCMT (Map Chunk Material Table) chunk was introduced in Cataclysm. It contains material identifiers for terrain textures used in the MCNK chunk. This chunk is found in the split file system, specifically in the tex0 files.

## Chunk Structure

### C++ Structure
```cpp
struct MCMT {
    uint8_t material_id[4]; // Material IDs per texture layer (references TerrainMaterial table)
};
```

### C# Structure
```csharp
public struct MCMT
{
    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 4)]
    public byte[] MaterialIds; // Material IDs per texture layer
}
```

## Fields

| Field | Type | Description |
|-------|------|-------------|
| material_id | uint8_t[4] | Material identifiers for up to 4 texture layers, referencing the TerrainMaterial DB2/DBC table |

## Related Chunks
- MCNK - Parent chunk that contains this subchunk
- MCLY - Texture layer definitions that correspond to these material IDs

## Notes
- This chunk was introduced in Cataclysm as part of the split file system.
- It is stored in the tex0 files.
- The material IDs reference the TerrainMaterial database table, which defines various terrain properties.
- Each material ID corresponds to a texture layer defined in the MCLY chunk.
- These material IDs can affect terrain properties like sound when walking, weather effects, etc.

## Version History
- **Cataclysm**: Introduced as part of the split file system

## References
- [ADT_v18.md documentation](../../docs/ADT_v18.md) 