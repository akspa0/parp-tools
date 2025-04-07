# L008: MAHO

## Type
WDL Chunk

## Source
WDL_v18.md

## Description
The MAHO (Map Height Offset) chunk contains an array of offsets to heightmap data (MAHE chunks) within the WDL file. Similar to the MAOF chunk, MAHO provides a lookup table for quickly accessing height data for specific map areas. This chunk is an alternative access method to heightmap data, complementing the references through MARE chunks.

## Structure
```csharp
struct MAHO
{
    /*0x00*/ uint32_t heightMapOffsets[64][64]; // Array of offsets to MAHE chunks
};
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| heightMapOffsets | uint32_t[64][64] | Array of offsets pointing to MAHE chunks within the file |

## Offset Values
- Each offset is a 32-bit unsigned integer (uint32_t)
- An offset of 0 indicates that no heightmap data exists for that map area
- Non-zero offsets represent the byte position of MAHE chunks within the file
- The array is organized as a 64×64 grid corresponding to the world map structure

## Dependencies
- MAHE (L007) - The height data chunks referenced by the offsets in MAHO
- MAOF (L005) - Similar structure; MAOF references MARE chunks which in turn reference MAHE chunks

## Implementation Notes
- The MAHO chunk provides direct access to heightmap data, bypassing the need to go through MARE chunks
- This creates an alternative access path: MAHO → MAHE vs. MAOF → MARE → MAHE
- The 64×64 grid corresponds to the same grid in the MAOF chunk and the MAIN chunk in WDT files
- Not all positions in the grid will have valid heightmap data
- The coordinate system follows the same conventions as other map data structures

## Implementation Example
```csharp
public class MAHO : IChunk
{
    public const int GRID_SIZE_X = 64;
    public const int GRID_SIZE_Y = 64;
    
    public uint[,] HeightMapOffsets { get; private set; }
    
    public MAHO()
    {
        HeightMapOffsets = new uint[GRID_SIZE_Y, GRID_SIZE_X];
    }
    
    public void Parse(BinaryReader reader, long size)
    {
        if (size != GRID_SIZE_X * GRID_SIZE_Y * sizeof(uint))
            throw new InvalidDataException($"MAHO chunk has invalid size: {size} (expected {GRID_SIZE_X * GRID_SIZE_Y * sizeof(uint)})");
        
        // Read all offsets
        for (int y = 0; y < GRID_SIZE_Y; y++)
        {
            for (int x = 0; x < GRID_SIZE_X; x++)
            {
                HeightMapOffsets[y, x] = reader.ReadUInt32();
            }
        }
    }
    
    public void Write(BinaryWriter writer)
    {
        // Write all offsets
        for (int y = 0; y < GRID_SIZE_Y; y++)
        {
            for (int x = 0; x < GRID_SIZE_X; x++)
            {
                writer.Write(HeightMapOffsets[y, x]);
            }
        }
    }
    
    // Helper method to check if a map area has heightmap data
    public bool HasHeightMapData(int x, int y)
    {
        if (x < 0 || x >= GRID_SIZE_X || y < 0 || y >= GRID_SIZE_Y)
            return false;
            
        return HeightMapOffsets[y, x] != 0;
    }
    
    // Helper method to get the offset to a heightmap
    public uint GetHeightMapOffset(int x, int y)
    {
        if (x < 0 || x >= GRID_SIZE_X || y < 0 || y >= GRID_SIZE_Y)
            return 0;
            
        return HeightMapOffsets[y, x];
    }
    
    // Helper method to set the offset to a heightmap
    public void SetHeightMapOffset(int x, int y, uint offset)
    {
        if (x < 0 || x >= GRID_SIZE_X || y < 0 || y >= GRID_SIZE_Y)
            return;
            
        HeightMapOffsets[y, x] = offset;
    }
    
    // Helper method to get a list of all valid map areas with height data
    public List<(int x, int y)> GetValidMapAreas()
    {
        var result = new List<(int x, int y)>();
        
        for (int y = 0; y < GRID_SIZE_Y; y++)
        {
            for (int x = 0; x < GRID_SIZE_X; x++)
            {
                if (HasHeightMapData(x, y))
                    result.Add((x, y));
            }
        }
        
        return result;
    }
}
```

## Access Patterns
The MAHO chunk provides two potential access patterns for retrieving heightmap data:

1. **Direct access**: MAHO → MAHE
   - Use the offset from MAHO to directly access the MAHE chunk
   - More efficient when only height data is needed

2. **Indirect access**: MAOF → MARE → MAHE
   - Use the offset from MAOF to access the MARE chunk
   - The MARE chunk provides the offset to the MAHE chunk, plus metadata
   - More efficient when metadata (like height scale) is also needed

## Coordinate System
The MAHO chunk uses a 64×64 coordinate system:
- X coordinate: 0 to 63, increasing from west to east
- Y coordinate: 0 to 63, increasing from north to south
- This matches the coordinate system used in the MAOF chunk and the MAIN chunk in WDT files

## Relationship to WDT
The MAHO chunk's 64×64 grid directly corresponds to the same grid in the WDT file:
- Each position (x, y) in MAHO refers to the same geographical area as position (x, y) in the WDT file
- If a WDT area has terrain (ADT file), the corresponding WDL position typically has height data

## Optimization Purpose
The MAHO chunk provides an optimization for client performance:
- Allows direct access to height data without parsing intermediate chunks
- Useful for distant terrain rendering where only basic height information is needed
- The client can quickly load low-resolution terrain for large view distances

## Version Differences
- The structure of MAHO remains consistent across different versions
- The format of the referenced MAHE chunks may vary between versions 