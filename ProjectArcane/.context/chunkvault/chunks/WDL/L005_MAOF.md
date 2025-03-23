# L005: MAOF

## Type
WDL Chunk

## Source
WDL_v18.md

## Description
The MAOF (Map Area OFfset) chunk contains an array of offsets to map area data (MARE chunks) within the WDL file. This is a key structural chunk that defines the 64×64 grid of the world map, with each cell corresponding to an ADT tile in the full-resolution map. The MAOF chunk enables efficient random access to specific map areas without having to parse the entire file.

## Structure
```csharp
struct MAOF
{
    /*0x00*/ uint32_t areaOffsets[64][64]; // Offsets to map areas (MARE chunks)
};
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| areaOffsets | uint32_t[64][64] | Array of offsets to MARE chunks, representing a 64×64 grid |

## Offset Values
- If the offset is 0, the corresponding map area does not exist
- If the offset is non-zero, it points to the position of a MARE chunk within the file
- Offsets are relative to the beginning of the file

## Dependencies
- MARE (L006) - Map area chunks referenced by the offsets in this chunk

## Implementation Notes
- The MAOF chunk defines the same 64×64 grid as the MAIN chunk in WDT files
- Each non-zero offset points to a MARE chunk containing low-resolution height data
- The grid coordinates correspond to the same map tiles in WDT/ADT files
- Zero offsets indicate areas without terrain (typically ocean or empty areas)
- Accessing a specific map area requires:
  1. Looking up the offset in the areaOffsets array
  2. Seeking to that position in the file
  3. Reading the MARE chunk at that location

## Implementation Example
```csharp
public class MAOF : IChunk
{
    public const int MAP_GRID_SIZE = 64;
    public uint[,] AreaOffsets { get; private set; }
    
    public MAOF()
    {
        AreaOffsets = new uint[MAP_GRID_SIZE, MAP_GRID_SIZE];
    }
    
    public void Parse(BinaryReader reader, long size)
    {
        // The size should be 64*64*4 = 16384 bytes
        if (size != MAP_GRID_SIZE * MAP_GRID_SIZE * sizeof(uint))
            throw new InvalidDataException($"MAOF chunk has invalid size: {size} (expected {MAP_GRID_SIZE * MAP_GRID_SIZE * sizeof(uint)})");
        
        // Read all offsets in row-major order
        for (int y = 0; y < MAP_GRID_SIZE; y++)
        {
            for (int x = 0; x < MAP_GRID_SIZE; x++)
            {
                AreaOffsets[y, x] = reader.ReadUInt32();
            }
        }
    }
    
    public void Write(BinaryWriter writer)
    {
        // Write all offsets in row-major order
        for (int y = 0; y < MAP_GRID_SIZE; y++)
        {
            for (int x = 0; x < MAP_GRID_SIZE; x++)
            {
                writer.Write(AreaOffsets[y, x]);
            }
        }
    }
    
    // Helper method to check if a map area exists
    public bool HasMapArea(int x, int y)
    {
        if (x < 0 || x >= MAP_GRID_SIZE || y < 0 || y >= MAP_GRID_SIZE)
            return false;
            
        return AreaOffsets[y, x] != 0;
    }
    
    // Helper method to get offset for a specific map area
    public uint GetAreaOffset(int x, int y)
    {
        if (x < 0 || x >= MAP_GRID_SIZE || y < 0 || y >= MAP_GRID_SIZE)
            return 0;
            
        return AreaOffsets[y, x];
    }
    
    // Helper method to set offset for a specific map area
    public void SetAreaOffset(int x, int y, uint offset)
    {
        if (x < 0 || x >= MAP_GRID_SIZE || y < 0 || y >= MAP_GRID_SIZE)
            return;
            
        AreaOffsets[y, x] = offset;
    }
    
    // Helper method to iterate over all existing map areas
    public IEnumerable<(int X, int Y, uint Offset)> GetExistingAreas()
    {
        for (int y = 0; y < MAP_GRID_SIZE; y++)
        {
            for (int x = 0; x < MAP_GRID_SIZE; x++)
            {
                uint offset = AreaOffsets[y, x];
                if (offset != 0)
                {
                    yield return (x, y, offset);
                }
            }
        }
    }
}
```

## Map Coordinates
The MAOF chunk's 64×64 grid corresponds to the global map coordinates used across World of Warcraft:

- The grid is laid out in row-major order, with the origin at the top-left
- The X-coordinate increases from west to east (left to right)
- The Y-coordinate increases from north to south (top to bottom)
- Each cell corresponds to an ADT tile in the high-resolution map
- Map coordinates in WDL, WDT, and ADT formats all use this same grid system

## Relationship to Other Formats
The MAOF chunk's grid directly corresponds to:
- The MAIN chunk in WDT files, which references external ADT files
- The ADT files themselves, which have filenames in the format `mapID_X_Y.adt`

However, while the WDT/ADT system provides detailed terrain data, the WDL format's MAOF/MARE system provides simplified low-resolution height data for distant rendering.

## Validation Requirements
- Must contain exactly 4096 (64×64) offset values
- Offsets must either be 0 or point to a valid position within the file 