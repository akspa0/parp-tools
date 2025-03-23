# C013: MFBO

## Type
ADT Chunk

## Source
ADT_v18.md

## Description
Flight Boundaries Object - defines the minimum and maximum heights for flying mounts in a specific area. This chunk was introduced in Burning Crusade and controls where players can fly with their flying mounts.

## Structure
```csharp
struct SMMapObjFlightBoundary
{
    /*0x00*/ int16_t heightMaximums[9][9];      // Maximum fly height for 9x9 grid
    /*0x81*/ int16_t heightMinimums[9][9];      // Minimum fly height for 9x9 grid
}
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| heightMaximums | int16[9][9] | 9x9 grid of maximum height values for flying |
| heightMinimums | int16[9][9] | 9x9 grid of minimum height values for flying |

## Dependencies
- MHDR (C002) - The MHDR.flags field must have bit 0 (mhdr_MFBO = 1) set to indicate that this chunk exists

## Implementation Notes
- Each ADT tile is divided into a 9x9 grid for flight boundaries
- Heights are specified as int16 values rather than floats to save space
- A height of 0 represents sea level
- Negative values represent areas below sea level
- The grid coordinates (0,0) represent the southwest corner of the ADT tile
- The grid coordinates (8,8) represent the northeast corner of the ADT tile
- The grid spans the entire ADT tile (533.33 x 533.33 yards)
- Height values are stored in row-major order (i.e., [y][x])
- If flying is not allowed at all, both heightMinimums and heightMaximums will typically have very restrictive values
- If flying is unrestricted, heightMaximums will be set to a very large value (typically 32767) and heightMinimums to a very small value (typically -32768)

## Implementation Example
```csharp
public class MFBO : IChunk
{
    public const int GRID_SIZE = 9;
    public const float TILE_SIZE = 533.33333f;
    public const float CELL_SIZE = TILE_SIZE / (GRID_SIZE - 1);
    
    public short[,] MaximumHeights { get; private set; }
    public short[,] MinimumHeights { get; private set; }
    
    public MFBO(BinaryReader reader, uint size)
    {
        // Initialize height grids
        MaximumHeights = new short[GRID_SIZE, GRID_SIZE];
        MinimumHeights = new short[GRID_SIZE, GRID_SIZE];
        
        // Read maximum heights (9x9 grid)
        for (int y = 0; y < GRID_SIZE; y++)
        {
            for (int x = 0; x < GRID_SIZE; x++)
            {
                MaximumHeights[y, x] = reader.ReadInt16();
            }
        }
        
        // Read minimum heights (9x9 grid)
        for (int y = 0; y < GRID_SIZE; y++)
        {
            for (int x = 0; x < GRID_SIZE; x++)
            {
                MinimumHeights[y, x] = reader.ReadInt16();
            }
        }
    }
    
    // Get the flight boundaries at a specific position within the ADT tile
    public (float Min, float Max) GetFlightBoundariesAt(float x, float y)
    {
        // Convert world coordinates to grid coordinates
        float gridX = x / CELL_SIZE;
        float gridY = y / CELL_SIZE;
        
        // Get grid cell indices
        int x1 = (int)Math.Floor(gridX);
        int y1 = (int)Math.Floor(gridY);
        int x2 = Math.Min(x1 + 1, GRID_SIZE - 1);
        int y2 = Math.Min(y1 + 1, GRID_SIZE - 1);
        
        // Clamp to valid range
        x1 = Math.Max(0, Math.Min(x1, GRID_SIZE - 1));
        y1 = Math.Max(0, Math.Min(y1, GRID_SIZE - 1));
        
        // Get fractional components for bilinear interpolation
        float fracX = gridX - x1;
        float fracY = gridY - y1;
        
        // Bilinearly interpolate minimum heights
        float minHeight = 
            MinimumHeights[y1, x1] * (1 - fracX) * (1 - fracY) +
            MinimumHeights[y1, x2] * fracX * (1 - fracY) +
            MinimumHeights[y2, x1] * (1 - fracX) * fracY +
            MinimumHeights[y2, x2] * fracX * fracY;
        
        // Bilinearly interpolate maximum heights
        float maxHeight = 
            MaximumHeights[y1, x1] * (1 - fracX) * (1 - fracY) +
            MaximumHeights[y1, x2] * fracX * (1 - fracY) +
            MaximumHeights[y2, x1] * (1 - fracX) * fracY +
            MaximumHeights[y2, x2] * fracX * fracY;
        
        return (minHeight, maxHeight);
    }
    
    // Check if flying is allowed at a specific position and height
    public bool IsFlightAllowed(float x, float y, float height)
    {
        var (min, max) = GetFlightBoundariesAt(x, y);
        return height >= min && height <= max;
    }
}
```

## Usage Context
The MFBO chunk is used to:

1. Define where players can fly with flying mounts
2. Enforce minimum and maximum flight altitudes in specific areas
3. Create invisible barriers for flying mounts
4. Prevent players from accessing unfinished or out-of-bounds areas

Examples of usage in-game:
- Preventing players from flying over mountains to skip content
- Restricting flight near the edges of the map
- Creating flight corridors through mountain passes
- Setting minimum altitudes over dangerous areas
- Restricting maximum flight height to prevent players from seeing unfinished areas

MFBO was introduced in Burning Crusade when flying mounts were first implemented, and provides a more sophisticated system than simple on/off flying restrictions. 