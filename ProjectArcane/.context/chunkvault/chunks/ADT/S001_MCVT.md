# S001: MCVT

## Type
MCNK Subchunk

## Source
ADT_v18.md

## Description
The MCVT (Map Chunk Vertex Table) subchunk contains the height map data for a single MCNK chunk. It consists of 145 height values that define the terrain shape.

## Structure
```csharp
struct MCVT
{
    /*0x00*/ float heights[145];  // 9x9 grid + middle points
};
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| heights | float[145] | Array of 145 height values defining the terrain shape |

## Height Map Layout
The 145 height values represent a 9×9 grid of vertices plus the middle points:
- 9×9 = 81 vertices at the corners of the grid cells
- 8×8 = 64 vertices at the midpoints of the grid cells

The height values are stored in the following order:
1. The outer 9×9 grid vertices (81 values)
2. The inner 8×8 midpoint vertices (64 values)

Visually, the height map can be represented as:
```
O-X-O-X-O-X-O-X-O
| | | | | | | | |
X-X-X-X-X-X-X-X-X
| | | | | | | | |
O-X-O-X-O-X-O-X-O
| | | | | | | | |
X-X-X-X-X-X-X-X-X
| | | | | | | | |
O-X-O-X-O-X-O-X-O
| | | | | | | | |
X-X-X-X-X-X-X-X-X
| | | | | | | | |
O-X-O-X-O-X-O-X-O
| | | | | | | | |
X-X-X-X-X-X-X-X-X
| | | | | | | | |
O-X-O-X-O-X-O-X-O
```
Where:
- 'O' represents corner vertices (9×9 grid)
- 'X' represents midpoint vertices (8×8 grid and the edge midpoints)
- '|' represents the connections between vertices

## Dependencies
- MCNK (C018) - Parent chunk that contains this subchunk
- MCNR (S002) - Normal vectors that correspond to each height value

## Height Calculation
The actual Z-coordinate in the world is calculated by combining:
1. The base Z coordinate from the MCNK header (heightmap_pos_z)
2. The height value from the MCVT heights array

The X and Y coordinates are determined by:
1. The ADT coordinates (32×32 yards per ADT)
2. The MCNK index within the ADT (each MCNK is 8 units wide)
3. The position of the vertex within the MCNK grid

Each MCNK spans 33.33 × 33.33 yards in the world.

## Implementation Notes
- Height values are absolute, not relative
- The first 81 values are for the corner vertices in the grid
- The next 64 values are for the inner midpoint vertices
- The height map is used for terrain rendering and collision detection
- The layout allows for more detailed terrain with fewer vertices
- The heights should be processed along with the MCNR normal vectors

## Implementation Example
```csharp
public class MCVT : IChunk
{
    public const int GRID_SIZE = 9;
    public const int VERTICES_COUNT = 145;
    public const int CORNER_VERTICES_COUNT = 81; // 9×9
    public const int MIDDLE_VERTICES_COUNT = 64; // 8×8
    
    public float[] Heights { get; set; } = new float[VERTICES_COUNT];
    
    public float GetHeight(int x, int y)
    {
        // Bounds check
        if (x < 0 || x >= GRID_SIZE || y < 0 || y >= GRID_SIZE)
            throw new ArgumentOutOfRangeException();
            
        // Get the vertex at the corners
        return Heights[y * GRID_SIZE + x];
    }
    
    public float GetMiddleHeight(int x, int y)
    {
        // Bounds check
        if (x < 0 || x >= GRID_SIZE - 1 || y < 0 || y >= GRID_SIZE - 1)
            throw new ArgumentOutOfRangeException();
            
        // Get the vertex at the middle point
        return Heights[CORNER_VERTICES_COUNT + y * (GRID_SIZE - 1) + x];
    }
    
    public void Parse(BinaryReader reader)
    {
        for (int i = 0; i < VERTICES_COUNT; i++)
        {
            Heights[i] = reader.ReadSingle();
        }
    }
}
```

## Usage Context
The MCVT subchunk is essential for terrain rendering in World of Warcraft. It defines the heights of each point in the terrain grid, which determines the shape of the landscape. When combined with texture information from MCLY and normal vectors from MCNR, it creates the 3D terrain visible in the game. The terrain system uses triangle strips to render the surface, with each pair of triangles formed by connecting four adjacent vertices. 