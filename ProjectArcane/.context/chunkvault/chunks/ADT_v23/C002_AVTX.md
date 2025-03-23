# AVTX - Vertex Height Data

## Type
ADT v23 Chunk

## Source
Referenced from `ADT_v23.md`

## Description
The AVTX (Vertex) chunk contains height data for all terrain vertices in the ADT v23 format. Unlike v18 where vertex data is distributed across multiple MCNK chunks, the v23 format consolidates all vertex heights into a single AVTX chunk. This chunk includes both the outer grid (129×129) and inner grid (128×128) vertices that define the terrain's 3D surface.

## Structure

```csharp
public struct AVTX
{
    // Outer grid vertices (129×129)
    public float[] outerGrid;   // Array of height values for the outer vertices
    
    // Inner grid vertices (128×128)
    public float[] innerGrid;   // Array of height values for the inner vertices
}
```

## Properties

| Name | Type | Description |
|------|------|-------------|
| outerGrid | float[] | Height values for the outer vertices (129×129 grid) |
| innerGrid | float[] | Height values for the inner vertices (128×128 grid) |

## Dependencies

- AHDR (C001) - Provides grid dimensions (vertices_x, vertices_y) needed to parse this chunk

## Implementation Notes

1. The AVTX chunk contains height data for all vertices in the terrain grid, unlike v18 where the data is distributed across MCVT subchunks within each MCNK.

2. The size of this chunk is variable and depends on the dimensions specified in the AHDR chunk. For standard ADTs, the size would be:
   - Outer grid: 129×129 = 16,641 vertices × 4 bytes = 66,564 bytes
   - Inner grid: 128×128 = 16,384 vertices × 4 bytes = 65,536 bytes
   - Total: 132,100 bytes (0x20404 bytes)

3. The outer grid vertices form the main terrain grid, while the inner grid vertices provide additional detail within each grid cell.

4. The outer and inner grids are stored as separate continuous arrays, with all outer grid vertices first, followed by all inner grid vertices. This differs from v18 where the data is interleaved.

5. In a standard ADT:
   - The outer grid (129×129) has one vertex at each corner of each chunk
   - The inner grid (128×128) has one vertex in the center of each outer grid cell

## Implementation Example

```csharp
public class AvtxChunk
{
    public float[] OuterGrid { get; private set; }
    public float[] InnerGrid { get; private set; }
    
    private int _outerGridWidth;
    private int _outerGridHeight;
    private int _innerGridWidth;
    private int _innerGridHeight;
    
    public AvtxChunk(int outerGridWidth, int outerGridHeight)
    {
        _outerGridWidth = outerGridWidth;
        _outerGridHeight = outerGridHeight;
        _innerGridWidth = outerGridWidth - 1;
        _innerGridHeight = outerGridHeight - 1;
        
        // Initialize the arrays
        OuterGrid = new float[_outerGridWidth * _outerGridHeight];
        InnerGrid = new float[_innerGridWidth * _innerGridHeight];
        
        // Default values (flat terrain)
        for (int i = 0; i < OuterGrid.Length; i++)
            OuterGrid[i] = 0.0f;
            
        for (int i = 0; i < InnerGrid.Length; i++)
            InnerGrid[i] = 0.0f;
    }
    
    public void Load(BinaryReader reader, long size)
    {
        // Read outer grid vertices
        for (int i = 0; i < OuterGrid.Length; i++)
        {
            OuterGrid[i] = reader.ReadSingle();
        }
        
        // Read inner grid vertices
        for (int i = 0; i < InnerGrid.Length; i++)
        {
            InnerGrid[i] = reader.ReadSingle();
        }
    }
    
    public void Save(BinaryWriter writer)
    {
        writer.Write("AVTX".ToCharArray());
        
        // Calculate size: 4 bytes per float for both grids
        uint size = (uint)((OuterGrid.Length + InnerGrid.Length) * 4);
        writer.Write(size);
        
        // Write outer grid heights
        for (int i = 0; i < OuterGrid.Length; i++)
        {
            writer.Write(OuterGrid[i]);
        }
        
        // Write inner grid heights
        for (int i = 0; i < InnerGrid.Length; i++)
        {
            writer.Write(InnerGrid[i]);
        }
    }
    
    // Get height at a specific outer grid position
    public float GetOuterHeight(int x, int y)
    {
        if (x < 0 || x >= _outerGridWidth || y < 0 || y >= _outerGridHeight)
            return 0.0f; // Default for out of bounds
            
        return OuterGrid[y * _outerGridWidth + x];
    }
    
    // Get height at a specific inner grid position
    public float GetInnerHeight(int x, int y)
    {
        if (x < 0 || x >= _innerGridWidth || y < 0 || y >= _innerGridHeight)
            return 0.0f; // Default for out of bounds
            
        return InnerGrid[y * _innerGridWidth + x];
    }
    
    // Get interpolated height at a normalized position (0.0-1.0)
    public float GetInterpolatedHeight(float normalizedX, float normalizedY)
    {
        // Convert normalized coordinates to grid coordinates
        float gridX = normalizedX * (_outerGridWidth - 1);
        float gridY = normalizedY * (_outerGridHeight - 1);
        
        // Get integer grid coordinates
        int x1 = (int)Math.Floor(gridX);
        int y1 = (int)Math.Floor(gridY);
        int x2 = Math.Min(x1 + 1, _outerGridWidth - 1);
        int y2 = Math.Min(y1 + 1, _outerGridHeight - 1);
        
        // Get fractional parts
        float fracX = gridX - x1;
        float fracY = gridY - y1;
        
        // Get the heights at the four corners
        float h11 = GetOuterHeight(x1, y1);
        float h21 = GetOuterHeight(x2, y1);
        float h12 = GetOuterHeight(x1, y2);
        float h22 = GetOuterHeight(x2, y2);
        
        // Check if we need to consider inner height
        if (fracX > 0 && fracY > 0 && fracX < 1 && fracY < 1)
        {
            // Use inner height instead
            float innerHeight = GetInnerHeight(x1, y1);
            if (fracX < 0.5f && fracY < 0.5f)
            {
                // Bottom-left quadrant
                return Lerp(Lerp(h11, innerHeight, fracX * 2), Lerp(h21, h12, fracY * 2), fracY * 2);
            }
            else if (fracX >= 0.5f && fracY < 0.5f)
            {
                // Bottom-right quadrant
                return Lerp(Lerp(innerHeight, h21, (fracX - 0.5f) * 2), Lerp(h12, h22, fracY * 2), fracY * 2);
            }
            else if (fracX < 0.5f && fracY >= 0.5f)
            {
                // Top-left quadrant
                return Lerp(Lerp(innerHeight, h12, (fracY - 0.5f) * 2), Lerp(h21, h22, fracX * 2), fracX * 2);
            }
            else
            {
                // Top-right quadrant
                return Lerp(Lerp(innerHeight, h22, ((fracX - 0.5f) + (fracY - 0.5f)) * 2), Lerp(h21, h12, (1 - fracX) * (1 - fracY) * 4), (1 - fracX) * (1 - fracY) * 4);
            }
        }
        
        // Simple bilinear interpolation for outer heights
        return Lerp(Lerp(h11, h21, fracX), Lerp(h12, h22, fracX), fracY);
    }
    
    // Linear interpolation helper
    private float Lerp(float a, float b, float t)
    {
        return a + t * (b - a);
    }
}
```

## Usage Context

The AVTX chunk is a fundamental part of the ADT v23 format's terrain representation, providing the height data necessary to construct the 3D terrain surface. Its role in the terrain rendering system includes:

1. **Surface Construction**: The height values in AVTX define the vertical position of each point on the terrain surface, which is then used by the rendering system to create the 3D mesh.

2. **Collision Detection**: The height data is essential for determining when characters or objects collide with the terrain surface.

3. **Pathfinding**: AI systems use the terrain height information for pathfinding calculations and determining traversable areas.

4. **Environmental Effects**: Water systems, particle effects, and other environmental features use terrain height data to properly interact with the landscape.

The v23 format's approach of consolidating all height data into a single AVTX chunk (rather than distributing it across multiple MCNK chunks as in v18) represents an experimental attempt to optimize memory access patterns and potentially improve rendering performance. By keeping outer and inner vertices in separate continuous arrays, the format may have been designed to allow for more efficient streaming and processing of terrain data.

Though never used in retail clients, this experimental approach provides insight into how Blizzard was exploring alternative terrain data organizations during the Cataclysm beta development period. 