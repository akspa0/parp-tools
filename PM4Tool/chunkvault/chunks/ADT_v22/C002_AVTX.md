# C002: AVTX

## Type
ADT v22 Chunk

## Source
ADT_v22.md

## Description
Vertex height data for the terrain in ADT v22 format.

## Structure
```csharp
struct AVTX
{
    float outer[129][129]; // Outer vertices grid
    float inner[128][128]; // Inner vertices grid
}
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| outer | float[129][129] | Height values for the outer vertices grid |
| inner | float[128][128] | Height values for the inner vertices grid |

## Dependencies
- AHDR (C001) - To determine the size and structure of the vertex data

## Implementation Notes
- Size: Variable based on the header (AHDR)
- Total size calculation: `header.vertices_x * header.vertices_y + (header.vertices_x-1) * (header.vertices_y-1) * 4 bytes`
- Important difference from ADT v18: Vertices are NOT mixed as in ADT v18, but instead organized with the 129×129 field first, then the 128×128 one
- Outer vertices define the coarse terrain grid
- Inner vertices provide additional detail between outer vertices
- Both together form the complete heightmap for the terrain

## Implementation Example
```csharp
public class AVTX
{
    public float[,] OuterVertices { get; set; } // 129×129 grid
    public float[,] InnerVertices { get; set; } // 128×128 grid
    
    // Constructor to initialize arrays based on AHDR data
    public AVTX(AHDR header)
    {
        // Initialize based on header dimensions
        int outerDimX = (int)header.VerticesX;
        int outerDimY = (int)header.VerticesY;
        int innerDimX = outerDimX - 1;
        int innerDimY = outerDimY - 1;
        
        OuterVertices = new float[outerDimX, outerDimY];
        InnerVertices = new float[innerDimX, innerDimY];
    }
    
    // Helper method to get the height at a specific point
    public float GetHeightAt(float normalizedX, float normalizedY)
    {
        // Convert from normalized [0,1] coordinates to grid indices
        float gridX = normalizedX * 128;
        float gridY = normalizedY * 128;
        
        // Get integer and fractional parts
        int gridX1 = (int)gridX;
        int gridY1 = (int)gridY;
        float fracX = gridX - gridX1;
        float fracY = gridY - gridY1;
        
        // Ensure we're within bounds
        gridX1 = Math.Min(gridX1, 127);
        gridY1 = Math.Min(gridY1, 127);
        
        // Get the four corner heights
        float h00 = OuterVertices[gridX1, gridY1];
        float h10 = OuterVertices[gridX1 + 1, gridY1];
        float h01 = OuterVertices[gridX1, gridY1 + 1];
        float h11 = OuterVertices[gridX1 + 1, gridY1 + 1];
        
        // Consider inner vertex if available
        if (gridX1 < 128 && gridY1 < 128)
        {
            float hm = InnerVertices[gridX1, gridY1];
            // Adjust interpolation to account for inner vertex
            // This is a simplified approach - actual implementation would be more complex
        }
        
        // Perform bilinear interpolation (simplified)
        float h0 = h00 * (1 - fracX) + h10 * fracX;
        float h1 = h01 * (1 - fracX) + h11 * fracX;
        return h0 * (1 - fracY) + h1 * fracY;
    }
}
```

## Usage Context
The AVTX chunk contains the height map data for the terrain. The vertices are organized in a grid structure with both outer and inner vertices. The outer vertices form a 129×129 grid, while the inner vertices form a 128×128 grid positioned between the outer vertices. This provides a detailed heightmap for the terrain rendering. 