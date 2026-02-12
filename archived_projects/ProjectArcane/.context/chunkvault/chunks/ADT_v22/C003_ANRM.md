# C003: ANRM

## Type
ADT v22 Chunk

## Source
ADT_v22.md

## Description
Normal vectors for terrain lighting in ADT v22 format.

## Structure
```csharp
struct ANRM
{
    // Each normal is stored as 3 signed bytes (-127 to 127)
    int8_t outer_normals[129][129][3]; // Outer vertices normals
    int8_t inner_normals[128][128][3]; // Inner vertices normals
}
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| outer_normals | int8[129][129][3] | Normal vectors for outer vertices grid |
| inner_normals | int8[128][128][3] | Normal vectors for inner vertices grid |

## Dependencies
- AHDR (C001) - To determine the size and structure of the normal data
- AVTX (C002) - Normals are typically calculated from vertex heights

## Implementation Notes
- Size: Variable based on the header (AHDR)
- Total size calculation: `header.vertices_x * header.vertices_y + (header.vertices_x-1) * (header.vertices_y-1) * 3 bytes`
- Like ADT v18, normals are stored as triples of signed chars
- 127 represents 1.0, -127 represents -1.0 in the normal components
- Normals follow the same structure as vertices in AVTX, with outer normals first, then inner normals
- Normals are used for terrain lighting calculations

## Implementation Example
```csharp
public class ANRM
{
    // Normalize (-127 to 127) to (-1.0 to 1.0)
    private const float NormalScale = 1.0f / 127.0f;
    
    // Byte representation as stored in the file
    public sbyte[,,] OuterNormalBytes { get; set; } // 129×129×3 grid
    public sbyte[,,] InnerNormalBytes { get; set; } // 128×128×3 grid
    
    // Calculated Vector3 normals for convenience
    public Vector3[,] OuterNormals { get; private set; }
    public Vector3[,] InnerNormals { get; private set; }
    
    // Constructor to initialize arrays based on AHDR data
    public ANRM(AHDR header)
    {
        // Initialize based on header dimensions
        int outerDimX = (int)header.VerticesX;
        int outerDimY = (int)header.VerticesY;
        int innerDimX = outerDimX - 1;
        int innerDimY = outerDimY - 1;
        
        OuterNormalBytes = new sbyte[outerDimX, outerDimY, 3];
        InnerNormalBytes = new sbyte[innerDimX, innerDimY, 3];
        
        OuterNormals = new Vector3[outerDimX, outerDimY];
        InnerNormals = new Vector3[innerDimX, innerDimY];
    }
    
    // Convert byte normal data to Vector3
    public void CalculateVectors()
    {
        // Process outer normals
        for (int y = 0; y < OuterNormalBytes.GetLength(1); y++)
        {
            for (int x = 0; x < OuterNormalBytes.GetLength(0); x++)
            {
                OuterNormals[x, y] = new Vector3(
                    OuterNormalBytes[x, y, 0] * NormalScale,
                    OuterNormalBytes[x, y, 1] * NormalScale,
                    OuterNormalBytes[x, y, 2] * NormalScale
                );
            }
        }
        
        // Process inner normals
        for (int y = 0; y < InnerNormalBytes.GetLength(1); y++)
        {
            for (int x = 0; x < InnerNormalBytes.GetLength(0); x++)
            {
                InnerNormals[x, y] = new Vector3(
                    InnerNormalBytes[x, y, 0] * NormalScale,
                    InnerNormalBytes[x, y, 1] * NormalScale,
                    InnerNormalBytes[x, y, 2] * NormalScale
                );
            }
        }
    }
    
    // Calculate normals from height data (if normal data isn't available)
    public static ANRM CalculateFromHeightMap(AHDR header, AVTX heightMap)
    {
        ANRM normals = new ANRM(header);
        
        // Calculate outer normals
        for (int y = 0; y < header.VerticesY; y++)
        {
            for (int x = 0; x < header.VerticesX; x++)
            {
                // Sample neighboring heights (with bounds checking)
                // Calculate normal using cross product of tangent vectors
                // Normalize and convert to byte representation
                // Simplified example - actual implementation would be more complex
            }
        }
        
        // Calculate inner normals similarly
        
        return normals;
    }
}
```

## Usage Context
The ANRM chunk contains normal vector data for terrain lighting calculations. Each normal vector is stored as three signed bytes, representing the X, Y, and Z components. The normal data follows the same structure as the vertex data in AVTX, with outer normals (129×129 grid) followed by inner normals (128×128 grid). Normal vectors are essential for realistic terrain lighting and shading. 