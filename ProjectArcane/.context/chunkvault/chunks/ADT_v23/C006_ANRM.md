# C006: ANRM

## Type
ADT v23 Chunk

## Source
ADT_v23.md

## Description
Normal vector data for ADT v23 format. Contains surface normal information for terrain vertices, enhanced for WoD's improved lighting model.

## Structure
```csharp
struct ANRM
{
    // 145×145 normal vectors for terrain vertices
    // Each normal is represented by 3 signed bytes (-127 to 127)
    // Enhanced precision in WoD (v23)
    struct {
        int8 nx;  // Normal X component
        int8 ny;  // Normal Y component
        int8 nz;  // Normal Z component
        int8 reserved; // Added in WoD for future expansion
    } normals[145 * 145];
}
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| normals | struct[21025] | Normal vectors for each vertex in the terrain grid (145×145) |

## Dependencies
- AHDR (C004) - References this chunk via offsets
- AVTX (C005) - Normals correspond to the same vertices defined in AVTX

## Implementation Notes
- Size: 84,100 bytes (21,025 normals × 4 bytes)
- Contains a 145×145 grid of normal vectors
- Each normal is a 3-component vector with values from -127 to 127
- In v23, a fourth byte is added for future expansion (was 3 bytes in v22)
- Enhanced in WoD with improved precision for better lighting calculations
- Normal vectors are normalized and scaled to fit in the -127 to 127 range
- To convert to actual normal vectors, divide components by 127.0f
- Used for lighting calculations and shading the terrain

## Implementation Example
```csharp
public class ANRM
{
    // Dimensions of the normal grid
    public const int GRID_SIZE = 145;
    
    // Structure to hold a compressed normal vector
    public struct CompressedNormal
    {
        public sbyte X { get; set; }
        public sbyte Y { get; set; }
        public sbyte Z { get; set; }
        public sbyte Reserved { get; set; }  // Added in WoD (v23)
        
        public CompressedNormal(sbyte x, sbyte y, sbyte z, sbyte reserved = 0)
        {
            X = x;
            Y = y;
            Z = z;
            Reserved = reserved;
        }
        
        // Convert to a normalized floating-point vector
        public System.Numerics.Vector3 ToNormalizedVector()
        {
            return new System.Numerics.Vector3(
                X / 127.0f,
                Y / 127.0f, 
                Z / 127.0f
            ).Normalize();
        }
    }
    
    // Store normals as a 2D grid
    private CompressedNormal[,] _normals;
    
    public ANRM(CompressedNormal[] normalData)
    {
        if (normalData.Length != GRID_SIZE * GRID_SIZE)
            throw new ArgumentException($"Normal data must contain {GRID_SIZE * GRID_SIZE} elements");
        
        // Convert linear array to 2D grid
        _normals = new CompressedNormal[GRID_SIZE, GRID_SIZE];
        for (int y = 0; y < GRID_SIZE; y++)
        {
            for (int x = 0; x < GRID_SIZE; x++)
            {
                _normals[x, y] = normalData[y * GRID_SIZE + x];
            }
        }
    }
    
    // Get compressed normal at specific grid coordinates
    public CompressedNormal GetCompressedNormalAt(int x, int y)
    {
        if (x < 0 || x >= GRID_SIZE || y < 0 || y >= GRID_SIZE)
            throw new ArgumentOutOfRangeException($"Coordinates ({x},{y}) out of range");
            
        return _normals[x, y];
    }
    
    // Get normalized vector at specific grid coordinates
    public System.Numerics.Vector3 GetNormalizedVectorAt(int x, int y)
    {
        return GetCompressedNormalAt(x, y).ToNormalizedVector();
    }
    
    // Get interpolated normal at a normalized position (0-1)
    public System.Numerics.Vector3 GetInterpolatedNormal(float normalizedX, float normalizedY)
    {
        float gridX = normalizedX * (GRID_SIZE - 1);
        float gridY = normalizedY * (GRID_SIZE - 1);
        
        int x1 = (int)Math.Floor(gridX);
        int y1 = (int)Math.Floor(gridY);
        int x2 = Math.Min(x1 + 1, GRID_SIZE - 1);
        int y2 = Math.Min(y1 + 1, GRID_SIZE - 1);
        
        float fracX = gridX - x1;
        float fracY = gridY - y1;
        
        var n11 = GetNormalizedVectorAt(x1, y1);
        var n21 = GetNormalizedVectorAt(x2, y1);
        var n12 = GetNormalizedVectorAt(x1, y2);
        var n22 = GetNormalizedVectorAt(x2, y2);
        
        // Bilinear interpolation
        var n1 = System.Numerics.Vector3.Lerp(n11, n21, fracX);
        var n2 = System.Numerics.Vector3.Lerp(n12, n22, fracX);
        var n = System.Numerics.Vector3.Lerp(n1, n2, fracY);
        
        // Renormalize after interpolation
        return System.Numerics.Vector3.Normalize(n);
    }
    
    // Generate normals from a height map (for editing)
    public static ANRM GenerateFromHeightMap(AVTX heightMap)
    {
        var normalData = new CompressedNormal[GRID_SIZE * GRID_SIZE];
        
        for (int y = 0; y < GRID_SIZE; y++)
        {
            for (int x = 0; x < GRID_SIZE; x++)
            {
                // Get heights of neighboring points
                float h = heightMap.GetHeightAt(x, y);
                float hLeft = x > 0 ? heightMap.GetHeightAt(x - 1, y) : h;
                float hRight = x < GRID_SIZE - 1 ? heightMap.GetHeightAt(x + 1, y) : h;
                float hUp = y > 0 ? heightMap.GetHeightAt(x, y - 1) : h;
                float hDown = y < GRID_SIZE - 1 ? heightMap.GetHeightAt(x, y + 1) : h;
                
                // Calculate partial derivatives
                float dhdx = (hRight - hLeft) / 2.0f;
                float dhdy = (hDown - hUp) / 2.0f;
                
                // Cross product of tangent vectors to get normal
                var normal = new System.Numerics.Vector3(-dhdx, 1.0f, -dhdy);
                normal = System.Numerics.Vector3.Normalize(normal);
                
                // Convert to compressed format
                normalData[y * GRID_SIZE + x] = new CompressedNormal(
                    (sbyte)(normal.X * 127),
                    (sbyte)(normal.Y * 127),
                    (sbyte)(normal.Z * 127),
                    0  // Reserved field (added in v23)
                );
            }
        }
        
        return new ANRM(normalData);
    }
    
    // Visualize normal map as RGB image
    public System.Drawing.Bitmap GenerateNormalMap()
    {
        var bitmap = new System.Drawing.Bitmap(GRID_SIZE, GRID_SIZE);
        
        for (int y = 0; y < GRID_SIZE; y++)
        {
            for (int x = 0; x < GRID_SIZE; x++)
            {
                var normal = GetNormalizedVectorAt(x, y);
                
                // Map from [-1,1] to [0,255]
                int r = (int)((normal.X + 1) * 127.5f);
                int g = (int)((normal.Y + 1) * 127.5f);
                int b = (int)((normal.Z + 1) * 127.5f);
                
                r = Math.Max(0, Math.Min(255, r));
                g = Math.Max(0, Math.Min(255, g));
                b = Math.Max(0, Math.Min(255, b));
                
                var color = System.Drawing.Color.FromArgb(r, g, b);
                bitmap.SetPixel(x, y, color);
            }
        }
        
        return bitmap;
    }
}
```

## Usage Context
The ANRM chunk contains normal vector information for the entire terrain tile, providing essential data for lighting and shading calculations. It defines a 145×145 grid of normal vectors that correspond to the height vertices in the AVTX chunk.

In the ADT v23 format used since Warlords of Draenor, the ANRM chunk has been enhanced with improved precision for normal vectors compared to v22, and an additional reserved byte has been added for future expansion. These improvements support WoD's enhanced lighting model, which included better dynamic lighting, shadows, and surface detail.

Normal vectors indicate the direction a surface is facing, which is crucial for determining how light interacts with the terrain. When combined with the height data (AVTX) and color/texture information, these normals enable realistic lighting effects such as diffuse lighting, specular highlights, and shadow calculations. The improved normal precision in v23 allows for more subtle terrain details to be properly lit, enhancing the visual quality of the terrain. 