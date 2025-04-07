# C003: ABSH

## Type
ADT v23 Chunk

## Source
ADT_v23.md

## Description
Blend Shadow data for the terrain tile. Contains shadow coefficients used for terrain shading and atmosphere blending.

## Structure
```csharp
struct ABSH
{
    // 8x8 grid of shadow coefficients for the entire tile
    float coefficients[8 * 8 * 4]; // 4 floats per point, 256 floats total
}
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| coefficients | float[256] | Shadow coefficients for atmospheric blending (8x8 grid with 4 values per point) |

## Dependencies
- None

## Implementation Notes
- Size: 1024 bytes (256 floats x 4 bytes)
- Contains an 8×8 grid of shadow coefficient points
- Each point has 4 floating-point values (x, y, z components + blend factor)
- Used for advanced terrain lighting and atmosphere blending
- Added in WoD (6.x) to enhance terrain lighting quality
- Works in conjunction with ACVT (Color Vertex data) for atmosphere effects

## Implementation Example
```csharp
public class ABSH
{
    // Grid dimensions
    public const int GRID_SIZE = 8;
    
    // Number of coefficients per point
    public const int COEFFICIENTS_PER_POINT = 4;
    
    // Shadow coefficient structure
    public struct ShadowCoefficient
    {
        public float X { get; set; }
        public float Y { get; set; }
        public float Z { get; set; }
        public float BlendFactor { get; set; }
        
        public ShadowCoefficient(float x, float y, float z, float blendFactor)
        {
            X = x;
            Y = y;
            Z = z;
            BlendFactor = blendFactor;
        }
        
        // Get the shadow color based on these coefficients
        public System.Numerics.Vector3 GetShadowColor()
        {
            return new System.Numerics.Vector3(X, Y, Z);
        }
    }
    
    // Raw coefficient data as stored in the file
    private float[] _rawCoefficients;
    
    // Parsed coefficient grid for easier access
    private ShadowCoefficient[,] _coefficientGrid;
    
    public ABSH(float[] coefficients)
    {
        if (coefficients.Length != GRID_SIZE * GRID_SIZE * COEFFICIENTS_PER_POINT)
            throw new ArgumentException($"Shadow coefficient array must have {GRID_SIZE * GRID_SIZE * COEFFICIENTS_PER_POINT} elements");
            
        _rawCoefficients = coefficients;
        
        // Convert to a 2D grid of coefficient objects
        _coefficientGrid = new ShadowCoefficient[GRID_SIZE, GRID_SIZE];
        
        for (int y = 0; y < GRID_SIZE; y++)
        {
            for (int x = 0; x < GRID_SIZE; x++)
            {
                int baseIndex = (y * GRID_SIZE + x) * COEFFICIENTS_PER_POINT;
                
                _coefficientGrid[x, y] = new ShadowCoefficient(
                    _rawCoefficients[baseIndex],
                    _rawCoefficients[baseIndex + 1],
                    _rawCoefficients[baseIndex + 2],
                    _rawCoefficients[baseIndex + 3]
                );
            }
        }
    }
    
    // Get raw coefficient data
    public float[] GetRawCoefficients()
    {
        return _rawCoefficients;
    }
    
    // Get coefficient at a specific grid position
    public ShadowCoefficient GetCoefficientAt(int x, int y)
    {
        if (x < 0 || x >= GRID_SIZE || y < 0 || y >= GRID_SIZE)
            throw new ArgumentOutOfRangeException($"Coordinates ({x},{y}) out of range");
            
        return _coefficientGrid[x, y];
    }
    
    // Get interpolated coefficient at a normalized position (0-1)
    public ShadowCoefficient GetInterpolatedCoefficient(float normalizedX, float normalizedY)
    {
        float gridX = normalizedX * (GRID_SIZE - 1);
        float gridY = normalizedY * (GRID_SIZE - 1);
        
        int x1 = (int)Math.Floor(gridX);
        int y1 = (int)Math.Floor(gridY);
        int x2 = Math.Min(x1 + 1, GRID_SIZE - 1);
        int y2 = Math.Min(y1 + 1, GRID_SIZE - 1);
        
        float fracX = gridX - x1;
        float fracY = gridY - y1;
        
        ShadowCoefficient c11 = _coefficientGrid[x1, y1];
        ShadowCoefficient c21 = _coefficientGrid[x2, y1];
        ShadowCoefficient c12 = _coefficientGrid[x1, y2];
        ShadowCoefficient c22 = _coefficientGrid[x2, y2];
        
        // Bilinear interpolation of each component
        float x = Interpolate(c11.X, c21.X, c12.X, c22.X, fracX, fracY);
        float y = Interpolate(c11.Y, c21.Y, c12.Y, c22.Y, fracX, fracY);
        float z = Interpolate(c11.Z, c21.Z, c12.Z, c22.Z, fracX, fracY);
        float b = Interpolate(c11.BlendFactor, c21.BlendFactor, c12.BlendFactor, c22.BlendFactor, fracX, fracY);
        
        return new ShadowCoefficient(x, y, z, b);
    }
    
    // Helper method for bilinear interpolation
    private float Interpolate(float v11, float v21, float v12, float v22, float fracX, float fracY)
    {
        float v1 = v11 * (1 - fracX) + v21 * fracX;
        float v2 = v12 * (1 - fracX) + v22 * fracX;
        return v1 * (1 - fracY) + v2 * fracY;
    }
    
    // Blend with color vertex data (from ACVT) to get final atmospheric colors
    public System.Numerics.Vector3 BlendWithCVT(System.Numerics.Vector3 cvtColor, float normalizedX, float normalizedY)
    {
        var coef = GetInterpolatedCoefficient(normalizedX, normalizedY);
        var shadowColor = coef.GetShadowColor();
        
        // Blend between CVT color and shadow color based on blend factor
        return System.Numerics.Vector3.Lerp(cvtColor, shadowColor, coef.BlendFactor);
    }
}
```

## Usage Context
The ABSH chunk contains shadow blending coefficients used for advanced terrain lighting and atmospheric effects. It was introduced in Warlords of Draenor (6.x) to enhance the visual quality of terrain. 

The chunk data consists of an 8×8 grid of points covering the entire ADT tile, with each point containing four floating-point values that define how shadows blend with the terrain colors. These shadow coefficients work in conjunction with the color vertex data (ACVT) to create more realistic terrain appearance, including atmospheric fog effects, distance fading, and time-of-day lighting.

During rendering, these shadow coefficients are interpolated across the terrain to provide smooth atmospheric blending. The coefficient components (X, Y, Z) typically represent a color value, while the fourth component represents a blend factor determining how much this shadow color affects the final terrain appearance. 