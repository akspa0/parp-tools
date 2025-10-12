# ANRM - Normal Vectors

## Type
ADT v23 Chunk

## Source
Referenced from `ADT_v23.md`

## Description
The ANRM (Normals) chunk contains normal vector data for all terrain vertices in the ADT v23 format. These normals are used for lighting calculations to determine how light interacts with the terrain surface. Like the AVTX chunk, ANRM consolidates all normal data into a single chunk rather than distributing it across multiple chunks as in v18.

## Structure

```csharp
public struct ANRM
{
    // Normal vectors for all vertices following the same layout as AVTX
    // Each normal is stored as three signed bytes
    // Outer grid normals (129×129 vertices)
    public sbyte[] outerGridNormals;   // 3 bytes per normal * 129*129 vertices
    
    // Inner grid normals (128×128 vertices)
    public sbyte[] innerGridNormals;   // 3 bytes per normal * 128*128 vertices
}
```

## Properties

| Name | Type | Description |
|------|------|-------------|
| outerGridNormals | sbyte[] | Normal vectors for the outer vertices (129×129 grid × 3 components) |
| innerGridNormals | sbyte[] | Normal vectors for the inner vertices (128×128 grid × 3 components) |

## Dependencies

- AHDR (C001) - Provides grid dimensions (vertices_x, vertices_y) needed to parse this chunk
- AVTX (C002) - The normals correspond to the terrain vertices defined in AVTX

## Implementation Notes

1. The ANRM chunk contains normal vectors for all vertices in the terrain grid, which correspond to the heights in the AVTX chunk.

2. Normal vectors are stored as 3 signed bytes (X, Y, Z) for each vertex. The values range from -127 to 127, with 127 representing 1.0 and -127 representing -1.0 in normalized vector terms.

3. The size of this chunk is variable and depends on the dimensions specified in the AHDR chunk. For standard ADTs, the size would be:
   - Outer grid: 129×129 = 16,641 vertices × 3 bytes = 49,923 bytes
   - Inner grid: 128×128 = 16,384 vertices × 3 bytes = 49,152 bytes
   - Total: 99,075 bytes (approximately 0x18303 bytes)

4. Like AVTX, the outer grid normals are stored first as a continuous array, followed by the inner grid normals, which differs from v18's interleaved approach.

5. Normal vectors should be normalized before use in lighting calculations to ensure correct results.

## Implementation Example

```csharp
public class AnrmChunk
{
    // Each normal has X, Y, Z components stored as bytes
    private const int COMPONENTS_PER_NORMAL = 3;
    
    // Raw byte data
    public sbyte[] OuterGridNormals { get; private set; }
    public sbyte[] InnerGridNormals { get; private set; }
    
    private int _outerGridWidth;
    private int _outerGridHeight;
    private int _innerGridWidth;
    private int _innerGridHeight;
    
    public AnrmChunk(int outerGridWidth, int outerGridHeight)
    {
        _outerGridWidth = outerGridWidth;
        _outerGridHeight = outerGridHeight;
        _innerGridWidth = outerGridWidth - 1;
        _innerGridHeight = outerGridHeight - 1;
        
        // Initialize the arrays
        OuterGridNormals = new sbyte[_outerGridWidth * _outerGridHeight * COMPONENTS_PER_NORMAL];
        InnerGridNormals = new sbyte[_innerGridWidth * _innerGridHeight * COMPONENTS_PER_NORMAL];
        
        // Default values (straight up normals)
        for (int i = 0; i < _outerGridWidth * _outerGridHeight; i++)
        {
            int index = i * COMPONENTS_PER_NORMAL;
            OuterGridNormals[index] = 0;     // X
            OuterGridNormals[index + 1] = 0; // Y
            OuterGridNormals[index + 2] = 127; // Z (up)
        }
        
        for (int i = 0; i < _innerGridWidth * _innerGridHeight; i++)
        {
            int index = i * COMPONENTS_PER_NORMAL;
            InnerGridNormals[index] = 0;     // X
            InnerGridNormals[index + 1] = 0; // Y
            InnerGridNormals[index + 2] = 127; // Z (up)
        }
    }
    
    public void Load(BinaryReader reader, long size)
    {
        // Read outer grid normals
        for (int i = 0; i < OuterGridNormals.Length; i++)
        {
            OuterGridNormals[i] = reader.ReadSByte();
        }
        
        // Read inner grid normals
        for (int i = 0; i < InnerGridNormals.Length; i++)
        {
            InnerGridNormals[i] = reader.ReadSByte();
        }
    }
    
    public void Save(BinaryWriter writer)
    {
        writer.Write("ANRM".ToCharArray());
        
        // Calculate size: 3 bytes per normal for both grids
        uint size = (uint)(OuterGridNormals.Length + InnerGridNormals.Length);
        writer.Write(size);
        
        // Write outer grid normals
        for (int i = 0; i < OuterGridNormals.Length; i++)
        {
            writer.Write(OuterGridNormals[i]);
        }
        
        // Write inner grid normals
        for (int i = 0; i < InnerGridNormals.Length; i++)
        {
            writer.Write(InnerGridNormals[i]);
        }
    }
    
    // Get normal vector for a specific outer grid position
    public Vector3 GetOuterNormal(int x, int y)
    {
        if (x < 0 || x >= _outerGridWidth || y < 0 || y >= _outerGridHeight)
            return Vector3.UnitZ; // Default to up vector for out of bounds
            
        int index = (y * _outerGridWidth + x) * COMPONENTS_PER_NORMAL;
        
        // Convert from -127..127 to -1.0..1.0
        float nx = OuterGridNormals[index] / 127.0f;
        float ny = OuterGridNormals[index + 1] / 127.0f;
        float nz = OuterGridNormals[index + 2] / 127.0f;
        
        // Create and normalize the vector
        Vector3 normal = new Vector3(nx, ny, nz);
        if (normal.LengthSquared() > 0)
            normal = Vector3.Normalize(normal);
        else
            normal = Vector3.UnitZ; // Fallback if normal is zero
            
        return normal;
    }
    
    // Get normal vector for a specific inner grid position
    public Vector3 GetInnerNormal(int x, int y)
    {
        if (x < 0 || x >= _innerGridWidth || y < 0 || y >= _innerGridHeight)
            return Vector3.UnitZ; // Default to up vector for out of bounds
            
        int index = (y * _innerGridWidth + x) * COMPONENTS_PER_NORMAL;
        
        // Convert from -127..127 to -1.0..1.0
        float nx = InnerGridNormals[index] / 127.0f;
        float ny = InnerGridNormals[index + 1] / 127.0f;
        float nz = InnerGridNormals[index + 2] / 127.0f;
        
        // Create and normalize the vector
        Vector3 normal = new Vector3(nx, ny, nz);
        if (normal.LengthSquared() > 0)
            normal = Vector3.Normalize(normal);
        else
            normal = Vector3.UnitZ; // Fallback if normal is zero
            
        return normal;
    }
    
    // Get interpolated normal at a normalized position (0.0-1.0)
    public Vector3 GetInterpolatedNormal(float normalizedX, float normalizedY)
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
        
        // Get the normals at the four corners
        Vector3 n11 = GetOuterNormal(x1, y1);
        Vector3 n21 = GetOuterNormal(x2, y1);
        Vector3 n12 = GetOuterNormal(x1, y2);
        Vector3 n22 = GetOuterNormal(x2, y2);
        
        // Check if we need to consider inner normal
        if (fracX > 0 && fracY > 0 && fracX < 1 && fracY < 1)
        {
            // Use inner normal
            Vector3 innerNormal = GetInnerNormal(x1, y1);
            
            // Simple interpolation for this example
            // A more accurate implementation would consider the quadrants as in AVTX
            Vector3 result = Vector3.Lerp(Vector3.Lerp(n11, n21, fracX), Vector3.Lerp(n12, n22, fracX), fracY);
            result = Vector3.Lerp(result, innerNormal, 0.5f); // Blend with inner normal
            
            if (result.LengthSquared() > 0)
                return Vector3.Normalize(result);
            else
                return Vector3.UnitZ;
        }
        
        // Simple bilinear interpolation for outer normals
        Vector3 interpolated = Vector3.Lerp(Vector3.Lerp(n11, n21, fracX), Vector3.Lerp(n12, n22, fracX), fracY);
        
        if (interpolated.LengthSquared() > 0)
            return Vector3.Normalize(interpolated);
        else
            return Vector3.UnitZ;
    }
    
    // Calculate normals from an AVTX chunk
    public void CalculateNormals(AvtxChunk avtxChunk)
    {
        // Calculate outer grid normals
        for (int y = 0; y < _outerGridHeight; y++)
        {
            for (int x = 0; x < _outerGridWidth; x++)
            {
                Vector3 normal = CalculateNormalAt(avtxChunk, x, y, true);
                SetOuterNormal(x, y, normal);
            }
        }
        
        // Calculate inner grid normals
        for (int y = 0; y < _innerGridHeight; y++)
        {
            for (int x = 0; x < _innerGridWidth; x++)
            {
                Vector3 normal = CalculateNormalAt(avtxChunk, x, y, false);
                SetInnerNormal(x, y, normal);
            }
        }
    }
    
    // Helper to calculate a normal at a specific position based on heights
    private Vector3 CalculateNormalAt(AvtxChunk avtxChunk, int x, int y, bool isOuter)
    {
        // Different approaches for outer and inner vertices...
        // Implementation would depend on how you want to calculate normals
        // Typically involves finding adjacent vertices and using cross products
        
        // For simplicity, just return an up vector in this example
        return Vector3.UnitZ;
    }
    
    // Set a normal in the outer grid
    private void SetOuterNormal(int x, int y, Vector3 normal)
    {
        int index = (y * _outerGridWidth + x) * COMPONENTS_PER_NORMAL;
        
        // Normalize and convert to -127..127 range
        normal = Vector3.Normalize(normal);
        OuterGridNormals[index] = (sbyte)(normal.X * 127);
        OuterGridNormals[index + 1] = (sbyte)(normal.Y * 127);
        OuterGridNormals[index + 2] = (sbyte)(normal.Z * 127);
    }
    
    // Set a normal in the inner grid
    private void SetInnerNormal(int x, int y, Vector3 normal)
    {
        int index = (y * _innerGridWidth + x) * COMPONENTS_PER_NORMAL;
        
        // Normalize and convert to -127..127 range
        normal = Vector3.Normalize(normal);
        InnerGridNormals[index] = (sbyte)(normal.X * 127);
        InnerGridNormals[index + 1] = (sbyte)(normal.Y * 127);
        InnerGridNormals[index + 2] = (sbyte)(normal.Z * 127);
    }
}
```

## Usage Context

The ANRM chunk plays a crucial role in rendering realistic terrain in the game by providing normal vectors used for lighting calculations. Its primary functions include:

1. **Light Interaction**: Normal vectors determine how light rays interact with the terrain surface, defining which areas are brightly lit and which are in shadow.

2. **Slope Visualization**: Normal data enhances the visual perception of terrain features, making slopes, ridges, and valleys more apparent to the player.

3. **Texture Mapping**: Normal vectors can influence how textures are applied to the terrain, including parallax effects and normal mapping.

4. **Shader Effects**: Many advanced shader effects in modern terrain rendering rely on accurate normal data, including ambient occlusion, subsurface scattering, and reflection effects.

The v23 format's approach of consolidating all normal data into a single ANRM chunk (similar to AVTX) represents an experimental approach to terrain data organization that might have been intended to improve memory locality and data access patterns. By keeping outer and inner grid normals in separate continuous arrays, the format may have been designed to enable more efficient rendering processes.

Although this format was ultimately not used in any retail release, it illustrates how Blizzard was experimenting with different ways to organize terrain data during the Cataclysm beta development period, potentially seeking performance improvements or preparing for future rendering enhancements. 