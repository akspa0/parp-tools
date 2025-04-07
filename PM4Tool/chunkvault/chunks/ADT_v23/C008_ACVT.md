# ACVT - Vertex Shading Data

## Type
ADT v23 Chunk

## Source
Referenced from `ADT_v23.md`

## Description
The ACVT (Color Vertex) chunk stores color/shading information for each vertex in the ADT tile, providing additional visual details for terrain rendering. This chunk centralizes vertex color data that was previously embedded in MCVT chunks in v18, creating a single global array of vertex colors for the entire ADT tile.

## Structure

```csharp
public struct ACVT
{
    // Color data for all vertices in the ADT tile
    // Organized in row-major order, from north to south, west to east
    // 145 × 145 vertices total (9 × 9 inner vertices per chunk, plus shared edges)
    public byte[] colorData;  // BGRA format, 4 bytes per vertex
}
```

## Properties

| Name | Type | Description |
|------|------|-------------|
| colorData | byte[] | Array of BGRA color values for all vertices in the ADT tile (145 × 145 × 4 bytes = 84,100 bytes) |

## Dependencies

- AVTX (C002) - The vertex color data corresponds to the same vertices defined in the height map

## Implementation Notes

1. The ACVT chunk stores color data for each vertex in the ADT tile in BGRA format (4 bytes per vertex).

2. There are 145 × 145 vertices in a standard ADT tile, resulting in a data size of 84,100 bytes.

3. The vertex colors provide additional visual detail such as darkness in shadowed areas, discoloration for different terrain types (snow, dirt, etc.), and other subtle shading effects.

4. This is a significant departure from the v18 format, where vertex color data was stored per-chunk in MCVT. The new global approach provides:
   - Potential memory savings through deduplication
   - Easier access to all vertex data at once for rendering
   - Better consistency across chunk boundaries

5. The vertex colors are applied during rendering and are typically blended with the base texture colors for the final terrain appearance.

6. Data is stored in row-major order: vertices proceed from north to south (outer loop), and from west to east (inner loop).

7. Each vertex has four color components: Blue, Green, Red, and Alpha, in that order.

## Implementation Example

```csharp
public class AcvtChunk
{
    // Constants for ADT dimensions
    private const int MAP_SIZE = 17; // Chunks per ADT side (always 17×17 chunks in a tile)
    private const int CHUNK_VERTICES = 9; // Vertices per chunk side (9×9 inner vertices)
    private const int VERTICES_PER_TILE_SIDE = (MAP_SIZE * (CHUNK_VERTICES - 1)) + 1; // 145 vertices per side

    // BGRA color data for all vertices
    public byte[] ColorData { get; private set; }
    
    public AcvtChunk()
    {
        // Initialize with default colors (white, fully opaque)
        ColorData = new byte[VERTICES_PER_TILE_SIDE * VERTICES_PER_TILE_SIDE * 4];
        for (int i = 0; i < ColorData.Length; i += 4)
        {
            ColorData[i] = 255;     // Blue
            ColorData[i + 1] = 255; // Green
            ColorData[i + 2] = 255; // Red
            ColorData[i + 3] = 255; // Alpha
        }
    }
    
    public void Load(BinaryReader reader)
    {
        int size = VERTICES_PER_TILE_SIDE * VERTICES_PER_TILE_SIDE * 4;
        ColorData = reader.ReadBytes(size);
    }
    
    public void Save(BinaryWriter writer)
    {
        writer.Write("ACVT".ToCharArray());
        writer.Write(ColorData.Length);
        writer.Write(ColorData);
    }
    
    // Get color for a specific vertex by its grid position
    public Color GetVertexColor(int x, int y)
    {
        if (x < 0 || x >= VERTICES_PER_TILE_SIDE || y < 0 || y >= VERTICES_PER_TILE_SIDE)
            throw new ArgumentOutOfRangeException("Vertex coordinates out of range");
            
        int index = (y * VERTICES_PER_TILE_SIDE + x) * 4;
        
        return new Color(
            r: ColorData[index + 2] / 255.0f,
            g: ColorData[index + 1] / 255.0f,
            b: ColorData[index] / 255.0f,
            a: ColorData[index + 3] / 255.0f
        );
    }
    
    // Set color for a specific vertex
    public void SetVertexColor(int x, int y, Color color)
    {
        if (x < 0 || x >= VERTICES_PER_TILE_SIDE || y < 0 || y >= VERTICES_PER_TILE_SIDE)
            throw new ArgumentOutOfRangeException("Vertex coordinates out of range");
            
        int index = (y * VERTICES_PER_TILE_SIDE + x) * 4;
        
        ColorData[index] = (byte)(color.B * 255);     // Blue
        ColorData[index + 1] = (byte)(color.G * 255); // Green
        ColorData[index + 2] = (byte)(color.R * 255); // Red
        ColorData[index + 3] = (byte)(color.A * 255); // Alpha
    }
    
    // Apply a gradient across the entire terrain
    public void ApplyGradient(Color topLeft, Color topRight, Color bottomLeft, Color bottomRight)
    {
        for (int y = 0; y < VERTICES_PER_TILE_SIDE; y++)
        {
            float yPercent = (float)y / (VERTICES_PER_TILE_SIDE - 1);
            
            // Interpolate colors along top and bottom edges
            Color topColor = Color.Lerp(topLeft, topRight, yPercent);
            Color bottomColor = Color.Lerp(bottomLeft, bottomRight, yPercent);
            
            for (int x = 0; x < VERTICES_PER_TILE_SIDE; x++)
            {
                float xPercent = (float)x / (VERTICES_PER_TILE_SIDE - 1);
                
                // Interpolate between top and bottom colors
                Color vertexColor = Color.Lerp(topColor, bottomColor, xPercent);
                SetVertexColor(x, y, vertexColor);
            }
        }
    }
    
    // Adjust brightness of an area
    public void AdjustBrightness(int startX, int startY, int endX, int endY, float factor)
    {
        // Clamp coordinates to valid range
        startX = Math.Max(0, Math.Min(startX, VERTICES_PER_TILE_SIDE - 1));
        startY = Math.Max(0, Math.Min(startY, VERTICES_PER_TILE_SIDE - 1));
        endX = Math.Max(0, Math.Min(endX, VERTICES_PER_TILE_SIDE - 1));
        endY = Math.Max(0, Math.Min(endY, VERTICES_PER_TILE_SIDE - 1));
        
        for (int y = startY; y <= endY; y++)
        {
            for (int x = startX; x <= endX; x++)
            {
                Color color = GetVertexColor(x, y);
                color = new Color(
                    r: Math.Min(1.0f, color.R * factor),
                    g: Math.Min(1.0f, color.G * factor),
                    b: Math.Min(1.0f, color.B * factor),
                    a: color.A
                );
                SetVertexColor(x, y, color);
            }
        }
    }
}
```

## Usage Context

The ACVT chunk plays a critical role in enhancing the visual appearance of terrain in World of Warcraft:

1. **Visual Detail Enhancement**: Vertex colors add subtle shading variations to terrain, creating more realistic and visually interesting landscapes by breaking up the uniformity of texture patterns.

2. **Environment Representation**: Colors can represent environmental effects such as snow, dirt, or shadowed areas, helping to blend different terrain types together naturally.

3. **Data Centralization**: By moving vertex colors from per-chunk storage (as in v18) to a global array, the v23 format enables more efficient memory use and potentially improves rendering performance.

4. **Global Terrain Effects**: The global approach allows for large-scale color effects that span multiple chunks, such as gradients, shadows from large objects, or smooth transitions between biomes.

5. **Extended Artistic Control**: Level designers can use vertex coloring to fine-tune the appearance of terrain without needing to create additional textures or modify the base terrain height.

The ACVT chunk represents an important part of the experimental v23 format's approach to terrain data centralization, moving away from the self-contained chunk design of previous formats toward a more unified, globally accessible data structure. While this format was never used in a retail release, it demonstrates Blizzard's exploration of alternative data organization methods during the Cataclysm beta development period. 