# C005: AVTX

## Type
ADT v23 Chunk

## Source
ADT_v23.md

## Description
Vertex data for ADT v23 format. Contains height information for the terrain vertices, enhanced with additional precision for WoD's higher detail terrain.

## Structure
```csharp
struct AVTX
{
    // 9×9 vertex grid for each of the 16×16 chunks = 145×145 vertices
    // Enhanced precision in WoD (v23)
    float heights[145 * 145]; // 145×145 grid of vertex heights
}
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| heights | float[21025] | Height values for each vertex in the terrain grid (145×145) |

## Dependencies
- AHDR (C004) - References this chunk via offsets

## Implementation Notes
- Size: 84,100 bytes (21,025 floats × 4 bytes)
- Contains a 145×145 grid of height values
- Each ACNK chunk is 9×9 vertices, with shared edges between chunks
- Enhanced in WoD with improved precision and support for more detailed height variations
- Used for high-detail terrain rendering
- Similar to AVTX in ADT v22, but with improved precision for height data

## Implementation Example
```csharp
public class AVTX
{
    // Dimensions of the vertex grid
    public const int GRID_SIZE = 145;
    
    // Number of vertices per chunk plus 1 (shared edges)
    public const int CHUNK_GRID_SIZE = 9;
    
    // Number of chunks in each dimension
    public const int CHUNKS_PER_SIDE = 16;
    
    // Height data as a 2D array
    private float[,] _heights;
    
    public AVTX(float[] heightData)
    {
        if (heightData.Length != GRID_SIZE * GRID_SIZE)
            throw new ArgumentException($"Height data must contain {GRID_SIZE * GRID_SIZE} elements");
        
        // Convert linear array to 2D grid
        _heights = new float[GRID_SIZE, GRID_SIZE];
        for (int y = 0; y < GRID_SIZE; y++)
        {
            for (int x = 0; x < GRID_SIZE; x++)
            {
                _heights[x, y] = heightData[y * GRID_SIZE + x];
            }
        }
    }
    
    // Get height at specific grid coordinates
    public float GetHeightAt(int x, int y)
    {
        if (x < 0 || x >= GRID_SIZE || y < 0 || y >= GRID_SIZE)
            throw new ArgumentOutOfRangeException($"Coordinates ({x},{y}) out of range");
            
        return _heights[x, y];
    }
    
    // Get interpolated height at a normalized position (0-1)
    public float GetInterpolatedHeight(float normalizedX, float normalizedY)
    {
        float gridX = normalizedX * (GRID_SIZE - 1);
        float gridY = normalizedY * (GRID_SIZE - 1);
        
        int x1 = (int)Math.Floor(gridX);
        int y1 = (int)Math.Floor(gridY);
        int x2 = Math.Min(x1 + 1, GRID_SIZE - 1);
        int y2 = Math.Min(y1 + 1, GRID_SIZE - 1);
        
        float fracX = gridX - x1;
        float fracY = gridY - y1;
        
        float h11 = _heights[x1, y1];
        float h21 = _heights[x2, y1];
        float h12 = _heights[x1, y2];
        float h22 = _heights[x2, y2];
        
        // Bilinear interpolation
        float h1 = h11 * (1 - fracX) + h21 * fracX;
        float h2 = h12 * (1 - fracX) + h22 * fracX;
        float h = h1 * (1 - fracY) + h2 * fracY;
        
        return h;
    }
    
    // Get height data for a specific chunk (9×9 grid)
    public float[,] GetChunkHeights(int chunkX, int chunkY)
    {
        if (chunkX < 0 || chunkX >= CHUNKS_PER_SIDE || chunkY < 0 || chunkY >= CHUNKS_PER_SIDE)
            throw new ArgumentOutOfRangeException($"Chunk coordinates ({chunkX},{chunkY}) out of range");
        
        float[,] chunkHeights = new float[CHUNK_GRID_SIZE, CHUNK_GRID_SIZE];
        
        int baseX = chunkX * (CHUNK_GRID_SIZE - 1);
        int baseY = chunkY * (CHUNK_GRID_SIZE - 1);
        
        for (int y = 0; y < CHUNK_GRID_SIZE; y++)
        {
            for (int x = 0; x < CHUNK_GRID_SIZE; x++)
            {
                chunkHeights[x, y] = _heights[baseX + x, baseY + y];
            }
        }
        
        return chunkHeights;
    }
    
    // Generate a height map visualization
    public System.Drawing.Bitmap GenerateHeightMap()
    {
        var bitmap = new System.Drawing.Bitmap(GRID_SIZE, GRID_SIZE);
        
        // Find height range for normalization
        float minHeight = float.MaxValue;
        float maxHeight = float.MinValue;
        
        for (int y = 0; y < GRID_SIZE; y++)
        {
            for (int x = 0; x < GRID_SIZE; x++)
            {
                minHeight = Math.Min(minHeight, _heights[x, y]);
                maxHeight = Math.Max(maxHeight, _heights[x, y]);
            }
        }
        
        float heightRange = maxHeight - minHeight;
        
        // Generate grayscale height map
        for (int y = 0; y < GRID_SIZE; y++)
        {
            for (int x = 0; x < GRID_SIZE; x++)
            {
                // Normalize height to 0-255 range
                int grayscale = (int)(255 * (_heights[x, y] - minHeight) / heightRange);
                grayscale = Math.Max(0, Math.Min(255, grayscale));
                
                var color = System.Drawing.Color.FromArgb(grayscale, grayscale, grayscale);
                bitmap.SetPixel(x, y, color);
            }
        }
        
        return bitmap;
    }
    
    // Apply a terrain smoothing filter
    public void SmoothTerrain(float strength = 0.5f)
    {
        float[,] smoothedHeights = new float[GRID_SIZE, GRID_SIZE];
        
        for (int y = 0; y < GRID_SIZE; y++)
        {
            for (int x = 0; x < GRID_SIZE; x++)
            {
                // Get neighboring heights
                float sum = 0;
                int count = 0;
                
                for (int dy = -1; dy <= 1; dy++)
                {
                    for (int dx = -1; dx <= 1; dx++)
                    {
                        int nx = x + dx;
                        int ny = y + dy;
                        
                        if (nx >= 0 && nx < GRID_SIZE && ny >= 0 && ny < GRID_SIZE)
                        {
                            sum += _heights[nx, ny];
                            count++;
                        }
                    }
                }
                
                // Weighted average with original height
                float average = sum / count;
                smoothedHeights[x, y] = _heights[x, y] * (1 - strength) + average * strength;
            }
        }
        
        // Update heights with smoothed values
        _heights = smoothedHeights;
    }
}
```

## Usage Context
The AVTX chunk contains height information for the entire terrain tile, providing the base data needed to construct the 3D terrain mesh. It defines a 145×145 grid of height values, representing a 9×9 grid for each of the 16×16 terrain chunks (with shared edges).

In the ADT v23 format used since Warlords of Draenor, the AVTX chunk has been enhanced with improved precision for height values compared to v22, allowing for more detailed terrain features and smoother height transitions. This supports WoD's graphical improvements, which included higher resolution terrain and more detailed environmental features.

The height values are used in conjunction with the normal data (ANRM) and texture information to render the terrain. Each vertex in the grid represents a point in 3D space, with the X and Z coordinates determined by the grid position and the Y coordinate provided by the height value. 