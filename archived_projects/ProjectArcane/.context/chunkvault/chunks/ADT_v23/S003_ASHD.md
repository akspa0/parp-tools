# ASHD - Shadow Map Data

## Type
ADT v23 ACNK Subchunk

## Source
Referenced from `ADT_v23.md`

## Description
The ASHD (Shadow) subchunk contains shadow map data for a specific terrain chunk. This data represents pre-calculated shadows cast by static terrain features, providing ambient occlusion and shadow information that enhances the visual quality of the terrain without requiring real-time shadow calculations. The shadow map stores an intensity value for each point in the chunk's grid.

## Structure

```csharp
public struct ASHD
{
    // Shadow intensity values for each point in the chunk grid
    // Typically a 64×64 or 128×128 grid of bytes
    public byte[] shadowData;
}
```

## Properties

| Name | Type | Description |
|------|------|-------------|
| shadowData | byte[] | Array of shadow intensity values (0 = full shadow, 255 = no shadow) |

## Dependencies

- ACNK (C006) - Parent chunk that contains this subchunk

## Implementation Notes

1. The ASHD subchunk contains a grid of shadow intensity values, with one byte per grid point.

2. The size of the shadow map is typically 64×64 (4,096 bytes) or 128×128 (16,384 bytes), depending on the detail level required.

3. Each byte represents the shadow intensity at that point, where 0 indicates complete shadow (darkest) and 255 indicates no shadow (lightest).

4. The shadow map is used to provide ambient occlusion and pre-baked shadows for static terrain features, enhancing depth perception without expensive real-time shadow calculations.

5. Unlike dynamic shadows that change with time of day, these shadows represent persistent features like valleys, overhangs, and depressions in the terrain.

6. The v23 format stores shadow maps at the chunk level in ASHD subchunks, while the v18 format uses global MCSH chunks.

7. The local approach in v23 potentially allows for more detailed shadows that are specific to each chunk's terrain features.

## Implementation Example

```csharp
public class AshdSubchunk
{
    // Constants
    private const int SHADOW_MAP_WIDTH = 64;   // Standard size for shadow maps
    private const int SHADOW_MAP_HEIGHT = 64;  // Can be increased for higher detail
    
    // Shadow map data
    public byte[] ShadowData { get; private set; }
    
    // Whether this is a high-resolution shadow map (128×128)
    private bool isHighResolution = false;
    
    public AshdSubchunk(bool highResolution = false)
    {
        isHighResolution = highResolution;
        int size = GetMapSize();
        ShadowData = new byte[size];
        
        // Initialize to no shadow (255)
        for (int i = 0; i < size; i++)
            ShadowData[i] = 255;
    }
    
    private int GetMapSize()
    {
        int width = isHighResolution ? SHADOW_MAP_WIDTH * 2 : SHADOW_MAP_WIDTH;
        int height = isHighResolution ? SHADOW_MAP_HEIGHT * 2 : SHADOW_MAP_HEIGHT;
        return width * height;
    }
    
    private int GetMapWidth()
    {
        return isHighResolution ? SHADOW_MAP_WIDTH * 2 : SHADOW_MAP_WIDTH;
    }
    
    private int GetMapHeight()
    {
        return isHighResolution ? SHADOW_MAP_HEIGHT * 2 : SHADOW_MAP_HEIGHT;
    }
    
    public void Load(BinaryReader reader, uint size)
    {
        // Determine if this is a high-resolution shadow map
        isHighResolution = size > (SHADOW_MAP_WIDTH * SHADOW_MAP_HEIGHT);
        
        // Ensure we don't read more data than the shadow map should contain
        int expectedSize = GetMapSize();
        int readSize = (int)Math.Min(size, expectedSize);
        
        // Read shadow data
        ShadowData = reader.ReadBytes(readSize);
        
        // If we read less than expected, fill the rest with 255 (no shadow)
        if (readSize < expectedSize)
        {
            byte[] fullData = new byte[expectedSize];
            Array.Copy(ShadowData, 0, fullData, 0, readSize);
            
            for (int i = readSize; i < expectedSize; i++)
                fullData[i] = 255;
                
            ShadowData = fullData;
        }
    }
    
    public void Save(BinaryWriter writer)
    {
        writer.Write("ASHD".ToCharArray());
        writer.Write(ShadowData.Length);
        writer.Write(ShadowData);
    }
    
    // Get shadow value at specific coordinates
    public byte GetShadow(int x, int y)
    {
        int width = GetMapWidth();
        int height = GetMapHeight();
        
        if (x < 0 || x >= width || y < 0 || y >= height)
            return 255; // Default to no shadow for out-of-bounds
            
        int index = y * width + x;
        return ShadowData[index];
    }
    
    // Set shadow value at specific coordinates
    public void SetShadow(int x, int y, byte value)
    {
        int width = GetMapWidth();
        int height = GetMapHeight();
        
        if (x < 0 || x >= width || y < 0 || y >= height)
            return; // Out of bounds
            
        int index = y * width + x;
        ShadowData[index] = value;
    }
    
    // Generate a simple shadow map based on height data
    public void GenerateFromHeightMap(float[,] heightMap)
    {
        int width = GetMapWidth();
        int height = GetMapHeight();
        
        // Very simple shadow algorithm (north-west light source)
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                // Sample height at current position
                float centerHeight = SampleHeightMap(heightMap, x, y);
                
                // Sample height at light direction (north-west)
                float nwHeight = SampleHeightMap(heightMap, x - 1, y - 1);
                
                // Calculate shadow based on height difference
                float heightDiff = nwHeight - centerHeight;
                
                if (heightDiff > 0)
                {
                    // Higher terrain to the north-west casts shadow
                    float shadowIntensity = Math.Min(1.0f, heightDiff * 2.0f);
                    byte shadowValue = (byte)(255 - (shadowIntensity * 192)); // Max 75% shadow
                    SetShadow(x, y, shadowValue);
                }
                else
                {
                    // No shadow
                    SetShadow(x, y, 255);
                }
            }
        }
    }
    
    // Helper method to sample height map with bounds checking
    private float SampleHeightMap(float[,] heightMap, int x, int y)
    {
        int hmWidth = heightMap.GetLength(0);
        int hmHeight = heightMap.GetLength(1);
        
        if (x < 0 || x >= hmWidth || y < 0 || y >= hmHeight)
            return 0.0f; // Default to lowest height for out-of-bounds
            
        return heightMap[x, y];
    }
    
    // Apply a blur filter to smooth the shadow map
    public void ApplyBlur(int radius = 1)
    {
        int width = GetMapWidth();
        int height = GetMapHeight();
        byte[] blurredData = new byte[ShadowData.Length];
        
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int sum = 0;
                int count = 0;
                
                // Average the surrounding pixels
                for (int dy = -radius; dy <= radius; dy++)
                {
                    for (int dx = -radius; dx <= radius; dx++)
                    {
                        int sx = x + dx;
                        int sy = y + dy;
                        
                        if (sx >= 0 && sx < width && sy >= 0 && sy < height)
                        {
                            sum += ShadowData[sy * width + sx];
                            count++;
                        }
                    }
                }
                
                // Calculate average
                blurredData[y * width + x] = (byte)(sum / count);
            }
        }
        
        // Update shadow data
        ShadowData = blurredData;
    }
    
    // Resize the shadow map (useful for converting between resolution levels)
    public void Resize(bool highResolution)
    {
        if (isHighResolution == highResolution)
            return; // No change needed
            
        int oldWidth = GetMapWidth();
        int oldHeight = GetMapHeight();
        
        // Change resolution setting
        isHighResolution = highResolution;
        
        int newWidth = GetMapWidth();
        int newHeight = GetMapHeight();
        byte[] newData = new byte[newWidth * newHeight];
        
        // Resample the shadow map
        for (int y = 0; y < newHeight; y++)
        {
            for (int x = 0; x < newWidth; x++)
            {
                // Map coordinates to old shadow map
                float oldX = (float)x * oldWidth / newWidth;
                float oldY = (float)y * oldHeight / newHeight;
                
                // Simple bilinear interpolation
                int x1 = (int)Math.Floor(oldX);
                int y1 = (int)Math.Floor(oldY);
                int x2 = Math.Min(x1 + 1, oldWidth - 1);
                int y2 = Math.Min(y1 + 1, oldHeight - 1);
                
                float fracX = oldX - x1;
                float fracY = oldY - y1;
                
                byte topLeft = ShadowData[y1 * oldWidth + x1];
                byte topRight = ShadowData[y1 * oldWidth + x2];
                byte bottomLeft = ShadowData[y2 * oldWidth + x1];
                byte bottomRight = ShadowData[y2 * oldWidth + x2];
                
                float top = topLeft * (1 - fracX) + topRight * fracX;
                float bottom = bottomLeft * (1 - fracX) + bottomRight * fracX;
                byte value = (byte)(top * (1 - fracY) + bottom * fracY);
                
                newData[y * newWidth + x] = value;
            }
        }
        
        // Update shadow data
        ShadowData = newData;
    }
}
```

## Usage Context

The ASHD subchunk plays a crucial role in enhancing the visual quality of terrain in World of Warcraft:

1. **Ambient Occlusion**: Provides subtle shading in crevices, under ledges, and in other areas that would naturally receive less ambient light, significantly enhancing depth perception.

2. **Pre-calculated Shadows**: Offers computationally efficient shadows for static terrain features without requiring expensive real-time shadow calculations.

3. **Visual Consistency**: Ensures consistent shadow representation regardless of time of day or weather conditions, creating a baseline of terrain definition.

4. **Performance Optimization**: By pre-calculating shadows for static terrain, reduces the computational load on the client during gameplay.

5. **Enhanced Realism**: Contributes to the overall realism of the terrain by emphasizing its three-dimensional structure through appropriate shading.

The ASHD subchunk in v23 represents an evolution from the MCSH chunk in v18, adopting a per-chunk approach that potentially allows for more detailed and localized shadow information. This local shadow data organization aligns with v23's general theme of organizing terrain data at the chunk level rather than globally. Though never used in a retail release, this experimental approach provides insight into Blizzard's exploration of alternative terrain data organization during the Cataclysm beta development period. 