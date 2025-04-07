# ASHD - Shadow Map Data

## Type
ADT v22 ACNK Subchunk

## Source
Referenced from `ADT_v22.md`

## Description
The ASHD (Shadow) subchunk contains shadow map data for a terrain chunk in the ADT v22 format. It defines how shadows are cast on the terrain, controlling the darkness and distribution of shadows from terrain features, objects, and global lighting. Each value in the shadow map represents the shadow intensity at a specific point on the terrain.

## Structure

```csharp
public struct ASHD
{
    // Shadow map data (commonly a 64×64 grid of 8-bit shadow values)
    public byte[] shadowData;
}
```

## Properties

| Name | Type | Description |
|------|------|-------------|
| shadowData | byte[] | Array of shadow intensity values (typically 64×64 = 4096 bytes) |

## Dependencies

- ACNK (C006) - Parent chunk that contains this subchunk

## Implementation Notes

1. The ASHD subchunk is similar to the MCSH subchunk in ADT v18, but with a different parent-child relationship.

2. The shadow map is typically a 64×64 grid of 8-bit values, matching the terrain chunk resolution.

3. Each byte in the shadow data represents the shadow intensity at a specific point on the terrain:
   - 0 = Fully shadowed (black)
   - 255 = No shadow (full brightness)
   - Values in between represent partial shadows

4. The shadow map is primarily used for static terrain shadows and ambient occlusion, not for dynamic shadows from moving objects or time-of-day lighting.

5. The shadow data size is typically 0x1000 (4096) bytes for a 64×64 grid, but the documentation suggests it might be 0x200 (512) bytes in some cases, which would imply a different organization or compression.

## Implementation Example

```csharp
public class AshdSubChunk
{
    // Shadow grid dimensions (typically 64×64)
    public const int SHADOW_SIZE = 64;
    
    // Raw shadow data
    public byte[] ShadowData { get; set; }
    
    // Constructor with empty shadow map
    public AshdSubChunk()
    {
        // Initialize with a fully lit shadow map (no shadows)
        ShadowData = new byte[SHADOW_SIZE * SHADOW_SIZE];
        for (int i = 0; i < ShadowData.Length; i++)
        {
            ShadowData[i] = 255; // Fully lit
        }
    }
    
    // Constructor with existing shadow data
    public AshdSubChunk(byte[] shadowData)
    {
        if (shadowData != null)
        {
            ShadowData = shadowData;
        }
        else
        {
            ShadowData = new byte[SHADOW_SIZE * SHADOW_SIZE];
            for (int i = 0; i < ShadowData.Length; i++)
            {
                ShadowData[i] = 255; // Fully lit
            }
        }
    }
    
    // Load shadow data from a binary reader
    public void Load(BinaryReader reader, long size)
    {
        ShadowData = reader.ReadBytes((int)size);
        
        // If the shadow data isn't the expected size, resize it
        if (ShadowData.Length != SHADOW_SIZE * SHADOW_SIZE)
        {
            // Handle different shadow map sizes - either expand or compress
            ResizeShadowMap();
        }
    }
    
    // Save shadow data to a binary writer
    public void Save(BinaryWriter writer)
    {
        writer.Write("ASHD".ToCharArray());
        writer.Write(ShadowData.Length);
        writer.Write(ShadowData);
    }
    
    // Resize shadow map to the expected size if it's different
    private void ResizeShadowMap()
    {
        // If the size is already correct, do nothing
        if (ShadowData.Length == SHADOW_SIZE * SHADOW_SIZE)
            return;
        
        byte[] resizedData = new byte[SHADOW_SIZE * SHADOW_SIZE];
        
        if (ShadowData.Length == 512) // 0x200 as mentioned in docs
        {
            // This might be a 32×16 shadow map or some other format
            // Simple linear scaling for demonstration purposes
            for (int y = 0; y < SHADOW_SIZE; y++)
            {
                for (int x = 0; x < SHADOW_SIZE; x++)
                {
                    // Scale coordinates to match the smaller map
                    int sourceX = x * 32 / SHADOW_SIZE;
                    int sourceY = y * 16 / SHADOW_SIZE;
                    int sourceIndex = sourceY * 32 + sourceX;
                    
                    if (sourceIndex < ShadowData.Length)
                    {
                        resizedData[y * SHADOW_SIZE + x] = ShadowData[sourceIndex];
                    }
                    else
                    {
                        resizedData[y * SHADOW_SIZE + x] = 255; // Default to fully lit
                    }
                }
            }
        }
        else
        {
            // For other sizes, just fill with default values
            for (int i = 0; i < resizedData.Length; i++)
            {
                resizedData[i] = 255; // Fully lit
            }
            
            // Copy as much data as possible
            int copyLength = Math.Min(ShadowData.Length, resizedData.Length);
            Array.Copy(ShadowData, resizedData, copyLength);
        }
        
        ShadowData = resizedData;
    }
    
    // Get shadow value at a specific position (0-63)
    public byte GetShadowAt(int x, int y)
    {
        if (x < 0) x = 0;
        if (x >= SHADOW_SIZE) x = SHADOW_SIZE - 1;
        if (y < 0) y = 0;
        if (y >= SHADOW_SIZE) y = SHADOW_SIZE - 1;
        
        int index = y * SHADOW_SIZE + x;
        if (index >= 0 && index < ShadowData.Length)
        {
            return ShadowData[index];
        }
        
        return 255; // Default to fully lit
    }
    
    // Get interpolated shadow value at a normalized position (0-1)
    public byte GetInterpolatedShadow(float normalizedX, float normalizedY)
    {
        float gridX = normalizedX * (SHADOW_SIZE - 1);
        float gridY = normalizedY * (SHADOW_SIZE - 1);
        
        int x1 = (int)Math.Floor(gridX);
        int y1 = (int)Math.Floor(gridY);
        int x2 = Math.Min(x1 + 1, SHADOW_SIZE - 1);
        int y2 = Math.Min(y1 + 1, SHADOW_SIZE - 1);
        
        float fracX = gridX - x1;
        float fracY = gridY - y1;
        
        // Get the four surrounding shadow values
        byte s11 = GetShadowAt(x1, y1);
        byte s21 = GetShadowAt(x2, y1);
        byte s12 = GetShadowAt(x1, y2);
        byte s22 = GetShadowAt(x2, y2);
        
        // Bilinear interpolation
        float s1 = s11 * (1 - fracX) + s21 * fracX;
        float s2 = s12 * (1 - fracX) + s22 * fracX;
        float s = s1 * (1 - fracY) + s2 * fracY;
        
        return (byte)Math.Round(s);
    }
}
```

## Usage Context

The ASHD subchunk plays a crucial role in creating visually realistic terrain by providing shadow information. These shadows help define the terrain's contours and features, making elevation changes more apparent and adding depth to the visual representation.

Shadow maps in World of Warcraft serve several purposes:

1. **Ambient Occlusion**: The shadow map provides subtle darkening in crevices, corners, and areas where light has difficulty reaching, enhancing the perception of depth and detail in the terrain.

2. **Terrain Definition**: Shadows highlight slopes, ridges, and valleys, making the shape of the terrain more visually apparent even without strong lighting.

3. **Visual Consistency**: By baking shadows into the terrain, the game maintains a consistent visual appearance regardless of the player's graphics settings or viewing distance.

The v22 format's approach to shadow mapping is similar to v18, but with the ASHD subchunk directly contained within the ACNK chunk, which potentially simplifies the relationship between terrain data and its associated shadows.

In the rendering pipeline, the shadow map is typically combined with dynamic lighting effects to produce the final lighting result for the terrain. The static shadow information in ASHD provides a baseline that is then modified by time-of-day lighting, weather effects, and other dynamic lighting factors. 