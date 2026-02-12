# AMAP - Alpha Map Data

## Type
ADT v23 ACNK Subchunk

## Source
Referenced from `ADT_v23.md`

## Description
The AMAP (Alpha Map) subchunk contains alpha (opacity) map data for texture layers defined in the ALYR subchunks. Each alpha map controls how the corresponding texture layer blends with layers beneath it. The AMAP subchunk typically follows a set of ALYR subchunks in the ACNK chunk and contains all alpha maps for those layers in a single consolidated data block.

## Structure

```csharp
public struct AMAP
{
    // Variable-length array containing all alpha map data for the chunk
    // Each texture layer that has the USE_ALPHA_MAP flag will have its alpha map 
    // stored at the offset specified in its ALYR.offsetInAMAP field
    public byte[] alphaMapData;
}
```

## Properties

| Name | Type | Description |
|------|------|-------------|
| alphaMapData | byte[] | Variable-length array containing alpha map data for all texture layers in the chunk |

## Dependencies

- ACNK (C006) - Parent chunk that contains this subchunk
- ALYR (S001) - Defines texture layers that reference this alpha map data

## Implementation Notes

1. The AMAP subchunk contains a single contiguous block of alpha map data for all texture layers in the chunk that use alpha maps.

2. The size of the AMAP subchunk depends on the number of alpha maps and their dimensions.

3. Each alpha map is typically a 64×64 grid of byte values, where each byte represents the opacity of the texture layer at that point (0 = fully transparent, 255 = fully opaque).

4. The location of each layer's alpha map within the AMAP data block is specified by the `offsetInAMAP` field in the corresponding ALYR subchunk.

5. Alpha maps may be stored in compressed or uncompressed format, as indicated by the `COMPRESSED_ALPHA` flag in the ALYR subchunk.

6. When rendering, these alpha values are used to blend multiple texture layers together, creating transitions between different terrain types (grass fading to dirt, etc.).

7. The v23 format consolidates all alpha maps into a single AMAP subchunk, unlike the v18 format where alpha maps were separate MCAL chunks.

8. This consolidated approach potentially allows for more efficient memory use and easier management of alpha map data.

## Implementation Example

```csharp
public class AmapSubchunk
{
    private const int ALPHA_MAP_SIZE = 64; // Standard size for alpha maps (64×64)
    
    // Alpha map data for all layers
    public byte[] AlphaMapData { get; private set; }
    
    // List of offsets where each layer's alpha map starts
    private List<int> layerOffsets = new List<int>();
    
    public AmapSubchunk()
    {
        AlphaMapData = new byte[0];
    }
    
    public void Load(BinaryReader reader, uint size)
    {
        AlphaMapData = reader.ReadBytes((int)size);
    }
    
    public void Save(BinaryWriter writer)
    {
        writer.Write("AMAP".ToCharArray());
        writer.Write(AlphaMapData.Length);
        writer.Write(AlphaMapData);
    }
    
    // Get alpha value for a specific layer at specific coordinates
    public byte GetAlpha(int layerIndex, int x, int y, int offset)
    {
        if (layerIndex < 0 || offset >= AlphaMapData.Length)
            return 0;
            
        if (x < 0 || x >= ALPHA_MAP_SIZE || y < 0 || y >= ALPHA_MAP_SIZE)
            return 0;
            
        int index = offset + (y * ALPHA_MAP_SIZE + x);
        
        if (index >= AlphaMapData.Length)
            return 0;
            
        return AlphaMapData[index];
    }
    
    // Set alpha value for a specific layer at specific coordinates
    public void SetAlpha(int layerIndex, int x, int y, byte value, int offset)
    {
        if (layerIndex < 0 || offset >= AlphaMapData.Length)
            return;
            
        if (x < 0 || x >= ALPHA_MAP_SIZE || y < 0 || y >= ALPHA_MAP_SIZE)
            return;
            
        int index = offset + (y * ALPHA_MAP_SIZE + x);
        
        if (index >= AlphaMapData.Length)
            return;
            
        AlphaMapData[index] = value;
    }
    
    // Add a new alpha map and return its offset in the data block
    public int AddAlphaMap(byte[] alphaMap = null)
    {
        int offset = AlphaMapData.Length;
        
        // Create a default alpha map if none provided
        if (alphaMap == null)
        {
            alphaMap = new byte[ALPHA_MAP_SIZE * ALPHA_MAP_SIZE];
            
            // Default to fully opaque
            for (int i = 0; i < alphaMap.Length; i++)
                alphaMap[i] = 255;
        }
        
        // Resize the data array to accommodate the new alpha map
        byte[] newData = new byte[AlphaMapData.Length + alphaMap.Length];
        
        // Copy existing data
        if (AlphaMapData.Length > 0)
            Array.Copy(AlphaMapData, 0, newData, 0, AlphaMapData.Length);
            
        // Append new alpha map
        Array.Copy(alphaMap, 0, newData, offset, alphaMap.Length);
        
        // Update the data array
        AlphaMapData = newData;
        
        // Track the offset of this layer
        layerOffsets.Add(offset);
        
        return offset;
    }
    
    // Create a smoothed transition between two terrain types
    public byte[] CreateTransitionAlphaMap(int width, bool invertGradient = false)
    {
        byte[] alphaMap = new byte[ALPHA_MAP_SIZE * ALPHA_MAP_SIZE];
        
        for (int y = 0; y < ALPHA_MAP_SIZE; y++)
        {
            for (int x = 0; x < ALPHA_MAP_SIZE; x++)
            {
                // Create a transition along the X axis
                float gradient = (float)x / ALPHA_MAP_SIZE;
                
                // Smooth the transition using a sigmoid-like function
                float smoothGradient = (float)(1.0 / (1.0 + Math.Exp(-10 * (gradient - 0.5))));
                
                // Apply width factor (narrower or wider transition zone)
                float adjustedGradient = (float)(1.0 / (1.0 + Math.Exp(-width * (gradient - 0.5))));
                
                // Convert to byte (0-255)
                byte alpha = (byte)(adjustedGradient * 255);
                
                // Invert if requested
                if (invertGradient)
                    alpha = (byte)(255 - alpha);
                    
                alphaMap[y * ALPHA_MAP_SIZE + x] = alpha;
            }
        }
        
        return alphaMap;
    }
    
    // Create a circular alpha map (useful for objects like lakes)
    public byte[] CreateCircularAlphaMap(float centerX = 0.5f, float centerY = 0.5f, float radius = 0.4f, bool invertGradient = false)
    {
        byte[] alphaMap = new byte[ALPHA_MAP_SIZE * ALPHA_MAP_SIZE];
        
        for (int y = 0; y < ALPHA_MAP_SIZE; y++)
        {
            float normalizedY = (float)y / ALPHA_MAP_SIZE;
            
            for (int x = 0; x < ALPHA_MAP_SIZE; x++)
            {
                float normalizedX = (float)x / ALPHA_MAP_SIZE;
                
                // Calculate distance from center
                float distance = (float)Math.Sqrt(
                    Math.Pow(normalizedX - centerX, 2) + 
                    Math.Pow(normalizedY - centerY, 2)
                );
                
                // Create alpha based on distance from center
                float alpha;
                if (distance <= radius)
                    alpha = 1.0f;
                else if (distance >= radius + 0.1f)
                    alpha = 0.0f;
                else
                    alpha = 1.0f - ((distance - radius) * 10.0f); // Smooth edge
                    
                // Invert if requested
                if (invertGradient)
                    alpha = 1.0f - alpha;
                    
                alphaMap[y * ALPHA_MAP_SIZE + x] = (byte)(alpha * 255);
            }
        }
        
        return alphaMap;
    }
    
    // Compress an alpha map (simple RLE compression as an example)
    public byte[] CompressAlphaMap(byte[] alphaMap)
    {
        if (alphaMap == null || alphaMap.Length == 0)
            return new byte[0];
            
        List<byte> compressed = new List<byte>();
        byte currentValue = alphaMap[0];
        byte runLength = 1;
        
        for (int i = 1; i < alphaMap.Length; i++)
        {
            if (alphaMap[i] == currentValue && runLength < 255)
            {
                runLength++;
            }
            else
            {
                // Store the run
                compressed.Add(runLength);
                compressed.Add(currentValue);
                
                // Start a new run
                currentValue = alphaMap[i];
                runLength = 1;
            }
        }
        
        // Add the last run
        compressed.Add(runLength);
        compressed.Add(currentValue);
        
        return compressed.ToArray();
    }
    
    // Decompress an alpha map (matching the above compression)
    public byte[] DecompressAlphaMap(byte[] compressedData, int offset, int size)
    {
        List<byte> decompressed = new List<byte>();
        
        for (int i = offset; i < offset + size; i += 2)
        {
            if (i + 1 >= offset + size)
                break;
                
            byte runLength = compressedData[i];
            byte value = compressedData[i + 1];
            
            for (int j = 0; j < runLength; j++)
                decompressed.Add(value);
        }
        
        return decompressed.ToArray();
    }
}
```

## Usage Context

The AMAP subchunk is essential for creating realistic and varied terrain in World of Warcraft, serving these key functions:

1. **Texture Blending**: Enables smooth transitions between different terrain types by controlling the opacity of each texture layer, creating natural-looking landscapes where grass gradually transitions to dirt, snow, rock, etc.

2. **Detail Control**: Allows artists to create specific terrain features like paths, riverbeds, or patches of different ground types by manipulating alpha values.

3. **Memory Efficiency**: By consolidating all alpha maps into a single data block, the v23 format potentially reduces memory overhead compared to the v18 format's separate chunks.

4. **Visual Coherence**: Enables creation of visually coherent landscapes by carefully blending terrain textures to match the environment's theme and climate.

5. **Optimization**: The consolidated structure may provide performance benefits for terrain rendering by keeping related data together in memory.

The AMAP subchunk in the v23 format represents a different approach to alpha map storage compared to the v18 format. Rather than having separate MCAL chunks for each alpha map, v23 consolidates all alpha maps into a single AMAP subchunk and uses offsets in the ALYR subchunks to locate specific alpha maps. This approach aligns with the general theme of v23's experiments with data centralization and more efficient memory usage during the Cataclysm beta development period. 