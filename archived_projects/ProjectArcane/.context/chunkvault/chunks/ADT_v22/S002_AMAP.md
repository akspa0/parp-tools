# AMAP - Alpha Map Data

## Type
ADT v22 ALYR Subchunk

## Source
Referenced from `ADT_v22.md`

## Description
The AMAP (Alpha Map) subchunk contains alpha map data for texture blending in a terrain chunk. It defines the transparency values used to blend texture layers together, allowing for smooth transitions between different ground textures. The AMAP subchunk is embedded within an ALYR subchunk when the ALYR has the appropriate flag set.

## Structure

```csharp
public struct AMAP
{
    // Raw alpha map data, either uncompressed 8-bit values or 4-bit RLE compressed data
    public byte[] alphaData;
}
```

## Properties

| Name | Type | Description |
|------|------|-------------|
| alphaData | byte[] | Raw alpha map data, either uncompressed or compressed depending on the parent ALYR's flags |

## Dependencies

- ALYR (S001) - Parent subchunk that contains this subchunk
- Only present if the ALYR's flags & 0x100 is set

## Implementation Notes

1. The AMAP subchunk is embedded within an ALYR subchunk when the ALYR has flags & 0x100 set.

2. The alpha map data may be either uncompressed 8-bit or compressed 4-bit RLE, depending on the WDT settings.

3. For uncompressed data, each byte represents a single alpha value (0 = fully transparent, 255 = fully opaque).

4. For compressed data, a simple 4-bit RLE compression scheme is used:
   - Each byte contains two 4-bit values
   - The high 4 bits represent the number of times to repeat the value
   - The low 4 bits represent the alpha value (0-15, which is scaled to 0-255 for rendering)

5. The size of the AMAP chunk is variable and depends on the compression method and the alpha map's complexity.

6. Alpha maps typically cover a 64×64 grid, matching the size of the terrain chunk they apply to.

## Implementation Example

```csharp
public class AmapSubChunk
{
    public byte[] AlphaData { get; set; }
    
    // Whether this alpha map is compressed
    private bool _isCompressed;
    
    public AmapSubChunk()
    {
    }
    
    public AmapSubChunk(bool isCompressed)
    {
        _isCompressed = isCompressed;
        
        // Default to a blank alpha map of the appropriate size
        if (!isCompressed)
        {
            // Uncompressed: 64×64 = 4096 bytes
            AlphaData = new byte[64 * 64];
        }
        else
        {
            // Compressed: empty map can be quite small
            AlphaData = new byte[0];
        }
    }
    
    public void Load(BinaryReader reader, long size)
    {
        AlphaData = reader.ReadBytes((int)size);
    }
    
    public void Save(BinaryWriter writer)
    {
        writer.Write("AMAP".ToCharArray());
        writer.Write(AlphaData.Length);
        writer.Write(AlphaData);
    }
    
    // Get the alpha value at a specific coordinate
    public byte GetAlphaAt(int x, int y, bool isCompressed)
    {
        if (AlphaData == null || AlphaData.Length == 0)
            return 255; // Fully opaque if no data
            
        if (!isCompressed)
        {
            // Uncompressed: direct lookup in the 64×64 grid
            int index = (y * 64) + x;
            if (index >= 0 && index < AlphaData.Length)
                return AlphaData[index];
            else
                return 255; // Fully opaque for out of bounds
        }
        else
        {
            // Decompress the RLE data to get the value at the specific position
            return DecompressRLEAt(x, y);
        }
    }
    
    // Decompress the RLE data to get a specific pixel
    private byte DecompressRLEAt(int x, int y)
    {
        int pixelIndex = (y * 64) + x;
        int currentPixel = 0;
        
        // Iterate through the compressed data to find our pixel
        for (int i = 0; i < AlphaData.Length; i++)
        {
            byte compressedByte = AlphaData[i];
            
            // High 4 bits = repeat count, Low 4 bits = alpha value
            int repeatCount = (compressedByte >> 4) & 0xF;
            int alphaValue = compressedByte & 0xF;
            
            // Scale alpha from 0-15 to 0-255
            byte scaledAlpha = (byte)(alphaValue * 17); // 17 = 255 / 15
            
            // Check if our target pixel is in this run
            if (currentPixel <= pixelIndex && pixelIndex < currentPixel + repeatCount)
                return scaledAlpha;
                
            currentPixel += repeatCount;
        }
        
        return 255; // Default to fully opaque if not found
    }
    
    // Create a compressed version of an uncompressed alpha map
    public static byte[] CompressAlphaMap(byte[] uncompressedData)
    {
        if (uncompressedData == null || uncompressedData.Length == 0)
            return new byte[0];
            
        List<byte> compressed = new List<byte>();
        
        int currentIndex = 0;
        while (currentIndex < uncompressedData.Length)
        {
            // Get the current value and scale it to 0-15
            byte currentValue = uncompressedData[currentIndex];
            byte scaledValue = (byte)(currentValue / 17); // 17 = 255 / 15
            
            // Count how many consecutive pixels have the same scaled value
            int repeatCount = 0;
            while (currentIndex + repeatCount < uncompressedData.Length && 
                   repeatCount < 15 && // Maximum repeat count is 15 (4 bits)
                   (uncompressedData[currentIndex + repeatCount] / 17) == scaledValue)
            {
                repeatCount++;
            }
            
            // Create the compressed byte (repeat count in high 4 bits, value in low 4 bits)
            byte compressedByte = (byte)((repeatCount << 4) | scaledValue);
            compressed.Add(compressedByte);
            
            currentIndex += repeatCount;
        }
        
        return compressed.ToArray();
    }
}
```

## Usage Context

The AMAP subchunk is crucial for creating realistic terrain in World of Warcraft, allowing for smooth transitions between different ground textures. When multiple texture layers are used on a terrain chunk, alpha maps control how these textures blend together.

For example:

1. A base rock texture might cover the entire chunk
2. A second dirt texture would show through only in certain areas, controlled by its alpha map
3. A third grass texture might appear on top of both, again controlled by its alpha map

This layering system creates visually rich terrain with natural-looking transitions between different surface materials. The alpha maps effectively act as masks that determine where each texture is visible.

The v22 format embeds the AMAP directly within the ALYR subchunk (when the appropriate flag is set), which differs from v18 where alpha maps are stored in separate MCAL subchunks. This embedding approach potentially simplifies parsing and reduces the need to jump between different parts of the file when processing a single texture layer.

The choice to support both uncompressed and RLE-compressed alpha maps allows for balancing between file size and processing speed, with simpler alpha maps being efficiently compressed while more complex ones can use the uncompressed format for better quality. 