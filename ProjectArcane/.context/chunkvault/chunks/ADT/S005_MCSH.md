# S005: MCSH

## Type
MCNK Subchunk

## Source
ADT_v18.md

## Description
The MCSH (Map Chunk Shadow Map) subchunk contains shadow information for an MCNK chunk. It stores pre-calculated shadow data used to create realistic shadowing effects on terrain, particularly for static shadows cast by terrain features.

## Structure
```csharp
struct MCSH
{
    /*0x00*/ byte[64*64/8] shadow_map;  // Bit-packed shadow map (512 bytes)
};
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| shadow_map | byte[512] | Bit-packed shadow map, 1 bit per cell (8x8 per byte) |

## Shadow Map Format
The shadow map is stored as a bit-packed array where:
- Each bit represents the shadow state of one cell in the shadow grid
- The grid size is 64x64 cells, covering the entire MCNK chunk
- A value of 1 indicates the cell is shadowed
- A value of 0 indicates the cell is not shadowed
- The bits are packed 8 per byte (64*64 = 4096 bits = 512 bytes)
- They are arranged in row-major order (rows are consecutive)

## Bit Packing Logic
Within each byte:
- Bit 0 (least significant bit) is the leftmost cell in the row
- Bit 7 (most significant bit) is the 8th cell from the left
- The next byte continues with the 9th cell, and so on

## Dependencies
- MCNK (C018) - Parent chunk that contains this subchunk
- MCNK.mcsh - Offset to this subchunk
- MCNK.flags - Bit 0x10000 indicates if shadows are enabled

## Presence Determination
This subchunk is only present when:
- MCNK.mcsh offset is non-zero
- MCNK.flags has the MCSH_SHADOW bit (0x10000) set

## Implementation Notes
- The shadow map provides pre-calculated shadow information
- Shadows are static and don't change based on time of day
- Used primarily for terrain self-shadowing (mountains, cliffs)
- The resolution of 64x64 per chunk provides a balance between detail and memory usage
- Shadow maps were more important in earlier versions of WoW before dynamic lighting was improved
- In later expansions, this data is sometimes used as a fallback or supplement to dynamic shadows

## Implementation Example
```csharp
public class MCSH : IChunk
{
    public const int SHADOW_MAP_SIZE = 64;
    public const int BYTES_PER_ROW = SHADOW_MAP_SIZE / 8;
    public const int TOTAL_BYTES = SHADOW_MAP_SIZE * BYTES_PER_ROW;
    
    public byte[] ShadowMapData { get; private set; } = new byte[TOTAL_BYTES];
    private bool[,] _shadowGrid = new bool[SHADOW_MAP_SIZE, SHADOW_MAP_SIZE];
    
    public void Parse(BinaryReader reader)
    {
        // Read the raw shadow map data
        ShadowMapData = reader.ReadBytes(TOTAL_BYTES);
        
        // Decode the bit-packed data into a 2D array for easier access
        for (int y = 0; y < SHADOW_MAP_SIZE; y++)
        {
            for (int x = 0; x < SHADOW_MAP_SIZE; x++)
            {
                int byteIndex = (y * BYTES_PER_ROW) + (x / 8);
                int bitIndex = x % 8;
                byte mask = (byte)(1 << bitIndex);
                
                _shadowGrid[y, x] = (ShadowMapData[byteIndex] & mask) != 0;
            }
        }
    }
    
    public void Write(BinaryWriter writer)
    {
        // Encode the 2D array back into bit-packed format
        for (int y = 0; y < SHADOW_MAP_SIZE; y++)
        {
            for (int x = 0; x < SHADOW_MAP_SIZE; x++)
            {
                int byteIndex = (y * BYTES_PER_ROW) + (x / 8);
                int bitIndex = x % 8;
                byte mask = (byte)(1 << bitIndex);
                
                if (_shadowGrid[y, x])
                {
                    ShadowMapData[byteIndex] |= mask;
                }
                else
                {
                    ShadowMapData[byteIndex] &= (byte)~mask;
                }
            }
        }
        
        // Write the shadow map data
        writer.Write(ShadowMapData);
    }
    
    // Helper method to get shadow state at specific coordinates
    public bool IsShadowed(int x, int y)
    {
        if (x < 0 || x >= SHADOW_MAP_SIZE || y < 0 || y >= SHADOW_MAP_SIZE)
            throw new ArgumentOutOfRangeException();
            
        return _shadowGrid[y, x];
    }
    
    // Helper method to set shadow state at specific coordinates
    public void SetShadow(int x, int y, bool shadowed)
    {
        if (x < 0 || x >= SHADOW_MAP_SIZE || y < 0 || y >= SHADOW_MAP_SIZE)
            throw new ArgumentOutOfRangeException();
            
        _shadowGrid[y, x] = shadowed;
        
        // Update the corresponding bit in the raw data
        int byteIndex = (y * BYTES_PER_ROW) + (x / 8);
        int bitIndex = x % 8;
        byte mask = (byte)(1 << bitIndex);
        
        if (shadowed)
        {
            ShadowMapData[byteIndex] |= mask;
        }
        else
        {
            ShadowMapData[byteIndex] &= (byte)~mask;
        }
    }
}
```

## Shadow Map Interpolation
When rendering terrain, the shadow map values are interpolated:
- Each point in the shadow grid corresponds to a specific spot on the terrain
- During rendering, the shadow value at a specific point is determined by sampling the nearest points in the shadow grid
- Linear or bilinear interpolation may be used to create smooth shadow transitions
- The shadow map is aligned with the terrain mesh but doesn't necessarily have the same resolution

## Version Information
- MCSH has been present since early versions of World of Warcraft
- Its importance has diminished in later versions as dynamic shadows improved
- Still used in some cases where pre-calculated shadows are more efficient
- The presence is determined by MCNK flags bit 0x10000

## Visual Effects
The MCSH subchunk contributes to several visual effects:
- **Self-Shadowing**: Terrain features casting shadows on other parts of the terrain
- **Ambient Occlusion**: Darker areas in crevices and corners
- **Static Shadow Casters**: Shadows from permanent terrain features like mountains and cliffs
- **Shadow Baking**: Pre-calculated shadows that don't require runtime computation
- **Detail Enhancement**: Adding visual depth to terrain without additional geometry

## Usage Context
The MCSH subchunk enhances the visual quality of terrain by providing shadow information that would otherwise be computationally expensive to calculate in real-time. These static shadow maps work in conjunction with dynamic lighting to create realistic shadowing effects on terrain surfaces. The shadow map is particularly important for older hardware or in areas with complex terrain where dynamic shadow calculations would be too performance-intensive.

The shadow data is typically generated during the terrain creation process by simulating sunlight at various angles and recording which areas become shadowed. This approach allows for efficient rendering while still maintaining visually convincing shadow effects across the landscape. 