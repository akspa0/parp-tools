# S006: MCAL

## Type
MCNK Subchunk

## Source
ADT_v18.md

## Description
The MCAL (Map Chunk Alpha Layer) subchunk contains alpha maps for blending between texture layers in an MCNK chunk. These alpha maps determine how the different texture layers blend together to create the final terrain appearance.

## Structure
```csharp
struct MCAL
{
    /*0x00*/ uint8_t alpha_map[?];  // Variable size, depends on compression and number of layers
};
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| alpha_map | uint8_t[] | Raw alpha map data for blending texture layers |

## Alpha Map Format
The alpha map data varies based on compression and the number of layers:

1. **Uncompressed Format** (when FLAG_COMPRESS_ALPHA_MAP is not set):
   - Each alpha value is stored as a single byte (0-255)
   - 64×64 = 4096 values per layer (except for the first layer)
   - Stored sequentially for each texture layer after the first
   - The first layer (index 0) doesn't need an alpha map as it's the base

2. **Compressed Format** (when FLAG_COMPRESS_ALPHA_MAP is set):
   - Values use 1 bit per pixel
   - 64×64 = 4096 bits = 512 bytes per layer
   - The bits are arranged in 2×2 blocks to allow for compression
   - A value of 0 means fully transparent, 1 means fully opaque

## Dependencies
- MCNK (C018) - Parent chunk that contains this subchunk
- MCLY (S003) - Contains offsetInMCAL values that point to alpha map data

## Alpha Map Layout
The alpha map uses a 64×64 grid that covers the entire MCNK:
- Each alpha value corresponds to a specific position in the MCNK
- The values determine how much of each texture layer is visible at that position
- The alpha values blend between texture layers for a smooth transition

## Implementation Notes
- Check the flags in the corresponding MCLY entry to determine compression
- The offsetInMCAL in each MCLY entry points to the start of its alpha map in MCAL
- The first texture layer (index 0) doesn't have an alpha map (it's the base layer)
- The total size of the MCAL chunk depends on:
  - Number of texture layers (from MCNK.layercount)
  - Whether compression is used for each layer

## Implementation Example
```csharp
public class MCAL : IChunk
{
    public const int ALPHA_MAP_SIZE = 64;
    public const int ALPHA_MAP_LENGTH = ALPHA_MAP_SIZE * ALPHA_MAP_SIZE;
    
    public byte[] RawData { get; set; }
    private Dictionary<int, byte[]> _decompressedAlphaMaps = new Dictionary<int, byte[]>();
    
    public void Parse(BinaryReader reader, int dataSize)
    {
        // Read the entire raw data
        RawData = reader.ReadBytes(dataSize);
    }
    
    public byte[] GetAlphaMap(MCLYEntry layerInfo)
    {
        // Cache the decompressed maps to avoid reprocessing
        if (_decompressedAlphaMaps.ContainsKey((int)layerInfo.OffsetInMCAL))
            return _decompressedAlphaMaps[(int)layerInfo.OffsetInMCAL];
            
        byte[] alphaMap = new byte[ALPHA_MAP_LENGTH];
        
        if (layerInfo.CompressAlphaMap)
        {
            // Decompress the alpha map (1 bit per pixel)
            for (int i = 0; i < ALPHA_MAP_LENGTH / 8; i++)
            {
                byte compressedByte = RawData[layerInfo.OffsetInMCAL + i];
                
                for (int bit = 0; bit < 8; bit++)
                {
                    bool isSet = (compressedByte & (1 << bit)) != 0;
                    alphaMap[i * 8 + bit] = isSet ? (byte)255 : (byte)0;
                }
            }
        }
        else
        {
            // Just copy the uncompressed alpha map
            Array.Copy(RawData, layerInfo.OffsetInMCAL, alphaMap, 0, ALPHA_MAP_LENGTH);
        }
        
        _decompressedAlphaMaps[(int)layerInfo.OffsetInMCAL] = alphaMap;
        return alphaMap;
    }
    
    public byte GetAlphaAt(MCLYEntry layerInfo, int x, int y)
    {
        if (x < 0 || x >= ALPHA_MAP_SIZE || y < 0 || y >= ALPHA_MAP_SIZE)
            throw new ArgumentOutOfRangeException();
            
        byte[] alphaMap = GetAlphaMap(layerInfo);
        return alphaMap[y * ALPHA_MAP_SIZE + x];
    }
}
```

## Alpha Map Interpolation
When rendering terrain, the alpha values are typically interpolated:
- The 64×64 alpha map is stretched across the MCNK
- Bilinear filtering is used to sample between alpha values
- This creates smooth transitions between textures

## Special Handling
- For older ADT files (pre-Cataclysm), the alpha maps are all stored in the main ADT file
- For newer ADT files (Cataclysm+), alpha maps may be stored in separate files (_tex0.adt)
- Some texture layers use vertex colors (MCCV) instead of alpha maps for blending
- The FLAG_USE_ALPHA_MAP in MCLY must be set for the alpha map to be used

## Usage Context
The MCAL subchunk enables the blending of multiple textures across terrain, creating realistic transitions between different ground types. For example, a grassy area can gradually blend into a rocky area using alpha maps to control the visibility of each texture. This blending system is fundamental to the terrain rendering in World of Warcraft, allowing for diverse and detailed landscapes without visible tiling or harsh texture boundaries. 