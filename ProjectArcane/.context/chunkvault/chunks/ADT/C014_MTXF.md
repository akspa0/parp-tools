# C014: MTXF

## Type
ADT Chunk

## Source
ADT_v18.md

## Description
Texture Flags - contains flags for terrain textures referenced in the MTEX chunk. This chunk was introduced in Wrath of the Lich King and provides additional rendering hints for terrain textures.

## Structure
```csharp
struct MTXF
{
    /*0x00*/ uint32_t flags[];    // Array of flags for each texture in MTEX
}
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| flags | uint32[] | Array of flags for textures, one entry per texture in MTEX |

## Flag Values
| Value | Name | Description |
|-------|------|-------------|
| 0x1 | TextureFlag_DisableTerrainShadows | Disable terrain shadows on this texture |
| 0x2 | TextureFlag_DisableDoodadShadows | Disable doodad shadows on this texture |
| 0x4 | TextureFlag_DisableShading | Disable shading on this texture (no lighting effects) |
| 0x8 | TextureFlag_DisableLiquidShadows | Disable liquid shadows on this texture |
| 0x10 | TextureFlag_HighResolution | Use high-resolution version of this texture if available |
| 0x20 | TextureFlag_NoCompression | Disable texture compression for this texture |
| 0x40 | TextureFlag_NoVertexAlpha | Disable vertex alpha for this texture |
| 0x80 | TextureFlag_Specular | Use specular lighting for this texture |

## Dependencies
- MTEX (C004) - Contains the texture filenames that correspond to these flags

## Implementation Notes
- The MTXF chunk contains one flag entry for each texture referenced in the MTEX chunk
- The flags determine special rendering behavior for each texture
- If the MTXF chunk is missing, the client assumes default flags (0) for all textures
- The MHDR chunk contains a pointer to this chunk (mtxf field)
- Some flags may override settings in the texture's BLP file
- Flags can be combined using bitwise OR operations
- The size of this chunk should be 4 bytes multiplied by the number of textures in MTEX

## Implementation Example
```csharp
public class MTXF : IChunk
{
    [Flags]
    public enum TextureFlags : uint
    {
        None = 0,
        DisableTerrainShadows = 0x1,
        DisableDoodadShadows = 0x2,
        DisableShading = 0x4,
        DisableLiquidShadows = 0x8,
        HighResolution = 0x10,
        NoCompression = 0x20,
        NoVertexAlpha = 0x40,
        Specular = 0x80
    }
    
    public List<TextureFlags> TextureFlagsList { get; private set; }
    
    public MTXF(BinaryReader reader, uint size)
    {
        int count = (int)(size / 4); // Each flag is 4 bytes
        TextureFlagsList = new List<TextureFlags>(count);
        
        for (int i = 0; i < count; i++)
        {
            TextureFlagsList.Add((TextureFlags)reader.ReadUInt32());
        }
    }
    
    // Helper method to check if a specific texture has a specific flag
    public bool HasFlag(int textureIndex, TextureFlags flag)
    {
        if (textureIndex < 0 || textureIndex >= TextureFlagsList.Count)
            return false;
            
        return (TextureFlagsList[textureIndex] & flag) != 0;
    }
    
    // Helper method to get all flags for a specific texture
    public TextureFlags GetFlags(int textureIndex)
    {
        if (textureIndex < 0 || textureIndex >= TextureFlagsList.Count)
            return TextureFlags.None;
            
        return TextureFlagsList[textureIndex];
    }
}
```

## Usage Context
The MTXF chunk is used to control texture rendering behavior in the game world. It affects how textures are displayed, particularly regarding shadows, lighting, and quality settings. This allows for:

1. Performance optimizations by disabling specific rendering features for certain textures
2. Visual enhancements by enabling high-resolution textures or specular lighting where appropriate
3. Artistic control over lighting and shadow behavior on specific terrain types
4. Custom rendering behavior for specialized terrain surfaces (like snow, water edges, or magical ground)

These flags are particularly important for:
- Terrain that should appear to glow or emit light (by disabling shading)
- Areas where shadow maps might cause artifacts (by disabling specific shadow types)
- Textures that benefit from special rendering effects like specular highlights
- Performance-critical areas where texture quality can be selectively adjusted

The MTXF chunk was added in Wrath of the Lich King to provide finer control over terrain rendering without requiring texture artists to create multiple versions of each texture. 