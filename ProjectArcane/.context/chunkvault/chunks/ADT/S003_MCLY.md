# S003: MCLY

## Type
MCNK Subchunk

## Source
ADT_v18.md

## Description
The MCLY (Map Chunk Layer) subchunk contains information about texture layers applied to the terrain. Each MCNK can have up to 4 texture layers that blend together to create the final terrain appearance.

## Structure
```csharp
struct MCLY_entry
{
    /*0x00*/ uint32_t textureId;       // Index into MTEX or file ID depending on version
    /*0x04*/ uint32_t flags;           // Texture flags (see below)
    /*0x08*/ uint32_t offsetInMCAL;    // Offset to corresponding alpha map in MCAL chunk
    /*0x0C*/ uint32_t effectId;        // Used to reference values in MTXP, if present
};

struct MCLY
{
    /*0x00*/ MCLY_entry layers[layerCount];  // layerCount from MCNK header
};
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| layers | MCLY_entry[] | Array of texture layers (size defined by layercount in MCNK header) |

## MCLY_entry Properties
| Name | Type | Description |
|------|------|-------------|
| textureId | uint32 | Index into MTEX array or file ID (in newer versions) |
| flags | uint32 | Texture flags (see below) |
| offsetInMCAL | uint32 | Offset to alpha map data in MCAL subchunk |
| effectId | uint32 | Effect ID used to reference values in MTXP, if present |

## Flag Values
| Value | Name | Description |
|-------|------|-------------|
| 0x1 | FLAG_ANIMATION_ENABLED | Animation enabled |
| 0x4 | FLAG_USE_ALPHA_MAP | Use alpha map (0 for ground, non-zero for detail blending) |
| 0x10 | FLAG_COMPRESS_ALPHA_MAP | Compress alpha values |
| 0x40 | FLAG_USE_CUBEMAP | Use as cube map, effect ID refers to MCSE cube map entries |
| 0x100 | FLAG_USE_MCCV_VERTEX_ALPHA | Alpha values are in the MCCV subchunk for this layer |
| 0x200 | FLAG_USE_TEXTURE_GRADIENT | Unknown, something about 'texture gradients' |

## Dependencies
- MCNK (C018) - Parent chunk that contains this subchunk
- MTEX (C004) - Texture filenames referenced by textureId (older versions)
- MDID (C013) - Texture file IDs referenced by textureId (8.1.0+)
- MCAL (S006) - Alpha map data referenced by offsetInMCAL
- MTXP (C017) - Texture parameters referenced by effectId

## Version Considerations
- In older versions (pre-8.1.0), textureId is an index into the MTEX string array
- In newer versions (8.1.0+), textureId is a file ID (direct reference)
- The presence of MTXP influences how effectId is interpreted

## Implementation Notes
- The layercount in the MCNK header determines how many MCLY entries to read
- The first layer is the base terrain texture, subsequent layers are blended on top
- The flags determine how textures are rendered and blended
- Each layer (except the first) requires an alpha map in MCAL for blending
- Layers are applied in order from bottom to top
- A maximum of 4 layers per MCNK is supported

## Alpha Map Handling
- The offsetInMCAL field points to the start of the alpha map data in the MCAL subchunk
- For the first layer (index 0), no alpha map is needed (it covers the entire MCNK)
- For subsequent layers, the alpha map determines how much of that texture is visible
- If the FLAG_COMPRESS_ALPHA_MAP flag is set, the alpha values use a different encoding

## Implementation Example
```csharp
[Flags]
public enum MCLYFlags : uint
{
    None = 0,
    AnimationEnabled = 0x1,
    UseAlphaMap = 0x4,
    CompressAlphaMap = 0x10,
    UseCubemap = 0x40,
    UseVertexAlpha = 0x100,
    UseTextureGradient = 0x200
}

public class MCLYEntry
{
    public uint TextureId { get; set; }
    public MCLYFlags Flags { get; set; }
    public uint OffsetInMCAL { get; set; }
    public uint EffectId { get; set; }
    
    // Helper properties
    public bool UseAlphaMap => (Flags & MCLYFlags.UseAlphaMap) != 0;
    public bool CompressAlphaMap => (Flags & MCLYFlags.CompressAlphaMap) != 0;
}

public class MCLY : IChunk
{
    public List<MCLYEntry> Layers { get; set; } = new List<MCLYEntry>();
    
    public void Parse(BinaryReader reader, uint layerCount)
    {
        for (int i = 0; i < layerCount; i++)
        {
            var entry = new MCLYEntry
            {
                TextureId = reader.ReadUInt32(),
                Flags = (MCLYFlags)reader.ReadUInt32(),
                OffsetInMCAL = reader.ReadUInt32(),
                EffectId = reader.ReadUInt32()
            };
            
            Layers.Add(entry);
        }
    }
}
```

## Texture Coordinates
The textures referenced by MCLY are mapped onto the terrain using a predefined texture coordinate system:
- Each MCNK is considered a square in texture space
- Textures are tiled across the MCNK based on their properties
- The texture coordinates are generated based on the vertex position within the MCNK

## Usage Context
The MCLY subchunk is vital for the visual appearance of terrain in World of Warcraft. It defines which textures are applied to the terrain and how they blend together. The system allows for detailed terrain with varied appearance by layering up to 4 textures per chunk. For example, a grassy area might have a base grass texture, with dirt paths, rocky outcroppings, and snow patches blended on top using alpha maps. This layering system creates the varied and detailed ground textures seen throughout the game world. 