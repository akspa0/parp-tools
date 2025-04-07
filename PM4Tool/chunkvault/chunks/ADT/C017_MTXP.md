# C017: MTXP

## Type
ADT Chunk

## Source
ADT_v18.md

## Description
Texture parameters for terrain textures, introduced in Mists of Pandaria.

## Applicability
This section only applies to versions ≥ Mists of Pandaria (MoP+).

## Structure
```csharp
struct MTXP
{
    struct TextureParams
    {
        float scale_x;       // X scaling factor
        float scale_y;       // Y scaling factor
        float offset_x;      // X offset for texture coordinates
        float offset_y;      // Y offset for texture coordinates
        uint32_t flags;      // Additional texture parameters
    } params[];              // One entry per texture in MTEX/MDID
}
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| params | TextureParams[] | Array of texture parameters, one per texture |

## TextureParams Properties
| Name | Type | Description |
|------|------|-------------|
| scale_x | float | X scaling factor for the texture |
| scale_y | float | Y scaling factor for the texture |
| offset_x | float | X offset for texture coordinates |
| offset_y | float | Y offset for texture coordinates |
| flags | uint32 | Additional flags for texture parameters |

## Flag Values
| Value | Description |
|-------|-------------|
| 0x1 | Unknown |
| 0x2 | Unknown |
| 0x4 | Unknown |
| 0x8 | Unknown |

## Dependencies
- MHDR (C002) - Contains the offset to this chunk
- MTEX (C004) or MDID (C013) - Must match the number of textures defined

## Implementation Notes
- Introduced in Mists of Pandaria
- Each entry corresponds to a texture in the MTEX or MDID chunk
- Provides scaling and offset parameters for texture coordinates
- Allows for more detailed control over how textures are applied to the terrain
- Used in the terrain shader for advanced texture mapping

## Implementation Example
```csharp
public class TextureParams
{
    public float ScaleX { get; set; }
    public float ScaleY { get; set; }
    public float OffsetX { get; set; }
    public float OffsetY { get; set; }
    public uint Flags { get; set; }
}

public class MTXP
{
    public List<TextureParams> Params { get; set; } = new List<TextureParams>();
}
```

## Parsing Example
```csharp
public MTXP ParseMTXP(byte[] data)
{
    var mtxp = new MTXP();
    using (var ms = new MemoryStream(data))
    using (var reader = new BinaryReader(ms))
    {
        while (ms.Position < ms.Length - 19) // Each entry is 20 bytes
        {
            var param = new TextureParams
            {
                ScaleX = reader.ReadSingle(),
                ScaleY = reader.ReadSingle(),
                OffsetX = reader.ReadSingle(),
                OffsetY = reader.ReadSingle(),
                Flags = reader.ReadUInt32()
            };
            mtxp.Params.Add(param);
        }
    }
    return mtxp;
}
```

## Usage Context
The MTXP chunk provides additional parameters for terrain textures, allowing for more detailed control over how textures are applied to the terrain. These parameters include scaling and offset values, which are used in the terrain shader for advanced texture mapping. This was introduced in Mists of Pandaria to support more complex terrain texturing. 