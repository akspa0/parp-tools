# C016: MTXF

## Type
ADT Chunk

## Source
ADT_v18.md

## Description
Extended texture flags for terrain textures, introduced in Wrath of the Lich King.

## Applicability
This section only applies to versions ≥ Wrath of the Lich King (WotLK+).

## Structure
```csharp
struct MTXF
{
    uint32_t flags[];  // One entry per texture in MTEX/MDID
}
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| flags | uint32[] | Array of texture flags, one per texture |

## Flag Values
| Value | Description |
|-------|-------------|
| 0x1 | Unknown |
| 0x2 | Unknown |
| 0x4 | Unknown |
| 0x8 | Unknown |
| 0x10 | Unknown |
| 0x20 | Unknown |
| 0x40 | Texture is HD (high resolution) |

## Dependencies
- MHDR (C002) - Contains the offset to this chunk
- MTEX (C004) or MDID (C013) - Must match the number of textures defined

## Implementation Notes
- Introduced in Wrath of the Lich King
- Each entry corresponds to a texture in the MTEX or MDID chunk
- The number of entries must match the number of textures
- Provides additional flags/properties for terrain textures beyond what's in MCLY
- Flag 0x40 indicates the texture is a high-resolution texture

## Implementation Example
```csharp
[Flags]
public enum TextureFlags : uint
{
    None = 0x0,
    Unknown1 = 0x1,
    Unknown2 = 0x2,
    Unknown4 = 0x4,
    Unknown8 = 0x8,
    Unknown10 = 0x10,
    Unknown20 = 0x20,
    HighResolution = 0x40
}

public class MTXF
{
    public List<TextureFlags> Flags { get; set; } = new List<TextureFlags>();
}
```

## Parsing Example
```csharp
public MTXF ParseMTXF(byte[] data)
{
    var mtxf = new MTXF();
    using (var ms = new MemoryStream(data))
    using (var reader = new BinaryReader(ms))
    {
        while (ms.Position < ms.Length)
        {
            mtxf.Flags.Add((TextureFlags)reader.ReadUInt32());
        }
    }
    return mtxf;
}
```

## Usage Context
The MTXF chunk provides extended flags for terrain textures, allowing for additional properties beyond what is stored in the MCLY chunk. This was introduced in Wrath of the Lich King to support new texture features. Most notably, it indicates which textures are high-resolution (HD). 