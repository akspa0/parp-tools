# S007: MCLQ

## Type
MCNK Subchunk

## Source
ADT_v18.md

## Description
The MCLQ (Map Chunk Liquid) subchunk contains liquid data for an MCNK chunk. It's the legacy format for liquid information, used in pre-Cataclysm versions of World of Warcraft. In later versions, this was replaced by the MH2O chunk.

## Structure
```csharp
struct SMMapChunkLiquidData
{
    /*0x00*/ float heightLevel1;          // Water level 1
    /*0x04*/ float heightLevel2;          // Water level 2
    /*0x08*/ uint8_t flags;               // Liquid flags
    /*0x09*/ uint8_t data;                // Unknown data value
    /*0x0A*/ float x;                     // X coordinate
    /*0x0E*/ float y;                     // Y coordinate
    /*0x12*/ uint8_t xOffset;             // X offset in the liquid texture
    /*0x13*/ uint8_t yOffset;             // Y offset in the liquid texture
    /*0x14*/ uint8_t width;               // Width of the liquid area
    /*0x15*/ uint8_t height;              // Height of the liquid area
    /*0x16*/ uint16_t liquidEntry;        // Liquid type ID from LiquidType.dbc
    /*0x18*/ uint8_t liquidVertexFormat;  // Determines format of vertex data that follows
    /*0x19*/ uint8_t liquidFlags;         // Additional liquid flags
    /*0x1A*/ uint16_t liquidType;         // Liquid type
    /*0x1C*/ float heightMap[9][9];       // 9x9 height map for liquid surface
    /*0x??*/ uint8_t alphaMap[8][8];      // 8x8 alpha map, only if MCLQ_HAS_ALPHA flag is set
};
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| heightLevel1 | float | First water level height |
| heightLevel2 | float | Second water level height |
| flags | uint8_t | Liquid flags (see Flag Values) |
| data | uint8_t | Unknown data value |
| x | float | X coordinate of liquid area |
| y | float | Y coordinate of liquid area |
| xOffset | uint8_t | X offset in the liquid texture |
| yOffset | uint8_t | Y offset in the liquid texture |
| width | uint8_t | Width of the liquid area |
| height | uint8_t | Height of the liquid area |
| liquidEntry | uint16_t | Liquid type ID from LiquidType.dbc |
| liquidVertexFormat | uint8_t | Format of vertex data |
| liquidFlags | uint8_t | Additional liquid flags |
| liquidType | uint16_t | Liquid type |
| heightMap | float[9][9] | Height map for liquid surface |
| alphaMap | uint8_t[8][8] | Alpha map for transparency (optional) |

## Flag Values
| Value | Name | Description |
|-------|------|-------------|
| 0x01 | MCLQ_HAS_LIQUID | This chunk has liquid |
| 0x02 | MCLQ_HIDDEN | Liquid is hidden |
| 0x04 | MCLQ_HAS_ALPHA | Liquid has alpha map |
| 0x08 | MCLQ_FISHABLE | Liquid is fishable |
| 0x10 | MCLQ_SHARED | Shared with adjacent chunks |

## Liquid Types
| Value | Name | Description |
|-------|------|-------------|
| 0 | Water | Standard water |
| 1 | Ocean | Ocean water |
| 2 | Magma | Lava/magma |
| 3 | Slime | Slime/ooze |
| 4 | WMO | WMO-specific liquid |

## Dependencies
- MCNK (C018) - Parent chunk that contains this subchunk
- MCNK.flags - Bits 2-5 indicate liquid type

## Presence Determination
This subchunk is only present when:
- MCNK.mclq offset is non-zero
- MCNK.flags indicate some type of liquid (bits 2-5)
- In pre-Cataclysm versions of the client

## Implementation Notes
- The legacy MCLQ format was used in vanilla through Wrath of the Lich King
- In Cataclysm and later, this was replaced by the MH2O chunk
- The heightMap defines the surface shape of the liquid
- The alphaMap is only present if the MCLQ_HAS_ALPHA flag is set
- The alpha map controls transparency of the liquid (for shorelines)
- Height values are absolute world heights, not relative to terrain
- Liquid types are connected to visual styles in LiquidType.dbc

## Implementation Example
```csharp
public class MCLQ : IChunk
{
    public const int HEIGHT_MAP_SIZE = 9;
    public const int ALPHA_MAP_SIZE = 8;
    
    // MCLQ flags
    [Flags]
    public enum MCLQFlags : byte
    {
        None = 0,
        HasLiquid = 0x01,
        Hidden = 0x02,
        HasAlpha = 0x04,
        Fishable = 0x08,
        Shared = 0x10
    }
    
    public float HeightLevel1 { get; set; }
    public float HeightLevel2 { get; set; }
    public MCLQFlags Flags { get; set; }
    public byte Data { get; set; }
    public float X { get; set; }
    public float Y { get; set; }
    public byte XOffset { get; set; }
    public byte YOffset { get; set; }
    public byte Width { get; set; }
    public byte Height { get; set; }
    public ushort LiquidEntry { get; set; }
    public byte LiquidVertexFormat { get; set; }
    public byte LiquidFlags { get; set; }
    public ushort LiquidType { get; set; }
    public float[,] HeightMap { get; set; } = new float[HEIGHT_MAP_SIZE, HEIGHT_MAP_SIZE];
    public byte[,] AlphaMap { get; set; } = new byte[ALPHA_MAP_SIZE, ALPHA_MAP_SIZE];
    
    public bool HasAlphaMap => (Flags & MCLQFlags.HasAlpha) != 0;
    
    public void Parse(BinaryReader reader)
    {
        HeightLevel1 = reader.ReadSingle();
        HeightLevel2 = reader.ReadSingle();
        Flags = (MCLQFlags)reader.ReadByte();
        Data = reader.ReadByte();
        X = reader.ReadSingle();
        Y = reader.ReadSingle();
        XOffset = reader.ReadByte();
        YOffset = reader.ReadByte();
        Width = reader.ReadByte();
        Height = reader.ReadByte();
        LiquidEntry = reader.ReadUInt16();
        LiquidVertexFormat = reader.ReadByte();
        LiquidFlags = reader.ReadByte();
        LiquidType = reader.ReadUInt16();
        
        // Read the height map - 9x9 grid
        for (int y = 0; y < HEIGHT_MAP_SIZE; y++)
        {
            for (int x = 0; x < HEIGHT_MAP_SIZE; x++)
            {
                HeightMap[x, y] = reader.ReadSingle();
            }
        }
        
        // Read the alpha map if present - 8x8 grid
        if (HasAlphaMap)
        {
            for (int y = 0; y < ALPHA_MAP_SIZE; y++)
            {
                for (int x = 0; x < ALPHA_MAP_SIZE; x++)
                {
                    AlphaMap[x, y] = reader.ReadByte();
                }
            }
        }
    }
    
    public void Write(BinaryWriter writer)
    {
        writer.Write(HeightLevel1);
        writer.Write(HeightLevel2);
        writer.Write((byte)Flags);
        writer.Write(Data);
        writer.Write(X);
        writer.Write(Y);
        writer.Write(XOffset);
        writer.Write(YOffset);
        writer.Write(Width);
        writer.Write(Height);
        writer.Write(LiquidEntry);
        writer.Write(LiquidVertexFormat);
        writer.Write(LiquidFlags);
        writer.Write(LiquidType);
        
        // Write the height map
        for (int y = 0; y < HEIGHT_MAP_SIZE; y++)
        {
            for (int x = 0; x < HEIGHT_MAP_SIZE; x++)
            {
                writer.Write(HeightMap[x, y]);
            }
        }
        
        // Write the alpha map if present
        if (HasAlphaMap)
        {
            for (int y = 0; y < ALPHA_MAP_SIZE; y++)
            {
                for (int x = 0; x < ALPHA_MAP_SIZE; x++)
                {
                    writer.Write(AlphaMap[x, y]);
                }
            }
        }
    }
}
```

## Liquid Rendering
The liquid data is used to render water, lava, and other liquid surfaces:
- The heightMap defines the actual liquid surface geometry
- The alphaMap controls transparency (for shorelines and shallow areas)
- Liquid type determines the texture and visual effects
- Liquid flags control properties like whether it's fishable

## Usage Context
The MCLQ subchunk was used in pre-Cataclysm versions of World of Warcraft to define and render liquid surfaces like water, lava, and slime. Each MCNK could contain one type of liquid. The liquid surface was rendered with its own height map, which could differ from the terrain height map, allowing for lakes, rivers, and other water features. In Cataclysm and later versions, this was replaced by the MH2O chunk, which offered more flexibility and features. 