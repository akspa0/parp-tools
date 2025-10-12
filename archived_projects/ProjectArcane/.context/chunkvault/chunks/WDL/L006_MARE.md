# L006: MARE

## Type
WDL Chunk

## Source
WDL_v18.md

## Description
The MARE (Map ARea) chunk represents a single map area (or quadrant) in the WDL format. Each MARE chunk corresponds to a specific cell in the 64×64 grid defined by the MAOF chunk. MARE chunks contain information about the low-resolution terrain for distant viewing, including heightmap offsets and various rendering attributes.

## Structure
```csharp
struct MARE
{
    /*0x00*/ uint32_t flags;                // Area flags
    /*0x04*/ uint32_t heightMapOffset;      // Offset to heightmap data (MAHE chunk)
    /*0x08*/ float heightScale;             // Scale factor for heightmap values
    /*0x0C*/ float heightMean;              // Mean height value for the area
    /*0x10*/ uint32_t areaID;               // ID of the area
    /*0x14*/ uint32_t numHeightTextures;    // Number of height textures
    /*0x18*/ uint32_t numUnk;               // Number of unknown values
    /*0x1C*/ uint32_t unk1;                 // Unknown value 1
    /*0x20*/ uint32_t unk2;                 // Unknown value 2
    /*0x24*/ uint32_t unk3;                 // Unknown value 3
};
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| flags | uint32_t | Flags that control rendering and behavior of the area |
| heightMapOffset | uint32_t | Offset to the heightmap data (MAHE chunk) for this area |
| heightScale | float | Scale factor to apply to heightmap values |
| heightMean | float | Mean height value for this area |
| areaID | uint32_t | ID of the area, may correspond to zone or subzone |
| numHeightTextures | uint32_t | Number of height textures for this area |
| numUnk | uint32_t | Number of unknown values |
| unk1 | uint32_t | Unknown value 1 |
| unk2 | uint32_t | Unknown value 2 |
| unk3 | uint32_t | Unknown value 3 |

## Flag Values
The flags field may include the following bits:

| Value | Name | Description |
|-------|------|-------------|
| 0x01 | MARE_HAS_HEIGHTMAP | Area has heightmap data |
| 0x02 | MARE_HAS_TEXTURES | Area has texture information |
| 0x04 | MARE_HAS_OBJECTS | Area has object information |
| 0x08 | MARE_UNDERWATER | Area is underwater |
| 0x10 | MARE_HIGHRES | Area has high-resolution data |
| 0x20 | MARE_NO_TRANSITION | Disable transition to ADT data |
| 0x40 | MARE_UNK6 | Unknown purpose |
| 0x80 | MARE_UNK7 | Unknown purpose |

These flag values are speculative and would require further research to confirm.

## Dependencies
- MAOF (L005) - Contains offsets to MARE chunks
- MAHE (L007) - Contains heightmap data referenced by heightMapOffset

## Implementation Notes
- Each MARE chunk represents one cell in the 64×64 world grid
- The heightMapOffset points to a MAHE chunk containing the actual height data
- The heightScale and heightMean are used to interpret the raw heightmap values
- A MARE chunk corresponds to the same map area as an ADT file in the high-resolution map
- The low-resolution heightmap is used for distant terrain rendering
- Areas without terrain may not have a MARE chunk (the MAOF offset will be 0)
- The actual size of heightmap data is typically 17×17 or 33×33 points

## Implementation Example
```csharp
public class MARE : IChunk
{
    public uint Flags { get; set; }
    public uint HeightMapOffset { get; set; }
    public float HeightScale { get; set; }
    public float HeightMean { get; set; }
    public uint AreaID { get; set; }
    public uint NumHeightTextures { get; set; }
    public uint NumUnknown { get; set; }
    public uint Unknown1 { get; set; }
    public uint Unknown2 { get; set; }
    public uint Unknown3 { get; set; }
    
    // Helper properties for flag checking
    public bool HasHeightmap => (Flags & 0x01) != 0;
    public bool HasTextures => (Flags & 0x02) != 0;
    public bool HasObjects => (Flags & 0x04) != 0;
    public bool IsUnderwater => (Flags & 0x08) != 0;
    public bool IsHighRes => (Flags & 0x10) != 0;
    public bool NoTransition => (Flags & 0x20) != 0;
    
    public void Parse(BinaryReader reader, long size)
    {
        // The size of a MARE chunk should be at least 40 bytes
        if (size < 40)
            throw new InvalidDataException($"MARE chunk has invalid size: {size} (expected at least 40)");
        
        Flags = reader.ReadUInt32();
        HeightMapOffset = reader.ReadUInt32();
        HeightScale = reader.ReadSingle();
        HeightMean = reader.ReadSingle();
        AreaID = reader.ReadUInt32();
        NumHeightTextures = reader.ReadUInt32();
        NumUnknown = reader.ReadUInt32();
        Unknown1 = reader.ReadUInt32();
        Unknown2 = reader.ReadUInt32();
        Unknown3 = reader.ReadUInt32();
    }
    
    public void Write(BinaryWriter writer)
    {
        writer.Write(Flags);
        writer.Write(HeightMapOffset);
        writer.Write(HeightScale);
        writer.Write(HeightMean);
        writer.Write(AreaID);
        writer.Write(NumHeightTextures);
        writer.Write(NumUnknown);
        writer.Write(Unknown1);
        writer.Write(Unknown2);
        writer.Write(Unknown3);
    }
    
    // Helper method to convert a raw height value to actual height
    public float ConvertHeight(float rawHeight)
    {
        return (rawHeight * HeightScale) + HeightMean;
    }
    
    // Helper method to check if the heightmap is valid
    public bool HasValidHeightmap()
    {
        return HasHeightmap && HeightMapOffset != 0;
    }
}
```

## Height Calculation
The low-resolution heightmap values in the MAHE chunk are converted to actual terrain heights using:

```
actualHeight = (rawHeight * heightScale) + heightMean
```

Where:
- `rawHeight` is the raw value from the MAHE chunk
- `heightScale` is the scale factor from the MARE chunk
- `heightMean` is the mean height from the MARE chunk
- `actualHeight` is the resulting terrain height in game units

## Map Cell Structure
Each WDL map cell (defined by a MARE chunk) typically represents the same geographical area as an ADT file, but with much lower resolution:

- ADT: 9×9 terrain chunks, each with 17×17 vertices = 145×145 vertices total
- WDL: Typically 17×17 or 33×33 vertices for the entire cell

This reduced resolution is sufficient for distant terrain rendering while requiring much less memory and processing power.

## Transition Between WDL and ADT
The WoW client uses the low-resolution WDL data for distant terrain, and then smoothly transitions to the high-resolution ADT data as the player approaches:

1. Distant terrain: Render using WDL height data
2. Mid-range terrain: Blend between WDL and ADT data
3. Near terrain: Render using full ADT data with textures and objects

This technique allows for much greater view distances without requiring all ADT files to be loaded. 