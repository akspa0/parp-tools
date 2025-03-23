# WA03: MAOH

## Type
Alpha WDT Chunk

## Source
Alpha.md

## Description
The MAOH (Map Object Header) chunk contains header information for map objects in the Alpha WDT format. This chunk provides metadata about the map objects that are indexed by MAOT and stored in MAOI. It likely contains global properties and settings that apply to all map objects or the map as a whole.

## Structure
```csharp
struct MAOH
{
    /*0x00*/ uint32_t flags;          // Global map flags
    /*0x04*/ uint32_t unk1;           // Unknown value 1
    /*0x08*/ uint32_t unk2;           // Unknown value 2
    /*0x0C*/ uint32_t version;        // Format version
    /*0x10*/ uint32_t num_objects;    // Number of map objects
    /*0x14*/ float global_min_height; // Minimum height value across all objects
    /*0x18*/ float global_max_height; // Maximum height value across all objects
    /*0x1C*/ uint32_t unk3;           // Unknown value 3
    /*0x20*/ uint32_t unk4;           // Unknown value 4
    /*0x24*/ uint32_t unk5;           // Unknown value 5
};
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| flags | uint32_t | Global flags for the map |
| unk1 | uint32_t | Unknown value 1 |
| unk2 | uint32_t | Unknown value 2 |
| version | uint32_t | Version of the map format |
| num_objects | uint32_t | Number of map objects (should match MAOT entry count) |
| global_min_height | float | Global minimum height value across all map objects |
| global_max_height | float | Global maximum height value across all map objects |
| unk3 | uint32_t | Unknown value 3 |
| unk4 | uint32_t | Unknown value 4 |
| unk5 | uint32_t | Unknown value 5 |

## Flag Values
The flags field may contain bits indicating various global map properties:

| Value | Name | Description |
|-------|------|-------------|
| 0x01 | MAP_UNKNOWN1 | Unknown purpose |
| 0x02 | MAP_HAS_TEXTURES | Map contains texture data |
| 0x04 | MAP_HAS_OBJECTS | Map contains object placements |
| 0x08 | MAP_UNKNOWN2 | Unknown purpose |
| 0x10 | MAP_OUTDOOR | Map is an outdoor zone |
| 0x20 | MAP_UNKNOWN3 | Unknown purpose |
| 0x40 | MAP_UNKNOWN4 | Unknown purpose |
| 0x80 | MAP_UNKNOWN5 | Unknown purpose |

These flag values are speculative and would require further research to confirm.

## Dependencies
- MAOT (WA01) - The number of objects should match the entry count in MAOT
- MAOI (WA02) - Contains the actual map object data that this header describes

## Implementation Notes
- The MAOH chunk provides global metadata for all map objects in the Alpha WDT file
- The version field likely indicates the version of the Alpha format
- The global height range (global_min_height to global_max_height) defines the overall vertical bounds of the terrain
- The unknown fields may contain additional metadata about the map that was relevant to the Alpha client
- This chunk serves a purpose similar to what MPHD does in the modern WDT format
- The flags likely control various global rendering and behavior aspects of the map
- The num_objects field should match the number of entries in the MAOT chunk

## Implementation Example
```csharp
public class MAOH : IChunk
{
    public uint Flags { get; private set; }
    public uint Unknown1 { get; private set; }
    public uint Unknown2 { get; private set; }
    public uint Version { get; private set; }
    public uint NumObjects { get; private set; }
    public float GlobalMinHeight { get; private set; }
    public float GlobalMaxHeight { get; private set; }
    public uint Unknown3 { get; private set; }
    public uint Unknown4 { get; private set; }
    public uint Unknown5 { get; private set; }
    
    // Helper properties for flag checking
    public bool HasUnknownFlag1 => (Flags & 0x01) != 0;
    public bool HasTextures => (Flags & 0x02) != 0;
    public bool HasObjects => (Flags & 0x04) != 0;
    public bool HasUnknownFlag2 => (Flags & 0x08) != 0;
    public bool IsOutdoor => (Flags & 0x10) != 0;
    public bool HasUnknownFlag3 => (Flags & 0x20) != 0;
    public bool HasUnknownFlag4 => (Flags & 0x40) != 0;
    public bool HasUnknownFlag5 => (Flags & 0x80) != 0;
    
    // Helper property for global height range
    public float GlobalHeightRange => GlobalMaxHeight - GlobalMinHeight;
    
    public void Parse(BinaryReader reader, long size)
    {
        Flags = reader.ReadUInt32();
        Unknown1 = reader.ReadUInt32();
        Unknown2 = reader.ReadUInt32();
        Version = reader.ReadUInt32();
        NumObjects = reader.ReadUInt32();
        GlobalMinHeight = reader.ReadSingle();
        GlobalMaxHeight = reader.ReadSingle();
        Unknown3 = reader.ReadUInt32();
        Unknown4 = reader.ReadUInt32();
        Unknown5 = reader.ReadUInt32();
    }
    
    public void Write(BinaryWriter writer)
    {
        writer.Write(Flags);
        writer.Write(Unknown1);
        writer.Write(Unknown2);
        writer.Write(Version);
        writer.Write(NumObjects);
        writer.Write(GlobalMinHeight);
        writer.Write(GlobalMaxHeight);
        writer.Write(Unknown3);
        writer.Write(Unknown4);
        writer.Write(Unknown5);
    }
    
    // Validation method to check if MAOH data matches MAOT
    public bool ValidateAgainstMAOT(MAOT maot)
    {
        return NumObjects == maot.NumEntries;
    }
    
    // Method to calculate height scale for normalized height values
    public float CalculateHeightScale()
    {
        return GlobalHeightRange > 0 ? 1.0f / GlobalHeightRange : 0;
    }
    
    // Method to convert a normalized height [0-1] to actual world height
    public float ConvertNormalizedHeight(float normalizedHeight)
    {
        return GlobalMinHeight + (normalizedHeight * GlobalHeightRange);
    }
}
```

## Version Information
- Present only in the Alpha version of the WDT format
- The version field may indicate the specific Alpha client build
- This chunk was replaced in later versions by the MPHD chunk in modern WDT

## Architectural Significance
The MAOH chunk provides context for understanding the Alpha WDT architecture:

1. **Global Metadata**: Provides map-wide settings and properties
2. **Height Range**: Defines the vertical bounds of the entire terrain
3. **Version Control**: Specifies the format version, aiding in backward compatibility
4. **Feature Flags**: Indicates what features or data types are present in the map

This contrasts with the modern approach where:
- MPHD provides global map flags
- Individual ADT files contain their own specific metadata
- The map grid is more rigidly defined 