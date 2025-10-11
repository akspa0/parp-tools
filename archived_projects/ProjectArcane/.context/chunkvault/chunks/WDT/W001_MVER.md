# W001: MVER

## Type
WDT Chunk

## Source
WDT.md

## Description
The MVER (Version) chunk contains version information for the WDT file. It defines the format version of the file, which determines how the client should parse the rest of the data.

## Structure
```csharp
struct MVER
{
    /*0x00*/ uint32_t version;  // File version (18, 22, or 23)
};
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| version | uint32_t | File version number (18, 22, or 23) |

## Version Values
| Value | Description |
|-------|-------------|
| 18 | Classic through Wrath of the Lich King |
| 22 | Cataclysm through Legion |
| 23 | Battle for Azeroth and later |

## Dependencies
- None - MVER has no dependencies on other chunks

## Implementation Notes
- MVER is always the first chunk in a WDT file
- The version number determines which other chunks may be present and their formats
- This chunk is required for proper parsing of the file
- All WDT files, regardless of version, contain this chunk
- The structure is identical to the MVER chunk in other file formats

## Implementation Example
```csharp
public class MVER : IChunk
{
    public uint Version { get; set; }
    
    public MVER()
    {
        Version = 18; // Default to version 18
    }
    
    public void Parse(BinaryReader reader)
    {
        Version = reader.ReadUInt32();
    }
    
    public void Write(BinaryWriter writer)
    {
        writer.Write(Version);
    }
    
    public bool IsValidVersion()
    {
        // Only certain versions are valid
        return Version == 18 || Version == 22 || Version == 23;
    }
}
```

## Version Information
The version number indicates which client expansion the file is designed for:
- Version 18: Classic through Wrath of the Lich King (3.3.5)
- Version 22: Cataclysm through Legion (7.3.5)
- Version 23: Battle for Azeroth (8.0.1) and later

These version changes correspond to significant changes in the file format structure.

## Usage Context
The MVER chunk is essential for proper parsing of the WDT file. The client reads this chunk first to determine how to interpret the remaining data in the file. Different versions may have different chunk structures or data formats, so knowing the version is crucial for correctly reading the file.

The version also indicates which game expansion the map was created for, which helps with compatibility checking when loading maps in different client versions. 