# O001: MVER

## Type
WMO Chunk

## Source
WMO.md

## Description
The MVER (Map VERsion) chunk defines the version of the WMO file. This is always the first chunk in a WMO file (both root and group files) and serves as an identifier for the file format and version. The version number helps the client determine how to parse the remaining chunks.

## Structure
```csharp
struct MVER
{
    /*0x00*/ uint32_t version;  // File version, typically 17
};
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| version | uint32_t | Version number of the WMO file format |

## Version Values
| Value | Description |
|-------|-------------|
| 14 | Alpha version (self-contained file with MOMO wrapper) |
| 17 | Classic through Battle for Azeroth |

## Dependencies
None - MVER is always the first chunk and has no dependencies on other chunks.

## Implementation Notes
- Always the first chunk in both WMO root and group files
- Simple structure with a single uint32_t value
- Used to determine how to parse the rest of the file
- Version 17 is most common in retail files
- Even though the format has changed over time (especially in Legion), Blizzard never updated the version number beyond 17

## Implementation Example
```csharp
public class MVER : IChunk
{
    public uint Version { get; private set; }
    
    public void Parse(BinaryReader reader, long size)
    {
        // The size should always be 4 bytes
        if (size != 4)
            throw new InvalidDataException($"MVER chunk has invalid size: {size} (expected 4)");
            
        Version = reader.ReadUInt32();
    }
    
    public void Write(BinaryWriter writer)
    {
        writer.Write(Version);
    }
    
    // Helper method to check for valid version
    public bool IsValidVersion()
    {
        // 17 is the version used in retail WMOs
        return Version == 17;
    }
    
    // Default constructor for creating new version chunk
    public MVER(uint version = 17)
    {
        Version = version;
    }
    
    // Constructor that takes no parameters creates version 17 by default
    public MVER() : this(17) { }
}
```

## Related Chunks in Other Formats
The MVER chunk appears in multiple World of Warcraft file formats with identical structure:
- ADT: Map tile data (A001_MVER)
- WDT: World definition table (W001_MVER)
- WDL: Low-resolution terrain data (L001_MVER)
- WMO: World map object files (both root and group files)

All these formats use the MVER chunk to define the file version, with the version number appropriate to each format.

## File Format Identification
The presence of an MVER chunk with a valid version number (typically 17) at the beginning of a file can be used, along with the file extension, to identify the file as a valid WMO file.

## Usage Context
The MVER chunk is essential for the proper parsing of WMO files. The client reads this chunk first to determine how to interpret the remaining data in the file. This ensures backward compatibility as the format evolves.

In practice, most WMO files in retail builds have used version 17 since Classic, even though significant changes have been made to the format over time. Blizzard chose to handle these changes through fallback mechanisms and optional chunks rather than by incrementing the version number. 