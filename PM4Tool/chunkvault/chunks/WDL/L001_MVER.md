# L001: MVER

## Type
WDL Chunk

## Source
WDL_v18.md

## Description
The MVER (Map VERsion) chunk defines the version of the WDL file format. This is always the first chunk in a WDL file and serves as an identifier for the file format and version. The version number helps the client determine how to parse the remaining chunks.

## Structure
```csharp
struct MVER
{
    /*0x00*/ uint32_t version; // Version number, typically 18
};
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| version | uint32_t | Version number of the WDL file format |

## Version Values
| Value | Description |
|-------|-------------|
| 18 | Original (Vanilla) WDL format |
| 22+ | Later versions with minor adjustments |

## Dependencies
None - MVER is always the first chunk and has no dependencies on other chunks.

## Implementation Notes
- Always the first chunk in a WDL file
- Simple structure with a single uint32_t value
- Used to determine how to parse the rest of the file
- Version 18 is most common in original (Vanilla) files
- Later versions may have slightly different structures for some chunks

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
        // 18 is the most common version for WDL
        // Later versions (22+) are also valid
        return Version >= 18;
    }
    
    // Default constructor for creating new version chunk
    public MVER(uint version = 18)
    {
        Version = version;
    }
    
    // Constructor that takes no parameters creates version 18 by default
    public MVER() : this(18) { }
}
```

## Related Chunks in Other Formats
The MVER chunk appears in multiple World of Warcraft file formats with identical structure:
- ADT: Map tile data (A001_MVER)
- WDT: World definition table (W001_MVER)
- WMO: World map object files
- M2: Model files

All these formats use the MVER chunk to define the file version, with the version number appropriate to each format.

## Validation Requirements
- Must always be present
- Must always be the first chunk in the file
- Must be exactly 4 bytes in size
- Version should be 18 or greater

## File Format Identification
The presence of an MVER chunk with a valid version number (typically 18) at the beginning of a file can be used, along with the file extension, to identify the file as a valid WDL file. 