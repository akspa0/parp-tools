# M001: MVER (Version)

## Type
PM4 Version Chunk

## Source
PM4 Format Documentation

## Description
The MVER chunk contains the version number of the PM4 file format. It is the first chunk in a PM4 file and is required for proper parsing. The chunk contains a single uint32_t value that indicates the format version.

## Structure
The MVER chunk has the following structure:

```csharp
struct MVER
{
    /*0x00*/ uint32_t version; // enum { version_48 = 48 };
}
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| version | uint32_t | Version number of the PM4 format (48 in version 6.0.1.18297, 6.0.1.18443) |

## Dependencies
None. MVER is a self-contained chunk with no dependencies on other chunks.

## Implementation Notes
- MVER must be the first chunk in the file
- Only version 48 has been observed in PM4 files
- This chunk is required for proper parsing
- Parsers should check the version number to ensure compatibility

## C# Implementation Example

```csharp
public class MverChunk : IChunk
{
    public const string Signature = "MVER";
    public uint Version { get; private set; }

    public MverChunk()
    {
        Version = 48; // Default to known version
    }

    public void Read(BinaryReader reader)
    {
        Version = reader.ReadUInt32();
    }

    public void Write(BinaryWriter writer)
    {
        writer.Write(Version);
    }

    public bool IsVersionSupported()
    {
        // Currently only version 48 is supported
        return Version == 48;
    }
}
```

## Related Information
- The MVER chunk is also present in other WoW file formats like ADT, WDT, and WMO
- In PM4/PD4 formats, MVER is always followed by MSHD (header chunk)
- The version number (48) is unique to PM4/PD4 formats and differs from other formats
- The companion format PD4 uses the same MVER structure and version number 