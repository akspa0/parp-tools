# VERS - MDX Version Chunk

## Type
MDX Main Chunk

## Source
MDX_index.md

## Description
The VERS (Version) chunk defines the version of the MDX file format. This is the first chunk in every MDX file after the "MDLX" identifier and is crucial for determining how to parse the rest of the file. Different versions of the MDX format have varying structures and features, particularly between the Warcraft 3 versions and the early World of Warcraft versions.

## Structure

```csharp
public struct VERS
{
    /// <summary>
    /// Format version number
    /// </summary>
    public uint version;
}
```

## Properties

| Offset | Name | Type | Description |
|--------|------|------|-------------|
| 0x00 | version | uint | Format version identifier. Possible values: 800 (WC3), 900/1000 (WC3 Reforged), 1300/1400/1500 (WoW Alpha/Beta) |

## Version Values

| Value | Game | Description |
|-------|------|-------------|
| 800 | Warcraft 3 Classic | Original Warcraft 3 version |
| 900 | Warcraft 3 Reforged | Early Warcraft 3 Reforged version |
| 1000 | Warcraft 3 Reforged | Later Warcraft 3 Reforged version |
| 1300 | WoW Alpha/Beta | Used in early WoW development (≤ 0.9.1.3810) |
| 1400 | WoW Alpha/Beta | Minor updates with path normalization |
| 1500 | WoW Alpha/Beta | Structural changes to MTLS and GEOS chunks |

## Dependencies
None. This is the first chunk in the file and does not rely on any other chunks.

## Implementation Notes
- The VERS chunk is mandatory and must be present in all valid MDX files
- The chunk is always 4 bytes in size (excluding the chunk header)
- The version number determines how to parse subsequent chunks, as structure changes exist between versions
- In WoW Alpha client versions (1300-1500), the format progressively evolved toward what would become the M2 format
- Version 1400 primarily normalized file paths but maintained the same chunk structure as version 1300
- Version 1500 introduced significant structural changes to the MTLS chunk and completely redesigned the GEOS chunk

## Usage Context
The version information is used to determine:
- Which chunks may be present in the file
- The structure of each chunk (as some changed between versions)
- How to interpret animation data
- How to handle texture references
- Whether certain features (like environment mapping in v1500) are supported

## Implementation Example

```csharp
public class VERSChunk : IMdxChunk
{
    public string ChunkId => "VERS";
    public uint Version { get; private set; }

    public void Parse(BinaryReader reader, long size)
    {
        if (size != 4)
        {
            throw new InvalidDataException($"VERS chunk size must be 4 bytes, found {size} bytes");
        }

        Version = reader.ReadUInt32();
        
        // Validate version is one of the known versions
        if (Version != 800 && Version != 900 && Version != 1000 &&
            Version != 1300 && Version != 1400 && Version != 1500)
        {
            throw new InvalidDataException($"Unknown MDX version: {Version}");
        }
    }
    
    public void Write(BinaryWriter writer)
    {
        // Write chunk header (ID and size) handled by caller
        writer.Write(Version);
    }
    
    public bool IsWarcraft3Version()
    {
        return Version == 800 || Version == 900 || Version == 1000;
    }
    
    public bool IsWowVersion()
    {
        return Version == 1300 || Version == 1400 || Version == 1500;
    }
}
``` 