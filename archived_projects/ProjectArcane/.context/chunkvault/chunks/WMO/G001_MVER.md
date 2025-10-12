# MVER - WMO Group Version

## Type
WMO Group Chunk

## Source
WMO.md

## Description
The MVER chunk defines the version of the WMO group file format. This is typically the first chunk in a WMO group file and indicates the structure and features available in the file. It follows the same pattern as the MVER chunk in root WMO files and other World of Warcraft file formats, ensuring consistent version handling across the game's assets.

## Structure

```csharp
public struct MVER
{
    public uint version; // Version number (typically 17)
}
```

## Properties

| Offset | Name | Type | Description |
|--------|------|------|-------------|
| 0x00 | version | uint | Version number of the WMO group file format. The standard value is 17 (0x11) for modern WMO group files. |

## Version Values

| Value | Description |
|-------|-------------|
| 17 (0x11) | Standard WMO group file version used in modern World of Warcraft |

## Dependencies
None. The MVER chunk is typically the first chunk in the file and does not depend on other chunks.

## Implementation Notes
- This chunk is always 4 bytes in size (not including the 8-byte chunk header).
- The version number is crucial for determining how to parse the rest of the file.
- While the version is typically 17 (0x11), it's good practice to check this value to ensure compatibility.
- The MVER chunk in group files is identical in structure to the MVER chunk in root WMO files.
- Group files with different versions may have different chunks or chunk structures.

## Implementation Example

```csharp
public class MVERChunk : IWmoGroupChunk
{
    public string ChunkId => "MVER";
    public uint Version { get; set; } = 17; // Default to version 17
    
    public void Read(BinaryReader reader, uint size)
    {
        if (size != 4)
        {
            throw new InvalidDataException($"MVER chunk size is expected to be 4 bytes, but got {size} bytes.");
        }
        
        Version = reader.ReadUInt32();
    }
    
    public void Write(BinaryWriter writer)
    {
        // Write the chunk header
        writer.Write(ChunkUtils.GetChunkIdBytes(ChunkId));
        writer.Write((uint)4); // Size is always 4 bytes
        
        // Write the version
        writer.Write(Version);
    }
}
```

## Validation Requirements
- The chunk size must be exactly 4 bytes.
- The version value should typically be 17 (0x11) for modern WMO group files.
- This should be the first chunk in the WMO group file.

## Usage Context
- **Version Checking:** Client applications use this to determine if they can properly parse the file.
- **Backward Compatibility:** Allows the game engine to handle different versions of WMO group files.
- **File Validation:** Helps verify that a file is indeed a valid WMO group file.
- **Parser Selection:** Used to select the appropriate parsing algorithm for the remaining chunks. 