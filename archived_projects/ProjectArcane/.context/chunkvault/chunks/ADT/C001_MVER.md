# C001: MVER

## Type
ADT Chunk

## Source
ADT_v18.md

## Description
Version information chunk for ADT files. This is always the first chunk in an ADT file and identifies the version of the ADT format. All modern ADT files use version 18, regardless of game expansion.

## Original Structure (C++)
```cpp
struct MVER 
{
    /*0x00*/ uint32_t version;  // File version, always 18 for current ADT files
};
```

## Properties
| Offset | Name | Type | Description |
|--------|------|------|-------------|
| 0x00 | version | uint32_t | The version of the file format, always 18 for modern ADT files. |

## Dependencies
None - MVER is always the first chunk in the file.

## Implementation Notes
- This is the first chunk in each ADT file
- Always check this chunk first to validate the file format version
- Split files (root, tex, obj) all include this chunk
- Despite various expansions adding new chunks and features, the version number has remained at 18

## C# Implementation
```csharp
public class MVER : IChunk
{
    public const uint FILE_VERSION = 18;
    
    public uint Version { get; private set; }
    
    public MVER()
    {
        Version = FILE_VERSION;
    }
    
    public void Parse(BinaryReader reader, long size)
    {
        if (size != 4)
            throw new InvalidDataException($"MVER chunk has invalid size: {size} bytes (expected 4)");
            
        Version = reader.ReadUInt32();
        
        if (Version != FILE_VERSION)
            throw new InvalidDataException($"MVER has invalid version: {Version} (expected {FILE_VERSION})");
    }
    
    public void Write(BinaryWriter writer)
    {
        writer.Write(Version);
    }
    
    public bool IsValid()
    {
        return Version == FILE_VERSION;
    }
}
```

## Usage Context
The MVER chunk is used to identify the version of the ADT file format. It must be parsed first to ensure the file structure is understood correctly. Despite later expansions adding new features and chunks, the ADT format version has remained at 18 since the original release. 