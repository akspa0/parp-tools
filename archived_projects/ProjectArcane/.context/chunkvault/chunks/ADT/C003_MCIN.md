# C003: MCIN

## Type
ADT Chunk

## Source
ADT_v18.md

## Description
Pointers to MCNK chunks and their sizes.

## Applicability
This section only applies to versions ≤ Cataclysm. No longer possible due to split files.

## Structure
```csharp
struct SMChunkInfo 
{ 
    uint32_t offset;   // absolute offset
    uint32_t size;     // the size of the MCNK chunk, this is referring to
    uint32_t flags;    // always 0. only set in the client., FLAG_LOADED = 1
    union 
    { 
        char pad[4]; 
        uint32_t asyncId; // not in the adt file. client use only 
    }; 
} mcin[16*16];
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| offset | uint32 | Absolute offset to the MCNK chunk in the file |
| size | uint32 | The size of the MCNK chunk this entry references |
| flags | uint32 | Always 0 in the file. Used in client with FLAG_LOADED = 1 |
| pad/asyncId | char[4]/uint32 | Padding/Client-only AsyncId field |

## Dependencies
- MHDR (C002) - Contains the offset to this chunk

## Implementation Notes
- Contains 256 entries (16x16 grid)
- Each entry refers to a corresponding MCNK chunk
- Not used in Cataclysm and later versions due to split files
- Offsets are absolute (from beginning of the file)

## Implementation Example
```csharp
public class MCINEntry
{
    public uint Offset { get; set; }
    public uint Size { get; set; }
    public uint Flags { get; set; }
    public uint Padding { get; set; }
}

public class MCIN
{
    public MCINEntry[] Entries { get; set; } = new MCINEntry[256];
}
```

## Parsing Example
```csharp
public MCIN ParseMCIN(byte[] data)
{
    var mcin = new MCIN();
    using (var ms = new MemoryStream(data))
    using (var reader = new BinaryReader(ms))
    {
        for (int i = 0; i < 256; i++)
        {
            mcin.Entries[i] = new MCINEntry
            {
                Offset = reader.ReadUInt32(),
                Size = reader.ReadUInt32(),
                Flags = reader.ReadUInt32(),
                Padding = reader.ReadUInt32()
            };
        }
    }
    return mcin;
}
```

## Usage Context
The MCIN chunk serves as a directory for the 256 MCNK chunks in the ADT file, providing their offsets and sizes for fast lookup without having to scan through the entire file. 