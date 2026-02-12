# C008: MWID

## Type
ADT Chunk

## Source
ADT_v18.md

## Description
Offsets for WMO filenames in the MWMO chunk.

## Structure
```csharp
struct MWID
{
    uint32_t offsets[0]; // offsets in the MWMO chunk. -1 means invalid
}
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| offsets | uint32[] | Variable-length array of offsets into the MWMO chunk |

## Dependencies
- MHDR (C002) - Contains the offset to this chunk
- MWMO (C007) - Contains the WMO filenames referenced by this chunk

## Implementation Notes
- Split files: appears in obj file
- The offsets are relative to the start of the MWMO data section
- An offset value of 0xFFFFFFFF (-1) means invalid/not used
- Used to quickly look up WMO filenames in the MWMO chunk
- Referenced by MODF chunk to identify which WMO model to place
- Similar structure to MMID, but for WMO models instead of M2 models

## Implementation Example
```csharp
public class MWID
{
    public List<uint> Offsets { get; set; } = new List<uint>();
}
```

## Parsing Example
```csharp
public MWID ParseMWID(byte[] data)
{
    var mwid = new MWID();
    using (var ms = new MemoryStream(data))
    using (var reader = new BinaryReader(ms))
    {
        while (ms.Position < ms.Length)
        {
            mwid.Offsets.Add(reader.ReadUInt32());
        }
    }
    return mwid;
}
```

## Usage Context
The MWID chunk provides a way to efficiently look up WMO filenames in the MWMO chunk. Each offset in MWID points to the start of a filename in the MWMO chunk. The MODF chunk uses indices into this MWID array to reference the WMO models that need to be placed in the world. 