# C006: MMID

## Type
ADT Chunk

## Source
ADT_v18.md

## Description
Offsets for model filenames in the MMDX chunk.

## Structure
```csharp
struct MMID
{
    uint32_t offsets[0]; // offsets in the MMDX chunk. -1 means invalid
}
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| offsets | uint32[] | Variable-length array of offsets into the MMDX chunk |

## Dependencies
- MHDR (C002) - Contains the offset to this chunk
- MMDX (C005) - Contains the model filenames referenced by this chunk

## Implementation Notes
- Split files: appears in obj file
- The offsets are relative to the start of the MMDX data section
- An offset value of 0xFFFFFFFF (-1) means invalid/not used
- Used to quickly look up model filenames in the MMDX chunk
- Referenced by MDDF chunk to identify which model to place

## Implementation Example
```csharp
public class MMID
{
    public List<uint> Offsets { get; set; } = new List<uint>();
}
```

## Parsing Example
```csharp
public MMID ParseMMID(byte[] data)
{
    var mmid = new MMID();
    using (var ms = new MemoryStream(data))
    using (var reader = new BinaryReader(ms))
    {
        while (ms.Position < ms.Length)
        {
            mmid.Offsets.Add(reader.ReadUInt32());
        }
    }
    return mmid;
}
```

## Usage Context
The MMID chunk provides a way to efficiently look up model filenames in the MMDX chunk. Each offset in MMID points to the start of a filename in the MMDX chunk. The MDDF chunk uses indices into this MMID array to reference the models that need to be placed in the world. 