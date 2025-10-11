# M003: MSPI (MSP Indices)

## Type
PM4 Geometry Chunk

## Source
PM4 Format Documentation

## Description
The MSPI chunk contains indices into the MSPV chunk, creating connections between vertices to form geometric shapes or paths. This index array allows multiple geometric elements to reference the same vertex data, reducing redundancy. The MSPI chunk consists of an array of 32-bit unsigned integers, each representing an index into the MSPV vertex array.

## Structure
The MSPI chunk has the following structure:

```csharp
struct MSPI
{
    /*0x00*/ uint32_t[] msp_indices; // index into #MSPV
}
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| msp_indices | uint32_t[] | Array of indices referencing vertices in the MSPV chunk |

## Dependencies
- **MSPV** - Contains vertices referenced by these indices

## Implementation Notes
- The number of indices is determined by the chunk size divided by 4 (size of uint32_t)
- Each index should be validated to ensure it's within the bounds of the MSPV vertex array
- The indices are typically accessed in ranges defined by the MSLK chunk
- Unlike some other formats, these indices are direct references (not offsets)

## C# Implementation Example

```csharp
public class MspiChunk : IChunk
{
    public const string Signature = "MSPI";
    public List<uint> Indices { get; private set; }

    public MspiChunk()
    {
        Indices = new List<uint>();
    }

    public void Read(BinaryReader reader, uint size)
    {
        // Calculate number of indices
        int indexCount = (int)(size / 4); // Each index is 4 bytes (uint32)
        Indices.Clear();

        for (int i = 0; i < indexCount; i++)
        {
            Indices.Add(reader.ReadUInt32());
        }
    }

    public void Write(BinaryWriter writer)
    {
        foreach (uint index in Indices)
        {
            writer.Write(index);
        }
    }

    public bool ValidateIndices(int vertexCount)
    {
        // Validate that all indices are within bounds of the MSPV array
        foreach (uint index in Indices)
        {
            if (index >= vertexCount)
            {
                return false;
            }
        }
        return true;
    }
}
```

## Related Information
- This chunk is referenced by the MSLK chunk, which specifies ranges of indices to use
- MSLK.MSPI_first_index specifies the starting index in this array
- MSLK.MSPI_index_count specifies how many consecutive indices to use
- This chunk is present in both PM4 and PD4 formats with identical structure
- The indices in this chunk are used to create edges/lines between vertices rather than triangles 