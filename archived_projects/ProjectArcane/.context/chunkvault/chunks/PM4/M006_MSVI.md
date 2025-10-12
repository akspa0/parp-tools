# M006: MSVI (Vertex Indices)

## Type
PM4 Geometry Chunk

## Source
PM4 Format Documentation

## Description
The MSVI chunk contains indices into the MSVT chunk, creating connections between vertices to form surfaces defined in the MSUR chunk. This index array allows multiple surface elements to reference the same vertex data, reducing redundancy. The MSVI chunk consists of an array of 32-bit unsigned integers, each representing an index into the MSVT vertex array.

## Structure
The MSVI chunk has the following structure:

```csharp
struct MSVI
{
    /*0x00*/ uint32_t[] msv_indices; // index into #MSVT
}
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| msv_indices | uint32_t[] | Array of indices referencing vertices in the MSVT chunk |

## Dependencies
- **MSVT** - Contains vertices referenced by these indices

## Implementation Notes
- The number of indices is determined by the chunk size divided by 4 (size of uint32_t)
- Each index should be validated to ensure it's within the bounds of the MSVT vertex array
- The documentation notes that these indices likely refer to quads (four-sided polygons) or n-gons rather than triangles
- The documentation suggests these may be organized as described by the MSUR chunk
- The indices are accessed in ranges defined by the MSUR chunk, particularly by the MSUR.MSVI_first_index field
- Unlike some other formats, these indices are direct references (not offsets)

## C# Implementation Example

```csharp
public class MsviChunk : IChunk
{
    public const string Signature = "MSVI";
    public List<uint> Indices { get; private set; }

    public MsviChunk()
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
        // Validate that all indices are within bounds of the MSVT array
        foreach (uint index in Indices)
        {
            if (index >= vertexCount)
            {
                return false;
            }
        }
        return true;
    }

    // Get indices for a specific surface from MSUR
    public List<uint> GetIndicesForSurface(uint firstIndex, uint count)
    {
        if (firstIndex >= Indices.Count)
            return new List<uint>();
            
        // Ensure we don't read beyond the array bounds
        count = Math.Min(count, (uint)(Indices.Count - firstIndex));
        
        return Indices.GetRange((int)firstIndex, (int)count);
    }
}
```

## Related Information
- This chunk is referenced by the MSUR chunk, which specifies ranges of indices to use for surfaces
- MSUR.MSVI_first_index specifies the starting index in this array
- MSUR._0x01 (count field) specifies how many consecutive indices to use
- According to documentation, this may define quads or n-gons rather than triangles
- This chunk is present in both PM4 and PD4 formats with identical structure
- Unlike MSPI (which likely defines edges), MSVI appears to define polygon faces 