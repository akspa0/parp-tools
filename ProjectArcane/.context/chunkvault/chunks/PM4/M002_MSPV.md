# M002: MSPV (MSP Vertices)

## Type
PM4 Geometry Chunk

## Source
PM4 Format Documentation

## Description
The MSPV chunk contains an array of vertices used for the MSP (Mesh Shape Points) system. These vertices are referenced by indices in the MSPI chunk and used for various geometric operations. The chunk consists of an array of C3Vectori structures representing 3D positions in integer format.

## Structure
The MSPV chunk has the following structure:

```csharp
struct MSPV
{
    /*0x00*/ C3Vectori[] msp_vertices;
}

struct C3Vectori
{
    /*0x00*/ int32_t x;
    /*0x04*/ int32_t y;
    /*0x08*/ int32_t z;
}
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| msp_vertices | C3Vectori[] | Array of 3D position vectors in integer format |

## Dependencies
None directly, but this chunk is referenced by:
- MSPI (MSP Indices) - Contains indices into this vertex array
- MSLK (Links) - References ranges of indices in MSPI that point to these vertices

## Implementation Notes
- The number of vertices is determined by the chunk size divided by the size of C3Vectori (12 bytes)
- Integer coordinates are used instead of floating point for precise positioning
- C3Vectori contains integer coordinates in the typical X, Y, Z order
- Unlike MSVT, no special coordinate transformation is needed for these vertices

## C# Implementation Example

```csharp
public class MspvChunk : IChunk
{
    public const string Signature = "MSPV";
    public List<C3Vectori> Vertices { get; private set; }

    public MspvChunk()
    {
        Vertices = new List<C3Vectori>();
    }

    public void Read(BinaryReader reader, uint size)
    {
        // Calculate number of vertices
        int vertexCount = (int)(size / 12); // Each vertex is 12 bytes (3 ints)
        Vertices.Clear();

        for (int i = 0; i < vertexCount; i++)
        {
            C3Vectori vertex = new C3Vectori
            {
                X = reader.ReadInt32(),
                Y = reader.ReadInt32(),
                Z = reader.ReadInt32()
            };
            Vertices.Add(vertex);
        }
    }

    public void Write(BinaryWriter writer)
    {
        foreach (var vertex in Vertices)
        {
            writer.Write(vertex.X);
            writer.Write(vertex.Y);
            writer.Write(vertex.Z);
        }
    }
}

public struct C3Vectori
{
    public int X { get; set; }
    public int Y { get; set; }
    public int Z { get; set; }

    public override string ToString() => $"({X}, {Y}, {Z})";
}
```

## Related Information
- MSPI chunk contains indices that reference vertices in this chunk
- MSLK chunk defines connections between vertices using the MSPI indices
- This chunk is present in both PM4 and PD4 formats with identical structure
- The MSP system likely represents a lower-detail collision or selection mesh 