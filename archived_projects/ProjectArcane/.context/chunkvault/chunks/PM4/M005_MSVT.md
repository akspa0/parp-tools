# M005: MSVT (Vertices)

## Type
PM4 Geometry Chunk

## Source
PM4 Format Documentation

## Description
The MSVT chunk contains an array of vertices used for the main geometry system. These vertices are referenced by indices in the MSVI chunk and are used to construct surfaces defined in the MSUR chunk. The MSVT chunk consists of an array of C3Vectori structures representing 3D positions in integer format with a special YXZ ordering.

## Structure
The MSVT chunk has the following structure:

```csharp
struct MSVT
{
    /*0x00*/ C3Vectori[] msvt; // t ≠ tangents. vt = vertices?
}

struct C3Vectori
{
    /*0x00*/ int32_t y; // Note: Ordered YXZ, not XYZ
    /*0x04*/ int32_t x;
    /*0x08*/ int32_t z;
}
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| msvt | C3Vectori[] | Array of 3D position vectors in integer format with YXZ ordering |

## Dependencies
None directly, but this chunk is referenced by:
- MSVI (Vertex Indices) - Contains indices into this vertex array
- MSUR (Surface Definitions) - References ranges of indices in MSVI that point to these vertices

## Implementation Notes
- The number of vertices is determined by the chunk size divided by the size of C3Vectori (12 bytes)
- The vertex coordinates are ordered YXZ, not the standard XYZ order
- The coordinates must be transformed to get the in-game world coordinates using these formulas:
  - `worldPos.y = 17066.666 - position.y;`
  - `worldPos.x = 17066.666 - position.x;`
  - `worldPos.z = position.z / 36.0f;` // Divide by 36 to convert internal inch height to yards
- Integer coordinates are used for precise positioning
- The coordinate system differs from that of MSPV

## C# Implementation Example

```csharp
public class MsvtChunk : IChunk
{
    public const string Signature = "MSVT";
    public List<MsvtVertex> Vertices { get; private set; }

    // Constants for coordinate transformation
    private const float CoordinateOffset = 17066.666f;
    private const float HeightScale = 36.0f;

    public MsvtChunk()
    {
        Vertices = new List<MsvtVertex>();
    }

    public void Read(BinaryReader reader, uint size)
    {
        // Calculate number of vertices
        int vertexCount = (int)(size / 12); // Each vertex is 12 bytes (3 ints)
        Vertices.Clear();

        for (int i = 0; i < vertexCount; i++)
        {
            // Note the YXZ order here
            MsvtVertex vertex = new MsvtVertex
            {
                Y = reader.ReadInt32(),
                X = reader.ReadInt32(),
                Z = reader.ReadInt32()
            };
            Vertices.Add(vertex);
        }
    }

    public void Write(BinaryWriter writer)
    {
        foreach (var vertex in Vertices)
        {
            // Note the YXZ order here
            writer.Write(vertex.Y);
            writer.Write(vertex.X);
            writer.Write(vertex.Z);
        }
    }

    // Converts file coordinates to world coordinates
    public Vector3 GetWorldCoordinates(MsvtVertex vertex)
    {
        return new Vector3(
            CoordinateOffset - vertex.X,
            CoordinateOffset - vertex.Y,
            vertex.Z / HeightScale
        );
    }

    // Converts world coordinates to file coordinates
    public MsvtVertex GetFileCoordinates(Vector3 worldPos)
    {
        return new MsvtVertex
        {
            X = (int)(CoordinateOffset - worldPos.X),
            Y = (int)(CoordinateOffset - worldPos.Y),
            Z = (int)(worldPos.Z * HeightScale)
        };
    }
}

public struct MsvtVertex
{
    public int Y { get; set; } // Note YXZ ordering
    public int X { get; set; }
    public int Z { get; set; }

    public override string ToString() => $"({X}, {Y}, {Z})";
}

public struct Vector3
{
    public float X { get; set; }
    public float Y { get; set; }
    public float Z { get; set; }

    public Vector3(float x, float y, float z)
    {
        X = x;
        Y = y;
        Z = z;
    }

    public override string ToString() => $"({X}, {Y}, {Z})";
}
```

## Related Information
- MSVI chunk contains indices that reference vertices in this chunk
- MSUR chunk defines surfaces using ranges of indices from MSVI
- The unusual YXZ ordering requires special handling during import/export
- The coordinate transformation formulas are essential for correct positioning in the game world
- This chunk is present in both PM4 and PD4 formats with identical structure
- The Z coordinate scale (dividing by 36) converts internal height units (inches) to yards 