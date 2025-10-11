# M008: MSCN (Normal Vectors)

## Type
PM4 Geometry Chunk

## Source
PM4 Format Documentation

## Description
The MSCN chunk contains an array of normal vectors used for surface orientation and lighting calculations. Unlike MSPV and MSVT vertices, these normal vectors are not directly referenced by indices but are likely associated with surfaces or vertices through their position in the array. The vectors represent direction rather than position and are used to determine how light interacts with surfaces.

## Structure
The MSCN chunk has the following structure:

```csharp
struct MSCN
{
    /*0x00*/ C3Vector[] mscn;
}

struct C3Vector
{
    /*0x00*/ float x;
    /*0x04*/ float y;
    /*0x08*/ float z;
}
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| mscn | C3Vector[] | Array of 3D normal vectors in floating-point format |

## Dependencies
None directly, but this chunk is likely referenced by other chunks that define surfaces or lighting.

## Implementation Notes
- The number of normal vectors is determined by the chunk size divided by the size of C3Vector (12 bytes)
- Unlike MSVT, these vectors use the standard XYZ order
- Normal vectors should be normalized (length = 1.0) for proper lighting calculations
- These normals may correspond to vertices in MSPV or MSVT, or to surfaces in MSUR
- Floating-point values are used for higher precision in direction representation

## C# Implementation Example

```csharp
public class MscnChunk : IChunk
{
    public const string Signature = "MSCN";
    public List<C3Vector> Normals { get; private set; }

    public MscnChunk()
    {
        Normals = new List<C3Vector>();
    }

    public void Read(BinaryReader reader, uint size)
    {
        // Calculate number of normal vectors
        int normalCount = (int)(size / 12); // Each normal is 12 bytes (3 floats)
        Normals.Clear();

        for (int i = 0; i < normalCount; i++)
        {
            C3Vector normal = new C3Vector
            {
                X = reader.ReadSingle(),
                Y = reader.ReadSingle(),
                Z = reader.ReadSingle()
            };
            Normals.Add(normal);
        }
    }

    public void Write(BinaryWriter writer)
    {
        foreach (var normal in Normals)
        {
            writer.Write(normal.X);
            writer.Write(normal.Y);
            writer.Write(normal.Z);
        }
    }

    // Normalize a vector (ensure length = 1.0)
    public C3Vector Normalize(C3Vector vector)
    {
        float length = (float)Math.Sqrt(vector.X * vector.X + vector.Y * vector.Y + vector.Z * vector.Z);
        if (length < 0.0001f) return new C3Vector { X = 0, Y = 0, Z = 1 }; // Default to up if zero length
        
        return new C3Vector
        {
            X = vector.X / length,
            Y = vector.Y / length,
            Z = vector.Z / length
        };
    }

    // Ensure all normals are normalized
    public void NormalizeAllVectors()
    {
        for (int i = 0; i < Normals.Count; i++)
        {
            Normals[i] = Normalize(Normals[i]);
        }
    }
}

public struct C3Vector
{
    public float X { get; set; }
    public float Y { get; set; }
    public float Z { get; set; }

    public override string ToString() => $"({X}, {Y}, {Z})";
}
```

## Related Information
- Unlike MSVT, this chunk uses floating-point values instead of integers for higher precision
- Normal vectors are used for lighting calculations and determining surface orientation
- The relationship between these normals and other geometric elements (MSPV/MSVT vertices or MSUR surfaces) is not explicitly documented
- This chunk is present in both PM4 and PD4 formats with identical structure
- No special coordinate transformation is documented for these normals, unlike MSVT vertices 