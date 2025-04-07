# MONR - WMO Group Normals

## Type
WMO Group Chunk

## Source
WMO.md

## Description
The MONR chunk contains the normal vectors for each vertex in the WMO group. Normal vectors are essential for proper lighting and shading calculations as they define the direction that a surface is facing. Each normal corresponds to a vertex in the MOVT chunk and should be in the same order. These normals are used by the rendering engine to calculate how light interacts with the surfaces of the model.

## Structure

```csharp
public struct MONR
{
    public Vector3[] normals; // Array of normal vectors
}

public struct Vector3
{
    public float x;
    public float y;
    public float z;
}
```

## Properties

| Offset | Name | Type | Description |
|--------|------|------|-------------|
| 0x00 | normals | Vector3[] | Array of 3D vectors representing the normal direction at each vertex. Each normal takes 12 bytes (3 floats, 4 bytes each). The number of normals can be calculated as chunk size / 12. Like vertices, normals are stored in (X, Z, -Y) order due to coordinate system differences. |

## Dependencies
- **MOVT**: The normals in this chunk should correspond to the vertices in the MOVT chunk in the same order.

## Implementation Notes
- The size of the chunk should be a multiple of 12 bytes (3 floats × 4 bytes each).
- Normal vectors should be normalized (length of 1.0) for correct lighting calculations.
- The coordinate system in WMO files has the Z-axis pointing up and the Y-axis pointing into the screen, which differs from OpenGL's coordinate system where Z points toward the viewer and Y points up. This results in the (X, Z, -Y) ordering.
- In some cases, WMO models may use "smooth" normals where the normal at a vertex is the average of the normals of all faces that share that vertex, creating a smoother appearance.
- Each normal in the array corresponds to the vertex at the same index in the MOVT chunk.
- Proper normal vectors are critical for lighting calculations, including ambient, diffuse, and specular lighting.

## Implementation Example

```csharp
public class MONRChunk : IWmoGroupChunk
{
    public string ChunkId => "MONR";
    public List<Vector3> Normals { get; private set; } = new List<Vector3>();

    public void Parse(BinaryReader reader, long size)
    {
        // Calculate the number of normals
        int normalCount = (int)(size / 12);
        
        // Read all normals
        for (int i = 0; i < normalCount; i++)
        {
            Vector3 normal = new Vector3
            {
                x = reader.ReadSingle(),
                y = reader.ReadSingle(),
                z = reader.ReadSingle()
            };
            
            Normals.Add(normal);
        }
        
        // Ensure we've read all the data
        if (reader.BaseStream.Position % 12 != 0)
        {
            throw new InvalidDataException("MONR chunk size is not a multiple of 12 bytes");
        }
    }
    
    public void Write(BinaryWriter writer)
    {
        foreach (var normal in Normals)
        {
            writer.Write(normal.x);
            writer.Write(normal.y);
            writer.Write(normal.z);
        }
    }
}
```

## Usage Context
- The normal vectors are essential for calculating lighting and shading on the WMO model.
- When rendering, these normals are used in lighting calculations to determine how light reflects off each point on the model's surface.
- The normals help create the visual depth and three-dimensionality of the model by determining how light interacts with surfaces.
- In some rendering techniques, normals may be interpolated across triangles to create smoother-looking surfaces.
- Normal maps (textures) may be used in conjunction with these vertex normals to add additional detail to the surface's appearance without increasing geometry complexity. 