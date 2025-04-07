# MOVT - WMO Group Vertices

## Type
WMO Group Chunk

## Source
WMO.md

## Description
The MOVT chunk contains the vertex coordinates for the 3D mesh of the WMO group. These vertices define the spatial structure of the model and are referenced by the index data in the MOVI chunk to form triangles. The vertex data is stored as an array of 3D vectors, with each vector containing the X, Y, and Z coordinates of a vertex.

## Structure

```csharp
public struct MOVT
{
    public Vector3[] vertices; // Array of vertex positions
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
| 0x00 | vertices | Vector3[] | Array of 3D coordinates, with each vertex taking 12 bytes (3 floats, 4 bytes each). The number of vertices can be calculated as chunk size / 12. The coordinates are stored in (X, Z, -Y) order due to coordinate system differences between the modelling tools and the game engine. |

## Dependencies
- **MOVI**: The indices in the MOVI chunk reference these vertices to form triangles.
- **MONR**: The normal vectors in the MONR chunk should correspond to these vertices in order.
- **MOTV**: The texture coordinates in the MOTV chunk should correspond to these vertices in order.

## Implementation Notes
- The size of the chunk should be a multiple of 12 bytes (3 floats × 4 bytes each).
- The coordinate system in WMO files has the Z-axis pointing up and the Y-axis pointing into the screen, which differs from OpenGL's coordinate system where Z points toward the viewer and Y points up. This results in the (X, Z, -Y) ordering.
- Vertices in a WMO group are usually within the bounding box defined in the MOGP header.
- WMO models typically use fewer vertices than M2 models as they represent static structures rather than animated characters.
- The vertex array may contain unused vertices that aren't referenced by any triangles, which is common in 3D models after optimization.

## Implementation Example

```csharp
public class MOVTChunk : IWmoGroupChunk
{
    public string ChunkId => "MOVT";
    public List<Vector3> Vertices { get; private set; } = new List<Vector3>();

    public void Parse(BinaryReader reader, long size)
    {
        // Calculate the number of vertices
        int vertexCount = (int)(size / 12);
        
        // Read all vertices
        for (int i = 0; i < vertexCount; i++)
        {
            Vector3 vertex = new Vector3
            {
                x = reader.ReadSingle(),
                y = reader.ReadSingle(),
                z = reader.ReadSingle()
            };
            
            Vertices.Add(vertex);
        }
        
        // Ensure we've read all the data
        if (reader.BaseStream.Position % 12 != 0)
        {
            throw new InvalidDataException("MOVT chunk size is not a multiple of 12 bytes");
        }
    }
    
    public void Write(BinaryWriter writer)
    {
        foreach (var vertex in Vertices)
        {
            writer.Write(vertex.x);
            writer.Write(vertex.y);
            writer.Write(vertex.z);
        }
    }
}
```

## Usage Context
- The vertices defined in this chunk form the base geometric structure of the WMO group.
- In combination with the MOVI chunk, these vertices form triangles that make up the visible surfaces of the model.
- When rendering, these vertices are transformed by the WMO's world matrix to position them correctly in the game world.
- For collision detection, these vertices form the collision mesh of the WMO.
- Level designers use these vertices to define the shape of buildings, terrain features, and other world structures. 