# MOVV - WMO Visible Vertices

## Type
WMO Root Chunk

## Source
WMO.md

## Description
The MOVV chunk contains an array of vertices that define visible blocks within the WMO model. These vertices are used to create bounding geometry for visibility testing and are referenced by the MOVB (Visible Blocks) chunk. This system helps to optimize rendering by quickly determining which parts of the world model are potentially visible from any given viewpoint.

## Structure

```csharp
public struct MOVV
{
    public Vector3[] visibleVertices; // Array of 3D vertices
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
| 0x00 | visibleVertices | Vector3[] | Array of 3D coordinate vectors, each consisting of three 32-bit floating point values (x, y, z). The array size is determined by the chunk size divided by 12 (3 floats × 4 bytes each). |

## Dependencies
- **MOHD**: The header chunk contains information about the number of visible blocks, which is related to these vertices.
- **MOVB**: The visible blocks chunk references these vertices to define visibility bounding geometry.

## Implementation Notes
- Each vertex is 12 bytes (3 floats × 4 bytes) in size.
- These vertices define points in 3D space that are used to construct visibility testing geometry.
- The vertices use the same coordinate system as the rest of the WMO model.
- The MOVV chunk works in conjunction with the MOVB chunk to implement a visibility optimization system.
- The number of vertices can be calculated from the chunk size: `vertexCount = chunkSize / 12`.
- These vertices are not the same as the vertices used for rendering the visual geometry of the model; they are specifically for visibility optimization.

## Implementation Example

```csharp
public class MOVVChunk : IWmoChunk
{
    public string ChunkId => "MOVV";
    public List<Vector3> VisibleVertices { get; set; } = new List<Vector3>();

    public void Read(BinaryReader reader, uint size)
    {
        // Calculate how many vertices are in the chunk
        int count = (int)(size / 12); // Each vertex is 12 bytes (3 floats × 4 bytes)
        
        for (int i = 0; i < count; i++)
        {
            Vector3 vertex = new Vector3
            {
                X = reader.ReadSingle(),
                Y = reader.ReadSingle(),
                Z = reader.ReadSingle()
            };
            VisibleVertices.Add(vertex);
        }
    }

    public void Write(BinaryWriter writer)
    {
        // Write the chunk header
        writer.Write(ChunkUtils.GetChunkIdBytes(ChunkId));
        
        // Calculate size (12 bytes per vertex)
        uint dataSize = (uint)(VisibleVertices.Count * 12);
        writer.Write(dataSize);
        
        // Write all vertices
        foreach (var vertex in VisibleVertices)
        {
            writer.Write(vertex.X);
            writer.Write(vertex.Y);
            writer.Write(vertex.Z);
        }
    }
    
    public class Vector3
    {
        public float X { get; set; }
        public float Y { get; set; }
        public float Z { get; set; }
    }
}
```

## Validation Requirements
- The number of vertices must be a whole number (chunk size must be divisible by 12).
- Vertices should be within a reasonable range for the WMO model's dimensions.
- The vertices should form valid bounding geometry when referenced by the MOVB chunk.
- Coordinate values should be finite (not NaN or infinite).
- The array should contain enough vertices to properly define the visibility volumes.

## Usage Context
- **Visibility Optimization:** The vertices define geometry used for quickly testing whether parts of the WMO are potentially visible.
- **Occlusion Testing:** Combined with the MOVB chunk, these vertices help determine which groups should be rendered based on the viewer's position.
- **Performance Scaling:** The visibility system enables WMOs to scale effectively on different hardware by limiting rendering to only what's potentially visible.
- **Level of Detail:** The visibility system can work in conjunction with level-of-detail mechanisms to further optimize rendering.
- **Culling System:** These vertices are part of a hierarchical culling system that works with portals and groups to manage rendering complexity. 