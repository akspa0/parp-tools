# MOVI - Map Object Vertex Indices

## Type
WMO Group Chunk

## Source
WMO.md

## Description
The MOVI (Map Object Vertex Indices) chunk contains vertex indices that form triangles in a WMO group. These indices reference vertices defined in the MOVT (vertices), MONR (normals), and MOTV (texture coordinates) chunks. Every three consecutive indices in this chunk define one triangle in the group.

## Structure

```csharp
public struct MOVI
{
    public ushort[] Indices; // Array of vertex indices
}
```

## Properties

| Offset | Name | Type | Description |
|--------|------|------|-------------|
| 0x00 | Indices | ushort[] | Array of vertex indices forming triangles (each three consecutive indices form one triangle) |

## Dependencies
- MOVT chunk (vertex positions referenced by these indices)
- MONR chunk (vertex normals referenced by these indices)
- MOTV chunk (texture coordinates referenced by these indices)
- MOPY chunk (material information for the triangles formed by these indices)

## Implementation Notes
- The number of elements in this chunk must be divisible by 3, as three indices are needed to form one triangle.
- The indices reference the corresponding entries in the MOVT, MONR, and MOTV chunks.
- All triangles use a right-handed coordinate system where vertices are specified in counter-clockwise order for front-facing triangles.
- When used in a left-handed coordinate system, the 2nd and 3rd vertex indices of each triangle should be swapped to maintain proper facing.
- In newer versions of the format (expansion level 9+), this chunk may be replaced by MOVX, which uses 32-bit indices instead of 16-bit indices to support larger vertex counts.
- The MOBA (render batches) chunk references ranges of indices in this chunk for efficient rendering.

## Implementation Example

```csharp
public class MOVIChunk : IWmoGroupChunk
{
    public string ChunkId => "MOVI";
    
    public List<ushort> Indices { get; } = new List<ushort>();
    
    public void Read(BinaryReader reader, uint size)
    {
        // Each index is 2 bytes, calculate the number of indices
        int indexCount = (int)(size / 2);
        Indices.Clear();
        
        for (int i = 0; i < indexCount; i++)
        {
            Indices.Add(reader.ReadUInt16());
        }
        
        // Verify that the number of indices is divisible by 3
        if (Indices.Count % 3 != 0)
        {
            throw new InvalidDataException($"MOVI chunk index count ({Indices.Count}) is not divisible by 3.");
        }
    }
    
    public void Write(BinaryWriter writer)
    {
        // Verify that the number of indices is divisible by 3
        if (Indices.Count % 3 != 0)
        {
            throw new InvalidDataException($"MOVI chunk index count ({Indices.Count}) is not divisible by 3.");
        }
        
        // Write chunk header
        writer.Write(ChunkUtils.GetChunkIdBytes(ChunkId));
        writer.Write((uint)(Indices.Count * 2)); // Size = indexCount * 2 bytes per index
        
        // Write indices
        foreach (var index in Indices)
        {
            writer.Write(index);
        }
    }
    
    // Helper method to get triangle indices
    public (ushort, ushort, ushort) GetTriangle(int triangleIndex)
    {
        int baseIndex = triangleIndex * 3;
        if (baseIndex + 2 >= Indices.Count)
        {
            throw new ArgumentOutOfRangeException(nameof(triangleIndex));
        }
        
        return (Indices[baseIndex], Indices[baseIndex + 1], Indices[baseIndex + 2]);
    }
    
    // Helper method to get the total number of triangles
    public int GetTriangleCount()
    {
        return Indices.Count / 3;
    }
    
    // Helper method to add a triangle
    public void AddTriangle(ushort index1, ushort index2, ushort index3)
    {
        Indices.Add(index1);
        Indices.Add(index2);
        Indices.Add(index3);
    }
}
```

## Validation Requirements
- The total number of indices must be divisible by 3.
- All indices should be within the valid range of vertices defined in the MOVT, MONR, and MOTV chunks.
- The triangles formed by these indices should be consistent with the material entries in the MOPY chunk.
- The MOBA chunk's references to ranges within this chunk should be valid.

## Usage Context
- **Triangle Definition**: Defines the triangles that make up the 3D geometry of the WMO group.
- **Vertex Referencing**: Connects vertices with their corresponding normals and texture coordinates.
- **Batch Organization**: Used by the MOBA chunk to group triangles for efficient rendering.
- **Collision Detection**: Defines the triangular mesh used for collision detection.
- **Rendering**: Provides the index buffer for GPU-accelerated rendering. 