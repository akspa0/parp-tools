# MOBR - WMO Group BSP Face References

## Type
WMO Group Chunk

## Source
WMO.md

## Description
The MOBR (Map Object BsP face References) chunk contains indices into the MOVI chunk that reference the actual triangles used by the BSP tree leaf nodes. It serves as a layer of indirection between the BSP tree nodes in the MOBN chunk and the actual triangle indices in the MOVI chunk. Each MOBN leaf node specifies a range of indices in the MOBR array, and those MOBR indices in turn point to triangle indices in the MOVI chunk.

## Structure

```csharp
public struct MOBR
{
    public ushort[] faceIndices; // Array of indices into the MOVI chunk
}
```

## Properties

| Offset | Name | Type | Description |
|--------|------|------|-------------|
| 0x00 | faceIndices | ushort[] | Array of indices into the MOVI chunk, which in turn contains the actual triangle indices |

## Dependencies
- **MOBN**: Contains the BSP tree nodes that reference ranges of indices in this chunk
- **MOVI**: Contains the actual triangle indices that this chunk references
- **MOGP**: The flag 0x00000001 (HasBSP) in the MOGP header indicates that this chunk is present

## Implementation Notes
- The MOBR chunk is simply an array of 16-bit unsigned integers (ushort).
- Each value in the array is an index into the MOVI chunk, which contains indices into the vertex array.
- The MOBR indices are referenced from the MOBN leaf nodes. Each leaf node contains a faceStart value and an nFaces value that together define a range of indices in the MOBR array.
- The indirection provided by MOBR allows the same triangle to be referenced by multiple BSP leaf nodes without duplicating the triangle data.
- When traversing the BSP tree to find triangles that might intersect with a ray or contain a point, you ultimately collect a set of MOBR indices from leaf nodes, then use those to index into MOVI to get the actual triangles.
- The chunk size should be a multiple of 2 bytes (sizeof(ushort)).
- For implementation, the typical usage pattern is to:
  1. Traverse the BSP tree to find relevant leaf nodes
  2. For each leaf node, get the range of MOBR indices using faceStart and nFaces
  3. Use each MOBR index to get a triangle index from MOVI
  4. Use the triangle index to get the three vertex indices from MOVI
  5. Use the vertex indices to get the actual vertex positions from MOVT

## Implementation Example

```csharp
public class MOBRChunk : IWmoGroupChunk
{
    public string ChunkId => "MOBR";
    public List<ushort> FaceIndices { get; private set; } = new List<ushort>();

    public void Parse(BinaryReader reader, long size)
    {
        // Each index is 2 bytes (ushort)
        int indexCount = (int)(size / 2);
        
        for (int i = 0; i < indexCount; i++)
        {
            FaceIndices.Add(reader.ReadUInt16());
        }
        
        // Ensure we've read all the data
        if (reader.BaseStream.Position % 2 != 0)
        {
            throw new InvalidDataException("MOBR chunk size is not a multiple of 2 bytes");
        }
    }
    
    public void Write(BinaryWriter writer)
    {
        foreach (var index in FaceIndices)
        {
            writer.Write(index);
        }
    }
    
    // Helper method to get the triangle indices for a BSP leaf node
    public List<ushort> GetTriangleIndicesForLeafNode(uint faceStart, ushort nFaces, List<ushort> moviIndices)
    {
        List<ushort> triangleIndices = new List<ushort>();
        
        // Ensure the range is valid
        if (faceStart >= FaceIndices.Count)
            return triangleIndices;
            
        // Calculate the end index, clamping to prevent going out of bounds
        uint endIndex = Math.Min(faceStart + nFaces, (uint)FaceIndices.Count);
        
        // Get the MOVI indices referenced by this leaf node
        for (uint i = faceStart; i < endIndex; i++)
        {
            // Get the index into MOVI
            ushort moviIndex = FaceIndices[(int)i];
            
            // Ensure the MOVI index is valid
            if (moviIndex < moviIndices.Count)
            {
                triangleIndices.Add(moviIndices[moviIndex]);
            }
        }
        
        return triangleIndices;
    }
}
```

## Usage Context
- The MOBR chunk is an essential part of the BSP tree structure used for collision detection and ray casting.
- When a ray intersects with a WMO, the BSP tree is traversed to find leaf nodes that might contain intersecting triangles. The MOBR indices from these leaf nodes are then used to find the actual triangles to test for intersection.
- For physics simulations and character movement, the MOBR chunk helps quickly identify which triangles a character or object might be colliding with.
- In rendering, the MOBR chunk can be used with the BSP tree for visibility determination and occlusion culling.
- The indirection through MOBR allows the BSP tree to efficiently reference subsets of triangles without duplicating triangle data, saving memory and improving cache coherence. 