# MOVB - WMO Visible Blocks

## Type
WMO Root Chunk

## Source
WMO.md

## Description
The MOVB chunk defines visibility blocks that help determine which parts of the WMO are visible from different viewpoints. It contains a list of indices referencing vertices from the MOVV chunk, forming convex volumes that can be used for efficient visibility testing. This system is an important part of the WMO's approach to occlusion culling and rendering optimization.

## Structure

```csharp
public struct MOVB
{
    public byte[][] visibleBlocks; // Array of visible block data
}
```

## Properties

| Offset | Name | Type | Description |
|--------|------|------|-------------|
| Varies | visibleBlocks | byte[][] | A series of visible block entries. Each entry begins with a 2-byte count (n) followed by n vertex indices (4 bytes each), referencing vertices in the MOVV chunk. |

## Dependencies
- **MOHD**: The header chunk contains information about the number of visible blocks.
- **MOVV**: Contains the vertices that are referenced by the indices in this chunk.

## Implementation Notes
- Each visible block defines a convex volume used for visibility testing.
- The blocks are defined by referencing vertices from the MOVV chunk.
- The structure of each visible block is:
  - A 16-bit unsigned integer (ushort) indicating the number of vertex indices (n)
  - n 32-bit unsigned integers (uint) serving as indices into the MOVV vertex array
- The size of each visible block entry is variable and depends on the number of vertices it references.
- Visible blocks are used in conjunction with the group information to determine which WMO groups need to be rendered from a given viewpoint.
- The total number of visible blocks should match the value specified in the MOHD chunk.
- These blocks help form a spatial partitioning scheme that allows for efficient visibility determination.

## Implementation Example

```csharp
public class MOVBChunk : IWmoChunk
{
    public string ChunkId => "MOVB";
    public List<VisibleBlock> VisibleBlocks { get; set; } = new List<VisibleBlock>();

    public void Read(BinaryReader reader, uint size)
    {
        long endPosition = reader.BaseStream.Position + size;
        
        // Read visible blocks until we reach the end of the chunk
        while (reader.BaseStream.Position < endPosition)
        {
            // Read the number of indices in this block
            ushort indexCount = reader.ReadUInt16();
            
            // Create a new visible block
            VisibleBlock block = new VisibleBlock();
            
            // Read all the indices
            for (int i = 0; i < indexCount; i++)
            {
                block.VertexIndices.Add(reader.ReadUInt32());
            }
            
            VisibleBlocks.Add(block);
        }
    }

    public void Write(BinaryWriter writer)
    {
        // Write the chunk header
        writer.Write(ChunkUtils.GetChunkIdBytes(ChunkId));
        
        // Calculate the size of all blocks
        uint dataSize = 0;
        foreach (var block in VisibleBlocks)
        {
            // 2 bytes for the count + 4 bytes per index
            dataSize += 2 + (uint)(block.VertexIndices.Count * 4);
        }
        writer.Write(dataSize);
        
        // Write all visible blocks
        foreach (var block in VisibleBlocks)
        {
            // Write the number of indices
            writer.Write((ushort)block.VertexIndices.Count);
            
            // Write each index
            foreach (var index in block.VertexIndices)
            {
                writer.Write(index);
            }
        }
    }
    
    public class VisibleBlock
    {
        public List<uint> VertexIndices { get; set; } = new List<uint>();
    }
}
```

## Validation Requirements
- The total number of visible blocks should match the count specified in the MOHD chunk.
- Each vertex index must be a valid index into the MOVV vertex array.
- Each visible block should define a valid convex volume (the vertices should form a closed, convex shape).
- The index count (n) for each block should be reasonable (typically between 4 and 8 vertices to define a convex volume).
- The total size of all blocks combined should match the chunk size.

## Usage Context
- **Occlusion Culling:** Visible blocks are used to determine which parts of the WMO are potentially visible from any given viewpoint.
- **Rendering Optimization:** By quickly eliminating groups that are definitely not visible, the rendering engine can focus on only the relevant portions of the model.
- **Performance Scaling:** This visibility system allows WMOs to perform well on a wide range of hardware by limiting the rendering workload.
- **Spatial Partitioning:** Visible blocks help partition the WMO into manageable sections for visibility determination.
- **View Frustum Culling:** The system works in conjunction with view frustum culling to efficiently render only what's necessary. 