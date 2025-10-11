# AHDR - Area Data Table Header

## Type
ADT v23 Chunk

## Source
Referenced from `ADT_v23.md`

## Description
The AHDR (Header) chunk contains basic information about the ADT tile, including its version, dimensions, and grid structure. This chunk defines the fundamental properties of the terrain grid and is essential for parsing the rest of the ADT file. The v23 format was an experimental format that appeared during the Cataclysm beta but was never used in the final release.

## Structure

```csharp
public struct AHDR
{
    public uint version;     // Always 23 for this format, mirrors MVER
    public uint vertices_x;  // Number of vertices in X direction, typically 129
    public uint vertices_y;  // Number of vertices in Y direction, typically 129
    public uint chunks_x;    // Number of chunks in X direction, typically 16
    public uint chunks_y;    // Number of chunks in Y direction, typically 16
    public uint padding1;    // Padding/reserved
    public uint padding2;    // Padding/reserved
    public uint padding3;    // Padding/reserved
    public uint padding4;    // Padding/reserved
    public uint padding5;    // Padding/reserved
    public uint padding6;    // Padding/reserved
    public uint padding7;    // Padding/reserved
    public uint padding8;    // Padding/reserved
    public uint padding9;    // Padding/reserved
    public uint padding10;   // Padding/reserved
    public uint padding11;   // Padding/reserved
}
```

## Properties

| Name | Type | Description |
|------|------|-------------|
| version | uint | Always 23 for this format, mirrors the version in MVER |
| vertices_x | uint | Number of vertices in X direction, typically 129 |
| vertices_y | uint | Number of vertices in Y direction, typically 129 |
| chunks_x | uint | Number of chunks in X direction, typically 16 |
| chunks_y | uint | Number of chunks in Y direction, typically 16 |

## Dependencies

No direct dependencies on other chunks.

## Implementation Notes

1. The AHDR chunk is similar to the MHDR chunk in ADT v18, but with a different structure and naming convention.

2. The vertex counts (vertices_x, vertices_y) define both the outer grid (129×129) and implicitly the inner grid (128×128).

3. The total size of this chunk is 0x40 (64) bytes, with 11 DWORDs of padding.

4. Unlike v18 where terrain data is interleaved, v23 keeps outer vertices (129×129) and inner vertices (128×128) as separate continuous arrays in the AVTX chunk.

5. The number of chunks (chunks_x, chunks_y) defines the grid of ACNK chunks in the ADT file. Each chunk typically represents a 16×16 yard piece of terrain.

## Implementation Example

```csharp
public class AhdrChunk
{
    public uint Version { get; set; }
    public uint VerticesX { get; set; }
    public uint VerticesY { get; set; }
    public uint ChunksX { get; set; }
    public uint ChunksY { get; set; }

    public AhdrChunk()
    {
        // Default values for a standard ADT
        Version = 23;
        VerticesX = 129;
        VerticesY = 129;
        ChunksX = 16;
        ChunksY = 16;
    }

    public void Load(BinaryReader reader)
    {
        Version = reader.ReadUInt32();
        VerticesX = reader.ReadUInt32();
        VerticesY = reader.ReadUInt32();
        ChunksX = reader.ReadUInt32();
        ChunksY = reader.ReadUInt32();
        
        // Skip padding fields (11 DWORDs)
        reader.BaseStream.Position += 44;
    }

    public void Save(BinaryWriter writer)
    {
        writer.Write("AHDR".ToCharArray());
        writer.Write(0x40); // Chunk size (64 bytes)
        
        writer.Write(Version);
        writer.Write(VerticesX);
        writer.Write(VerticesY);
        writer.Write(ChunksX);
        writer.Write(ChunksY);
        
        // Write padding fields
        for (int i = 0; i < 11; i++)
        {
            writer.Write(0); // 11 DWORDs of padding
        }
    }
    
    // Helper methods for accessing grid information
    
    // Get total vertices count
    public int GetTotalVerticesCount()
    {
        return (int)(VerticesX * VerticesY + (VerticesX - 1) * (VerticesY - 1));
    }
    
    // Get outer vertices count
    public int GetOuterVerticesCount()
    {
        return (int)(VerticesX * VerticesY);
    }
    
    // Get inner vertices count
    public int GetInnerVerticesCount()
    {
        return (int)((VerticesX - 1) * (VerticesY - 1));
    }
    
    // Get total chunks count
    public int GetTotalChunksCount()
    {
        return (int)(ChunksX * ChunksY);
    }
}
```

## Usage Context

The AHDR chunk is the first main chunk in an ADT v23 file and provides the fundamental dimensions and structure used by other chunks. It defines:

1. **Grid Layout**: The vertices_x and vertices_y fields define the dimensions of the height grid used for the terrain.

2. **Chunk Organization**: The chunks_x and chunks_y fields define how many terrain chunks are in the ADT tile, typically arranged in a 16×16 grid.

3. **Spatial Structure**: By defining the organization of the terrain data, AHDR enables spatial operations and queries on the terrain.

The v23 format's AHDR chunk maintains the same general purpose as the MHDR chunk in v18, but with a reorganized structure and the A-prefix naming convention. While the v23 format was ultimately abandoned and never used in retail clients, it represents an interesting experimental approach to terrain data organization during the Cataclysm beta development period.

The separation of inner and outer vertices arrays in v23 (as defined in AHDR and implemented in AVTX) differs from the interleaved approach of v18, potentially offering more efficient memory access patterns when processing the terrain grid. 