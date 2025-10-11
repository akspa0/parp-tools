# C001: AHDR

## Type
ADT v22 Chunk

## Source
ADT_v22.md

## Description
Header chunk for ADT v22 files containing structure information.

## Structure
```csharp
struct AHDR 
{
    uint32_t version;    // 22, mirrors MVER
    uint32_t vertices_x; // 129
    uint32_t vertices_y; // 129
    uint32_t chunks_x;   // 16
    uint32_t chunks_y;   // 16
    uint32_t unused[11]; // Padding/unused data
}
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| version | uint32 | The version (22), mirrors MVER chunk value |
| vertices_x | uint32 | Number of vertices in X dimension (129) |
| vertices_y | uint32 | Number of vertices in Y dimension (129) |
| chunks_x | uint32 | Number of chunks in X dimension (16) |
| chunks_y | uint32 | Number of chunks in Y dimension (16) |
| unused | uint32[11] | Unused data, likely reserved for future use |

## Dependencies
- MVER (must be read first to confirm file version)

## Implementation Notes
- Size: 0x40 (64 bytes)
- Replaces MHDR from ADT v18
- This chunk defines the grid structure of the ADT file
- The vertices are arranged in a grid of (vertices_x × vertices_y)
- Each chunk covers a 16×16 grid area

## Implementation Example
```csharp
public class AHDR
{
    public uint Version { get; set; }     // Should be 22
    public uint VerticesX { get; set; }   // 129
    public uint VerticesY { get; set; }   // 129
    public uint ChunksX { get; set; }     // 16
    public uint ChunksY { get; set; }     // 16
    public uint[] Unused { get; set; } = new uint[11];

    // Helper property to get total vertex count
    public uint TotalVertices => VerticesX * VerticesY;
    
    // Helper property to get total chunk count
    public uint TotalChunks => ChunksX * ChunksY;
}
```

## Usage Context
The AHDR chunk defines the structure of the ADT file in the v22 format. It specifies the grid dimensions for both vertices and chunks. This information is essential for parsing the subsequent vertex data chunks (AVTX, ANRM). 