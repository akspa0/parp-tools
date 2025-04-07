# G015: MLVR

## Type
WMO Group Chunk

## Source
WMO.md

## Description
The MLVR (Map Object Liquid Vertices Render) chunk contains additional vertex data for rendering liquid surfaces in WMO groups. This chunk extends the basic liquid vertex data from MLIQ with rendering-specific information. It works in conjunction with the MLIQ chunk to provide complete liquid surface rendering capabilities.

## Structure
```cpp
struct {
    SMOLiquidRenderVertex[] vertices;  // Array of liquid render vertices
} mapObjectLiquidRenderVertices;

struct SMOLiquidRenderVertex
{
    float height;        // Vertex height for rendering
    uint8_t flags;      // Render flags
    uint8_t padding[3]; // Padding to maintain alignment
};
```

### C# Structure
```csharp
struct MLVR
{
    LiquidRenderVertex[] Vertices;  // Array of liquid render vertices
}

struct LiquidRenderVertex
{
    /*0x00*/ float Height;        // Vertex height for rendering
    /*0x04*/ byte Flags;         // Render flags
    /*0x05*/ byte[] Padding;     // 3 bytes of padding
};
```

## Properties

### LiquidRenderVertex Structure
| Offset | Name | Type | Description |
|--------|------|------|-------------|
| 0x00 | Height | float | Height of the liquid vertex for rendering |
| 0x04 | Flags | uint8_t | Rendering flags for the vertex |
| 0x05 | Padding | uint8_t[3] | Padding bytes to maintain 4-byte alignment |

## Dependencies
- MLIQ: Contains the base liquid vertex data that this chunk extends
- MOGP: The groupLiquid field indicates if liquid is present

## Implementation Notes
- Each vertex entry is 8 bytes (0x08)
- The number of vertices should match the vertex count from MLIQ
- Vertex data is typically stored in column-major order
- Padding bytes should be preserved but not modified
- Height values should be within the group's bounding box
- Flags may control rendering effects like transparency or animation

## Implementation Example
```csharp
public class MLVR : IChunk
{
    public List<LiquidRenderVertex> Vertices { get; private set; }
    
    public MLVR()
    {
        Vertices = new List<LiquidRenderVertex>();
    }
    
    public void Parse(BinaryReader reader, long size)
    {
        // Calculate how many vertices we expect
        int vertexCount = (int)(size / 8); // Each vertex is 8 bytes
        
        Vertices.Clear();
        
        for (int i = 0; i < vertexCount; i++)
        {
            var vertex = new LiquidRenderVertex
            {
                Height = reader.ReadSingle(),
                Flags = reader.ReadByte(),
                Padding = reader.ReadBytes(3)
            };
            
            Vertices.Add(vertex);
        }
    }
    
    public void Write(BinaryWriter writer)
    {
        foreach (var vertex in Vertices)
        {
            writer.Write(vertex.Height);
            writer.Write(vertex.Flags);
            writer.Write(vertex.Padding);
        }
    }
}

public class LiquidRenderVertex
{
    public float Height { get; set; }
    public byte Flags { get; set; }
    public byte[] Padding { get; set; }
    
    public LiquidRenderVertex()
    {
        Padding = new byte[3];
    }
}
```

## Validation Requirements
- The chunk size must be a multiple of 8 bytes
- The number of vertices should match MLIQ's vertex count
- Height values should be within reasonable bounds
- Padding bytes should be preserved
- Flags should be valid for the liquid type

## Usage Context
The MLVR chunk provides rendering-specific data for liquid surfaces:

1. **Rendering Control**:
   - Height values may be interpolated for smooth rendering
   - Flags control visual effects and behavior
   - Works with MLIQ data to create complete liquid surfaces

2. **Visual Effects**:
   - May control transparency levels
   - Could influence wave animations
   - Might affect surface reflections

3. **Integration**:
   - Used in conjunction with MLIQ chunk
   - Extends basic liquid data with render-specific information
   - Part of WMO's liquid rendering system

The rendering system uses this data to:
1. Determine final liquid surface appearance
2. Apply visual effects and animations
3. Control liquid surface behavior
4. Integrate with the WMO's overall rendering pipeline 