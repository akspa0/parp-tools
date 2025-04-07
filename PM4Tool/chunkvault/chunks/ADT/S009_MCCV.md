# S009: MCCV

## Type
MCNK Subchunk

## Source
ADT_v18.md

## Description
The MCCV (Map Chunk Vertex Colors) subchunk contains vertex color information for an MCNK chunk. These colors are used to add additional shading and coloring to the terrain, allowing for more visual detail without additional textures.

## Structure
```csharp
struct MCCV
{
    /*0x00*/ uint8_t vertex_colors[145][4];  // BGRA colors for each vertex (145 vertices total)
};
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| vertex_colors | uint8_t[145][4] | BGRA color values for each terrain vertex |

## Color Format
- Each vertex has 4 color components: Blue, Green, Red, Alpha
- Each component is a value from 0-255
- The colors are applied to the same vertices as MCVT and MCNR
- The 145 vertices follow the same layout as in MCVT (9×9 grid + middle points)

## Dependencies
- MCNK (C018) - Parent chunk that contains this subchunk
- MCVT (S001) - Vertices corresponding to the color values
- MCNK.flags - Flag 0x40 (HasMCCV) indicates presence of this chunk

## Presence Determination
This subchunk is only present when:
- MCNK.flags has the HasMCCV flag set (0x40)
- MCNK.mccv offset is non-zero

## Implementation Notes
- The vertex colors are multiplied with the terrain texture colors during rendering
- Introduced in Wrath of the Lich King expansion
- Used for terrain color variation without requiring additional texture layers
- Can provide subtle weather or biome transitions
- The colors can provide environmental effects like snow, ash, or muddy areas
- Vertex colors can be used for areas where terrain textures wouldn't provide enough variation

## Implementation Example
```csharp
public class MCCV : IChunk
{
    public const int VERTICES_COUNT = 145;
    
    public struct Color
    {
        public byte B { get; set; }
        public byte G { get; set; }
        public byte R { get; set; }
        public byte A { get; set; }
        
        public Color(byte r, byte g, byte b, byte a)
        {
            R = r;
            G = g;
            B = b;
            A = a;
        }
    }
    
    public Color[] VertexColors { get; set; } = new Color[VERTICES_COUNT];
    
    public void Parse(BinaryReader reader)
    {
        for (int i = 0; i < VERTICES_COUNT; i++)
        {
            byte b = reader.ReadByte();
            byte g = reader.ReadByte();
            byte r = reader.ReadByte();
            byte a = reader.ReadByte();
            
            VertexColors[i] = new Color(r, g, b, a);
        }
    }
    
    public void Write(BinaryWriter writer)
    {
        foreach (var color in VertexColors)
        {
            writer.Write(color.B);
            writer.Write(color.G);
            writer.Write(color.R);
            writer.Write(color.A);
        }
    }
    
    public Color GetVertexColor(int x, int y)
    {
        if (x < 0 || x >= 9 || y < 0 || y >= 9)
            throw new ArgumentOutOfRangeException();
            
        return VertexColors[y * 9 + x];
    }
    
    public Color GetMiddleVertexColor(int x, int y)
    {
        if (x < 0 || x >= 8 || y < 0 || y >= 8)
            throw new ArgumentOutOfRangeException();
            
        return VertexColors[81 + y * 8 + x];
    }
}
```

## Vertex Color Interpolation
When rendering terrain:
- Colors are interpolated across triangles
- Colors are multiplied with the texture colors
- Alpha values can be used for blending or special effects
- This creates a smooth color gradient across the terrain

## Version Information
- MCCV was introduced in Wrath of the Lich King (version 8)
- It's optional - chunks may or may not have vertex colors
- The presence is indicated by the HasMCCV flag in MCNK.flags

## Visual Effects
Vertex colors enable several visual effects:
- Adding moss or darkening to cracks and crevices
- Snow patches that blend naturally with the terrain
- Gradual transitions between biomes
- Localized darkening to simulate charred earth or shadows
- Enhancing the color variation of terrain without using additional texture layers

## Usage Context
The MCCV subchunk allows for more detailed and varied terrain coloration by providing per-vertex color information. This technique adds visual richness without requiring additional textures, which helps with performance and memory usage. By modifying the color of the terrain at a granular level, map designers can create more realistic and varied landscapes. The effect is especially useful for creating transitions between different terrain types or adding localized details like patches of moss, ash, or snow. 