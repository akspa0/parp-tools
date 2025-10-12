# S010: MCLV

## Type
MCNK Subchunk

## Source
ADT_v18.md

## Description
The MCLV (Map Chunk Light Values) subchunk contains per-vertex lighting information for an MCNK chunk. It was introduced in Cataclysm to provide more advanced lighting capabilities for terrain.

## Structure
```csharp
struct MCLV
{
    /*0x00*/ uint8_t light_values[145][3];  // RGB color values for each vertex
};
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| light_values | uint8_t[145][3] | RGB light values for each terrain vertex |

## Light Value Format
- Each vertex has 3 color components: Red, Green, Blue
- Each component is a value from 0-255
- The light values are applied to the same vertices as MCVT
- The 145 vertices follow the same layout as in MCVT (9×9 grid + middle points)
- Unlike MCCV, there is no alpha component

## Dependencies
- MCNK (C018) - Parent chunk that contains this subchunk
- MCVT (S001) - Vertices corresponding to the light values
- MCNK.mclv - Offset to this subchunk

## Presence Determination
This subchunk is only present when:
- MCNK.mclv offset is non-zero
- The ADT is from Cataclysm or later (version >= 12)

## Implementation Notes
- The light values are used in combination with normal vectors (MCNR) for terrain lighting
- Introduced in Cataclysm expansion
- Used for more advanced lighting effects and local illumination
- Can represent ambient lighting, colored lights, or shadow effects
- Different from MCCV, which modifies the texture coloration
- Light values are applied during the lighting calculation phase of rendering

## Implementation Example
```csharp
public class MCLV : IChunk
{
    public const int VERTICES_COUNT = 145;
    
    public struct LightValue
    {
        public byte R { get; set; }
        public byte G { get; set; }
        public byte B { get; set; }
        
        public LightValue(byte r, byte g, byte b)
        {
            R = r;
            G = g;
            B = b;
        }
    }
    
    public LightValue[] LightValues { get; set; } = new LightValue[VERTICES_COUNT];
    
    public void Parse(BinaryReader reader)
    {
        for (int i = 0; i < VERTICES_COUNT; i++)
        {
            byte r = reader.ReadByte();
            byte g = reader.ReadByte();
            byte b = reader.ReadByte();
            
            LightValues[i] = new LightValue(r, g, b);
        }
    }
    
    public void Write(BinaryWriter writer)
    {
        foreach (var light in LightValues)
        {
            writer.Write(light.R);
            writer.Write(light.G);
            writer.Write(light.B);
        }
    }
    
    public LightValue GetLightValue(int x, int y)
    {
        if (x < 0 || x >= 9 || y < 0 || y >= 9)
            throw new ArgumentOutOfRangeException();
            
        return LightValues[y * 9 + x];
    }
    
    public LightValue GetMiddleLightValue(int x, int y)
    {
        if (x < 0 || x >= 8 || y < 0 || y >= 8)
            throw new ArgumentOutOfRangeException();
            
        return LightValues[81 + y * 8 + x];
    }
}
```

## Light Interpolation
When rendering terrain:
- Light values are interpolated across the triangles
- The interpolated light values are applied during the lighting calculation
- This creates smooth lighting transitions across the terrain

## Version Information
- MCLV was introduced in Cataclysm (version 12)
- It's optional - chunks may or may not have light values
- The presence is determined by a non-zero MCNK.mclv offset

## Difference from MCCV
- MCCV provides color modifications to the texture
- MCLV provides lighting information separate from texture colors
- Both can be present in the same chunk for different effects
- MCLV has no alpha component

## Visual Effects
The MCLV subchunk enables several lighting effects:
- Gradual lighting transitions between different areas
- Local light sources affecting terrain
- Light bleeding from detailed objects onto terrain
- Light/shadow effects independent of the texture color
- Enhanced ambient occlusion at a per-vertex level

## Usage Context
The MCLV subchunk enhances the visual quality of terrain by providing localized lighting information. This allows for more detailed and realistic lighting effects without requiring dynamic lighting calculations at runtime. By having pre-calculated per-vertex lighting, the game can render more complex lighting scenarios efficiently. The MCLV data complements other terrain features like height maps, normal vectors, and texture layers to create a cohesive and visually rich environment. 