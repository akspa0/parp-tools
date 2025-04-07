# O012: MOLP

## Type
WMO Root Chunk

## Source
WMO.md

## Description
The MOLP (Map Object Light Parameters) chunk defines point lights in the WMO. Each point light has properties such as position, color, intensity, and attenuation ranges that control how the light affects the surrounding environment. This chunk was introduced in expansion level 7 or possibly earlier.

## Structure
```cpp
struct {
    uint32_t unk;
    CImVector color; 
    C3Vector pos; //position of light
    float intensity; 
    float attenStart;
    float attenEnd;
    float unk4;   //Only seen zeros here 
    uint32_t unk5;
    uint32_t unk6; //CArgb?
} map_object_point_lights[];
```

### C# Structure
```csharp
struct MOLP
{
    SMOPointLight[] pointLights;  // Array of point light definitions
};

struct SMOPointLight
{
    /*0x00*/ uint32_t unknown1;      // Unknown value
    /*0x04*/ CImVector color;        // Light color (BGRA)
    /*0x08*/ C3Vector position;      // Position of light in 3D space
    /*0x14*/ float intensity;        // Light intensity
    /*0x18*/ float attenStart;       // Attenuation start distance
    /*0x1C*/ float attenEnd;         // Attenuation end distance
    /*0x20*/ float unknown4;         // Unknown value (typically zero)
    /*0x24*/ uint32_t unknown5;      // Unknown value
    /*0x28*/ uint32_t unknown6;      // Unknown value (possibly CArgb)
};
```

## Properties

### SMOPointLight Structure
| Offset | Name | Type | Description |
|--------|------|------|-------------|
| 0x00 | unknown1 | uint32_t | Unknown value |
| 0x04 | color | CImVector | Light color in BGRA format |
| 0x08 | position | C3Vector | 3D position of the light |
| 0x14 | intensity | float | Light intensity value |
| 0x18 | attenStart | float | Distance where light attenuation begins |
| 0x1C | attenEnd | float | Distance where light attenuation ends |
| 0x20 | unknown4 | float | Unknown value (typically zero) |
| 0x24 | unknown5 | uint32_t | Unknown value |
| 0x28 | unknown6 | uint32_t | Unknown value (possibly CArgb) |

## Dependencies
- MOHD: The nLights field indicates how many point lights should be present
- MOLT: May reference these lights for additional properties or relationships

## Implementation Notes
- Each point light definition is 40 bytes (0x28)
- The attenuation range defines how the light fades with distance:
  - Full intensity from light position to attenStart
  - Linear falloff from attenStart to attenEnd
  - No light beyond attenEnd
- Intensity must be non-negative
- Attenuation distances must be non-negative and attenEnd must be greater than attenStart
- The unknown4 field is typically zero in most WMO files
- The unknown6 field might be a secondary color or additional light properties

## Implementation Example
```csharp
public class MOLP : IChunk
{
    public List<PointLight> PointLights { get; private set; }
    
    public MOLP()
    {
        PointLights = new List<PointLight>();
    }
    
    public void Parse(BinaryReader reader, long size)
    {
        // Calculate how many point lights we expect
        int lightCount = (int)(size / 40); // Each light is 40 bytes
        
        PointLights.Clear();
        
        for (int i = 0; i < lightCount; i++)
        {
            PointLight light = new PointLight();
            
            light.Unknown1 = reader.ReadUInt32();
            light.Color = new Color(reader.ReadUInt32());
            light.Position = new Vector3(
                reader.ReadSingle(),
                reader.ReadSingle(),
                reader.ReadSingle()
            );
            light.Intensity = reader.ReadSingle();
            light.AttenuationStart = reader.ReadSingle();
            light.AttenuationEnd = reader.ReadSingle();
            light.Unknown4 = reader.ReadSingle();
            light.Unknown5 = reader.ReadUInt32();
            light.Unknown6 = reader.ReadUInt32();
            
            PointLights.Add(light);
        }
    }
    
    public void Write(BinaryWriter writer)
    {
        foreach (PointLight light in PointLights)
        {
            writer.Write(light.Unknown1);
            writer.Write(light.Color.ToUInt32());
            writer.Write(light.Position.X);
            writer.Write(light.Position.Y);
            writer.Write(light.Position.Z);
            writer.Write(light.Intensity);
            writer.Write(light.AttenuationStart);
            writer.Write(light.AttenuationEnd);
            writer.Write(light.Unknown4);
            writer.Write(light.Unknown5);
            writer.Write(light.Unknown6);
        }
    }
}

public class PointLight
{
    public uint Unknown1 { get; set; }
    public Color Color { get; set; }
    public Vector3 Position { get; set; }
    public float Intensity { get; set; }
    public float AttenuationStart { get; set; }
    public float AttenuationEnd { get; set; }
    public float Unknown4 { get; set; }
    public uint Unknown5 { get; set; }
    public uint Unknown6 { get; set; }
}
```

## Validation Requirements
- The number of point lights should match expectations from MOHD
- Each light's intensity must be non-negative
- Attenuation distances must be valid:
  - AttenuationStart must be non-negative
  - AttenuationEnd must be greater than AttenuationStart
- Position values should be within reasonable bounds of the WMO
- Color values should be valid BGRA format

## Usage Context
Point lights in WMOs serve several purposes:

1. **Local Illumination**: Provide localized lighting for specific areas within the WMO
2. **Ambient Enhancement**: Add depth and atmosphere to indoor and outdoor spaces
3. **Visual Indicators**: Mark important locations or create visual interest
4. **Dynamic Effects**: Can be used for flickering or pulsing light effects

Common uses include:
- Torch and lantern lighting
- Window light sources
- Magical effects
- Architectural lighting
- Ambient fill lighting

The lighting system works by:
1. Each point light defines a sphere of influence (based on attenuation)
2. Light intensity falls off linearly within the attenuation range
3. Multiple lights can affect the same area, their effects are combined
4. Colors and intensities can be used to create different moods or effects

Point lights are particularly important for:
- Indoor spaces where global illumination isn't sufficient
- Creating atmosphere and depth in dark areas
- Highlighting architectural features or points of interest
- Providing visual cues to players 