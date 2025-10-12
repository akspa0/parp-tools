# O012: MOLT

## Type
WMO Root Chunk

## Source
WMO.md

## Description
The MOLT (Map Object LighTs) chunk defines light sources within the WMO. Each light entry specifies position, color, intensity, and other properties that determine how the light affects the surrounding environment. These lights illuminate the interior and exterior of WMO buildings, creating atmosphere and visual depth through dynamic lighting.

## Structure
```csharp
struct MOLT
{
    SMOLight[] lights;  // Array of light definitions
};

struct SMOLight
{
    /*0x00*/ uint8_t lightType;        // Light type (point, spot, etc.)
    /*0x01*/ uint8_t useAttenuation;   // Whether light attenuates with distance
    /*0x02*/ uint8_t padding;          // Padding byte
    /*0x03*/ uint8_t useUnknown;       // Unknown usage flag
    /*0x04*/ uint32_t color;           // Light color (BGRA)
    /*0x08*/ C3Vector position;        // Position in model space
    /*0x14*/ float intensity;          // Light intensity multiplier
    /*0x18*/ float attenStart;         // Attenuation start distance
    /*0x1C*/ float attenEnd;           // Attenuation end distance
    /*0x20*/ float unk1;               // Unknown float 1
    /*0x24*/ float unk2;               // Unknown float 2
    /*0x28*/ float unk3;               // Unknown float 3
    /*0x2C*/ float unk4;               // Unknown float 4
};
```

## Properties

### SMOLight Structure
| Offset | Name | Type | Description |
|--------|------|------|-------------|
| 0x00 | lightType | uint8_t | Type of light (see Light Types table) |
| 0x01 | useAttenuation | uint8_t | Whether light uses distance-based attenuation (0 or 1) |
| 0x02 | padding | uint8_t | Padding byte (unused) |
| 0x03 | useUnknown | uint8_t | Unknown usage flag (possible shadow control) |
| 0x04 | color | uint32_t | Light color in BGRA format |
| 0x08 | position | C3Vector | Position of the light in model space |
| 0x14 | intensity | float | Light intensity/brightness multiplier |
| 0x18 | attenStart | float | Distance at which attenuation begins |
| 0x1C | attenEnd | float | Distance at which attenuation ends (light reaches zero) |
| 0x20 | unk1 | float | Unknown float (for spot lights: inner angle) |
| 0x24 | unk2 | float | Unknown float (for spot lights: outer angle) |
| 0x28 | unk3 | float | Unknown float (for spot lights: direction X) |
| 0x2C | unk4 | float | Unknown float (for spot lights: direction Z) |

### Light Types
| Value | Name | Description |
|-------|------|-------------|
| 0 | OMNI | Omnidirectional point light (radiates in all directions) |
| 1 | SPOT | Spot light (cone-shaped light with direction) |
| 2 | DIRECT | Directional light (parallel rays, like sunlight) |
| 3 | AMBIENT | Ambient light (general illumination with no specific source) |

## Dependencies
- MOHD: The nLights field indicates how many light definitions should be present

## Implementation Notes
- Each light definition is 48 bytes (0x30) in size
- The lightType field determines how the light behaves and which fields are relevant
- The useAttenuation flag enables distance-based attenuation (light fades with distance)
- The color is stored in BGRA format (blue in the lowest byte, alpha in the highest)
- For spot lights, unk1 and unk2 are used for inner and outer cone angles
- For spot lights, unk3 and unk4 provide the X and Z components of the direction vector (Y can be derived)
- The intensity field acts as a multiplier for the light's brightness
- The attenStart and attenEnd fields define the distance range over which the light fades
- If useAttenuation is 0, attenStart and attenEnd are ignored
- The number of lights must match the nLights field in the MOHD chunk
- The unknown fields may have different meanings based on light type

## Implementation Example
```csharp
public class MOLT : IChunk
{
    public List<Light> Lights { get; private set; }
    
    public MOLT()
    {
        Lights = new List<Light>();
    }
    
    public void Parse(BinaryReader reader, long size)
    {
        // Calculate how many lights we expect
        int lightCount = (int)(size / 0x30); // Each light is 48 bytes
        
        Lights.Clear();
        
        for (int i = 0; i < lightCount; i++)
        {
            Light light = new Light();
            
            light.Type = (LightType)reader.ReadByte();
            light.UseAttenuation = reader.ReadByte() != 0;
            reader.ReadByte(); // Skip padding byte
            light.UseUnknown = reader.ReadByte() != 0;
            
            light.Color = reader.ReadUInt32();
            
            // Read position vector
            light.Position = new Vector3(
                reader.ReadSingle(),
                reader.ReadSingle(),
                reader.ReadSingle()
            );
            
            light.Intensity = reader.ReadSingle();
            light.AttenuationStart = reader.ReadSingle();
            light.AttenuationEnd = reader.ReadSingle();
            
            // Read unknown floats (may be spot light parameters)
            light.Unknown1 = reader.ReadSingle();
            light.Unknown2 = reader.ReadSingle();
            light.Unknown3 = reader.ReadSingle();
            light.Unknown4 = reader.ReadSingle();
            
            Lights.Add(light);
        }
    }
    
    public void Write(BinaryWriter writer)
    {
        foreach (Light light in Lights)
        {
            writer.Write((byte)light.Type);
            writer.Write((byte)(light.UseAttenuation ? 1 : 0));
            writer.Write((byte)0); // Padding
            writer.Write((byte)(light.UseUnknown ? 1 : 0));
            
            writer.Write(light.Color);
            
            // Write position vector
            writer.Write(light.Position.X);
            writer.Write(light.Position.Y);
            writer.Write(light.Position.Z);
            
            writer.Write(light.Intensity);
            writer.Write(light.AttenuationStart);
            writer.Write(light.AttenuationEnd);
            
            // Write unknown floats
            writer.Write(light.Unknown1);
            writer.Write(light.Unknown2);
            writer.Write(light.Unknown3);
            writer.Write(light.Unknown4);
        }
    }
    
    public Light GetLight(int index)
    {
        if (index >= 0 && index < Lights.Count)
        {
            return Lights[index];
        }
        
        throw new IndexOutOfRangeException($"Light index {index} is out of range. Valid range: 0-{Lights.Count - 1}");
    }
    
    public void AddLight(Light light)
    {
        Lights.Add(light);
    }
}

public enum LightType : byte
{
    Omni = 0,      // Point light
    Spot = 1,      // Spot light
    Direct = 2,    // Directional light
    Ambient = 3    // Ambient light
}

public class Light
{
    public LightType Type { get; set; }
    public bool UseAttenuation { get; set; }
    public bool UseUnknown { get; set; }
    public uint Color { get; set; }
    public Vector3 Position { get; set; }
    public float Intensity { get; set; }
    public float AttenuationStart { get; set; }
    public float AttenuationEnd { get; set; }
    public float Unknown1 { get; set; } // For spot: inner angle
    public float Unknown2 { get; set; } // For spot: outer angle
    public float Unknown3 { get; set; } // For spot: direction X
    public float Unknown4 { get; set; } // For spot: direction Z
    
    // Helper for color conversion
    public Color GetRGBAColor()
    {
        return new Color(
            (byte)((Color >> 16) & 0xFF),  // R
            (byte)((Color >> 8) & 0xFF),   // G
            (byte)(Color & 0xFF),          // B
            (byte)((Color >> 24) & 0xFF)   // A
        );
    }
    
    public void SetRGBAColor(Color color)
    {
        Color = (uint)(
            (color.B) |
            (color.G << 8) |
            (color.R << 16) |
            (color.A << 24)
        );
    }
    
    // Helper for spot light direction
    public Vector3 GetSpotDirection()
    {
        if (Type != LightType.Spot)
        {
            return Vector3.Zero;
        }
        
        // For spot lights, compute the Y component from X and Z
        // Direction is a normalized vector, so X^2 + Y^2 + Z^2 = 1
        float x = Unknown3;
        float z = Unknown4;
        float y = (float)Math.Sqrt(1.0f - (x * x + z * z));
        
        return new Vector3(x, y, z);
    }
    
    public void SetSpotDirection(Vector3 direction)
    {
        if (Type != LightType.Spot)
        {
            return;
        }
        
        // Normalize the direction vector
        Vector3 normalized = Vector3.Normalize(direction);
        
        // Store X and Z components
        Unknown3 = normalized.X;
        Unknown4 = normalized.Z;
    }
    
    public Light()
    {
        // Initialize with defaults
        Type = LightType.Omni;
        UseAttenuation = true;
        UseUnknown = false;
        Color = 0xFFFFFFFF; // White, fully opaque
        Position = new Vector3(0, 0, 0);
        Intensity = 1.0f;
        AttenuationStart = 0.0f;
        AttenuationEnd = 20.0f;
        Unknown1 = 0.0f;
        Unknown2 = 0.0f;
        Unknown3 = 0.0f;
        Unknown4 = 0.0f;
    }
}
```

## Validation Requirements
- The number of light definitions should match the nLights field in the MOHD chunk
- The lightType value should be within the valid range (0-3)
- The intensity value should be positive
- If useAttenuation is true, attenEnd should be greater than attenStart
- The position coordinates should be within the overall bounding box of the WMO
- For spot lights, the direction components should form a normalized vector

## Usage Context
The MOLT chunk provides lighting information that dramatically affects the visual appearance of WMOs:

1. **Atmospheric Lighting**: Creates mood and atmosphere within WMO interiors
2. **Dynamic Illumination**: Illuminates surfaces based on light position and properties
3. **Visual Guidance**: Draws player attention to important areas or paths
4. **World Integration**: Connects the WMO to the broader world's lighting and time of day
5. **Special Effects**: Can be used for magical effects, fire, or other special lighting

Common light usage patterns include:
- **Torch/Lantern Lights**: Warm, flickering point lights attached to wall fixtures
- **Window Lights**: Directional lights simulating sunlight through windows
- **Ambient Interior**: Soft ambient lighting to ensure interiors aren't too dark
- **Ceiling Fixtures**: Omnidirectional lights hanging from ceilings
- **Spotlights**: Focused lighting to highlight specific features or areas

When rendering a WMO, the client:
1. Processes each light defined in this chunk
2. Calculates how each light affects surfaces based on position, type, and properties
3. Applies lighting calculations to vertices and/or fragments during rendering
4. Combines multiple light contributions for final illumination

The lighting system in WMOs allows for complex, realistic lighting scenarios that enhance the immersive quality of the game's environments. 