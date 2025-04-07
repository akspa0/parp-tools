# LITE - MDX Lights Chunk

## Type
MDX Main Chunk

## Source
MDX_index.md

## Description
The LITE (Lights) chunk defines all the light sources in the model. Each light provides illumination for the model and surrounding environment, with properties such as color, intensity, attenuation, and animation. Lights can be static or animated, and may be attached to specific nodes in the model hierarchy.

## Structure

```csharp
public struct LITE
{
    /// <summary>
    /// Array of light definitions
    /// </summary>
    // MDLLIGHT lights[numLights] follows
}

public struct MDLLIGHT : MDLGENOBJECT
{
    /// <summary>
    /// Light type (point, spot, etc.)
    /// </summary>
    public uint type;
    
    /// <summary>
    /// Attenuation start distance
    /// </summary>
    public float attenuationStart;
    
    /// <summary>
    /// Attenuation end distance
    /// </summary>
    public float attenuationEnd;
    
    /// <summary>
    /// Color of the light (RGB)
    /// </summary>
    public Vector3 color;
    
    /// <summary>
    /// Intensity of the light
    /// </summary>
    public float intensity;
    
    /// <summary>
    /// Ambient color (RGB)
    /// </summary>
    public Vector3 ambColor;
    
    /// <summary>
    /// Ambient intensity
    /// </summary>
    public float ambIntensity;
    
    /// <summary>
    /// Visibility of the light
    /// </summary>
    public uint visibility;
    
    /// <summary>
    /// Animation data for the light properties
    /// </summary>
    // MDLKEYTRACK animations follow
}
```

## Properties

### MDLLIGHT Structure

| Offset | Name | Type | Description |
|--------|------|------|-------------|
| 0x00..0x58 | MDLGENOBJECT | struct | Base generic object (see MDLGENOBJECT structure) |
| 0x5C | type | uint | Light type (see Light Types) |
| 0x60 | attenuationStart | float | Distance at which light starts to fade |
| 0x64 | attenuationEnd | float | Distance at which light ends completely |
| 0x68 | color | Vector3 | RGB color of the light (0.0-1.0) |
| 0x74 | intensity | float | Brightness of the light |
| 0x78 | ambColor | Vector3 | RGB color of ambient component (0.0-1.0) |
| 0x84 | ambIntensity | float | Brightness of ambient component |
| 0x88 | visibility | uint | Visibility flag (0 = hidden, 1 = visible) |
| 0x8C | ... | ... | Animation tracks follow |

## Light Types

| Value | Name | Description |
|-------|------|-------------|
| 0 | Omni | Omnidirectional point light (radiates in all directions) |
| 1 | Directional | Directional light (parallel rays like sunlight) |
| 2 | Ambient | Ambient light (global illumination with no direction) |

## Animation Tracks
After the base properties, several animation tracks may follow:

- Translation track (Vector3 XYZ)
- Rotation track (Quaternion XYZW)
- Scaling track (Vector3 XYZ)
- Color track (Vector3 RGB)
- Intensity track (float)
- Attenuation start track (float)
- Attenuation end track (float)
- Ambient color track (Vector3 RGB)
- Ambient intensity track (float)
- Visibility track (int, 0 or 1)

## Version Differences

| Version | Changes |
|---------|---------|
| 800-1000 (WC3) | Base structure as described |
| 1300-1500 (WoW Alpha) | Same structure with additional support for animated lights |

## Dependencies
- MDLGENOBJECT - All lights inherit from the generic object structure
- MDLKEYTRACK - Used for animation tracks within the structure
- BONE - Lights can be attached to bones in the model hierarchy via the parentId

## Implementation Notes
- Lights use the MDLGENOBJECT structure for base properties (name, ID, parent, flags)
- The visibility field can be animated to show/hide lights during specific animations
- Light colors and intensities can be animated for effects like pulsing or flickering
- The attenuation distances define a falloff range for the light's influence
- For directional lights, the translation tracks define the light's direction vector
- For Warcraft 3 models, lights are typically used for localized effects like glowing weapons
- In WoW Alpha models, more complex light setups might be used for environmental effects

## Usage Context
Lights in MDX models serve several purposes:
- Adding highlights or glows to specific parts of a model
- Creating ambient illumination for the model
- Simulating effects like fire, magic, or energy sources
- Enhancing visibility of important model parts
- Creating atmospheric effects through color and intensity

## Implementation Example

```csharp
public class LITEChunk : IMdxChunk
{
    public string ChunkId => "LITE";
    
    public List<MdxLight> Lights { get; private set; } = new List<MdxLight>();
    
    public void Parse(BinaryReader reader, long totalSize)
    {
        long startPosition = reader.BaseStream.Position;
        long endPosition = startPosition + totalSize;
        
        // Clear any existing lights
        Lights.Clear();
        
        // Read lights until we reach the end of the chunk
        while (reader.BaseStream.Position < endPosition)
        {
            var light = new MdxLight();
            
            // Read base object properties
            light.ParseBaseObject(reader);
            
            // Read light specific properties
            light.Type = reader.ReadUInt32();
            light.AttenuationStart = reader.ReadSingle();
            light.AttenuationEnd = reader.ReadSingle();
            
            // Read color and intensity
            light.Color = new Vector3(
                reader.ReadSingle(),
                reader.ReadSingle(),
                reader.ReadSingle()
            );
            light.Intensity = reader.ReadSingle();
            
            // Read ambient color and intensity
            light.AmbientColor = new Vector3(
                reader.ReadSingle(),
                reader.ReadSingle(),
                reader.ReadSingle()
            );
            light.AmbientIntensity = reader.ReadSingle();
            
            // Read visibility
            light.Visibility = reader.ReadUInt32();
            
            // Read animation tracks
            // Translation
            light.TranslationTrack = new MdxKeyTrack<Vector3>();
            light.TranslationTrack.Parse(reader, r => new Vector3(r.ReadSingle(), r.ReadSingle(), r.ReadSingle()));
            
            // Rotation
            light.RotationTrack = new MdxKeyTrack<Quaternion>();
            light.RotationTrack.Parse(reader, r => new Quaternion(r.ReadSingle(), r.ReadSingle(), r.ReadSingle(), r.ReadSingle()));
            
            // Color
            light.ColorTrack = new MdxKeyTrack<Vector3>();
            light.ColorTrack.Parse(reader, r => new Vector3(r.ReadSingle(), r.ReadSingle(), r.ReadSingle()));
            
            // Intensity
            light.IntensityTrack = new MdxKeyTrack<float>();
            light.IntensityTrack.Parse(reader, r => r.ReadSingle());
            
            // Attenuation start
            light.AttenuationStartTrack = new MdxKeyTrack<float>();
            light.AttenuationStartTrack.Parse(reader, r => r.ReadSingle());
            
            // Attenuation end
            light.AttenuationEndTrack = new MdxKeyTrack<float>();
            light.AttenuationEndTrack.Parse(reader, r => r.ReadSingle());
            
            // Ambient color
            light.AmbientColorTrack = new MdxKeyTrack<Vector3>();
            light.AmbientColorTrack.Parse(reader, r => new Vector3(r.ReadSingle(), r.ReadSingle(), r.ReadSingle()));
            
            // Ambient intensity
            light.AmbientIntensityTrack = new MdxKeyTrack<float>();
            light.AmbientIntensityTrack.Parse(reader, r => r.ReadSingle());
            
            // Visibility
            light.VisibilityTrack = new MdxKeyTrack<int>();
            light.VisibilityTrack.Parse(reader, r => r.ReadInt32());
            
            Lights.Add(light);
        }
    }
    
    public void Write(BinaryWriter writer)
    {
        foreach (var light in Lights)
        {
            // Write base object properties
            light.WriteBaseObject(writer);
            
            // Write light specific properties
            writer.Write(light.Type);
            writer.Write(light.AttenuationStart);
            writer.Write(light.AttenuationEnd);
            
            // Write color and intensity
            writer.Write(light.Color.X);
            writer.Write(light.Color.Y);
            writer.Write(light.Color.Z);
            writer.Write(light.Intensity);
            
            // Write ambient color and intensity
            writer.Write(light.AmbientColor.X);
            writer.Write(light.AmbientColor.Y);
            writer.Write(light.AmbientColor.Z);
            writer.Write(light.AmbientIntensity);
            
            // Write visibility
            writer.Write(light.Visibility);
            
            // Write animation tracks
            light.TranslationTrack.Write(writer, (w, v) => { w.Write(v.X); w.Write(v.Y); w.Write(v.Z); });
            light.RotationTrack.Write(writer, (w, q) => { w.Write(q.X); w.Write(q.Y); w.Write(q.Z); w.Write(q.W); });
            light.ColorTrack.Write(writer, (w, v) => { w.Write(v.X); w.Write(v.Y); w.Write(v.Z); });
            light.IntensityTrack.Write(writer, (w, f) => w.Write(f));
            light.AttenuationStartTrack.Write(writer, (w, f) => w.Write(f));
            light.AttenuationEndTrack.Write(writer, (w, f) => w.Write(f));
            light.AmbientColorTrack.Write(writer, (w, v) => { w.Write(v.X); w.Write(v.Y); w.Write(v.Z); });
            light.AmbientIntensityTrack.Write(writer, (w, f) => w.Write(f));
            light.VisibilityTrack.Write(writer, (w, i) => w.Write(i));
        }
    }
    
    /// <summary>
    /// Gets the current effective lighting parameters for a light
    /// </summary>
    /// <param name="lightIndex">Index of the light</param>
    /// <param name="time">Current animation time in milliseconds</param>
    /// <param name="sequenceDuration">Duration of the current sequence</param>
    /// <param name="globalSequences">Dictionary of global sequence durations</param>
    /// <returns>The effective light parameters, or null if the light is invisible</returns>
    public MdxLightParams GetEffectiveLightParams(int lightIndex, uint time, uint sequenceDuration, Dictionary<uint, uint> globalSequences)
    {
        if (lightIndex < 0 || lightIndex >= Lights.Count)
        {
            return null;
        }
        
        var light = Lights[lightIndex];
        var result = new MdxLightParams();
        
        // Check visibility
        bool isVisible = light.Visibility == 1;
        if (light.VisibilityTrack.NumKeys > 0)
        {
            isVisible = light.VisibilityTrack.Evaluate(time, sequenceDuration, globalSequences) > 0;
        }
        
        if (!isVisible)
        {
            return null;
        }
        
        // Get current values for all animated properties
        result.Type = (LightType)light.Type;
        
        result.Color = light.Color;
        if (light.ColorTrack.NumKeys > 0)
        {
            result.Color = light.ColorTrack.Evaluate(time, sequenceDuration, globalSequences);
        }
        
        result.Intensity = light.Intensity;
        if (light.IntensityTrack.NumKeys > 0)
        {
            result.Intensity = light.IntensityTrack.Evaluate(time, sequenceDuration, globalSequences);
        }
        
        result.AttenuationStart = light.AttenuationStart;
        if (light.AttenuationStartTrack.NumKeys > 0)
        {
            result.AttenuationStart = light.AttenuationStartTrack.Evaluate(time, sequenceDuration, globalSequences);
        }
        
        result.AttenuationEnd = light.AttenuationEnd;
        if (light.AttenuationEndTrack.NumKeys > 0)
        {
            result.AttenuationEnd = light.AttenuationEndTrack.Evaluate(time, sequenceDuration, globalSequences);
        }
        
        result.AmbientColor = light.AmbientColor;
        if (light.AmbientColorTrack.NumKeys > 0)
        {
            result.AmbientColor = light.AmbientColorTrack.Evaluate(time, sequenceDuration, globalSequences);
        }
        
        result.AmbientIntensity = light.AmbientIntensity;
        if (light.AmbientIntensityTrack.NumKeys > 0)
        {
            result.AmbientIntensity = light.AmbientIntensityTrack.Evaluate(time, sequenceDuration, globalSequences);
        }
        
        return result;
    }
}

public class MdxLight : MdxGenericObject
{
    public uint Type { get; set; }
    public float AttenuationStart { get; set; }
    public float AttenuationEnd { get; set; }
    public Vector3 Color { get; set; }
    public float Intensity { get; set; }
    public Vector3 AmbientColor { get; set; }
    public float AmbientIntensity { get; set; }
    public uint Visibility { get; set; }
    
    public MdxKeyTrack<Vector3> ColorTrack { get; set; }
    public MdxKeyTrack<float> IntensityTrack { get; set; }
    public MdxKeyTrack<float> AttenuationStartTrack { get; set; }
    public MdxKeyTrack<float> AttenuationEndTrack { get; set; }
    public MdxKeyTrack<Vector3> AmbientColorTrack { get; set; }
    public MdxKeyTrack<float> AmbientIntensityTrack { get; set; }
    public MdxKeyTrack<int> VisibilityTrack { get; set; }
    
    public override void ParseAnimationData(BinaryReader reader, uint version)
    {
        // Base translation, rotation, and scaling tracks are parsed by the base class
        base.ParseAnimationData(reader, version);
        
        // Parse light-specific animation tracks
        // (Implementation would read additional tracks from the reader)
    }
    
    public override void WriteAnimationData(BinaryWriter writer, uint version)
    {
        // Base translation, rotation, and scaling tracks are written by the base class
        base.WriteAnimationData(writer, version);
        
        // Write light-specific animation tracks
        // (Implementation would write additional tracks to the writer)
    }
}

public enum LightType
{
    Omni = 0,
    Directional = 1,
    Ambient = 2
}

public class MdxLightParams
{
    public LightType Type { get; set; }
    public Vector3 Color { get; set; }
    public float Intensity { get; set; }
    public float AttenuationStart { get; set; }
    public float AttenuationEnd { get; set; }
    public Vector3 AmbientColor { get; set; }
    public float AmbientIntensity { get; set; }
} 