# C017: MTCG

## Type
ADT Chunk

## Source
ADT_v18.md

## Description
Map Texture Color Gradient - contains color gradient information for terrain textures. This chunk was introduced in Shadowlands and provides a way to apply color variations across terrain textures for better visual blending and environmental effects.

## Structure
```csharp
struct MTCG
{
    /*0x00*/ uint32_t numEntries;              // Number of gradient entries
    /*0x04*/ ColorGradientEntry entries[];     // Color gradient entries
}

struct ColorGradientEntry
{
    /*0x00*/ uint32_t textureId;               // Index of the texture in MTEX or FileDataID in MDID
    /*0x04*/ uint32_t gradientFlags;           // Flags controlling the gradient behavior
    /*0x08*/ C4Vector startColor;              // Starting color (RGBA, each component 0.0-1.0)
    /*0x18*/ C4Vector endColor;                // Ending color (RGBA, each component 0.0-1.0)
    /*0x28*/ float startHeight;                // Starting height for the gradient
    /*0x2C*/ float endHeight;                  // Ending height for the gradient
    /*0x30*/ float startAngle;                 // Starting angle for the gradient (radians)
    /*0x34*/ float endAngle;                   // Ending angle for the gradient (radians)
}
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| numEntries | uint32 | Number of color gradient entries |
| entries | ColorGradientEntry[] | Array of color gradient entries |
| - textureId | uint32 | Index of the texture in MTEX or FileDataID in MDID |
| - gradientFlags | uint32 | Flags controlling the gradient behavior |
| - startColor | C4Vector | Starting color (RGBA, each component 0.0-1.0) |
| - endColor | C4Vector | Ending color (RGBA, each component 0.0-1.0) |
| - startHeight | float | Starting height for the gradient |
| - endHeight | float | Ending height for the gradient |
| - startAngle | float | Starting angle for the gradient (radians) |
| - endAngle | float | Ending angle for the gradient (radians) |

## Gradient Flag Values
| Value | Name | Description |
|-------|------|-------------|
| 0x1 | GradientFlag_UseHeight | Use height for gradient calculation |
| 0x2 | GradientFlag_UseAngle | Use angle for gradient calculation |
| 0x4 | GradientFlag_BlendMultiply | Use multiplicative blending instead of additive |
| 0x8 | GradientFlag_AffectDiffuse | Apply gradient to diffuse color |
| 0x10 | GradientFlag_AffectSpecular | Apply gradient to specular component |
| 0x20 | GradientFlag_AffectEmissive | Apply gradient to emissive component |

## Dependencies
- MTEX (C004) - Contains the texture filenames that correspond to these gradients (pre-8.1.0)
- MDID (C016) - Contains the texture FileDataIDs that correspond to these gradients (post-8.1.0)

## Implementation Notes
- The MTCG chunk provides a way to apply color variations to terrain textures
- Each entry applies to a specific texture referenced either by index or FileDataID
- The gradient can be based on height, angle, or both, as specified by the flags
- Height-based gradients are useful for snow caps, shorelines, etc.
- Angle-based gradients are useful for simulating different lighting conditions based on terrain orientation
- The gradient colors are interpolated between startColor and endColor based on the specified parameters
- If both height and angle flags are set, the results are combined
- The gradient effect is applied in the fragment/pixel shader when rendering the terrain
- This feature allows for more dynamic terrain appearance without requiring additional texture assets

## Implementation Example
```csharp
public class MTCG : IChunk
{
    [Flags]
    public enum GradientFlags : uint
    {
        None = 0,
        UseHeight = 0x1,
        UseAngle = 0x2,
        BlendMultiply = 0x4,
        AffectDiffuse = 0x8,
        AffectSpecular = 0x10,
        AffectEmissive = 0x20
    }
    
    public class ColorGradientEntry
    {
        public uint TextureId { get; set; }
        public GradientFlags Flags { get; set; }
        public Vector4 StartColor { get; set; }
        public Vector4 EndColor { get; set; }
        public float StartHeight { get; set; }
        public float EndHeight { get; set; }
        public float StartAngle { get; set; }
        public float EndAngle { get; set; }
        
        public ColorGradientEntry(BinaryReader reader)
        {
            TextureId = reader.ReadUInt32();
            Flags = (GradientFlags)reader.ReadUInt32();
            
            // Read start color (RGBA)
            float startR = reader.ReadSingle();
            float startG = reader.ReadSingle();
            float startB = reader.ReadSingle();
            float startA = reader.ReadSingle();
            StartColor = new Vector4(startR, startG, startB, startA);
            
            // Read end color (RGBA)
            float endR = reader.ReadSingle();
            float endG = reader.ReadSingle();
            float endB = reader.ReadSingle();
            float endA = reader.ReadSingle();
            EndColor = new Vector4(endR, endG, endB, endA);
            
            StartHeight = reader.ReadSingle();
            EndHeight = reader.ReadSingle();
            StartAngle = reader.ReadSingle();
            EndAngle = reader.ReadSingle();
        }
        
        // Calculate the blend factor based on height and angle
        public float CalculateBlendFactor(float height, float angle)
        {
            float heightFactor = 0.0f;
            float angleFactor = 0.0f;
            bool useHeight = (Flags & GradientFlags.UseHeight) != 0;
            bool useAngle = (Flags & GradientFlags.UseAngle) != 0;
            
            // Calculate height factor if needed
            if (useHeight)
            {
                if (EndHeight > StartHeight)
                {
                    heightFactor = (height - StartHeight) / (EndHeight - StartHeight);
                    heightFactor = Math.Clamp(heightFactor, 0.0f, 1.0f);
                }
                else
                {
                    heightFactor = 1.0f;
                }
            }
            
            // Calculate angle factor if needed
            if (useAngle)
            {
                // Normalize angle to [0, 2π]
                angle = angle % (2.0f * (float)Math.PI);
                if (angle < 0) angle += 2.0f * (float)Math.PI;
                
                // Check if angle is in the range [startAngle, endAngle]
                if (EndAngle > StartAngle)
                {
                    if (angle >= StartAngle && angle <= EndAngle)
                    {
                        angleFactor = (angle - StartAngle) / (EndAngle - StartAngle);
                    }
                }
                else // Handle wrap-around case
                {
                    if (angle >= StartAngle || angle <= EndAngle)
                    {
                        float angleRange = 2.0f * (float)Math.PI - StartAngle + EndAngle;
                        if (angle >= StartAngle)
                        {
                            angleFactor = (angle - StartAngle) / angleRange;
                        }
                        else
                        {
                            angleFactor = (2.0f * (float)Math.PI - StartAngle + angle) / angleRange;
                        }
                    }
                }
                
                angleFactor = Math.Clamp(angleFactor, 0.0f, 1.0f);
            }
            
            // Combine factors
            if (useHeight && useAngle)
            {
                return (heightFactor + angleFactor) / 2.0f;
            }
            else if (useHeight)
            {
                return heightFactor;
            }
            else if (useAngle)
            {
                return angleFactor;
            }
            
            return 0.0f;
        }
        
        // Get the color based on height and angle
        public Vector4 GetColor(float height, float angle)
        {
            float factor = CalculateBlendFactor(height, angle);
            
            // Interpolate between start and end colors
            return Vector4.Lerp(StartColor, EndColor, factor);
        }
    }
    
    public List<ColorGradientEntry> GradientEntries { get; private set; }
    
    public MTCG(BinaryReader reader, uint size)
    {
        uint numEntries = reader.ReadUInt32();
        GradientEntries = new List<ColorGradientEntry>((int)numEntries);
        
        for (int i = 0; i < numEntries; i++)
        {
            GradientEntries.Add(new ColorGradientEntry(reader));
        }
    }
    
    // Find all gradient entries for a specific texture
    public List<ColorGradientEntry> GetEntriesForTexture(uint textureId)
    {
        return GradientEntries.Where(entry => entry.TextureId == textureId).ToList();
    }
}
```

## Usage in Shader
```glsl
// Example shader code applying the gradient to a texture
struct GradientInfo {
    vec4 startColor;
    vec4 endColor;
    float startHeight;
    float endHeight;
    float startAngle;
    float endAngle;
    uint flags;
};

uniform sampler2D u_texture;
uniform GradientInfo u_gradient;

vec4 ApplyGradient(vec4 texColor, float height, float angle)
{
    float factor = 0.0;
    bool useHeight = (u_gradient.flags & 0x1) != 0;
    bool useAngle = (u_gradient.flags & 0x2) != 0;
    bool blendMultiply = (u_gradient.flags & 0x4) != 0;
    
    // Calculate blend factor
    if (useHeight && useAngle) {
        float heightFactor = clamp((height - u_gradient.startHeight) / 
                                  (u_gradient.endHeight - u_gradient.startHeight), 0.0, 1.0);
        
        float angleRange = u_gradient.endAngle - u_gradient.startAngle;
        if (angleRange < 0.0) angleRange += 2.0 * 3.14159;
        float angleFactor = clamp((angle - u_gradient.startAngle) / angleRange, 0.0, 1.0);
        
        factor = (heightFactor + angleFactor) / 2.0;
    } else if (useHeight) {
        factor = clamp((height - u_gradient.startHeight) / 
                      (u_gradient.endHeight - u_gradient.startHeight), 0.0, 1.0);
    } else if (useAngle) {
        float angleRange = u_gradient.endAngle - u_gradient.startAngle;
        if (angleRange < 0.0) angleRange += 2.0 * 3.14159;
        factor = clamp((angle - u_gradient.startAngle) / angleRange, 0.0, 1.0);
    }
    
    // Interpolate between start and end colors
    vec4 gradientColor = mix(u_gradient.startColor, u_gradient.endColor, factor);
    
    // Apply gradient color to texture
    if (blendMultiply) {
        return texColor * gradientColor;
    } else {
        return texColor + gradientColor;
    }
}
```

## Usage Context
The MTCG chunk is used to create more dynamic and visually interesting terrain by applying color variations based on height and angle:

1. Snow caps on mountains (white gradient based on height)
2. Shoreline transitions (blue gradient based on height)
3. Moss or vegetation growth on north-facing slopes (green gradient based on angle)
4. Sunlit areas on south-facing slopes (yellow/bright gradient based on angle)
5. Fire/heat effects on volcanic terrain (red/orange gradient based on height and emissive properties)

These gradients enhance the visual quality of the terrain without requiring additional texture assets or manual texture painting, allowing for more realistic and varied environments. The chunk was introduced in Shadowlands as part of the graphical enhancements to make the terrain in the new zones more visually impressive. 