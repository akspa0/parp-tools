# C016: MAMP

## Type
ADT Chunk

## Source
ADT_v18.md

## Description
Map Amplify - contains an amplification factor for texture coordinates used in terrain rendering. This chunk was introduced in Cataclysm and helps improve the visual quality of terrain textures by controlling their scaling.

## Structure
```csharp
struct MAMP
{
    /*0x00*/ float amplifyFactor;    // Amplification factor for texture coordinates
}
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| amplifyFactor | float | Amplification factor for texture coordinates (default: 1.0) |

## Dependencies
- MHDR (C002) - The MHDR.mamp_value field can provide a default value if this chunk is missing

## Implementation Notes
- The MAMP chunk contains a single float value that scales texture coordinates
- If this chunk is missing, the client uses the value from MHDR.mamp_value instead
- Values greater than 1.0 make textures appear smaller/more detailed
- Values less than 1.0 make textures appear larger/less detailed
- This value affects all textures in the ADT tile
- The amplify factor is applied to texture coordinates before sampling textures
- This chunk provides a way to globally adjust texture detail without modifying individual textures
- Typical values range from 0.5 to 2.0, with 1.0 being the default/no-change value

## Implementation Example
```csharp
public class MAMP : IChunk
{
    public float AmplifyFactor { get; private set; }
    
    public MAMP(BinaryReader reader, uint size)
    {
        // Read the amplification factor
        AmplifyFactor = reader.ReadSingle();
    }
    
    // Apply the amplification factor to texture coordinates
    public Vector2 ApplyToTexCoord(Vector2 texCoord)
    {
        return texCoord * AmplifyFactor;
    }
    
    // Get the actual amplification factor, considering default value if necessary
    public static float GetEffectiveAmplifyFactor(MAMP mampChunk, MHDR mhdrChunk)
    {
        // If MAMP chunk exists, use its value
        if (mampChunk != null)
            return mampChunk.AmplifyFactor;
            
        // Otherwise, use the value from MHDR if available
        if (mhdrChunk != null)
            return mhdrChunk.MampValue;
            
        // Default value if neither is available
        return 1.0f;
    }
}
```

## Usage in Shader
```glsl
// Example shader code using the amplify factor
uniform float u_amplifyFactor;

vec4 SampleTerrainTexture(sampler2D tex, vec2 texCoord)
{
    // Apply amplification to texture coordinates
    vec2 amplifiedTexCoord = texCoord * u_amplifyFactor;
    
    // Sample the texture with amplified coordinates
    return texture(tex, amplifiedTexCoord);
}
```

## Usage Context
The MAMP chunk is used to control the overall texture detail level for an ADT tile, which helps:

1. Improve visual quality in areas where players spend more time
2. Reduce texture tiling artifacts by adjusting repetition frequency
3. Create visual consistency across different terrain types
4. Optimize performance by adjusting texture detail based on area importance

This parameter is particularly useful for:
- Hero zones where higher texture detail is desired
- Large, open areas where texture tiling would be more noticeable
- Transitional areas between different terrain types
- Optimizing visual quality vs. performance across different expansion areas

The MAMP chunk was added in Cataclysm as part of the graphical enhancements that came with the world revamp, allowing for more control over terrain texture appearance without needing to create new texture assets. 