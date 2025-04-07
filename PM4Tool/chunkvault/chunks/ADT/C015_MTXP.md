# C015: MTXP

## Type
ADT Chunk

## Source
ADT_v18.md

## Description
Texture Parameters - contains additional rendering parameters for terrain textures referenced in the MTEX chunk. This chunk was introduced in Mists of Pandaria and provides control over advanced shader effects for terrain textures.

## Structure
```csharp
struct MTXP
{
    /*0x00*/ ShaderParameters shaderParams[];    // Array of shader parameters for each texture in MTEX
}

struct ShaderParameters
{
    /*0x00*/ float heightScale;            // Height scaling factor for parallax/displacement mapping
    /*0x04*/ float heightOffset;           // Height offset for parallax/displacement mapping
    /*0x08*/ float unknownA;               // Unknown parameter (possibly related to normal mapping)
    /*0x0C*/ float unknownB;               // Unknown parameter (possibly related to specular mapping)
}
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| shaderParams | ShaderParameters[] | Array of shader parameters, one entry per texture in MTEX |
| - heightScale | float | Height scaling factor for parallax/displacement mapping |
| - heightOffset | float | Height offset for parallax/displacement mapping |
| - unknownA | float | Unknown parameter (possibly related to normal mapping) |
| - unknownB | float | Unknown parameter (possibly related to specular mapping) |

## Dependencies
- MTEX (C004) - Contains the texture filenames that correspond to these parameters
- MTXF (C014) - Contains flags that may affect how these parameters are applied

## Implementation Notes
- The MTXP chunk contains one entry for each texture referenced in the MTEX chunk
- The parameters control advanced shader effects, particularly parallax mapping and displacement mapping
- If the MTXP chunk is missing, the client assumes default parameters for all textures
- The size of this chunk should be 16 bytes (4 floats) multiplied by the number of textures in MTEX
- Parameters are used by fragment/pixel shaders to create more realistic terrain textures
- Height scaling and offset are used with texture height maps (_h.blp files) to create parallax effects
- Some parameters may only be used if certain flags are set in the MTXF chunk
- Parameters are particularly important for realistic rendering of rocky, sandy, or uneven terrain

## Shader Usage
The texture parameters in MTXP are used by terrain shaders to create various visual effects:

```glsl
// Excerpt from a Legion terrain shader showing how these parameters might be used
vec2 GetParallaxOffset(vec2 texCoord, vec3 viewDir)
{
    // Height scale and offset from MTXP
    float heightScale = u_textureParams[textureIndex].x;
    float heightOffset = u_textureParams[textureIndex].y;
    
    // Sample the height map
    float height = texture(u_heightMaps[textureIndex], texCoord).r;
    
    // Apply height scale and offset
    height = height * heightScale + heightOffset;
    
    // Calculate parallax offset
    vec2 parallaxOffset = (height * viewDir.xy) / viewDir.z;
    return parallaxOffset;
}
```

## Implementation Example
```csharp
public class MTXP : IChunk
{
    public struct ShaderParameters
    {
        public float HeightScale;     // Height scaling factor for parallax mapping
        public float HeightOffset;    // Height offset for parallax mapping
        public float UnknownA;        // Unknown parameter A
        public float UnknownB;        // Unknown parameter B
        
        public ShaderParameters(BinaryReader reader)
        {
            HeightScale = reader.ReadSingle();
            HeightOffset = reader.ReadSingle();
            UnknownA = reader.ReadSingle();
            UnknownB = reader.ReadSingle();
        }
    }
    
    public List<ShaderParameters> TextureParameters { get; private set; }
    
    public MTXP(BinaryReader reader, uint size)
    {
        int count = (int)(size / 16); // Each parameters set is 16 bytes (4 floats)
        TextureParameters = new List<ShaderParameters>(count);
        
        for (int i = 0; i < count; i++)
        {
            TextureParameters.Add(new ShaderParameters(reader));
        }
    }
    
    // Helper method to get parameters for a specific texture
    public ShaderParameters GetParameters(int textureIndex)
    {
        if (textureIndex < 0 || textureIndex >= TextureParameters.Count)
            return new ShaderParameters(); // Return default values
            
        return TextureParameters[textureIndex];
    }
    
    // Helper method to apply parameters to a material
    public void ApplyToMaterial(int textureIndex, Material material)
    {
        if (textureIndex < 0 || textureIndex >= TextureParameters.Count)
            return;
            
        var parameters = TextureParameters[textureIndex];
        
        material.SetFloat("_HeightScale", parameters.HeightScale);
        material.SetFloat("_HeightOffset", parameters.HeightOffset);
        material.SetFloat("_ParameterA", parameters.UnknownA);
        material.SetFloat("_ParameterB", parameters.UnknownB);
    }
}
```

## Usage Context
The MTXP chunk is used to enhance the visual quality of terrain textures through advanced rendering techniques:

1. Parallax mapping - Creates the illusion of depth within a texture
2. Displacement mapping - Physically alters the geometry based on texture information
3. Normal mapping - Enhances lighting detail on surfaces
4. Specular mapping - Controls the shininess and highlight properties of surfaces

These parameters are particularly important for:
- Rocky terrain where depth and surface irregularity are important
- Sandy areas where texture depth creates a more realistic appearance
- Snow-covered regions where depth affects how light interacts with the surface
- Any terrain where a flat texture would look unrealistic

The MTXP chunk was added in Mists of Pandaria as part of the graphical improvements to the terrain rendering system, which introduced more advanced shader techniques to enhance the game's visual quality. 