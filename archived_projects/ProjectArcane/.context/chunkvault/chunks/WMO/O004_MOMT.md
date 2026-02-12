# O004: MOMT

## Type
WMO Root Chunk

## Source
WMO.md

## Description
The MOMT (Map Object MaTerial) chunk defines the materials used in the WMO. Each material specifies the textures to use, blending modes, shader options, and other rendering properties. These materials are referenced by group geometry to determine how surfaces are rendered in the game.

## Structure
```csharp
struct MOMT
{
    Material[] materials;  // Array of material structures
};

struct Material
{
    /*0x00*/ uint32_t flags;          // Material flags
    /*0x04*/ uint32_t shader;         // Shader ID
    /*0x08*/ uint32_t blendMode;      // Blending mode / render flags
    /*0x0C*/ uint32_t texture1;       // Offset to texture 1 filename in MOTX
    /*0x10*/ uint32_t color1;         // Color 1 (BGRA)
    /*0x14*/ uint32_t flags1;         // Texture flags 1
    /*0x18*/ uint32_t texture2;       // Offset to texture 2 filename in MOTX
    /*0x1C*/ uint32_t color2;         // Color 2 (BGRA)
    /*0x20*/ uint32_t flags2;         // Texture flags 2
    /*0x24*/ uint32_t texture3;       // Offset to texture 3 filename in MOTX
    /*0x28*/ uint32_t color3;         // Color 3 (BGRA)
    /*0x2C*/ uint32_t flags3;         // Texture flags 3
    /*0x30*/ float    runTimeData[4]; // Runtime data (shininess, etc.)
    /*0x40*/ uint32_t diffColor;      // Diffuse color (BGRA) (extended versions only)
    /*0x44*/ uint32_t groundType;     // Ground type (extended versions only)
};
```

## Properties

### Material Structure
| Offset | Name | Type | Description |
|--------|------|------|-------------|
| 0x00 | flags | uint32_t | Material flags (see table below) |
| 0x04 | shader | uint32_t | Shader ID (determines rendering technique) |
| 0x08 | blendMode | uint32_t | Blending mode and render flags |
| 0x0C | texture1 | uint32_t | Offset to first texture's filename in MOTX chunk |
| 0x10 | color1 | uint32_t | First color value (BGRA) |
| 0x14 | flags1 | uint32_t | First texture's flags (see texture flags table) |
| 0x18 | texture2 | uint32_t | Offset to second texture's filename in MOTX chunk |
| 0x1C | color2 | uint32_t | Second color value (BGRA) |
| 0x20 | flags2 | uint32_t | Second texture's flags |
| 0x24 | texture3 | uint32_t | Offset to third texture's filename in MOTX chunk |
| 0x28 | color3 | uint32_t | Third color value (BGRA) |
| 0x2C | flags3 | uint32_t | Third texture's flags |
| 0x30 | runTimeData | float[4] | Runtime data (specular, emissive properties) |
| 0x40 | diffColor | uint32_t | Diffuse color (BGRA) - Extended versions only |
| 0x44 | groundType | uint32_t | Ground type ID - Extended versions only |

### Material Flags
| Flag Value | Name | Description |
|------------|------|-------------|
| 0x01 | UNLIT | Material is unaffected by lighting |
| 0x02 | UNFOGGED | Material is unaffected by fog |
| 0x04 | UNCULLED | Two-sided material (disables backface culling) |
| 0x08 | DEPTH_WRITE | Material writes to depth buffer |
| 0x10 | UNUSED | Unused flag |
| 0x20 | UNUSED | Unused flag |
| 0x40 | UNUSED | Unused flag |
| 0x80 | UNUSED | Unused flag |

### Blend Modes
| Value | Name | Description |
|-------|------|-------------|
| 0 | BM_OPAQUE | Opaque blending (no transparency) |
| 1 | BM_TRANSPARENT | Alpha-blended transparency |
| 2 | BM_ALPHA_BLEND | Soft alpha blending |
| 3 | BM_ADDITIVE | Additive blending (adds source to destination) |
| 4 | BM_ADDITIVE_ALPHA | Additive blending with alpha |
| 5 | BM_MODULATE | Modulate blending (multiplies source with destination) |
| 6 | BM_MODULATE2X | Modulate blending with 2x multiplier |

### Shader IDs
| Value | Name | Description |
|-------|------|-------------|
| 0 | SHADER_DIFFUSE | Basic diffuse shader |
| 1 | SHADER_SPECULAR | Specular shader |
| 2 | SHADER_METAL | Metal shader (specular with environment mapping) |
| 3 | SHADER_ENV | Environment mapped shader |
| 4 | SHADER_OPAQUE | Opaque shader with cutout (typically for foliage) |
| 5 | SHADER_ENVIRONMENT_METAL | Environment mapped metal |
| 6 | SHADER_TRANSPARENT_METAL | Transparent metal shader |
| 7 | SHADER_TRANSPARENT_ENV | Transparent environment mapping |
| 8 | SHADER_TRANSPARENT | Basic transparent shader |
| 9 | SHADER_TRANSPARENT_SPECULAR | Transparent specular shader |
| 10 | SHADER_TRANSPARENT_EMISSIVE | Transparent emissive shader |
| 11 | SHADER_MCS1 | MCS1 shader (Mists of Pandaria+) |
| 12 | SHADER_MCS2 | MCS2 shader (Mists of Pandaria+) |
| 13 | SHADER_MCS3 | MCS3 shader (Mists of Pandaria+) |
| 14 | SHADER_MCS4 | MCS4 shader (Mists of Pandaria+) |

### Texture Flags
| Flag Value | Name | Description |
|------------|------|-------------|
| 0x01 | TEXTURE_WRAP_X | Texture wraps in X direction (horizontal) |
| 0x02 | TEXTURE_WRAP_Y | Texture wraps in Y direction (vertical) |

## Dependencies
- MOHD: The nTextures field indicates how many distinct textures are used
- MOTX: Contains the texture filenames referenced by offset in material definitions

## Implementation Notes
- The structure size may vary based on the WMO version; older versions may not have the diffColor and groundType fields
- Texture offsets point to the start of null-terminated filenames in the MOTX chunk
- An offset of 0 in a texture field indicates no texture for that stage
- The color values are stored in BGRA format (blue in least significant byte)
- The runTimeData array typically contains shininess and emissive material properties
- The groundType field is used for footstep sounds and similar effects
- WMOs in later expansions may have extended material structures not documented here

## Implementation Example
```csharp
public class MOMT : IChunk
{
    public List<Material> Materials { get; private set; }
    
    public MOMT()
    {
        Materials = new List<Material>();
    }
    
    public void Parse(BinaryReader reader, long size)
    {
        // Calculate how many materials we expect
        int materialSize = 0x40; // Standard size is 64 bytes
        
        // Check if we have the extended material format (added in later versions)
        if (size % materialSize != 0)
        {
            materialSize = 0x48; // Extended size is 72 bytes
        }
        
        int materialCount = (int)(size / materialSize);
        
        Materials.Clear();
        
        for (int i = 0; i < materialCount; i++)
        {
            Material material = new Material();
            
            material.Flags = reader.ReadUInt32();
            material.Shader = reader.ReadUInt32();
            material.BlendMode = reader.ReadUInt32();
            material.Texture1Offset = reader.ReadUInt32();
            material.Color1 = reader.ReadUInt32();
            material.Texture1Flags = reader.ReadUInt32();
            material.Texture2Offset = reader.ReadUInt32();
            material.Color2 = reader.ReadUInt32();
            material.Texture2Flags = reader.ReadUInt32();
            material.Texture3Offset = reader.ReadUInt32();
            material.Color3 = reader.ReadUInt32();
            material.Texture3Flags = reader.ReadUInt32();
            
            // Read runtime data array
            material.RuntimeData = new float[4];
            for (int j = 0; j < 4; j++)
            {
                material.RuntimeData[j] = reader.ReadSingle();
            }
            
            // Check if we have the extended material format
            if (materialSize == 0x48)
            {
                material.DiffuseColor = reader.ReadUInt32();
                material.GroundType = reader.ReadUInt32();
            }
            else
            {
                material.DiffuseColor = 0xFFFFFFFF; // Default to white
                material.GroundType = 0;
            }
            
            Materials.Add(material);
        }
    }
    
    public void Write(BinaryWriter writer)
    {
        // Determine if we need to write extended material format
        bool extendedFormat = false;
        foreach (Material material in Materials)
        {
            if (material.DiffuseColor != 0xFFFFFFFF || material.GroundType != 0)
            {
                extendedFormat = true;
                break;
            }
        }
        
        foreach (Material material in Materials)
        {
            writer.Write(material.Flags);
            writer.Write(material.Shader);
            writer.Write(material.BlendMode);
            writer.Write(material.Texture1Offset);
            writer.Write(material.Color1);
            writer.Write(material.Texture1Flags);
            writer.Write(material.Texture2Offset);
            writer.Write(material.Color2);
            writer.Write(material.Texture2Flags);
            writer.Write(material.Texture3Offset);
            writer.Write(material.Color3);
            writer.Write(material.Texture3Flags);
            
            // Write runtime data array
            for (int i = 0; i < 4; i++)
            {
                writer.Write(material.RuntimeData[i]);
            }
            
            // Write extended format fields if needed
            if (extendedFormat)
            {
                writer.Write(material.DiffuseColor);
                writer.Write(material.GroundType);
            }
        }
    }
}

public class Material
{
    // Material properties
    public uint Flags { get; set; }
    public uint Shader { get; set; }
    public uint BlendMode { get; set; }
    
    // Texture 1
    public uint Texture1Offset { get; set; }
    public uint Color1 { get; set; }
    public uint Texture1Flags { get; set; }
    
    // Texture 2
    public uint Texture2Offset { get; set; }
    public uint Color2 { get; set; }
    public uint Texture2Flags { get; set; }
    
    // Texture 3
    public uint Texture3Offset { get; set; }
    public uint Color3 { get; set; }
    public uint Texture3Flags { get; set; }
    
    // Additional data
    public float[] RuntimeData { get; set; }
    public uint DiffuseColor { get; set; }
    public uint GroundType { get; set; }
    
    // Helper properties for color conversion
    public Color GetColor1()
    {
        return new Color(
            (byte)((Color1 >> 16) & 0xFF),  // R
            (byte)((Color1 >> 8) & 0xFF),   // G
            (byte)(Color1 & 0xFF),          // B
            (byte)((Color1 >> 24) & 0xFF)   // A
        );
    }
    
    public void SetColor1(Color color)
    {
        Color1 = (uint)(
            (color.B) |
            (color.G << 8) |
            (color.R << 16) |
            (color.A << 24)
        );
    }
    
    // Similar methods for Color2 and Color3...
    
    public Material()
    {
        // Initialize with defaults
        Flags = 0;
        Shader = 0;
        BlendMode = 0;
        Texture1Offset = 0;
        Color1 = 0xFFFFFFFF; // White, fully opaque
        Texture1Flags = 0;
        Texture2Offset = 0;
        Color2 = 0xFFFFFFFF;
        Texture2Flags = 0;
        Texture3Offset = 0;
        Color3 = 0xFFFFFFFF;
        Texture3Flags = 0;
        RuntimeData = new float[4] { 0, 0, 0, 0 };
        DiffuseColor = 0xFFFFFFFF;
        GroundType = 0;
    }
}
```

## Validation Requirements
- The number of materials should match the requirements of the geometry
- Texture offsets must point to valid offsets within the MOTX chunk
- Shader IDs should be valid for the WoW version
- Blend modes should be valid for the WoW version
- If using extended material format, all materials should have the same structure size

## Usage Context
The MOMT chunk defines how surfaces in the WMO are rendered:

1. **Texture Mapping**: Specifies which textures to apply to model surfaces
2. **Visual Properties**: Defines how materials interact with light and the environment
3. **Rendering Effects**: Controls transparency, shading, and special effects
4. **Physical Properties**: Through the groundType field, influences how characters interact with surfaces

Each material can have up to three texture stages, allowing for complex effects like:
- Base diffuse texture for surface color
- Normal maps for detailed surface relief
- Specular maps for controlling shininess
- Environment maps for reflections
- Detail textures for close-up refinement

The shader and blend mode settings determine how these textures interact and are rendered by the game engine. This system allows for a wide range of visual effects, from simple opaque surfaces to complex translucent shaders with dynamic lighting responses. 