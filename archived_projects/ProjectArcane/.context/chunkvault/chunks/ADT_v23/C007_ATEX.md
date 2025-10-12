# C007: ATEX

## Type
ADT v23 Chunk

## Source
ADT_v23.md

## Description
Texture definitions for ADT v23 format. Contains references to texture filenames used in the terrain tile, enhanced with support for higher resolution textures and advanced shading.

## Structure
```csharp
struct ATEX
{
    struct TextureEntry
    {
        char filename[260];      // BLP texture filename (null-terminated)
        uint32 flags;            // Texture flags (enhanced in WoD)
        uint32 materialType;     // Material type ID (added in WoD)
        float specularMultiplier; // Specular intensity (added in WoD)
    } textures[]; // Array of texture entries
}
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| textures | TextureEntry[] | Array of texture entries used in this terrain tile |

## Dependencies
- AHDR (C004) - References this chunk via offsets
- ALYR (S001) - References these textures by index

## Implementation Notes
- Size: Variable, depends on number of texture entries
- Enhanced in WoD (v23) with:
  - Additional flags for advanced rendering
  - Material type field for physical properties
  - Specular multiplier for improved lighting
- Each texture entry contains a full path to a BLP texture file
- Maximum path length increased to accommodate longer paths
- Supports higher resolution textures (up to 2048×2048)
- Used to determine which textures are applied to the terrain

## Implementation Example
```csharp
[Flags]
public enum ATEXFlags
{
    None = 0,
    HighResolution = 0x1,    // High-resolution texture (2048×2048)
    SpecularEnabled = 0x2,   // Has specular lighting
    TerrainBlended = 0x4,    // Uses terrain blending
    UseParallaxMapping = 0x8, // Uses parallax mapping (added in WoD)
    HasNormalMap = 0x10,     // Has normal map texture (added in WoD)
    IsWater = 0x20,          // Is a water texture (added in WoD)
    IsLava = 0x40            // Is a lava texture (added in WoD)
}

public enum MaterialType
{
    Generic = 0,
    Stone = 1,
    Dirt = 2,
    Grass = 3,
    Metal = 4,
    Wood = 5,
    Water = 6,
    Lava = 7,
    Snow = 8,
    Sand = 9
    // Added in WoD (v23)
}

public class ATEX
{
    public class TextureEntry
    {
        public string Filename { get; set; }
        public ATEXFlags Flags { get; set; }
        public MaterialType MaterialType { get; set; } // Added in WoD
        public float SpecularMultiplier { get; set; }  // Added in WoD
        
        public TextureEntry(string filename, ATEXFlags flags = ATEXFlags.None, 
                           MaterialType materialType = MaterialType.Generic,
                           float specularMultiplier = 1.0f)
        {
            Filename = filename;
            Flags = flags;
            MaterialType = materialType;
            SpecularMultiplier = specularMultiplier;
        }
        
        // Derive normal map filename from base texture
        public string GetNormalMapFilename()
        {
            if (!HasNormalMap)
                return null;
                
            // Normal maps typically have _n suffix before file extension
            int extIndex = Filename.LastIndexOf('.');
            if (extIndex > 0)
            {
                return Filename.Substring(0, extIndex) + "_n" + Filename.Substring(extIndex);
            }
            
            return Filename + "_n";
        }
        
        // Derive specular map filename from base texture
        public string GetSpecularMapFilename()
        {
            if (!UsesSpecular)
                return null;
                
            // Specular maps typically have _s suffix before file extension
            int extIndex = Filename.LastIndexOf('.');
            if (extIndex > 0)
            {
                return Filename.Substring(0, extIndex) + "_s" + Filename.Substring(extIndex);
            }
            
            return Filename + "_s";
        }
        
        // Helper properties
        public bool IsHighResolution => (Flags & ATEXFlags.HighResolution) != 0;
        public bool UsesSpecular => (Flags & ATEXFlags.SpecularEnabled) != 0;
        public bool UsesTerrainBlending => (Flags & ATEXFlags.TerrainBlended) != 0;
        public bool UsesParallaxMapping => (Flags & ATEXFlags.UseParallaxMapping) != 0;
        public bool HasNormalMap => (Flags & ATEXFlags.HasNormalMap) != 0;
        public bool IsWaterTexture => (Flags & ATEXFlags.IsWater) != 0;
        public bool IsLavaTexture => (Flags & ATEXFlags.IsLava) != 0;
        
        // Get texture dimensions based on flags
        public System.Drawing.Size GetTextureDimensions()
        {
            return IsHighResolution ? new System.Drawing.Size(2048, 2048) : new System.Drawing.Size(1024, 1024);
        }
        
        // Get material properties based on type and flags
        public MaterialProperties GetMaterialProperties()
        {
            var props = new MaterialProperties();
            
            // Set base properties by material type
            switch (MaterialType)
            {
                case MaterialType.Stone:
                    props.Hardness = 0.9f;
                    props.Roughness = 0.8f;
                    props.FootstepSound = "stone";
                    break;
                case MaterialType.Dirt:
                    props.Hardness = 0.3f;
                    props.Roughness = 0.7f;
                    props.FootstepSound = "dirt";
                    break;
                case MaterialType.Grass:
                    props.Hardness = 0.2f;
                    props.Roughness = 0.4f;
                    props.FootstepSound = "grass";
                    break;
                case MaterialType.Metal:
                    props.Hardness = 1.0f;
                    props.Roughness = 0.2f;
                    props.FootstepSound = "metal";
                    break;
                // Additional cases for other material types
            }
            
            // Apply specular multiplier
            props.SpecularIntensity = SpecularMultiplier;
            
            return props;
        }
    }
    
    // Material properties structure for physics/sound
    public class MaterialProperties
    {
        public float Hardness { get; set; }         // 0-1 scale
        public float Roughness { get; set; }        // 0-1 scale
        public float SpecularIntensity { get; set; } // 0+ scale
        public string FootstepSound { get; set; }   // Sound ID
    }
    
    // List of texture entries
    public List<TextureEntry> TextureEntries { get; private set; }
    
    public ATEX()
    {
        TextureEntries = new List<TextureEntry>();
    }
    
    public ATEX(List<TextureEntry> entries)
    {
        TextureEntries = entries;
    }
    
    // Add a new texture entry
    public int AddTextureEntry(TextureEntry entry)
    {
        TextureEntries.Add(entry);
        return TextureEntries.Count - 1;
    }
    
    // Get texture entry by index
    public TextureEntry GetTextureEntry(int index)
    {
        if (index < 0 || index >= TextureEntries.Count)
            throw new ArgumentOutOfRangeException($"Invalid texture index: {index}");
            
        return TextureEntries[index];
    }
    
    // Find texture index by filename
    public int FindTextureIndex(string filename)
    {
        return TextureEntries.FindIndex(e => 
            string.Equals(e.Filename, filename, StringComparison.OrdinalIgnoreCase));
    }
}
```

## Usage Context
The ATEX chunk contains definitions for all textures used in the terrain tile. Each entry includes a filename pointing to a BLP texture file, along with flags and properties that determine how the texture is rendered. These textures are referenced by the ALYR (texture layer) subchunks to define which textures apply to specific areas of the terrain.

In the ADT v23 format used since Warlords of Draenor, the ATEX chunk has been enhanced with additional fields for material properties and advanced rendering techniques. The new materialType field categorizes textures by their physical properties (stone, dirt, grass, etc.), while the specularMultiplier controls the intensity of specular highlights. New flags support features like parallax mapping, normal maps, and special handling for water and lava textures.

These enhancements support WoD's improved rendering system, allowing for more realistic terrain with varying material properties. For example, stone surfaces can now have different physical properties and reflection characteristics than grass or dirt, enhancing both visual quality and gameplay immersion. 