# MTLS - MDX Materials Chunk

## Type
MDX Main Chunk

## Source
MDX_index.md

## Description
The MTLS (Materials) chunk defines the materials used to render the model. Each material contains information about textures, rendering modes, alpha blending, culling, and various shader parameters. Materials are referenced by geosets in the GEOS chunk to determine how geometry is rendered.

## Structure

```csharp
public struct MTLS
{
    /// <summary>
    /// Array of materials
    /// </summary>
    // MDLMATERIAL materials[numMaterials] follows
}

public struct MDLMATERIAL
{
    /// <summary>
    /// Priority plane for render ordering
    /// </summary>
    public uint priorityPlane;
    
    /// <summary>
    /// Render flags (two-sided, etc.)
    /// </summary>
    public uint renderFlags;
    
    /// <summary>
    /// Number of texture references in this material
    /// </summary>
    public uint numTextures;
    
    /// <summary>
    /// Array of texture references
    /// </summary>
    // MDLTEXUNIT textures[numTextures] follows

    /// <summary>
    /// Animation data for material properties
    /// </summary>
    // MDLKEYTRACK animations follow
}

public struct MDLTEXUNIT
{
    /// <summary>
    /// ID of the texture in the TEXS chunk
    /// </summary>
    public uint textureId;
    
    /// <summary>
    /// Flags for this texture unit (wrap modes, etc.)
    /// </summary>
    public uint flags;
    
    /// <summary>
    /// Animation data for texture transforms
    /// </summary>
    // MDLKEYTRACK animations follow
}
```

## Properties

### MDLMATERIAL Structure

| Offset | Name | Type | Description |
|--------|------|------|-------------|
| 0x00 | priorityPlane | uint | Priority for render ordering |
| 0x04 | renderFlags | uint | Rendering flags (see Material Flags) |
| 0x08 | numTextures | uint | Number of texture units in this material |
| 0x0C | ... | ... | Array of MDLTEXUNIT structures follows |
| varies | ... | ... | Animation tracks for material properties follow |

### MDLTEXUNIT Structure

| Offset | Name | Type | Description |
|--------|------|------|-------------|
| 0x00 | textureId | uint | ID of texture in TEXS chunk |
| 0x04 | flags | uint | Texture unit flags (see Texture Flags) |
| 0x08 | ... | ... | Animation tracks for texture transforms follow |

## Material Flags

| Bit | Name | Description |
|-----|------|-------------|
| 0 | ConstantColor | Use constant color instead of vertex colors |
| 1 | Unshaded | Don't apply lighting to this material |
| 2 | TwoSided | Disable backface culling (render both sides) |
| 3 | Unfogged | Don't apply fog effects |
| 4 | NoDepthTest | Disable depth testing |
| 5 | NoDepthWrite | Disable writing to depth buffer |
| 6-31 | Reserved | Reserved for future use |

## Texture Flags

| Bit | Name | Description |
|-----|------|-------------|
| 0 | WrapWidth | Wrap texture horizontally |
| 1 | WrapHeight | Wrap texture vertically |
| 2-31 | Reserved | Reserved for future use |

## Material Layers
Materials can have multiple texture units that are applied in sequence:
1. First texture unit is the base layer
2. Second texture unit is typically a detail/decal layer
3. Third texture unit is usually for team colors (replaceable textures)
4. Additional texture units for special effects

## Animation Tracks
After the basic material and texture data, several animation tracks may follow:

### Material Animation Tracks
- Emissive color (Vector3 RGB, 0-255)
- Alpha value (float, 0.0-1.0)

### Texture Animation Tracks
For each texture unit:
- Translation (Vector3 XYZ)
- Rotation (Quaternion XYZW)
- Scaling (Vector3 XYZ)

## Version Differences

| Version | Changes |
|---------|---------|
| 800-1000 (WC3) | Base structure as described |
| 1300-1400 (WoW Alpha) | Same basic structure |
| 1500 (WoW Alpha) | Added support for more texture stages and complex shaders |

## Dependencies
- TEXS - Materials reference textures by ID from the TEXS chunk
- GEOS - Geosets reference materials by ID from the MTLS chunk

## Implementation Notes
- The priorityPlane field is used for render ordering (higher values render last/on top)
- Each material can contain multiple texture units for multi-texturing effects
- Texture animations (translation, rotation, scaling) are used for effects like scrolling textures
- Material animations (color, alpha) are used for fading, glowing, and other effects
- The exact number of animation tracks can vary based on the material's complexity
- In Warcraft 3 models, most materials use 1-3 texture units
- WoW Alpha models (v1500) may use more complex texture stage arrangements

## Usage Context
The MTLS chunk:
- Defines all material properties for model rendering
- Controls blending modes and transparency
- Sets up multi-texturing effects
- Provides animation data for material properties
- Establishes render ordering via priority planes

## Implementation Example

```csharp
public class MTLSChunk : IMdxChunk
{
    public string ChunkId => "MTLS";
    
    public List<MdxMaterial> Materials { get; private set; } = new List<MdxMaterial>();
    
    public void Parse(BinaryReader reader, long totalSize)
    {
        long startPosition = reader.BaseStream.Position;
        long endPosition = startPosition + totalSize;
        
        // Clear any existing materials
        Materials.Clear();
        
        // Read materials until we reach the end of the chunk
        while (reader.BaseStream.Position < endPosition)
        {
            var material = new MdxMaterial();
            
            // Read basic properties
            material.PriorityPlane = reader.ReadUInt32();
            material.RenderFlags = reader.ReadUInt32();
            uint numTextures = reader.ReadUInt32();
            
            // Read texture units
            for (int i = 0; i < numTextures; i++)
            {
                var texUnit = new MdxTextureUnit();
                texUnit.TextureId = reader.ReadUInt32();
                texUnit.Flags = reader.ReadUInt32();
                
                // Read animation tracks for this texture unit
                // Translation
                texUnit.TranslationTrack = new MdxKeyTrack<Vector3>();
                texUnit.TranslationTrack.Parse(reader, r => new Vector3(r.ReadSingle(), r.ReadSingle(), r.ReadSingle()));
                
                // Rotation
                texUnit.RotationTrack = new MdxKeyTrack<Quaternion>();
                texUnit.RotationTrack.Parse(reader, r => new Quaternion(r.ReadSingle(), r.ReadSingle(), r.ReadSingle(), r.ReadSingle()));
                
                // Scaling
                texUnit.ScalingTrack = new MdxKeyTrack<Vector3>();
                texUnit.ScalingTrack.Parse(reader, r => new Vector3(r.ReadSingle(), r.ReadSingle(), r.ReadSingle()));
                
                material.TextureUnits.Add(texUnit);
            }
            
            // Read material animation tracks
            // Emissive color
            material.EmissiveColorTrack = new MdxKeyTrack<Vector3>();
            material.EmissiveColorTrack.Parse(reader, r => new Vector3(r.ReadSingle(), r.ReadSingle(), r.ReadSingle()));
            
            // Alpha
            material.AlphaTrack = new MdxKeyTrack<float>();
            material.AlphaTrack.Parse(reader, r => r.ReadSingle());
            
            Materials.Add(material);
        }
    }
    
    public void Write(BinaryWriter writer)
    {
        foreach (var material in Materials)
        {
            // Write basic properties
            writer.Write(material.PriorityPlane);
            writer.Write(material.RenderFlags);
            writer.Write((uint)material.TextureUnits.Count);
            
            // Write texture units
            foreach (var texUnit in material.TextureUnits)
            {
                writer.Write(texUnit.TextureId);
                writer.Write(texUnit.Flags);
                
                // Write animation tracks for this texture unit
                texUnit.TranslationTrack.Write(writer, (w, v) => { w.Write(v.X); w.Write(v.Y); w.Write(v.Z); });
                texUnit.RotationTrack.Write(writer, (w, q) => { w.Write(q.X); w.Write(q.Y); w.Write(q.Z); w.Write(q.W); });
                texUnit.ScalingTrack.Write(writer, (w, v) => { w.Write(v.X); w.Write(v.Y); w.Write(v.Z); });
            }
            
            // Write material animation tracks
            material.EmissiveColorTrack.Write(writer, (w, v) => { w.Write(v.X); w.Write(v.Y); w.Write(v.Z); });
            material.AlphaTrack.Write(writer, (w, f) => w.Write(f));
        }
    }
}

public class MdxMaterial
{
    public uint PriorityPlane { get; set; }
    public uint RenderFlags { get; set; }
    public List<MdxTextureUnit> TextureUnits { get; set; } = new List<MdxTextureUnit>();
    public MdxKeyTrack<Vector3> EmissiveColorTrack { get; set; }
    public MdxKeyTrack<float> AlphaTrack { get; set; }
    
    // Flag accessors
    public bool ConstantColor => (RenderFlags & 0x1) != 0;
    public bool Unshaded => (RenderFlags & 0x2) != 0;
    public bool TwoSided => (RenderFlags & 0x4) != 0;
    public bool Unfogged => (RenderFlags & 0x8) != 0;
    public bool NoDepthTest => (RenderFlags & 0x10) != 0;
    public bool NoDepthWrite => (RenderFlags & 0x20) != 0;
}

public class MdxTextureUnit
{
    public uint TextureId { get; set; }
    public uint Flags { get; set; }
    public MdxKeyTrack<Vector3> TranslationTrack { get; set; }
    public MdxKeyTrack<Quaternion> RotationTrack { get; set; }
    public MdxKeyTrack<Vector3> ScalingTrack { get; set; }
    
    // Flag accessors
    public bool WrapWidth => (Flags & 0x1) != 0;
    public bool WrapHeight => (Flags & 0x2) != 0;
} 