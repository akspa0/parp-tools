# C009: ACNK

## Type
ADT v23 Chunk

## Source
ADT_v23.md

## Description
Map chunk data container for ADT v23 format. Each ACNK represents one of the 16×16 terrain chunks that make up an ADT tile, enhanced with WoD-specific features for improved rendering and environmental effects.

## Structure
```csharp
struct ACNK
{
    // If size is bigger than 0x48, it has a header
    int indexX;                 // X index of this chunk
    int indexY;                 // Y index of this chunk
    uint32_t flags;             // Chunk flags (enhanced in WoD)
    int areaId;                 // Area ID for this chunk
    uint16_t lowDetailTextureMapping;  // Low detail texture mapping
    uint16_t heightTextureId;   // Height texture ID (added in WoD)
    uint8_t groundEffectId;     // Ground effect ID (added in WoD)
    uint8_t windAnimId;         // Wind animation ID (added in WoD)    
    uint16_t detailLayerMask;   // Detail layer mask (added in WoD)
    uint32_t reserved[3];       // Reserved data
    // Followed by subchunks: ALYR, ASHD, ACDO (same as v22)
}
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| indexX | int | X index of this chunk in the ADT grid (0-15) |
| indexY | int | Y index of this chunk in the ADT grid (0-15) |
| flags | uint32 | Flags for this chunk (enhanced in WoD) |
| areaId | int | Zone/area ID for this chunk |
| lowDetailTextureMapping | uint16 | Low detail texture mapping information |
| heightTextureId | uint16 | Height texture ID for displacement mapping (added in WoD) |
| groundEffectId | uint8 | Ground effect ID for particle effects (added in WoD) |
| windAnimId | uint8 | Wind animation ID for vegetation movement (added in WoD) |
| detailLayerMask | uint16 | Mask for detail texture layers (added in WoD) |
| reserved | uint32[3] | Reserved data fields |

## Dependencies
- AHDR (C004) - For overall tile structure
- Contains subchunks: ALYR, ASHD, ACDO (same as v22)

## Implementation Notes
- Size: Variable (larger than v22 header due to additional fields)
- Each ADT tile contains 256 (16×16) ACNK chunks
- Header is present if the chunk size is larger than 0x48 bytes (was 0x40 in v22)
- New fields added in WoD (v23):
  - heightTextureId: References a height texture for displacement mapping
  - groundEffectId: Defines particle effects for the terrain (dust, leaves, etc.)
  - windAnimId: Controls vegetation movement animation
  - detailLayerMask: Controls which detail layers are visible
- Container for terrain data for a specific section of the map
- Similar to ACNK in ADT v22, but with additional fields for WoD features
- Subchunks within ACNK provide texture, shadow, and object data (same as v22)

## Implementation Example
```csharp
[Flags]
public enum ACNKFlags
{
    None = 0,
    HasShadowMap = 0x1,         // Has shadow map (ASHD)
    HasAlphaMap = 0x100,        // Has alpha map (AMAP within ALYR)
    HasHighResHoles = 0x10000,  // Has high resolution terrain holes
    UsesMaterialBlending = 0x20000, // Uses material blending (WoD)
    HasHeightTexture = 0x40000, // Uses height texture for displacement (WoD)
    HasParticleEffects = 0x80000, // Has ground particle effects (WoD)
    UsesWindAnimation = 0x100000 // Uses wind animation (WoD)
}

public enum GroundEffectType
{
    None = 0,
    Dust = 1,
    Leaves = 2,
    Snow = 3,
    Sand = 4,
    RainSplash = 5,
    FireEmbers = 6,
    Petals = 7,
    SparkleGlow = 8  // Added in WoD
}

public enum WindAnimationType
{
    None = 0,
    Light = 1,
    Medium = 2,
    Strong = 3,
    Gusting = 4,
    Swirling = 5    // Added in WoD
}

public class ACNK
{
    // Header properties
    public int IndexX { get; set; }
    public int IndexY { get; set; }
    public ACNKFlags Flags { get; set; }
    public int AreaId { get; set; }
    public ushort LowDetailTextureMapping { get; set; }
    
    // New properties in v23 (WoD)
    public ushort HeightTextureId { get; set; }  // For displacement mapping
    public byte GroundEffectId { get; set; }     // For particle effects
    public byte WindAnimId { get; set; }         // For vegetation movement
    public ushort DetailLayerMask { get; set; }  // For detail texture layers
    
    public uint[] Reserved { get; set; } = new uint[3];
    
    // Subchunks
    public List<ALYR> TextureLayers { get; set; } = new List<ALYR>();
    public ASHD ShadowMap { get; set; }
    public List<ACDO> ObjectDefinitions { get; set; } = new List<ACDO>();
    
    // Helper properties
    public bool HasShadowMap => (Flags & ACNKFlags.HasShadowMap) != 0;
    public bool HasAlphaMap => (Flags & ACNKFlags.HasAlphaMap) != 0;
    public bool HasHighResHoles => (Flags & ACNKFlags.HasHighResHoles) != 0;
    public bool UsesMaterialBlending => (Flags & ACNKFlags.UsesMaterialBlending) != 0;
    public bool HasHeightTexture => (Flags & ACNKFlags.HasHeightTexture) != 0;
    public bool HasParticleEffects => (Flags & ACNKFlags.HasParticleEffects) != 0;
    public bool UsesWindAnimation => (Flags & ACNKFlags.UsesWindAnimation) != 0;
    
    // Get ground effect type
    public GroundEffectType GetGroundEffectType()
    {
        return (GroundEffectType)GroundEffectId;
    }
    
    // Get wind animation type
    public WindAnimationType GetWindAnimationType()
    {
        return (WindAnimationType)WindAnimId;
    }
    
    // Calculated position in world coordinates
    public float WorldX => IndexX * 33.3333f;
    public float WorldY => IndexY * 33.3333f;
    
    // Check if a specific detail layer is enabled
    public bool IsDetailLayerEnabled(int layerIndex)
    {
        if (layerIndex < 0 || layerIndex >= 16)
            throw new ArgumentOutOfRangeException("Layer index must be between 0 and 15");
            
        return (DetailLayerMask & (1 << layerIndex)) != 0;
    }
    
    // Helper method to get all used textures by this chunk
    public IEnumerable<int> GetTextureIndices()
    {
        return TextureLayers.Select(layer => layer.TextureID).Distinct();
    }
    
    // Helper method to check if a point is within this chunk
    public bool ContainsPoint(float x, float y)
    {
        float minX = WorldX;
        float minY = WorldY;
        float maxX = minX + 33.3333f;
        float maxY = minY + 33.3333f;
        
        return x >= minX && x < maxX && y >= minY && y < maxY;
    }
    
    // Configure ground particle effect system
    public ParticleSystem ConfigureGroundParticles()
    {
        if (!HasParticleEffects)
            return null;
            
        var effectType = GetGroundEffectType();
        var system = new ParticleSystem();
        
        switch (effectType)
        {
            case GroundEffectType.Dust:
                system.ParticleTexture = "particles/dust.blp";
                system.EmissionRate = 2.0f;
                system.ParticleLifetime = 3.0f;
                system.ParticleSize = 0.2f;
                system.ParticleColor = new System.Drawing.Color(180, 170, 160, 128);
                break;
                
            case GroundEffectType.Leaves:
                system.ParticleTexture = "particles/leaf.blp";
                system.EmissionRate = 1.0f;
                system.ParticleLifetime = 4.0f;
                system.ParticleSize = 0.15f;
                system.ParticleColor = new System.Drawing.Color(255, 255, 255, 200);
                break;
                
            case GroundEffectType.Snow:
                system.ParticleTexture = "particles/snow.blp";
                system.EmissionRate = 5.0f;
                system.ParticleLifetime = 6.0f;
                system.ParticleSize = 0.1f;
                system.ParticleColor = new System.Drawing.Color(255, 255, 255, 180);
                break;
                
            // Additional cases for other effect types
                
            case GroundEffectType.SparkleGlow: // New in WoD
                system.ParticleTexture = "particles/sparkle.blp";
                system.EmissionRate = 0.5f;
                system.ParticleLifetime = 2.0f;
                system.ParticleSize = 0.3f;
                system.ParticleColor = new System.Drawing.Color(220, 220, 255, 200);
                system.UsePointLighting = true;
                break;
        }
        
        return system;
    }
    
    // Configure wind animation parameters
    public WindParameters ConfigureWindAnimation()
    {
        if (!UsesWindAnimation)
            return null;
            
        var animType = GetWindAnimationType();
        var params = new WindParameters();
        
        switch (animType)
        {
            case WindAnimationType.Light:
                params.Strength = 0.2f;
                params.Variability = 0.1f;
                params.SwayFrequency = 0.5f;
                break;
                
            case WindAnimationType.Medium:
                params.Strength = 0.4f;
                params.Variability = 0.2f;
                params.SwayFrequency = 0.8f;
                break;
                
            case WindAnimationType.Strong:
                params.Strength = 0.7f;
                params.Variability = 0.4f;
                params.SwayFrequency = 1.2f;
                break;
                
            case WindAnimationType.Gusting:
                params.Strength = 0.6f;
                params.Variability = 0.8f;
                params.SwayFrequency = 1.0f;
                params.UseGusting = true;
                break;
                
            case WindAnimationType.Swirling: // New in WoD
                params.Strength = 0.5f;
                params.Variability = 0.6f;
                params.SwayFrequency = 1.1f;
                params.UseVortex = true;
                break;
        }
        
        return params;
    }
    
    // Helper classes for implementation
    public class ParticleSystem
    {
        public string ParticleTexture { get; set; }
        public float EmissionRate { get; set; }
        public float ParticleLifetime { get; set; }
        public float ParticleSize { get; set; }
        public System.Drawing.Color ParticleColor { get; set; }
        public bool UsePointLighting { get; set; }
    }
    
    public class WindParameters
    {
        public float Strength { get; set; }
        public float Variability { get; set; }
        public float SwayFrequency { get; set; }
        public bool UseGusting { get; set; }
        public bool UseVortex { get; set; }
    }
}
```

## Usage Context
The ACNK chunk represents a single terrain chunk in the 16×16 grid that makes up an ADT tile. Each ACNK contains information about the terrain in that specific section, including texture layers, shadow maps, and object placements. It is a container chunk that holds several subchunks (ALYR, ASHD, ACDO) with more specific data.

In the ADT v23 format used since Warlords of Draenor, the ACNK chunk has been enhanced with additional fields to support WoD's improved environmental effects and rendering techniques. The new heightTextureId field enables displacement mapping for more detailed terrain surfaces. The groundEffectId and windAnimId fields control particle effects and vegetation animation, adding dynamic elements to the environment. The detailLayerMask provides finer control over which detail textures are applied to the terrain.

These enhancements support WoD's focus on more immersive and dynamic environments, allowing terrain to respond to weather conditions, display appropriate particle effects based on surface type, and provide more realistic vegetation movement based on wind conditions. 