# ALYR - Texture Layer Information

## Type
ADT v23 ACNK Subchunk

## Source
Referenced from `ADT_v23.md`

## Description
The ALYR (Alpha Layer) subchunk defines individual texture layers applied to a terrain chunk. Each layer references a texture from the ATEX chunk and contains properties that control how the texture is applied, including texture coordinates, rotation, and flags. The ALYR subchunk is typically followed by an AMAP subchunk that contains the actual alpha (opacity) map for the layer.

## Structure

```csharp
public struct ALYR
{
    public uint textureId;     // Index into the ATEX chunk's texture array
    public uint flags;         // Texture layer flags
    public uint offsetInAMAP;  // Offset to alpha map in the corresponding AMAP subchunk
    public float startX;       // Starting X coordinate for texture mapping (0.0 to 1.0)
    public float startY;       // Starting Y coordinate for texture mapping (0.0 to 1.0)
    public float endX;         // Ending X coordinate for texture mapping (0.0 to 1.0)
    public float endY;         // Ending Y coordinate for texture mapping (0.0 to 1.0)
    public float alphaLevel;   // Base alpha level for the texture (0.0 to 1.0)
    public float rotation;     // Texture rotation in radians
    public uint effectId;      // Special effect ID (water, etc.)
}
```

## Properties

| Name | Type | Description |
|------|------|-------------|
| textureId | uint | Index into the ATEX chunk's texture array to identify which texture to use |
| flags | uint | Bitfield for layer properties (see table below) |
| offsetInAMAP | uint | Offset to this layer's alpha map data in the AMAP subchunk that follows |
| startX | float | Starting X texture coordinate (0.0 to 1.0) |
| startY | float | Starting Y texture coordinate (0.0 to 1.0) |
| endX | float | Ending X texture coordinate (0.0 to 1.0) |
| endY | float | Ending Y texture coordinate (0.0 to 1.0) |
| alphaLevel | float | Base alpha (opacity) level for the entire layer (0.0 to 1.0) |
| rotation | float | Rotation angle of the texture in radians |
| effectId | uint | ID for special rendering effects like water or lava |

### Layer Flags

| Flag | Value | Description |
|------|-------|-------------|
| ANIMATE_45 | 0x01 | Animate this layer by rotating 45 degrees back and forth |
| ANIMATE_90 | 0x02 | Animate this layer by rotating 90 degrees back and forth |
| ANIMATE_180 | 0x04 | Animate this layer by rotating 180 degrees back and forth |
| ANIMATE_FAST | 0x08 | Increase the animation speed |
| ANIMATE_SLOW | 0x10 | Decrease the animation speed |
| WATER_SHALLOW | 0x20 | This is a shallow water layer |
| WATER_DEEP | 0x40 | This is a deep water layer |
| USE_ALPHA_MAP | 0x80 | Use the alpha map from AMAP (if not set, use uniform alphaLevel) |
| COMPRESSED_ALPHA | 0x100 | Alpha map is stored in a compressed format |
| TERRAIN_SPECULAR | 0x200 | Enable specular highlights for this terrain layer |

## Dependencies

- ACNK (C006) - Parent chunk that contains this subchunk
- ATEX (C004) - Contains the texture filenames referenced by textureId
- AMAP (S002) - Contains the alpha map data referenced by offsetInAMAP

## Implementation Notes

1. Each ALYR subchunk defines a single texture layer for a terrain chunk, with multiple layers typically stacked to create the final terrain appearance.

2. The size of each ALYR subchunk is fixed at 40 bytes (not including the subchunk header).

3. The textureId field references an index in the ATEX chunk's array of textures. This is a change from the v18 format, which used direct string references.

4. The texture coordinates (startX, startY, endX, endY) control how the texture is mapped onto the terrain, allowing for stretching, tiling, or focusing on specific portions of the texture.

5. The alpha level provides a base opacity for the entire layer, which can be further modified by the alpha map in the AMAP subchunk.

6. The rotation value allows textures to be rotated, adding variety to terrain appearance.

7. Typically, the first texture layer serves as the base, with full opacity (no alpha map), while additional layers are blended on top using alpha maps.

8. The offsetInAMAP field points to the alpha map data in the AMAP subchunk that typically follows the ALYR subchunks. This is a more structured approach compared to the v18 format, where alpha maps were separate chunks.

## Implementation Example

```csharp
public class AlyrSubchunk
{
    // Constants
    private const int ALYR_SIZE = 40; // Size of ALYR structure in bytes
    
    // Layer flags
    public static class LayerFlags
    {
        public const uint ANIMATE_45 = 0x01;
        public const uint ANIMATE_90 = 0x02;
        public const uint ANIMATE_180 = 0x04;
        public const uint ANIMATE_FAST = 0x08;
        public const uint ANIMATE_SLOW = 0x10;
        public const uint WATER_SHALLOW = 0x20;
        public const uint WATER_DEEP = 0x40;
        public const uint USE_ALPHA_MAP = 0x80;
        public const uint COMPRESSED_ALPHA = 0x100;
        public const uint TERRAIN_SPECULAR = 0x200;
    }
    
    // Properties
    public uint TextureId { get; set; }
    public uint Flags { get; set; }
    public uint OffsetInAMAP { get; set; }
    public float StartX { get; set; }
    public float StartY { get; set; }
    public float EndX { get; set; }
    public float EndY { get; set; }
    public float AlphaLevel { get; set; }
    public float Rotation { get; set; }
    public uint EffectId { get; set; }
    
    // Helper properties
    public bool HasAlphaMap => (Flags & LayerFlags.USE_ALPHA_MAP) != 0;
    public bool IsCompressedAlpha => (Flags & LayerFlags.COMPRESSED_ALPHA) != 0;
    public bool IsWater => (Flags & (LayerFlags.WATER_SHALLOW | LayerFlags.WATER_DEEP)) != 0;
    public bool IsAnimated => (Flags & (LayerFlags.ANIMATE_45 | LayerFlags.ANIMATE_90 | LayerFlags.ANIMATE_180)) != 0;
    
    public AlyrSubchunk()
    {
        // Default values
        TextureId = 0;
        Flags = 0;
        OffsetInAMAP = 0;
        StartX = 0.0f;
        StartY = 0.0f;
        EndX = 1.0f;
        EndY = 1.0f;
        AlphaLevel = 1.0f;
        Rotation = 0.0f;
        EffectId = 0;
    }
    
    public void Load(BinaryReader reader)
    {
        TextureId = reader.ReadUInt32();
        Flags = reader.ReadUInt32();
        OffsetInAMAP = reader.ReadUInt32();
        StartX = reader.ReadSingle();
        StartY = reader.ReadSingle();
        EndX = reader.ReadSingle();
        EndY = reader.ReadSingle();
        AlphaLevel = reader.ReadSingle();
        Rotation = reader.ReadSingle();
        EffectId = reader.ReadUInt32();
    }
    
    public void Save(BinaryWriter writer)
    {
        writer.Write("ALYR".ToCharArray());
        writer.Write(ALYR_SIZE);
        
        writer.Write(TextureId);
        writer.Write(Flags);
        writer.Write(OffsetInAMAP);
        writer.Write(StartX);
        writer.Write(StartY);
        writer.Write(EndX);
        writer.Write(EndY);
        writer.Write(AlphaLevel);
        writer.Write(Rotation);
        writer.Write(EffectId);
    }
    
    // Helper methods
    
    public void SetTextureCoordinates(float startX, float startY, float endX, float endY)
    {
        StartX = Math.Max(0.0f, Math.Min(1.0f, startX));
        StartY = Math.Max(0.0f, Math.Min(1.0f, startY));
        EndX = Math.Max(0.0f, Math.Min(1.0f, endX));
        EndY = Math.Max(0.0f, Math.Min(1.0f, endY));
    }
    
    public void SetRotation(float degrees)
    {
        // Convert degrees to radians
        Rotation = degrees * (float)Math.PI / 180.0f;
    }
    
    public void EnableWater(bool isDeep = false)
    {
        if (isDeep)
        {
            Flags |= LayerFlags.WATER_DEEP;
            Flags &= ~LayerFlags.WATER_SHALLOW;
        }
        else
        {
            Flags |= LayerFlags.WATER_SHALLOW;
            Flags &= ~LayerFlags.WATER_DEEP;
        }
    }
    
    public void EnableAnimation(int degrees, bool fast = false)
    {
        // Clear existing animation flags
        Flags &= ~(LayerFlags.ANIMATE_45 | LayerFlags.ANIMATE_90 | LayerFlags.ANIMATE_180);
        
        // Set new animation flags
        switch (degrees)
        {
            case 45:
                Flags |= LayerFlags.ANIMATE_45;
                break;
            case 90:
                Flags |= LayerFlags.ANIMATE_90;
                break;
            case 180:
                Flags |= LayerFlags.ANIMATE_180;
                break;
            default:
                // No animation
                break;
        }
        
        // Set speed
        if (fast)
        {
            Flags |= LayerFlags.ANIMATE_FAST;
            Flags &= ~LayerFlags.ANIMATE_SLOW;
        }
        else
        {
            Flags |= LayerFlags.ANIMATE_SLOW;
            Flags &= ~LayerFlags.ANIMATE_FAST;
        }
    }
}
```

## Usage Context

The ALYR subchunk is fundamental to terrain rendering in World of Warcraft, serving several key functions:

1. **Texture Placement**: Controls how textures are mapped onto terrain, allowing for precise positioning, scaling, and rotation to create natural-looking landscapes.

2. **Texture Blending**: Through reference to alpha maps in the AMAP subchunk, enables smooth blending between different terrain types (grass, dirt, snow, etc.).

3. **Special Effects**: Supports water surfaces, animated textures, and special rendering effects through flags and effect IDs.

4. **Terrain Detail Control**: By allowing multiple texture layers per chunk, enables highly detailed and varied terrain with minimal repetition.

5. **Memory Optimization**: The reference-based approach to textures (via textureId into ATEX) allows the same texture to be reused across multiple chunks without duplication.

The ALYR subchunk in the v23 format represents an evolution from the MCLY subchunk in v18, with additional properties for more precise texture control and a clearer relationship with its alpha map data. While the v23 format was not ultimately used in a retail release, its approach to texture layering demonstrates Blizzard's continued refinement of terrain rendering techniques during the Cataclysm beta development period. 