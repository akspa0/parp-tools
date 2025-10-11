# PADC: Particle Animation Data

## Identification
- **Chunk ID**: PADC
- **Parent Format**: M2
- **Source**: M2 file format documentation

## Description
The PADC chunk (Particle Animation Data) was introduced in patch 7.3 and contains texture weight information for particle animations. It essentially moves the texture weight data from the main M2 file into a separate chunk, providing a replacement for the header.texture_weights data.

## Structure
```cpp
struct PADCChunk {
    M2Array<M2TextureWeight> texture_weights;  // Array of texture weight animation tracks
}
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| texture_weights | M2Array<M2TextureWeight> | Array of texture weight animation tracks that control transparency of textures over time |

## Dependencies
- Depends on the MD21 chunk which contains the basic texture information.
- Related to the particle and material systems defined in the main M2 data.
- Provides texture weight data that would otherwise be in the main M2 header.

## Implementation Notes
1. This chunk is identified in the client as "M2InitParentAnimData: parentTextureWeights".
2. It's unclear why this data was moved from the main header into a separate chunk.
3. The M2TextureWeight structure contains an animation track (M2Track) of fixed16 values that control transparency over time.
4. If this chunk is present, it replaces the header.texture_weights data from the main M2 file.
5. This chunk is part of a series of optimizations and refactorings in the particle system introduced in patch 7.3.

## Usage Context
The PADC chunk is used for:
- Controlling the transparency of textures over time for particle effects
- Animating the opacity of particle textures independent of the main color/alpha tracks
- Providing a separate animation channel for texture blending
- Potentially allowing for shared texture weight data between related models

This chunk represents part of the advanced animation system introduced in patch 7.3, providing more modular control over texture animation for particle systems. 