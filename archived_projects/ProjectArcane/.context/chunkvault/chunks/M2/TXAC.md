# TXAC: Texture Combiner Combos

## Identification
- **Chunk ID**: TXAC
- **Parent Format**: M2
- **Source**: M2 file format documentation

## Description
The TXAC chunk contains information related to texture animation and setup for materials and particles. It was introduced in Legion and appears to be used in shader setup, particularly for configuring texture transforms in the rendering pipeline.

## Structure
```cpp
struct TXACEntry {
    char unk[2];  // Likely used in CM2SceneRender::SetupTextureTransforms and uploaded to the shader directly
};

struct TXACChunk {
    TXACEntry texture_ac[m2data.header.materials.count + m2data.header.particles.count];
}
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| unk | char[2] | Unknown values used in texture transform setup; typically 0 when not actively used |

## Dependencies
- Depends on the MD21 chunk to determine the number of materials and particles.
- Related to shader selection and texture rendering pipeline.
- May affect the CParticleEmitter2 behavior for vertex buffer format selection.

## Implementation Notes
1. The exact purpose of this chunk isn't fully documented but appears to relate to shader configuration.
2. The total number of entries equals the sum of material count and particle emitter count from the M2 header.
3. Each entry is 2 bytes and typically contains zeros unless specific shader behaviors are needed.
4. The chunk appears to be used during the texture transform setup in the rendering pipeline.
5. It may influence particle rendering by affecting vertex buffer format selection.

## Usage Context
The TXAC chunk is used in the rendering pipeline for:
- Configuring texture transform operations
- Potentially providing shader parameters
- Influencing particle emitter vertex buffer formats
- Customizing material rendering behavior

This chunk appears to be part of the rendering optimization and customization systems added in Legion to give artists more control over how textures are combined and displayed on models. 