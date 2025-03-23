# EXP2: Extended Particle 2

## Identification
- **Chunk ID**: EXP2
- **Parent Format**: M2
- **Source**: M2 file format documentation

## Description
The EXP2 chunk is an enhanced version of the EXPT chunk introduced in patch 7.3. It contains extended particle data with additional animation tracks for alpha cutoff values, providing more advanced control over particle rendering and alpha testing.

## Structure
```cpp
struct M2ExtendedParticle {
    float zSource;                            // Z-axis source position for particles
    float colorMult;                          // Color multiplier value for particle rendering
    float alphaMult;                          // Alpha/opacity multiplier value for particle rendering
    M2PartTrack<fixed16> alphaCutoff;         // Track for alpha testing cutoff values based on particle lifetime
};

struct M2InitExtendedParticleArray {
    M2Array<M2ExtendedParticle> content;      // Array of extended particle data
};

struct EXP2Chunk {
    M2InitExtendedParticleArray exp2;         // Extended particle array container
}
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| zSource | float | When greater than 0, affects the initial velocity calculation of the particle |
| colorMult | float | Multiplier applied to the particle's diffuse color |
| alphaMult | float | Multiplier applied to the particle's opacity |
| alphaCutoff | M2PartTrack<fixed16> | Animation track for alpha testing cutoff values based on particle lifetime |

## Dependencies
- Depends on the MD21 chunk to determine the number of particle emitters.
- Supersedes the EXPT chunk with more advanced functionality.
- Related to the particle emitter system defined in the main M2 data.

## Implementation Notes
1. The length of the `content` array matches the number of particle emitters in the model.
2. The `alphaCutoff` track provides per-particle alpha testing based on the particle's current lifetime.
3. The `colorMult` is applied as a multiplier against the particle's diffuse color.
4. The `alphaMult` is applied as a multiplier against the particle's opacity.
5. If this chunk is present, it takes precedence over the EXPT chunk. If not present but EXPT exists, the client tries to reconstruct EXP2 data from EXPT data.

## Usage Context
The EXP2 chunk provides enhanced control over particle emitters for:
- Dynamic alpha testing based on particle lifetime
- More precise control over particle color and opacity
- Custom particle emission velocities
- Advanced visual effects like dissolve, fade, and reveal effects

This chunk represents an advancement in the particle system introduced in patch 7.3, giving artists more control over particle behavior and appearance without modifying the core particle emitter properties. 