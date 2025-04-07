# EXPT: Extended Particle

## Identification
- **Chunk ID**: EXPT
- **Parent Format**: M2
- **Source**: M2 file format documentation

## Description
The EXPT chunk contains additional particle data that extends the functionality of the standard M2 particle system. It was introduced in Legion and partially replaces variables from the M2ParticleOld structure with more flexible parameters. This chunk is considered outdated after the introduction of the EXP2 chunk in patch 7.3.

## Structure
```cpp
struct ExtendedParticleEntry {
    float zSource;     // Z-axis source position for particles
    float colorMult;   // Color multiplier value for particle rendering
    float alphaMult;   // Alpha/opacity multiplier value for particle rendering
};

struct EXPTChunk {
    ExtendedParticleEntry extended_particle[m2data.header.particles.count];
}
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| zSource | float | When greater than 0, affects the initial velocity calculation of the particle |
| colorMult | float | Multiplier applied to the particle's diffuse color |
| alphaMult | float | Multiplier applied to the particle's opacity |

## Dependencies
- Depends on the MD21 chunk to determine the number of particle emitters.
- Related to the particle emitter system defined in the main M2 data.
- Superseded by the EXP2 chunk in patch 7.3+.

## Implementation Notes
1. If the EXP2 chunk doesn't exist, the client tries to reconstruct it using data from this EXPT chunk.
2. The `zSource` parameter affects initial particle velocity: when greater than 0, the initial velocity is calculated as `(particle.position - C3Vector(0, 0, zSource)).Normalize()`.
3. The `colorMult` is applied as a multiplier against the particle's diffuse color.
4. The `alphaMult` is applied as a multiplier against the particle's opacity.
5. Each entry corresponds to a particle emitter in the M2 data.

## Usage Context
The EXPT chunk provides extended control over particle emitters for:
- More precise control over particle color and opacity
- Custom particle emission velocities
- Visual adjustments to particle systems without modifying the core emitter properties

While this chunk is considered outdated since patch 7.3, it's still processed for backward compatibility with models created between Legion's launch and patch 7.3. 