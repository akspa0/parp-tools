# RPID: Recursive Particle Model File IDs

## Identification
- **Chunk ID**: RPID
- **Parent Format**: M2
- **Source**: M2 file format documentation

## Description
The RPID chunk was introduced in Battle for Azeroth (patch 8.1.0) and replaces the `recursion_model_filename` field in the M2ParticleOld structure. It contains FileDataIDs for models used as recursive particle effects, allowing particle emitters to spawn other models with their own particle effects.

## Structure
```cpp
struct RPIDChunk {
    struct {
        uint32_t fileDataID;  // FileDataID referencing an M2 model
    } recursive_particle_models[particle_count];
}
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| fileDataID | uint32_t | FileDataID referencing an M2 model to be used as a recursive particle effect |

## Dependencies
- Depends on the MD21 chunk to determine the number of particle emitters.
- Related to the particle emitter system defined in the main M2 data.
- The M2 models referenced are loaded when the particle effect emits particles.

## Implementation Notes
1. The number of entries matches the number of particle emitters in the M2 model.
2. When a particle emitter has a recursive model, it acts as an alias for up to 4 emitters from the referenced model.
3. Prior to this chunk, recursive models were referenced by filename in the `recursion_model_filename` field.
4. A fileDataID of 0 indicates no recursive model for that particle emitter.
5. Recursive particle models can create complex effects like chain reactions, cascading explosions, or compound particle systems.

## Usage Context
The RPID chunk is used for:
- Creating complex particle effects that spawn additional models
- Building layered visual effects with multiple stages
- Chaining particle systems together for elaborate visual sequences
- Reusing existing particle effects as components in larger effects

This chunk is part of the transition to the FileDataID system, replacing direct filename references with database IDs to improve asset management and localization support. 