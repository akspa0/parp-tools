# GPID Chunk (Geometry Particle IDs)

## Overview
The GPID chunk contains FileDataIDs for geometry particle models used by the particle emitters in an M2 model. It was introduced in Battle for Azeroth (8.1.0.27826) and replaces the `geometry_model_filename` field previously found in the M2ParticleOld structure.

## Structure
```cpp
struct GPID_Entry {
  uint32_t fileDataID;  // FileDataID referring to an M2 model
};

struct GPID_Chunk {
  GPID_Entry geometry_particle_models[];  // Array of M2 model FileDataIDs
};
```

## Fields
- **geometry_particle_models**: An array of FileDataIDs that reference M2 models used as geometry particles. The array length corresponds to the number of particle emitters in the model that use geometry particles.

## Dependencies
- Requires the MD21 chunk, specifically the particle emitters section
- The number of entries should match the number of particle emitters that use geometry particles

## Usage
The GPID chunk is used to:
- Reference M2 models that are spawned as geometry particles
- Support complex particle effects that use full 3D models instead of simple sprites
- Create effects like debris, projectiles, or other complex particle elements

## Legacy Support
- Not present in pre-BfA (8.1) M2 files
- In older versions, geometry model filenames were stored directly in the M2ParticleOld structure
- Replaced the string-based filename approach with FileDataIDs

## Implementation Notes
- Each FileDataID points to another M2 model file
- These referenced M2 models are spawned by the particle emitter during runtime
- The particle system handles positioning, orientation, and lifetime of these geometry particle instances
- Geometry particles are more resource-intensive than standard particle sprites
- Typically used for effects where simple sprites wouldn't be sufficient
- When loading a model with the GPID chunk, you should prepare to load the referenced geometry models

## Version History
- Introduced in Battle for Azeroth (8.1.0.27826)
- Part of Blizzard's ongoing transition to FileDataID-based resource references
- Replaced the string-based geometry model filename system 