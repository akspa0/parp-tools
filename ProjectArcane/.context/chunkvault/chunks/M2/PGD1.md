# PGD1 Chunk (Particle Geoset Data Version 1)

## Overview
The PGD1 (Particle Geoset Data Version 1) chunk associates particle emitters with specific geosets in an M2 model. This chunk was added in Classic WoW (1.13.2.30172), allowing particle effects to be controlled by the same geoset visibility rules that govern model geometry.

## Structure
```cpp
struct PGD1Entry {
  uint16_t geoset;  // Geoset ID that controls this particle emitter
};

struct PGD1_Chunk {
  M2Array<PGD1Entry> p_g_d_v1;  // Array of geoset assignments for particle emitters
};
```

## Fields
- **p_g_d_v1**: An array of PGD1Entry structures, where each entry corresponds to a particle emitter in the model. The count of entries matches the number of particle emitters.
- **geoset**: The geoset ID assigned to each particle emitter, determining when the emitter is visible based on geoset visibility rules.

## Dependencies
- Requires the MD21 chunk, specifically the particle emitters section
- Interacts with the skin file (.skin) geoset visibility system
- The number of entries must match the number of particle emitters defined in the model

## Usage
The PGD1 chunk is used to:
- Associate particle emitters with specific geosets
- Control particle emitter visibility based on geoset selection
- Allow particle effects to be toggled along with specific model components
- Support equipment-dependent particle effects and other conditional visual elements

## Legacy Support
- Added in Classic WoW (1.13.2.30172)
- Enhances the particle system's integration with the model's component visibility system
- Allows for more precise control of particle effects compared to earlier implementations

## Implementation Notes
- The array length should match the number of particle emitters in the model
- Each entry's geoset value determines which geoset controls the visibility of the corresponding particle emitter
- When loading a model with the PGD1 chunk, the implementation should apply the same geoset visibility rules to both model geometry and particle emitters
- Implementation value is stored in `m_emitterSelectionGroup` in the internal structure
- This allows particle effects to be turned on or off based on equipment choices, character customization, animation state, or other factors that control geoset visibility

## Version History
- Introduced in Classic WoW (1.13.2.30172)
- Provides a mechanism to control particle emitter visibility using the existing geoset system
- Integrates particle effects more tightly with the model's component visibility system 