# LDV1 Chunk (Level of Detail Data Version 1)

## Overview
The LDV1 chunk defines Level of Detail (LOD) information for the M2 model. It was introduced in Battle for Azeroth (8.0.1.26629) and provides data for managing different detail levels of the model based on distance and performance requirements.

## Structure
```cpp
struct LDV1_Chunk {
  uint16_t unk0;                 // Unknown value
  uint16_t lodCount;             // Number of LOD levels (maxLod = lodCount-1)
  float unk2_f;                  // Unknown float value used in distance calculations
  uint8_t particleBoneLod[4];    // LOD levels for particle bone visibility
  uint32_t unk4;                 // Unknown value
};
```

## Fields
- **unk0**: Unknown value, purpose not yet determined
- **lodCount**: Number of LOD levels available for this model
- **unk2_f**: Float value used in LOD distance calculations (formula: fmaxf(fminf(740.0 / unk2_f, 5.0), 0.5))
- **particleBoneLod**: Array of 4 bytes that define particle bone LOD levels
- **unk4**: Unknown 32-bit value

## Dependencies
- Works in conjunction with the SFID chunk, which includes references to _lod%d.skin files
- Relates to the bone structure defined in the MD21 chunk
- Affects particle emitters defined in the model

## Usage
The LDV1 chunk is used to:
- Define how many LOD levels the model has
- Control which skin files are used at different detail levels
- Determine bone and particle visibility at different LOD levels
- Calculate distance thresholds for switching between detail levels

## Legacy Support
- Not present in pre-BfA M2 files
- In older versions, LOD might have been handled differently or with more limited capability

## Implementation Notes
- The particleBoneLod array is used to determine visibility of particle emitters at different LOD levels
- The LOD system uses both the primary skins and specific _lod#.skin files
- For the pandarenfemale.m2 example, with lodCount=4, the SFID chunk contains 7 files:
  - First 4 are standard .skin files
  - Last 3 are _lod1.skin, _lod2.skin, and _lod3.skin files
- The particleBoneLod values determine bone flags that disable particle emitters at specific LOD levels
- Implementation formula for bone visibility flags:
  ```cpp
  if (lod < 1)
    result = 0;
  
  if (LodData)
    result = (0x10000 << LodData->particleBoneLod[lod]);
  else
    result = (0x10000 << (lod - 1));
  
  // For each ParticleEmitter and related M2Particle record
  if (result & M2CompBone[M2Particle->old.boneIndex].flags) {
    // Do not animate this emitter
  }
  ```

## Version History
- Introduced in Battle for Azeroth (8.0.1.26629)
- Part of Blizzard's improvements to the LOD system for better performance scaling 