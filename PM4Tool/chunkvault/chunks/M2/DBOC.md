# DBOC Chunk (Dynamic Bounding Object Control)

## Overview
The DBOC (Dynamic Bounding Object Control) chunk was introduced in Shadowlands (9.0.1.33978) and appears to contain parameters related to dynamic object bounding. The purpose and full functionality of this chunk is not completely documented, but it has been observed in various models including cloud effects and banner models.

## Structure
```cpp
struct DBOC_Entry {
  float unk1_1;    // First unknown float value
  float unk1_2;    // Second unknown float value
  uint32_t unk1_3; // First unknown 32-bit value
  uint32_t unk1_4; // Second unknown 32-bit value
};

struct DBOC_Chunk {
  DBOC_Entry entries[];  // Array of DBOC entries
}
```

## Fields
- **unk1_1**: First unknown float value, purpose not fully documented
- **unk1_2**: Second unknown float value, purpose not fully documented
- **unk1_3**: First unknown 32-bit value, purpose not fully documented
- **unk1_4**: Second unknown 32-bit value, purpose not fully documented

## Dependencies
- Requires the MD21 chunk for basic model data
- May interact with model collision or bounding information

## Usage
The DBOC chunk has been observed in:
- Nebula cloud effect models (e.g., 9ARD_NebulaCloud_B01.m2)
- Banner models (e.g., 10be_bloodelf_rf_banner01.m2)

Though its exact purpose is not well-documented, the name and structure suggest it may relate to:
- Dynamic adjustment of model bounding volumes
- Special handling for objects with changing boundaries
- Parameter control for objects that need dynamic collision or culling

## Legacy Support
- Not present in pre-Shadowlands M2 files
- The functionality it provides may have been handled differently or not available in earlier versions

## Implementation Notes
- The chunk size varies between models, with some having 32 bytes (possibly 2 entries) and others having 16 bytes (1 entry)
- The implementation should be flexible enough to handle variable chunk sizes
- Without complete documentation, the exact application of these values requires experimentation and observation
- May be related to dynamic object culling or specialized collision detection

## Version History
- Introduced in Shadowlands (9.0.1.33978)
- Observed in various model types with potentially dynamic boundaries
- Part of ongoing specialized rendering and simulation features in the M2 format 