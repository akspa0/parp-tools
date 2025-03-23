# DETL Chunk (Detail Lights)

## Overview
The DETL (Detail Lights) chunk was introduced in Shadowlands (9.0.1.34365) and contains additional detail parameters for light sources in M2 models. The chunk provides various settings that affect how lights are rendered, including scale factors for shadow rendering and color multipliers.

## Structure
```cpp
struct DETL_Entry {
  /*0x00*/  uint16_t flags;                      // Flags controlling light behavior
  /*0x02*/  float16 scale;                       // Scale for shadow RT matrix in half-float format
  /*0x04*/  float16 diffuseColorMultiplier;      // Multiplier for M2Light.diffuse_color in half-float format
  /*0x06*/  uint16_t unknown0;                   // Unknown value
  /*0x08*/  uint32_t unknown1;                   // Unknown value
};

struct DETL_Chunk {
  DETL_Entry entries[m2data.header.lights.count]; // One entry per light in the model
}
```

## Fields
- **flags**: Bit flags controlling the behavior of the light
- **scale**: Scale factor for the shadow render target matrix, stored in half-float format
- **diffuseColorMultiplier**: Multiplier applied to the light's diffuse color, stored in half-float format
- **unknown0**: Unknown 16-bit value, purpose not documented
- **unknown1**: Unknown 32-bit value, purpose not documented

## Dependencies
- Requires the MD21 chunk for basic model data
- Corresponds to light definitions in the main M2 data
- The number of entries matches the number of lights defined in the model

## Usage
The DETL chunk is used to:
- Provide enhanced control over light rendering in models
- Adjust shadow rendering parameters for specific lights
- Fine-tune the color output of model lights
- Potentially control other advanced lighting features

## Legacy Support
- Not present in pre-Shadowlands M2 files
- Earlier versions used only the basic light parameters defined in the M2Light structure

## Implementation Notes
- Each entry corresponds to a light defined in the model's lights array
- The half-float format values need to be properly converted during rendering
- The scale value affects the shadow render target matrix calculation
- The diffuseColorMultiplier is applied to the light's diffuse color value
- The flags may control various aspects of the light's behavior, though specific flag meanings are not documented
- Implementation requires supporting half-float values and integrating with the shadow rendering system

## Version History
- Introduced in Shadowlands (9.0.1.34365)
- Represents an enhancement to the M2 format's lighting capabilities
- Part of ongoing improvements to lighting quality and performance in WoW 