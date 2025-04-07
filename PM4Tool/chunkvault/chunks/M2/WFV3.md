# WFV3 Chunk (Waterfall Version 3)

## Overview
The WFV3 (Waterfall Version 3) chunk represents the most advanced iteration of the specialized PBR (Physically Based Rendering) system for M2 models. It was introduced in Shadowlands (9.0.1.33978) and extends the functionality of the WFV1 and WFV2 chunks with additional rendering parameters and greater control over the visual appearance.

## Structure
```cpp
struct WFV3_Chunk {
  float bumpScale;       // Normal map scale factor, passed to vertex shader
  float value0_x;        // Shader parameter
  float value0_y;        // Shader parameter
  float value0_z;        // Shader parameter
  float value1_w;        // Shader parameter
  float value0_w;        // Shader parameter
  float value1_x;        // Shader parameter
  float value1_y;        // Shader parameter
  float value2_w;        // Shader parameter
  float value3_y;        // Shader parameter
  float value3_x;        // Shader parameter
  CImVector basecolor;   // Base color in RGBA format (not BGRA)
  uint16_t flags;        // Rendering flags
  uint16_t unk0;         // Unknown value
  float values3_w;       // Shader parameter
  float values3_z;       // Shader parameter
  float values4_y;       // Shader parameter
  float unk1;            // Unknown value
  float unk2;            // Unknown value
  float unk3;            // Unknown value
  float unk4;            // Unknown value
};
```

## Fields
- **bumpScale**: Controls the intensity of normal mapping, passed directly to the vertex shader
- **value0_x/y/z**, **value0_w**, **value1_x/y**, **value1_w**, **value2_w**, **value3_x/y**, **values3_z/w**, **values4_y**: Shader parameters passed directly to the fragment shader for rendering calculations
- **basecolor**: Base color of the material in RGBA format (differs from the usual BGRA format in WoW)
- **flags**: Rendering flags that modify the behavior of the shader
- **unk0**, **unk1**, **unk2**, **unk3**, **unk4**: Unknown values, possibly reserved for future use or internal calculations

## Dependencies
- Requires the MD21 chunk for basic model data
- Works with the TXID chunk for texture references, particularly normal maps
- May depend on specific textures in specific slots for normal mapping and other PBR effects

## Usage
The WFV3 chunk is used to:
- Enable advanced PBR rendering for select models
- Provide precise control over material properties and visual appearance
- Configure normal mapping intensity and other visual parameters
- Support specialized effects for environmental models like foliage, water, and atmospheric elements

## Legacy Support
- Not present in pre-Shadowlands M2 files
- Supersedes the WFV1 and WFV2 chunks from Battle for Azeroth
- Represents a significant advancement in M2 model rendering capability

## Implementation Notes
- Despite being named "Waterfall", this technology was used for various environmental elements in Shadowlands, not just water effects
- The various value fields are passed directly to shader programs with minimal processing
- FileDataID 3445776 is noted as an example model using this chunk in Shadowlands
- The base color is stored in RGBA format, unlike most other color values in WoW which use BGRA
- Implementation should handle the transition between WFV versions appropriately, using the correct shader variant for each
- The chunk name continues the "waterfall" terminology from earlier iterations despite broader application

## Version History
- Introduced in Shadowlands (9.0.1.33978)
- Builds upon the foundation established by WFV1 and WFV2 in Battle for Azeroth
- Represents the most advanced version of this specialized rendering system as of Shadowlands 