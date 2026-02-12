# WoW Alpha Client Deep Dive Analysis

This directory contains comprehensive documentation about the internal workings of the World of Warcraft Alpha client (version 0.5.3), derived from reverse engineering the WoWClient.exe binary using Ghidra.

## Table of Contents

1. [MDX Animation System](./01_MDX_Animation_System.md)
   - Animation chunk structure and parsing
   - Keyframe types and interpolation
   - Animation sequences and playback
   - Geoset animations and visibility

2. [Particle System](./02_Particle_System.md)
   - Particle emitter types and configuration
   - Particle properties and physics
   - Texture sheet animation
   - Wind and follow effects

3. [BLP Texture Format](./03_BLP_Texture_Format.md)
   - BLP header structure
   - Color encoding and compression
   - Alpha channel handling
   - Mipmap generation and selection

4. [Shader and Rendering System](./04_Shader_Rendering_System.md)
   - Blend modes and transparency
   - Shader program management
   - Texture binding and rendering
   - Material properties

5. [Model Rendering Pipeline](./05_Model_Rendering_Pipeline.md)
   - Model loading and initialization
   - Animation system integration
   - Transparency sorting
   - Optimization techniques

## Key Findings Summary

### Animation System
- Uses a hierarchical animation track system (ATSQ chunk)
- Supports four interpolation types: Linear, Hermite, Bezier, and Constant
- Quaternion-based rotation with compressed storage
- Global sequence support for synchronized animations
- Geoset animations for visibility, color, and alpha effects

### Particle System
- Multiple emitter types: Base, Plane, Sphere, Spline
- Quad and vertex particle geometry
- Texture sheet animation for sprite effects
- Wind vector and follow model support
- Particle keyframe animations for dynamic effects

### BLP Texture Format
- DXT1/DXT3/DXT5 compression support
- Fallback formats for unsupported compression
- Alpha channel detection and handling
- Mipmap-based LOD selection

### Rendering System
- Four blend modes: Disable, Blend, Add, AlphaKey
- Hardware-accelerated rendering via DirectX
- Shader-based material system
- Efficient rendering with visibility culling

## Tools Used
- Ghidra 11.x for binary reverse engineering
- IDA Pro for verification (if available)
- Custom analysis scripts for chunk identification

## Methodology
1. Function identification via pattern matching
2. Decompilation and structural analysis
3. Cross-reference tracing
4. Consistency verification across related functions
5. Documentation with code references

## Revision History
- v1.0 - Initial documentation (2026-02-07)
  - Animation system analysis
  - Particle system structure
  - BLP texture handling
  - Shader and blend modes
