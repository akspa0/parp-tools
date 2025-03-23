# ArcaneFileParser.Core Project Context

## Project Overview
ArcaneFileParser.Core is a .NET library for parsing World of Warcraft file formats. It provides a robust, extensible framework for reading and interpreting various WoW file formats including WDT, WMO, M2, and MDX files.

## Core Components

### Common Types
- [x] Vector2F - 2D vector type
- [x] Vector3F - 3D vector type
- [x] Matrix4x4F - 4x4 transformation matrix
- [x] QuaternionF - Quaternion for rotations
- [x] ColorRGBA/ColorBGRA - Color types
- [x] BoundingBox - Spatial bounds

### IO System
- [x] BinaryReaderExtensions - Core reading utilities
  - [x] Basic type reading (strings, arrays)
  - [x] Math type reading (vectors, matrices)
  - [x] Chunk reading helpers
  - [x] Padding and alignment utilities

### Chunk System
- [x] IChunk - Base chunk interface
- [x] ChunkBase - Common chunk functionality
- [x] VersionedChunkBase - Version-aware chunks
- [x] ChunkSignature - FourCC utilities

### File Format System
- [x] FileFormatBase - Base format handler
  - [x] Chunk management
  - [x] Parsing infrastructure
  - [x] Reporting capabilities

### Asset Path Validation
- [x] ListfileManager - FileDataID lookup and path validation
- [x] AssetPathValidator - Advanced path validation and pattern matching
  - [x] Legacy format support (.mdx -> .m2)
  - [x] Path pattern validation
  - [x] Missing file tracking
  - [x] Suggested fixes for common issues
  - [x] Detailed validation reporting

### File Format Handlers
- [x] WDT Format
  - [x] MVER Chunk - Version information
  - [x] MAIN Chunk - Map tile data
  - [x] MPHD Chunk - Map header information
  - [x] MODF Chunk - Model placement data
  - [x] MWID Chunk - Model path validation
    - [x] Path validation and reporting
    - [x] FileDataID lookup
    - [x] Legacy format support
    - [x] Pattern matching
    - [x] Missing file tracking

- [x] WMO Format
  - [x] Version Support
    - [x] v14 (Alpha) Format
      - [x] MOMO Container Chunk
      - [x] MOLM/MOLD Lightmap System
      - [x] Alpha-specific Chunks
    - [x] v17 (Retail) Format
      - [x] Modern Chunk Structure
      - [x] Vertex Lighting System
  - [x] Common Chunks
    - [x] MVER - Version Information
    - [x] MOHD - Header Information
    - [x] MOTX - Texture Names
    - [x] MOMT - Materials
    - [x] MOGN - Group Names
    - [x] MOGI - Group Information
    - [x] MOSB/MOSI - Skybox
    - [x] MODS - Doodad Sets
    - [x] MODN/MODI - Model Names
    - [x] MODD - Doodad Data
    - [x] MFOG - Fog Information
    - [x] MCVP - Convex Volume Planes
  - [x] Group File Support
    - [x] MOGP - Group Header
    - [x] MOPY - Material Info
    - [x] MOVI - Indices
    - [x] MOVT - Vertices
    - [x] MONR - Normals
    - [x] MOTV - Texture Coordinates
    - [x] MOBA - Render Batches

- [ ] M2 Format
  - [ ] Version Support
    - [ ] Pre-Legion (Non-chunked)
      - [ ] Classic (v256-257)
      - [ ] TBC (v260-263)
      - [ ] WotLK (v264)
      - [ ] Cata-WoD (v265-272)
    - [ ] Legion+ (Chunked)
      - [ ] Legion-Shadowlands (v272-274)
  - [ ] Core Chunks
    - [ ] MD20/MD21 - Main Data
    - [ ] SFID - Skin File IDs
    - [ ] AFID - Animation File IDs
    - [ ] BFID - Bone File IDs
    - [ ] TXAC - Texture Animation
    - [ ] EXPT/EXP2 - Extended Particles
  - [ ] Animation System
    - [ ] Global Sequences
    - [ ] Animation Sequences
    - [ ] Bones and Attachments
    - [ ] Vertex Deformation
  - [ ] Rendering
    - [ ] Materials and Textures
    - [ ] Transparency
    - [ ] Particle Systems
    - [ ] Ribbons

- [ ] MDX Format
  - [ ] Core Structure
    - [ ] MDLX Header
    - [ ] Model Info Block
    - [ ] Sequence Block
    - [ ] Global Sequence Block
    - [ ] Texture Block
    - [ ] Layer Block
    - [ ] Material Block
    - [ ] Geoset Block
    - [ ] GeosetAnimation Block
  - [ ] Animation Components
    - [ ] Bone Block
    - [ ] Light Block
    - [ ] Helper Block
    - [ ] Attachment Block
    - [ ] PivotPoint Block
    - [ ] EventObject Block
    - [ ] CollisionShape Block
  - [ ] Particle Systems
    - [ ] ParticleEmitter Block
    - [ ] ParticleEmitter2 Block
    - [ ] RibbonEmitter Block
  - [ ] Camera System
    - [ ] Camera Block
    - [ ] CameraTrack Block
  - [ ] Version Handling
    - [ ] Format Variations
    - [ ] Compatibility Checks
  - [ ] Resource Management
    - [ ] Texture Path Resolution
    - [ ] External Resource Loading
    - [ ] Team Color Support

## Initial Questions
1. Which file format should we implement first?
2. Do we need additional mathematical operations in our core types?
3. Should we add validation for chunk relationships?
4. Do we need to implement caching for shared resources?
5. Should we add support for writing validated paths back to files?
6. Do we need additional pattern matching for specific asset types?
7. How should we handle version-specific features in WMO and M2 formats?
8. Should we implement a unified animation system across formats?
9. How should we handle MDX-specific features not present in M2 format?
10. Do we need specialized texture handling for MDX team colors?

## Closing Questions
1. Are all file format handlers properly tested?
2. Have we documented format-specific quirks and edge cases?
3. Is error handling comprehensive across all parsers?
4. Do we need additional reporting or debugging features?
5. Should we implement batch validation for multiple files?
6. Do we need to optimize the path validation system for large files?
7. Are version-specific features properly isolated?
8. Do we need migration tools between format versions?

## Implementation Notes
- Using row-major order for matrices to match WoW format
- BGRA color format support for WoW's preferred ordering
- Versioned chunk system for format evolution support
- Efficient stream positioning and chunk navigation
- Legacy format support for pre-WotLK assets (.mdx -> .m2)
- Comprehensive path validation with pattern matching
- Missing file tracking and reporting system
- Support for community listfile integration
- Version-specific feature isolation
- Unified animation system design
- Efficient chunk relationship validation
- Resource caching strategy
- Dedicated MDX parsing pipeline
- Team color texture management
- MDX-specific animation handling

# WMO Format Parser Project

## Project Overview
The WMO Format Parser is a comprehensive C# implementation for parsing and manipulating World of Warcraft's WMO (World Map Object) file format. It supports versions from v14 through v17+ with full backwards compatibility.

## Current Status
- Implementation Progress: 34/44 chunks completed (77%)
- Version Support: v14 - v17+
- Testing Status: Basic validation implemented
- Documentation: In progress

## Completed Features
1. Core Functionality
   - Version detection and handling
   - Basic chunk parsing
   - File reading/writing
   - Validation system

2. Portal System
   - Portal definitions and references
   - Culling logic
   - Portal-group relationships

3. Group Data
   - Group information and headers
   - Material mappings
   - Batch rendering data

4. Visibility System
   - Visible vertex handling
   - Block visibility
   - Render batch processing

5. Legacy Lighting
   - Lightmap vertices (v14)
   - Index lists
   - Material attributes

6. Version-Specific Features
   - UV animations
   - Skybox information
   - Doodad placement

7. Volume System
   - Box volumes
   - Ambient volumes
   - Volume groups
   - Validation

8. Modern Lighting System
   - Spot lights (MOLS)
   - Light points (MOLP)
   - Light references (MOLR)
   - Light validation

## In Progress
1. Volume References
   - Box volume references (MBVR)
   - Fog volume references (MFVR)
   - Ambient volume references (MAVR)

2. Testing and Validation
   - Unit test development
   - Integration testing
   - Performance benchmarks

## Implementation Status

### WMO Format Support
- Version 14 (Alpha): ✅ Complete
- Version 17 (Retail): ✅ Complete
- Total Chunks: 48/48 implemented (100%)

### Features
- Core WMO Structure: ✅ Complete
- Group System: ✅ Complete
- Portal System: ✅ Complete
- Modern Lighting System: ✅ Complete
- Volume System: ✅ Complete
- Legacy Support: ✅ Complete
- Conversion Utilities: 🔄 In Progress

### Next Phase
- Implement WMO v17 to v14 conversion utility
- Add comprehensive validation
- Optimize chunk processing
- Document cross-chunk dependencies

### Version Support Matrix
| Feature            | v14 | v17 |
|-------------------|-----|-----|
| Core Chunks       | ✅  | ✅  |
| Group System      | ✅  | ✅  |
| Portal System     | ✅  | ✅  |
| Lighting System   | ✅  | ✅  |
| Volume System     | ❌  | ✅  |
| Material System   | ✅  | ✅  |
| Doodad System     | ✅  | ✅  |
| Fog System        | ✅  | ✅  |

## Implementation Approach
1. Versioning
   - Clear version detection
   - Backwards compatibility
   - Version-specific validation

2. Validation
   - Per-chunk validation
   - Cross-chunk reference validation
   - Version-specific rules

3. Performance
   - Efficient memory usage
   - Optimized reading/writing
   - Lazy loading where appropriate

4. Extensibility
   - Clean interfaces
   - Modular design
   - Easy version updates

## Usage Requirements
- .NET 8.0+
- C# 12.0+
- Windows/Linux compatible

## Contributing
1. Follow C# coding standards
2. Include XML documentation
3. Add appropriate tests
4. Update validation rules

## Questions to Ask
1. Are there any undocumented version differences?
2. How should we handle unknown future versions?
3. What performance optimizations are needed?
4. Are there any missing validation rules? 

!!! 23 March 2025 !!!
# Session Summary - WMO v17 to v14 Converter Implementation

## Key Accomplishments
1. Implemented complete `WmoV17ToV14Converter` with collision validation
2. Added flag conversion mappings for MOHD and MOGP chunks
3. Implemented conversion methods for all critical WMO chunks
4. Added validation checks for collision integrity

## Technical Details
- Implemented conversion for root chunks:
  - MVER (version set to 14)
  - MOHD (header with v14 flags)
  - Portal chunks (MOPT, MOPV, MOPR)
  - Lighting chunks (MOLT → MOLM/MOLD)
  - Doodad chunks (MODS, MODN, MODD)

- Implemented group chunk conversion:
  - MOGP with v14 flags
  - Geometry chunks (MOPY, MOVI, MOVT, MONR, MOTV)
  - BSP tree chunks (MOBN, MOBR) for collision
  - Batch conversion (MOBA → v14 format)

## Implementation Status
- ✅ Collision validation
- ✅ Root chunk conversion
- ✅ Group chunk conversion
- ✅ Flag mappings
- ✅ Lighting conversion
- ✅ Portal system preservation
- ✅ BSP tree integrity

## Next Steps
1. Consider adding unit tests for collision validation
2. Implement performance optimizations if needed
3. Add detailed logging for conversion process
4. Consider adding batch processing capabilities

## Open Questions
1. Should we add more detailed validation for specific chunk types?
2. Do we need to optimize the lightmap conversion process?
3. Should we add support for custom validation rules?