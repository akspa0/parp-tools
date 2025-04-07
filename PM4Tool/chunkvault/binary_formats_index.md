# Binary Formats Index

## Format Overview

| Format | Status | Description | Documentation |
|--------|--------|-------------|---------------|
| BLP | ✅ Complete | Blizzard's texture format | [BLP.md](BLP.md) |
| LIT | ✅ Complete | Light placement format | [LIT.md](LIT.md) |
| TEX | ✅ Complete | Texture data format | [TEX.md](TEX.md) |
| WFX | ✅ Complete | Weather effects format | [WFX.md](WFX.md) |
| WLW | ✅ Complete | Water Level Water format | [WLW.md](WLW.md) |
| WLM | ✅ Complete | Water Level Magma format | [WLM.md](WLM.md) |
| WLQ | ✅ Complete | Water Level Quality format | [WLQ.md](WLQ.md) |

## Implementation Status

### Texture Formats
- BLP: ✅ Documentation complete, ✅ Parser implemented
- TEX: ✅ Documentation complete, ⏳ Parser in progress

### Lighting Formats
- LIT: ✅ Documentation complete, 🔲 Parser planned

### Effect Formats
- WFX: ✅ Documentation complete, 🔲 Parser planned

### Liquid Formats
- WLW: ✅ Documentation complete, 🔲 Parser planned
- WLM: ✅ Documentation complete, 🔲 Parser planned
- WLQ: ✅ Documentation complete, 🔲 Parser planned

## Format Relationships

### Liquid Format Group
- **WLW** (Water Level Water)
  - Base format for water surfaces
  - Contains heightmap and basic properties
  - Used for most water bodies except oceans

- **WLM** (Water Level Magma)
  - Variant of WLW for magma/lava
  - Identical structure to WLW
  - Fixed liquid type (6)

- **WLQ** (Water Level Quality)
  - Companion format to WLW/WLM
  - Contains additional properties
  - Must have matching WLW/WLM file
  - Different liquid type enumeration

### Texture Format Group
- **BLP**: Primary texture format
- **TEX**: Alternative texture format

### Lighting Format Group
- **LIT**: Light placement and properties

### Effects Format Group
- **WFX**: Weather and environmental effects

## Implementation Priority
1. Texture Formats (BLP, TEX)
2. Liquid Formats (WLW, WLM, WLQ)
3. Lighting Format (LIT)
4. Effects Format (WFX)

## Notes
- All formats follow a consistent documentation structure
- Implementation focuses on format accuracy
- Parsers should handle all documented versions
- Error handling for corrupt or incomplete files
- Cross-format validation where applicable

## Water-Related Formats

### WLW (Water Level Water)
- Purpose: Defines heightmaps for water bodies in the game (except oceans)
- Version History: 0, 1, 2
- Key Features:
  - Grid-based height data
  - Liquid type definitions
  - Block-based structure
  - Coordinate mapping

### WLM (Water Level Magma)
- Purpose: Magma variant of the WLW format
- Version: Same as WLW
- Key Features:
  - Identical structure to WLW
  - Fixed liquid type (magma = 6)
  - Used for lava and magma effects

### WLQ (Water Level Quality)
- Purpose: Quality/property data for water bodies
- Version: Single version
- Key Features:
  - Paired with WLW files
  - Enhanced liquid type definitions
  - Additional property blocks
  - Extended liquid parameters

## Texture Formats

### BLP (Blizzard Picture)
- Purpose: Stores textures with precalculated mipmaps
- Version: BLP2 (version 1)
- Key Features:
  - Multiple compression formats (DXT1, DXT3, DXT5)
  - Palettized and direct color modes
  - Alpha channel support (0-bit, 1-bit, 8-bit)
  - Mipmap level management

### TEX (Texture Blob)
- Purpose: Low-resolution texture storage for distant rendering
- Version: 0 (pre-8.1.5), 1 (post-8.1.5)
- Key Features:
  - Efficient storage of low-res textures
  - Multiple mipmap levels
  - DXT compression support
  - FileDataID integration

## Lighting and Effect Formats

### LIT (Lighting)
- Purpose: Legacy lighting information storage
- Version: 8.3, 8.4, 8.5
- Key Features:
  - Sky color definitions
  - Light source data
  - Time-based color transitions
  - Lighting parameters

### WFX (Warcraft Effects)
- Purpose: Shader definitions for surface rendering
- Version: Single version
- Key Features:
  - Fixed function pipeline support
  - Shader program definitions
  - Render state management
  - Multi-pass rendering

## Format Relationships
- WLW and WLQ files are paired 1:1 with matching paths and filenames
- WLM follows WLW structure but is specifically for magma
- All formats use similar block-based data organization
- BLP files are referenced by most other formats for texturing
- TEX files provide optimized versions of BLP textures
- LIT data now stored in DBC files (Light, LightParams, etc.)
- WFX shaders reference BLS (shader) files

## Implementation Considerations
1. **Version Handling**
   - WLW supports multiple versions (0-2)
   - WLQ has a single version
   - Version affects liquid type interpretation

2. **Block Management**
   - Fixed-size blocks for height data
   - Grid-based vertex organization
   - Efficient memory layout
   - Coordinate system considerations

3. **Liquid Types**
   - WLW/WLM: Basic liquid types
   - WLQ: Extended liquid properties
   - Version-dependent type interpretation
   - Database integration (DB/LiquidType)

4. **Texture Management**
   - BLP loading prioritization
   - TEX fallback system
   - Mipmap generation and validation
   - Compression format handling

5. **Lighting System**
   - Legacy LIT support
   - Modern DBC lighting integration
   - Time-based interpolation
   - Performance optimization

6. **Shader Integration**
   - WFX parsing and validation
   - Render state management
   - Multi-pass coordination
   - Fixed function fallbacks 