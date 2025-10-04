# Architecture Documentation

## Overview

This directory contains comprehensive design documentation for the WoWRollback overlay system and data extraction pipeline.

---

## Documents

### 0. [PLAN_SUMMARY.md](PLAN_SUMMARY.md) 📋 **START HERE**

**Purpose**: Executive summary of the complete MCNK & AreaID overlay plan

**Contents**:
- What changed (minimal → expanded scope)
- Complete CSV schema (23 columns)
- JSON structure for all 4 overlay types
- Visualization design with color schemes
- Implementation timeline (~21 hours)
- Success criteria checklist

**Read this first** for a quick overview of the expanded plan.

---

### 1. [overlay-system-architecture.md](overlay-system-architecture.md)

**Purpose**: Complete architectural overview of the 3-stage overlay pipeline

**Contents**:
- System overview and data flow
- Stage 1: Extraction (AlphaWDTAnalysisTool)
- Stage 2: Transformation (WoWRollback.Core)
- Stage 3: Visualization (ViewerAssets)
- Design principles and conventions
- Performance guidelines
- Checklist for adding new overlays

**Read this first** to understand the overall system design.

---

### 2. [mcnk-flags-overlay.md](mcnk-flags-overlay.md)

**Purpose**: Original implementation specification for MCNK terrain flags (impassible + holes only)

**Contents**:
- ADT structure reference (MCNK chunk, flags, holes)
- CSV schema design (minimal: impassible + holes)
- Extraction implementation (`McnkFlagsExtractor.cs`)
- Transformation implementation (`McnkFlagsOverlayBuilder.cs`)
- Visualization implementation (`terrainFlagsLayer.js`)

**Note**: This is the **minimal** implementation. See `mcnk-complete-overlay.md` for the **expanded** version.

---

### 2b. [mcnk-complete-overlay.md](mcnk-complete-overlay.md) ⭐

**Purpose**: **Complete** MCNK terrain extraction including all flags, liquids, and AreaID

**Contents**:
- Complete MCNK structure (all 32 flag bits + metadata fields)
- Expanded CSV schema (23 columns: all flags, liquids, holes, AreaID, positions)
- Extraction implementation (`McnkTerrainExtractor.cs`)
- Multi-category overlay builders (terrain properties, liquids, holes)
- Visualization modules (4 separate layer types)
- Complete UI control panel design
- Color scheme definitions

**Use this** for the actual implementation - it supersedes `mcnk-flags-overlay.md`.

---

### 2c. [areaid-overlay.md](areaid-overlay.md) ⭐

**Purpose**: AreaID boundary detection and visualization

**Contents**:
- Alpha vs. LK AreaID encoding explanation
- Boundary detection algorithm
- Area name lookup from AreaTable.dbc
- AreaID overlay JSON schema
- Visualization with boundary lines and labels
- AreaTable config generation
- Integration with existing `areaid_verify_*.csv` data

**Key Feature**: Visualizes zone boundaries as colored lines with area names.

---

### 2d. [mcsh-shadows-overlay.md](mcsh-shadows-overlay.md) ⭐ **NEW**

**Purpose**: MCSH baked shadow map extraction and visualization

**Contents**:
- MCSH subchunk structure (64×64 bit shadow map per chunk)
- Shadow bitmap extraction from MCSH data
- Base64 encoding for CSV storage
- Server-side shadow compositing (16×16 chunks → 1024×1024 PNG)
- PNG data URL generation for viewer
- Image overlay rendering with opacity control
- Alternative client-side canvas rendering approach

**Key Feature**: Composites 256 chunk shadow bitmaps into a single 1024×1024 grayscale PNG overlay per tile, rendered as semi-transparent image layer.

---

### 3. [IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md)

**Purpose**: Step-by-step implementation plan for **complete** MCNK & AreaID overlays

**Contents**:
- Phase breakdown (Design → Extraction → Transformation → Visualization → Testing)
- Complete task lists with file-level detail
- Code snippets for all 5 overlay types (including shadows)
- Integration with VersionComparisonService
- Success criteria
- Timeline estimates (~25 hours total)

**Follow this roadmap** when ready to implement. Updated to reflect the expanded scope (all MCNK flags + AreaID boundaries + shadow maps).

---

### 4. [viewer-enhancements.md](viewer-enhancements.md) ⭐ **NEW**

**Purpose**: UI/UX improvements and quality-of-life features for the viewer

**Contents**:
- **Statistics Panel**: Real-time asset counter showing visible objects (M2, WMO), tile counts, zoom level, coordinates
- **Complete Map Processing**: Auto-discovery of all maps from Map.dbc, batch processing, minimap validation
- Parallel processing support for faster extraction
- Progress reporting and filtering options
- Integration with WoWRollback for auto-discovery

**Key Features**: 
- Real-time viewport statistics with collapsible panel
- Process all maps automatically instead of hardcoded subset
- Batch extraction with progress tracking

---

## Quick Start Guide

### Adding a New Overlay Type

1. **Study existing implementation**
   - Read `overlay-system-architecture.md` for overall design
   - Review `mcnk-complete-overlay.md` and `areaid-overlay.md` as comprehensive examples
   
2. **Create design document**
   - Copy `mcnk-flags-overlay.md` as a template
   - Document ADT structure from `reference_data/wowdev.wiki/`
   - Design CSV schema
   - Design JSON overlay format
   
3. **Follow 5-phase implementation**
   - Phase 1: Design (create docs)
   - Phase 2: Extraction (AlphaWDTAnalysisTool)
   - Phase 3: Transformation (WoWRollback.Core)
   - Phase 4: Visualization (ViewerAssets)
   - Phase 5: Testing & Documentation
   
4. **Update READMEs**
   - Add feature to `AlphaWDTAnalysisTool/README.md`
   - Add overlay type to `WoWRollback/README.md`
   - Add usage examples

---

## Architecture Principles

### 1. Separation of Concerns
```
Binary ADT Files → CSV (Extraction)
CSV Files → JSON (Transformation)
JSON → Interactive Viewer (Visualization)
```

### 2. CSV as Interchange Format
- Human-readable for debugging
- Versionable in Git
- Tool-agnostic consumption

### 3. Tile-Based Design
- All data relative to tile [row, col]
- Lazy loading for performance
- Parallel processing

### 4. Coordinate System Consistency
- WoW world coords: ±17066.66656 yards
- Tile indices: 0-63 (64×64 grid)
- Tile-relative: 0.0-1.0 (normalized)
- Pixel coords: 0 to (width-1)

---

## File Organization

```
docs/architecture/
├── README.md                           # This file
├── overlay-system-architecture.md      # System overview
├── mcnk-flags-overlay.md              # MCNK implementation spec
├── IMPLEMENTATION_ROADMAP.md          # Step-by-step guide
└── [future overlay designs...]        # liquid-overlay.md, area-overlay.md, etc.
```

---

## Reference Materials

### ADT Format Documentation
`reference_data/wowdev.wiki/ADT_v18.md`

Key sections:
- MHDR chunk (lines 80-119): Main header
- MDDF chunk (lines 170-338): M2 doodad placements
- MODF chunk: WMO placements
- MH2O chunk (lines 347-486): Liquid data
- MCNK chunk (lines 487-598): Map chunk data
  - Flags (lines 500-515): Terrain properties
  - Holes (lines 538, 525, 557-566): Terrain holes

### Coordinate System Reference
`reference_data/wowdev.wiki/ADT_v18.md` (lines 52-58, 283-338)

---

## Common Patterns

### Binary Parsing (Extraction)
```csharp
// Read chunk header
uint flags = BitConverter.ToUInt32(chunkData, 0);

// Extract bit flags
bool impassible = (flags & 0x2) != 0;

// Read coordinates
float worldX = BitConverter.ToSingle(chunkData, 0x08);
```

### Coordinate Transformation
```csharp
// Alpha corner-relative → WoW world coords
const double MAP_HALF_SIZE = 17066.66656;
float worldX = (float)(MAP_HALF_SIZE - alphaX);

// World coords → Tile indices
int tileCol = (int)Math.Floor(32.0 - (worldX / 533.33333));

// World coords → Tile-relative (0-1)
double localX = Frac(32.0 - (worldX / 533.33333));

// Tile-relative → Pixel coords
double pixelX = localX * (width - 1);
```

### JSON Overlay Structure
```json
{
  "map": "Azeroth",
  "tile": {"row": 31, "col": 34},
  "minimap": {"width": 512, "height": 512},
  "layers": [
    {
      "version": "0.5.3",
      "<overlay_type>": {
        // Overlay-specific data
      }
    }
  ]
}
```

---

## Contributing

When adding new architecture documentation:

1. Follow the existing document structure
2. Include code examples
3. Reference specific line numbers from ADT_v18.md
4. Provide testing checklist
5. Update this README with new document link

---

## Questions?

For implementation questions:
1. Check existing design docs
2. Review ADT_v18.md reference material
3. Examine current codebase (placements overlay)
4. Create new design document if breaking new ground
