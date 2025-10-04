# Architecture Documentation

## Overview

This directory contains comprehensive design documentation for the WoWRollback overlay system and data extraction pipeline.

---

## Documents

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

**Purpose**: Complete implementation specification for MCNK terrain flags overlay

**Contents**:
- ADT structure reference (MCNK chunk, flags, holes)
- CSV schema design
- Extraction implementation (`McnkFlagsExtractor.cs`)
- Transformation implementation (`McnkFlagsOverlayBuilder.cs`)
- Visualization implementation (`terrainFlagsLayer.js`)
- UI controls and styling
- Testing checklist

**Use this as a template** when implementing new overlay types.

---

### 3. [IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md)

**Purpose**: Step-by-step implementation plan for MCNK flags overlay

**Contents**:
- Phase breakdown (Design → Extraction → Transformation → Visualization → Testing)
- Task lists with file-level detail
- Code snippets and examples
- Success criteria
- Timeline estimates

**Follow this roadmap** when ready to implement the MCNK flags overlay.

---

## Quick Start Guide

### Adding a New Overlay Type

1. **Study existing implementation**
   - Read `overlay-system-architecture.md` for overall design
   - Review `mcnk-flags-overlay.md` as a template
   
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
