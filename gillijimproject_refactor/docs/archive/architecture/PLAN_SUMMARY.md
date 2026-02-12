# Expanded MCNK & AreaID Overlay Plan - Summary

## Overview

This document summarizes the **expanded** design for complete MCNK terrain and AreaID boundary visualization in WoWRollback.

---

## What Changed?

### Original Scope (mcnk-flags-overlay.md)
- **Minimal** MCNK extraction: impassible flag + holes only
- 2 overlay types (impassible, holes)
- 10-column CSV

### Expanded Scope (mcnk-complete-overlay.md + areaid-overlay.md + mcsh-shadows-overlay.md) ⭐
- **Complete** MCNK extraction: ALL flags + liquids + AreaID + positions
- **5 overlay categories**:
  1. **Terrain Properties** (impassible, vertex colors)
  2. **Baked Shadows (MCSH)** ⭐ 64×64 shadow bitmaps composited to 1024×1024 PNG per tile
  3. **Liquids** (river, ocean, magma, slime)
  4. **Holes** (terrain holes with resolution detection)
  5. **AreaID Boundaries** (zone transitions with area names)
- 23-column CSV with complete metadata
- Separate shadow CSV with base64-encoded bitmaps

---

## Architecture

### Data Pipeline

```
┌─────────────────────────────────────────────────────┐
│  Stage 1: Extraction (AlphaWDTAnalysisTool)         │
├─────────────────────────────────────────────────────┤
│  Parse MCNK chunks from ADT files                   │
│  Extract ALL 32 flag bits + metadata fields         │
│  Output: <map>_mcnk_terrain.csv (23 columns)        │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│  Stage 2: Transformation (WoWRollback.Core)         │
├─────────────────────────────────────────────────────┤
│  Read CSV → Group by tile → Detect boundaries       │
│  Build 5 overlay types (terrain, shadows, liquids, holes, areas)
│  Composite shadow bitmaps → 1024×1024 PNG data URL  │
│  Output: terrain_complete/tile_r*_c*.json           │
│          shadows/tile_r*_c*.json (PNG data URLs)     │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│  Stage 3: Visualization (ViewerAssets)              │
├─────────────────────────────────────────────────────┤
│  Load overlay JSON per visible tile                 │
│  Render 5 layer groups:                             │
│    - Terrain properties (rectangles)                │
│    - Baked shadows (1024×1024 PNG image overlay)    │
│    - Liquids (colored rectangles)                   │
│    - Holes (black rectangles)                       │
│    - AreaID boundaries (lines + labels)             │
│  UI: Toggles + opacity slider for shadows           │
└─────────────────────────────────────────────────────┘
```

---

## CSV Schema: `<map>_mcnk_terrain.csv`

| # | Column | Type | Description |
|---|--------|------|-------------|
| 1 | `map` | string | Map name (e.g., "Azeroth") |
| 2 | `tile_row` | int | ADT tile row (0-63) |
| 3 | `tile_col` | int | ADT tile column (0-63) |
| 4 | `chunk_row` | int | MCNK chunk row within tile (0-15) |
| 5 | `chunk_col` | int | MCNK chunk column within tile (0-15) |
| 6 | `flags_raw` | hex | Full 32-bit flags value |
| 7 | `has_mcsh` | bool | Has shadow map (MCSH) |
| 8 | `impassible` | bool | ⭐ Terrain is impassible |
| 9 | `lq_river` | bool | ⭐ Contains river liquid |
| 10 | `lq_ocean` | bool | ⭐ Contains ocean liquid |
| 11 | `lq_magma` | bool | ⭐ Contains magma liquid |
| 12 | `lq_slime` | bool | ⭐ Contains slime liquid |
| 13 | `has_mccv` | bool | Has vertex colors (MCCV) |
| 14 | `high_res_holes` | bool | Uses 64-bit hole map |
| 15 | `areaid` | int | ⭐ AreaTable.dbc ID |
| 16 | `num_layers` | int | Number of texture layers (0-4) |
| 17 | `has_holes` | bool | ⭐ Has terrain holes |
| 18 | `hole_type` | enum | "none", "low_res", "high_res" |
| 19 | `hole_bitmap_hex` | hex | Hole bitmap (16 or 64 bits) |
| 20 | `hole_count` | int | Number of holes |
| 21 | `position_x` | float | ⭐ World X coordinate of chunk center |
| 22 | `position_y` | float | ⭐ World Y coordinate of chunk center |
| 23 | `position_z` | float | ⭐ World Z coordinate of chunk center |

**⭐ = New/expanded fields beyond minimal implementation**

---

## Overlay JSON Structure

### Combined Overlay (Single File)

```json
{
  "map": "Azeroth",
  "tile": {"row": 31, "col": 34},
  "minimap": {"width": 512, "height": 512},
  "chunk_size": 32,
  "layers": [
    {
      "version": "0.5.3",
      "terrain_properties": {
        "impassible": [{"row": 0, "col": 0}],
        "shadow_mapped": [{"row": 1, "col": 1}],
        "vertex_colored": [{"row": 2, "col": 2}]
      },
      "liquids": {
        "river": [{"row": 3, "col": 3}],
        "ocean": [{"row": 4, "col": 4}],
        "magma": [],
        "slime": []
      },
      "holes": [
        {"row": 5, "col": 5, "type": "low_res", "holes": [0, 1, 4, 5]}
      ],
      "area_ids": {
        "chunks": [
          {"row": 0, "col": 0, "area_id": 1519, "area_name": "Stormwind City"}
        ],
        "boundaries": [
          {
            "from_area": 1519,
            "from_name": "Stormwind City",
            "to_area": 12,
            "to_name": "Elwynn Forest",
            "chunk_row": 0,
            "chunk_col": 2,
            "edge": "east"
          }
        ]
      }
    }
  ]
}
```

**Output Path**: `overlays/<version>/<map>/terrain_complete/tile_r<row>_c<col>.json`

---

## Visualization Design

### 1. Terrain Properties Layer

**Toggle**: "Terrain Properties"  
**Sub-toggles**: Impassible, Shadow Mapped, Vertex Colored

**Rendering**:
- Impassible: Red rectangle, `rgba(255, 0, 0, 0.3)`
- Shadow Mapped: Gray rectangle, `rgba(128, 128, 128, 0.2)`
- Vertex Colored: Blue rectangle, `rgba(0, 128, 255, 0.2)`

### 2. Liquids Layer

**Toggle**: "Liquids"  
**Sub-toggles**: Rivers, Oceans, Magma, Slime

**Rendering**:
- River: Light blue, `rgba(64, 164, 223, 0.4)`
- Ocean: Deep blue, `rgba(0, 64, 164, 0.5)`
- Magma: Orange-red, `rgba(255, 69, 0, 0.6)`
- Slime: Green, `rgba(0, 255, 0, 0.4)`

### 3. Holes Layer

**Toggle**: "Terrain Holes"

**Rendering**:
- Holes: Black rectangles, `rgba(0, 0, 0, 0.7)`
- Grid: 4×4 (low-res) or 8×8 (high-res) per chunk

### 4. AreaID Boundaries Layer

**Toggle**: "Area Boundaries"  
**Sub-toggles**: Show Names, Color By Area, Show Boundary Lines

**Rendering**:
- Boundary lines: Gold, `rgba(255, 215, 0, 0.8)`, 3px width
- Area fill (optional): Hash AreaID to consistent color, `opacity: 0.1`
- Labels: Area name at chunk center, dark background with gold text
- Popups: Show from/to area names and IDs

---

## Implementation Estimate

| Phase | Tasks | Time Estimate |
|-------|-------|---------------|
| Phase 1 | Design & Documentation | **2 hours** ✅ COMPLETE |
| Phase 2 | Extraction (AlphaWDTAnalysisTool) | **6 hours** |
| | - McnkTerrainExtractor.cs | 3 hours |
| | - CSV writer + CLI integration | 2 hours |
| | - README updates | 1 hour |
| Phase 3 | Transformation (WoWRollback.Core) | **5 hours** |
| | - CSV reader + models | 1 hour |
| | - 4 overlay builders | 3 hours |
| | - VersionComparisonService integration | 1 hour |
| Phase 4 | Visualization (ViewerAssets) | **6 hours** |
| | - terrainPropertiesLayer.js | 1.5 hours |
| | - liquidsLayer.js | 1.5 hours |
| | - holesLayer.js | 1 hour |
| | - areaIdLayer.js | 1.5 hours |
| | - UI controls + CSS | 0.5 hours |
| Phase 5 | Testing & Documentation | **2 hours** |
| | - End-to-end testing | 1 hour |
| | - README updates | 1 hour |
| **TOTAL** | | **~21 hours** |

---

## Key Features

### 1. Complete Flag Coverage
- All 32 MCNK flag bits extracted
- Future-proof for additional flag types

### 2. Liquid Type Differentiation
- Separate visualization for each liquid type
- Color-coded for easy identification

### 3. AreaID Intelligence
- Automatic boundary detection between zones
- Integration with existing AreaTable mapping
- Area name display from AreaTable.dbc

### 4. Flexible UI
- Independent toggles for each overlay type
- Sub-toggles for granular control
- Persistent layer state across navigation

### 5. Performance Optimized
- Lazy loading per visible tile
- 500ms debouncing on pan/zoom
- Aggressive tile unloading (>2 tiles away)
- Combined JSON reduces HTTP requests

---

## Files Created

### Design Documents (Phase 1) ✅
- `docs/architecture/overlay-system-architecture.md`
- `docs/architecture/mcnk-flags-overlay.md` (minimal version)
- `docs/architecture/mcnk-complete-overlay.md` ⭐ (complete version)
- `docs/architecture/areaid-overlay.md` ⭐ (area boundaries)
- `docs/architecture/IMPLEMENTATION_ROADMAP.md` (updated for expanded scope)
- `docs/architecture/README.md` (updated)
- `docs/architecture/PLAN_SUMMARY.md` (this file)

### To Be Created (Phases 2-4)
- `AlphaWdtAnalyzer.Core/McnkTerrainExtractor.cs`
- `WoWRollback.Core/Models/McnkModels.cs`
- `WoWRollback.Core/Services/McnkTerrainCsvReader.cs`
- `WoWRollback.Core/Services/Viewer/TerrainPropertiesOverlayBuilder.cs`
- `WoWRollback.Core/Services/Viewer/LiquidsOverlayBuilder.cs`
- `WoWRollback.Core/Services/Viewer/HolesOverlayBuilder.cs`
- `WoWRollback.Core/Services/Viewer/AreaIdOverlayBuilder.cs`
- `ViewerAssets/js/overlays/terrainPropertiesLayer.js`
- `ViewerAssets/js/overlays/liquidsLayer.js`
- `ViewerAssets/js/overlays/holesLayer.js`
- `ViewerAssets/js/overlays/areaIdLayer.js`
- `ViewerAssets/js/overlayManager.js`

---

## Success Criteria

- [ ] All MCNK flags extract correctly from Alpha ADTs
- [ ] CSV output validates against 23-column schema
- [ ] AreaID values match existing `areaid_verify_*.csv` data
- [ ] All 4 overlay types build successfully
- [ ] Terrain properties render correctly
- [ ] Liquid types display with correct colors
- [ ] Holes render at correct positions (4×4 and 8×8)
- [ ] AreaID boundaries detect correctly between zones
- [ ] Area names display from AreaTable.dbc
- [ ] All UI toggles work independently
- [ ] Performance < 100ms per tile (all overlays combined)
- [ ] Multi-overlay combinations display correctly
- [ ] Documentation complete and examples working

---

## Additional Features

### Asset Counter / Statistics Panel

**Purpose**: Real-time viewport statistics

**Features**:
- Object count (total, M2, WMO)
- Tiles loaded
- Current version/map
- Zoom level and coordinates
- Collapsible panel (top-right)

**Implementation**: `ViewerAssets/js/statsPanel.js`

---

### Complete Map Processing

**Purpose**: Process ALL maps from Map.dbc, not just a subset

**Features**:
- Auto-discovery from Map.dbc
- Minimap validation (skip maps without minimaps)
- Batch/parallel processing
- Progress reporting
- Map type filtering (continents, dungeons, raids, etc.)

**CLI Command**:
```bash
--process-all-maps --dbc-path Map.dbc --wdt-root <path>
```

**Implementation**: `AlphaWdtAnalyzer.Core/MapDiscovery.cs`

---

### AreaTable Integration

**Important**: AreaID overlay uses **existing AreaTable CSVs** from AlphaWDTAnalysisTool

**Files Generated by AlphaWDTAnalysisTool**:
- `AreaTable_Alpha.csv` (Alpha 0.5.3/0.5.5 area names)
- `AreaTable_335.csv` (LK 3.3.5 area names)

**WoWRollback Integration**:
- Read existing CSVs (don't re-extract from DBC)
- Generate combined `areatables.json` for viewer
- Support both Alpha and LK area names
- Display Alpha names by default (with LK fallback)

**Benefits**:
- No duplicate DBC parsing
- Consistent with existing workflow
- Shows original Alpha area names
- LK names available for reference

---

## Next Steps

**Type `ACT` when ready to begin Phase 2 implementation** (Extraction).

The roadmap is ready to execute with:
- Complete specifications for all 5 overlay types
- Asset counter design
- All-maps processing design
