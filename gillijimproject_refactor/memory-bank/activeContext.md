# Active Context

## Current Focus (2025-10-20)

### ✅ COMPLETED: MCNK Format Fixes
- **Fixed** `AlphaMcnkBuilder.cs` to correctly handle Alpha v18 format
- **Removed** headers from MCSH/MCAL/MCSE (Alpha expects raw data)
- **Fixed** offset calculations to use raw sizes instead of wrapped sizes
- **Added** `ValidateMclyMcalIntegrity()` to detect offset corruption bugs
- **Build** succeeded with 6 warnings

### 🎯 NEW FOCUS: Layer-Based UniqueID Filtering System
- **Created** `WoWDataPlot` project - lightweight CLI tool for UniqueID archaeology
- **Implemented** proper WoW coordinate system transformations:
  - Added `CoordinateTransform` helper with proper formulas from wowdev.wiki
  - Uses `floor(32 - (axis / 533.33))` for tile index calculation
  - Transform world coords to plot coords for correct top-down orientation
  - Top=North, Right=East, Bottom=South, Left=West (matches in-game maps)
- **Implemented** tile-specific layer detection:
  - Each tile gets its own independent UniqueID layers
  - Layers auto-detected by clustering IDs within tile
  - No more global layers applied to all tiles!
- **Commands**:
  - `visualize`: **UNIFIED PIPELINE** - One command does everything!
    - Extracts placements
    - Detects tile-specific layers
    - Generates per-tile PNGs
    - Creates map-wide overview
    - Saves analysis JSON
  - `export-csv`: Raw data export
  - `plot-uniqueid`: Simple plot (legacy)
- **Uses** existing gillijimproject parsers (WdtAlpha/AdtAlpha) via reflection
- **Workflow**: Single `visualize` command → Review images → Filter → Rollback

## Key Discovery (Gray Dot → Half-Circles Bug)
**Symptom:** Gray texture dot appears as two back-to-back half-circles in-game

**Root Cause:** MCLY `offsetInMCAL` pointing to middle of alpha map data (offset by 2048 bytes)

**Fix Applied:**
1. Removed incorrect headers from MCSH/MCAL/MCSE in Alpha format
2. Fixed offset calculations in `AlphaMcnkBuilder.cs`
3. Added validation to detect when `offsetInMCAL` is exactly 2048 (half of 4096-byte map)

## Format Summary
**Alpha v18 Format:**
- MCLY, MCRF: ✅ Have FourCC+size headers
- MCVT, MCNR, MCSH, MCAL, MCSE, MCLQ: ❌ NO headers (raw data only)

**LK 3.3.5 Format:**
- ALL sub-chunks: ✅ Have FourCC+size headers

## Documentation Created
1. `docs/MCNK-SubChunk-Audit.md` - Complete format comparison
2. `docs/MCAL-Data-Orientation-Bug.md` - Analysis of visual corruption
3. `docs/DataVisualizationTool-Design.md` - Visualization tool architecture
4. `WoWDataPlot/README.md` - Tool usage guide
5. `WoWDataPlot/LAYER_WORKFLOW.md` - Layer-based filtering workflow
6. `WoWDataPlot/Models/LayerInfo.cs` - Layer data models
7. `WoWDataPlot/Extractors/AlphaPlacementExtractor.cs` - Reuses gillijimproject parsers

## Next Steps
1. **Test MCNK fixes** with real Alpha 0.5.3 WDT → in-game client
2. **Build and test WoWDataPlot** with real Kalidar data
3. **Run layer analysis** on Kalidar to create JSON metadata
4. **Generate tile-layer images** for visual inspection
5. **Integrate layer filtering** into WoWRollback.Cli for selective rollback
6. **Create web UI** for interactive layer selection (future)
