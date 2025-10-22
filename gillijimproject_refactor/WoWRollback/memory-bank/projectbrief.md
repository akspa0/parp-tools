# WoWRollback - Time-Travel Your WoW Maps

## Project Vision
WoWRollback is a **map modification tool** that enables users to "roll back" World of Warcraft maps to earlier development states by selectively removing object placements based on UniqueID thresholds. Think of it as a time machine for WoW's game world.

## Core Purpose
1. **Analyze Maps**: Scan WDT/ADT files to discover UniqueID distribution (what content exists and when it was added)
2. **Visualize Layers**: Pre-generate minimap overlays showing different historical "snapshots" of content
3. **Modify Maps**: Bury unwanted objects below the terrain by UniqueID threshold
4. **Fix Terrain**: Clear terrain hole flags where buildings were removed, so ground appears intact

## Key Features
- **Works on 0.5.3 through 3.3.5** - Tested and confirmed working on earliest Alpha builds!
- **Pre-generated Overlays** - No on-the-fly rendering; fast, lightweight viewer
- **Terrain Hole Management** - Automatically fixes MCNK flags when buildings are removed
- **Shadow Disabling** - Optional: remove baked shadows (MCSH) that might look weird
- **MD5 Checksums** - Auto-generates .md5 files for minimap compatibility
- **Stupid Easy UI** - Drag a slider, pick a version, click a button. Done.
- **Unified Pipeline (Alpha→LK)** - One command: Alpha WDT → patched Alpha WDT → patched LK ADTs
- **AreaTable Mapping** - Apply `MCNK.AreaId` via JSON mapping or auto-fill from LK client MPQs

## Use Cases
- **Empty World Screenshots**: Remove all objects for terrain-only views
- **Historical Comparisons**: See what Azeroth looked like in patch 0.5.3 vs 1.0 vs 3.3.5
- **Content Analysis**: Identify which objects were added in which development phases
- **Machinima/Photography**: Create clean environments for video production

## Success Criteria
- ✅ Successfully bury objects in 0.5.3 WDT files (PROVEN - works on Kalimdor!)
- ✅ Clear terrain holes conservatively (only chunks whose referenced objects were all buried)
- ✅ Export LK ADTs with correct indices and applied AreaIDs (via mapping)
- ⏳ Generate pre-rendered overlay images for all UniqueID ranges
- ⏳ Build lightweight HTML viewer with slider control
- ⏳ One-button rollback with all options exposed
