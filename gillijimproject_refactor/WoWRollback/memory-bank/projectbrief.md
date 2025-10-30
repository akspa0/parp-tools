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
- **Terrain Hole Management** - Automatically fixes MCNK flags when buildings were removed, so ground appears intact
- **Shadow Disabling** - Optional: remove baked shadows (MCSH) that might look weird
- **MD5 Checksums** - Auto-generates .md5 files for minimap compatibility
- **Stupid Easy UI** - Drag a slider, pick a version, click a button. Done.
- **Unified Pipeline (Alpha→LK)** - One command: Alpha WDT → patched Alpha WDT → patched LK ADTs
- **CSV Crosswalk AreaIDs** - Patch `MCNK.AreaId` using CSV crosswalks (`--crosswalk-dir|--crosswalk-file`). No DBCTool dependency at runtime.
- **Mapping Fallbacks** - Support `--area-remap-json` or write 0 for unmapped; use `Map.dbc` only to resolve target map guard (no heuristics).

### Latest Direction (2025-10-27)
- CLI-first with GUI as runner (Load → Build → Layers); no modals; overlay + inline logs; auto-tab navigation.
- Energy-efficient preflight: skip-if-exists for LK ADTs, crosswalk CSVs, tile layers, and layers.json; reuse cache to save time/energy.
- Presets: management in Settings; Load Preset control on Load page.
- CSV handling: adopt CsvHelper in GUI for robust parsing; keep CSV schemas stable.
- BYOD: do not include copyrighted game data; rely on user-provided sources.

### Latest Direction (2025-10-29)
- CASC/DB2 discovery: when Map.dbc is unavailable, read Map.db2 via DBCD over CASC with FDID listfile; fallback to WDT scan using listfile enumeration.
- Versioned listfiles:
  - JSON snapshots per client (`snapshot-listfile --client-path <dir> --alias <major.minor.patch.build> --out <json>`)
  - Diff utilities (`diff-listfiles --a <fileA> --b <fileB> --out <dir>`) for added/removed/changed FDIDs
- Asset gating for recompile/export:
  - Outputs must reference assets present in the 3.3.5 target listfile; non-present assets are dropped and reported
  - Integrated in `pack-monolithic-alpha-wdt` for MDNM/MONM; placements to follow
- Alias policy: use full build strings (major.minor.patch.build) from .build.info/DBD/path heuristics

### Latest Direction (2025-10-25)
- **CLI-first**: `WoWRollback.Cli` is the primary entrypoint. Orchestrator is legacy.
- **Strict AreaID mapping (non-pivot by default)**: map-locked numeric → target map name → per-source-map numeric → exact-by-src-number only if strict=false; 0.6.0 pivot opt-in via `--chain-via-060`.
- **LK WDT emission**: Write `<Map>.wdt` alongside LK ADTs in the output folder.

## Use Cases
- **Empty World Screenshots**: Remove all objects for terrain-only views
- **Historical Comparisons**: See what Azeroth looked like in patch 0.5.3 vs 1.0 vs 3.3.5
- **Content Analysis**: Identify which objects were added in which development phases
- **Machinima/Photography**: Create clean environments for video production

## Success Criteria
- ✅ Successfully bury objects in 0.5.3 WDT files (PROVEN - works on Kalimdor!)
- ✅ Clear terrain holes conservatively (only chunks whose referenced objects were all buried)
- ✅ Export LK ADTs with correct indices and AreaIDs via CSV crosswalks; explicitly write 0 when unmapped
- ✅ Emit LK WDT (`<Map>.wdt`) together with LK ADTs
- ✅ Enforce strict, non-pivot AreaID mapping order; pivot gated by `--chain-via-060`
- ⏳ Generate pre-rendered overlay images for all UniqueID ranges
- ⏳ Build lightweight HTML viewer with slider control
- ⏳ One-button rollback with all options exposed

## Next Milestone
- Run the pipeline directly from the viewer UI (Tools panel): configure global/per-tile UniqueID ranges, terrain options, crosswalk rules; launch builds and stream logs; open results in viewer.
