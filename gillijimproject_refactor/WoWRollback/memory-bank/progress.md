# Progress - WoWRollback

## ✅ Completed (2025-10-12)

### Viewer Overlay Coordinate System Fix
- ✅ Fixed critical bug: Objects/clusters appearing in wrong tiles
- ✅ Implemented `ComputeActualTile()` to compute owning tile from coordinates
- ✅ ADT coordinates are ABSOLUTE from map corner (0,0), not tile-local
- ✅ Removed double-filtering bug in `ViewerReportWriter`
- ✅ `ClusterOverlayBuilder` now groups by computed tile
- ✅ Filter dummy markers early (UID=0 spam eliminated)
- ✅ Fixed `TerrainOverlayBuilder` CSV parsing column indices
- ✅ Viewer now correctly displays objects and clusters

### Files Modified
- `WoWRollback.Core/Services/Viewer/OverlayBuilder.cs`
- `WoWRollback.Core/Services/Viewer/ViewerReportWriter.cs`
- `WoWRollback.Core/Services/Viewer/ClusterOverlayBuilder.cs`
- `WoWRollback.Core/Services/Viewer/TerrainOverlayBuilder.cs`
- `WoWRollback.AnalysisModule/AdtTerrainExtractor.cs`

### Git Status
- Branch: `wrb-poc3b`
- Last Commit: `f2ee2f8` - "Minor fixes to cluster overlays"

## ✅ Completed (2025-10-07 - Previous Session)

### Phase 1: Module Architecture (Day 1)
- ✅ Created `WoWRollback.DbcModule` - Wraps DBCTool.V2 as library API
- ✅ Created `WoWRollback.AdtModule` - Wraps AlphaWdtAnalyzer.Core
- ✅ Created `WoWRollback.ViewerModule` - Embedded HTTP server with HttpListener
- ✅ All three modules build successfully

### Phase 2: Infrastructure (Day 2)
- ✅ Populated `WoWRollback.Core` with shared utilities:
  - `IO/FileHelpers.cs` - Directory operations
  - `Logging/ConsoleLogger.cs` - Structured logging
  - `Models/SessionManifest.cs` - Session metadata
- ✅ Fixed `SessionManager` to use correct output structure:
  - Numbered directories: `01_dbcs/`, `02_crosswalks/`, `03_adts/`, `04_analysis/`, `05_viewer/`
  - Removed wrong `shared_outputs/` concept
- ✅ Updated `DbcStageRunner`, `AdtStageRunner`, `ManifestWriter` to use new paths
- ✅ All projects build successfully

### Phase 3: Wire Modules into Orchestrator (Day 3)
- ✅ Refactored `DbcStageRunner` to use `DbcOrchestrator` API
  - No more direct CLI command instantiation
  - Calls `DumpAreaTables()` and `GenerateCrosswalks()` library methods
- ✅ Refactored `AdtStageRunner` to use `AdtOrchestrator` API
  - Simplified to call `ConvertAlphaToLk()` with `ConversionOptions`
  - Returns structured result with tile/area counts
- ✅ Implemented `ViewerStageRunner` with HTML and overlay generation
  - Generates `index.html` with session summary
  - Creates `overlays/metadata.json` with ADT results
- ✅ Wired `ViewerServer` into `Program.cs`
  - Starts HTTP server if `--serve` flag provided
  - Blocks until Ctrl+C for graceful shutdown

## 🎯 Next Steps (Next Session)

### Immediate Priorities
1. **Fix terrain extraction bug** - `AdtTerrainExtractor` returning 0 chunks
2. **Remove `terrain_complete` viewer code** - `terrainPropertiesLayer.js` and references
3. **Create `IArchiveSource` abstraction** - Wrap existing `MpqArchive` with loose file priority

### Enhanced Archive Analysis (See `plans/enhanced-archive-analysis.md`)
1. Phase 1: Archive reading (MPQ + loose files)
2. Phase 2: DBC export & map discovery
3. Phase 3: WDT parsing (map types)
4. Phase 4: Detailed terrain analysis (full MCNK subchunks)
5. Phase 5: CLI redesign (`analyze-archive` command)

## 📊 Current Status

**Progress**: ~83% Complete (10/12 tasks done)

### Architecture Status
```
WoWRollback/
├─ DbcModule/          ✅ Created & builds
├─ AdtModule/          ✅ Created & builds
├─ ViewerModule/       ✅ Created & builds with HTTP server
├─ Core/               ✅ Populated with utilities
└─ Orchestrator/       ✅ REFACTORED
   ├─ DbcStageRunner   ✅ Uses DbcOrchestrator API
   ├─ AdtStageRunner   ✅ Uses AdtOrchestrator API
   ├─ ViewerStageRunner ✅ Generates HTML + metadata
   └─ Program.cs        ✅ Wired ViewerServer with --serve
```

### Output Structure Status
✅ **Fixed**: Now matches spec exactly
```
parp_out/
└─ session_YYYYMMDD_HHMMSS/
   ├─ 01_dbcs/           ✅ Correct
   ├─ 02_crosswalks/     ✅ Correct
   ├─ 03_adts/           ✅ Correct
   ├─ 04_analysis/       ✅ Correct
   ├─ 05_viewer/         ✅ Correct
   ├─ logs/              ✅ Correct
   └─ manifest.json      ✅ Correct
```

## 🐛 Known Issues
- ❌ **Terrain extraction returns 0 chunks** - `AdtFormatDetector.EnumerateMapTiles()` not finding files
- ❌ **terrain_complete overlay broken** - Needs removal and replacement
- ❌ **No MPQ reading** - Only works with extracted/loose files (but `StormLibWrapper` exists!)
- ❌ **No loose file priority** - WoW checks Data/ folders BEFORE MPQs
- ❌ **No WDT parsing** - Can't detect WMO-only maps (Karazhan, instances)
- ❌ **Basic MCNK extraction only** - Missing subchunk data (MCVT, MCLY, MCLQ, etc.)

## ✨ Current Capabilities
- [x] Analyze extracted ADT files
- [x] Generate viewer with correct overlay coordinates
- [x] Serve viewer at http://localhost:8080
- [x] Cross-tile object deduplication
- [x] Cluster spatial analysis
- [ ] MPQ archive reading (infrastructure exists, not integrated)
- [ ] Loose file priority handling
- [ ] DBC export to JSON
- [ ] WDT parsing for map type detection
- [ ] Detailed MCNK terrain analysis
- [ ] WMO-only map support (instances)
