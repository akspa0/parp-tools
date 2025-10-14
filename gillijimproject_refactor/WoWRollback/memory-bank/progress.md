# Progress - WoWRollback

## ✅ Completed (2025-10-14)

### 3D Terrain Mesh Extraction Pipeline
- ✅ Created `AdtMpqTerrainExtractor` - Extracts MCNK data from ADTs in MPQs
- ✅ Created `AdtMeshExtractor` - Generates GLB 3D terrain meshes per tile
- ✅ Integrated WoWFormatLib for ADT geometry parsing
- ✅ Added SharpGLTF for GLB export
- ✅ Implemented mesh manifest JSON generation
- ✅ Added Step 5 (terrain) & Step 6 (mesh) to analysis pipeline
- ✅ Integrated mesh copying into unified viewer workflow
- ✅ Fixed Sedimentary Layers performance (97% reduction via viewport culling)
- ✅ Added shift-click range selection to layer checkboxes

### New Files Created
- `WoWRollback.AnalysisModule/AdtMpqTerrainExtractor.cs` - MCNK extraction from MPQs
- `WoWRollback.AnalysisModule/AdtMeshExtractor.cs` - GLB mesh generation

### Files Modified
- `WoWRollback.AnalysisModule/WoWRollback.AnalysisModule.csproj` - Added WoWFormatLib + SharpGLTF
- `WoWRollback.AnalysisModule/AnalysisViewerAdapter.cs` - Added `CopyTerrainMeshesToViewer()`
- `WoWRollback.Cli/Program.cs` - Added Step 5 & 6 to `AnalyzeSingleMapNoViewer()`
- `ViewerAssets/js/sedimentary-layers-csv.js` - Performance fix + shift-click selection
- `README.md` - Documented 3D mesh extraction feature

### Git Status
- Branch: `wrb-poc3b`
- Last Commit: `1ecd378` - "Terrain MCNK layers refactor"

## ✅ Completed (2025-10-12 - Previous Session)

### Viewer Overlay Coordinate System Fix
- ✅ Fixed critical bug: Objects/clusters appearing in wrong tiles
- ✅ Implemented `ComputeActualTile()` to compute owning tile from coordinates
- ✅ ADT coordinates are ABSOLUTE from map corner (0,0), not tile-local
- ✅ Removed double-filtering bug in `ViewerReportWriter`
- ✅ `ClusterOverlayBuilder` now groups by computed tile
- ✅ Filter dummy markers early (UID=0 spam eliminated)
- ✅ Fixed `TerrainOverlayBuilder` CSV parsing column indices
- ✅ Viewer now correctly displays objects and clusters

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
1. ✅ ~~Fix terrain extraction bug~~ - DONE! `AdtMpqTerrainExtractor` working
2. ✅ ~~Create `IArchiveSource` abstraction~~ - Already existed!
3. ✅ ~~Implement mesh extraction~~ - DONE! `AdtMeshExtractor` working
4. **Build 3D viewer** - Three.js/Babylon.js viewer for GLB meshes
5. **Test with large maps** - Verify performance with Azeroth/Kalimdor

### Future: 3D Viewer
- Load GLB meshes on-demand from `mesh_manifest.json`
- Render placement markers in 3D space
- Camera controls (orbit, pan, zoom)
- Reuse 2D viewer placement data
- Toggle between 2D and 3D views

## 📊 Current Status

**Progress**: ~90% Complete (Core features implemented)

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
