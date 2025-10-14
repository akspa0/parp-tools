# Active Context - WoWRollback Terrain Mesh Extraction & 3D Visualization

## Current Focus (2025-10-14)
**3D Terrain Mesh Extraction & Viewer Integration**

Successfully implemented complete terrain mesh extraction pipeline that generates GLB files for 3D visualization. System now extracts placements, terrain data (MCNK), AND 3D meshes from ADTs in MPQs, with full integration into the unified viewer workflow.

## What We Just Fixed (Session: 2025-10-12)

### ✅ Critical Bug Fixed: Overlay Coordinate System

**The Problem:**
- ADT placement coordinates are **absolute offsets from map corner (0,0)**, not tile-local
- Dark Portal at coords `(1086, 1099)` was stored in ADT files `development_1_1.adt` and `development_2_1.adt`
- But actual world coords `(15980, 15966)` place it in tile `(2,2)`!
- Objects and clusters were appearing in wrong tiles on the viewer

**The Root Cause:**
- `ViewerReportWriter` was pre-filtering entries by CSV tile (from ADT filename)
- `OverlayBuilder` was re-filtering by computed actual tile (from coordinates)
- **Double filtering = nothing matched!**

**The Fix:**
1. Created `ComputeActualTile()` - computes owning tile from coordinates instead of trusting ADT filename
2. Removed pre-filtering in `ViewerReportWriter` - passes ALL version entries to OverlayBuilder
3. `OverlayBuilder` filters by computed tile, correctly placing objects
4. `ClusterOverlayBuilder` now also groups by computed tile (same fix)
5. Filter dummy markers early to prevent UID=0 spam

### ✅ Key Technical Insights

1. **ADT Coordinate System**
   - Placement coords in MDDF/MODF chunks are ABSOLUTE from map NW corner `(0,0)`
   - NOT relative to the ADT tile!
   - Must compute: `tileCol = floor(32 - worldX/533.33)`, `tileRow = floor(32 - worldY/533.33)`

2. **Cross-Tile Objects**
   - Objects spanning tile boundaries appear in MULTIPLE ADT files
   - Same UniqueID, same coordinates, different ADT files
   - Deduplication by UniqueID prevents duplicates

3. **Dummy Tile Markers**
   - Tiles without placements get dummy entry: `UID=0`, `AssetPath="_dummy_tile_marker"`
   - Must filter early to avoid processing overhead

## What We Just Implemented (Session: 2025-10-14)

### ✅ 3D Terrain Mesh Extraction Pipeline

**New Components:**
1. **`AdtMpqTerrainExtractor.cs`** - Extracts MCNK terrain data from ADTs in MPQs
   - Reads terrain CSV: AreaID, flags, liquids, holes, impassible
   - Outputs: `{mapName}_terrain.csv`

2. **`AdtMeshExtractor.cs`** - Generates 3D terrain meshes from ADTs
   - Uses WoWFormatLib's `ADTReader` to parse ADT geometry
   - Extracts vertices (17x17 grid per chunk, 145 vertices with padding)
   - Builds triangle indices (skips holes)
   - Exports to GLB format using SharpGLTF
   - Outputs: `{mapName}_mesh/tile_{x}_{y}.glb` + `mesh_manifest.json`

3. **Analysis Pipeline Integration** - Added Step 5 & 6 to `AnalyzeSingleMapNoViewer()`
   - Step 5: Extract terrain data (MCNK chunks)
   - Step 6: Extract terrain meshes (GLB)

4. **Unified Viewer Integration** - `GenerateUnifiedViewer()` now:
   - Generates terrain overlays (AreaIDs, liquids, holes)
   - Generates cluster overlays
   - **Copies mesh files to viewer output** ← NEW!

**Output Structure:**
```
{outputDir}/
├── {mapName}_placements.csv
├── {mapName}_terrain.csv
├── {mapName}_mesh/                    ← NEW!
│   ├── tile_30_41.glb
│   ├── tile_30_42.glb
│   └── mesh_manifest.json
└── viewer/
    └── overlays/{version}/{mapName}/
        ├── terrain_complete/
        ├── clusters/
        └── mesh/                      ← NEW!
            ├── tile_30_41.glb
            └── mesh_manifest.json
```

### ✅ MPQ Archive Reading
- `IArchiveSource` abstraction already existed in `WoWRollback.Core.Services.Archive`
- `PrioritizedArchiveSource` handles loose files + MPQ priority
- `AdtMpqTerrainExtractor` and `AdtMeshExtractor` use `IArchiveSource`
- Full MPQ reading support implemented!

## Next Steps (For Fresh Session)

### Immediate Tasks
1. ✅ ~~Fix terrain extraction~~ - DONE! `AdtMpqTerrainExtractor` working
2. ✅ ~~Create `IArchiveSource` abstraction~~ - Already existed!
3. ✅ ~~Implement mesh extraction~~ - DONE! `AdtMeshExtractor` working
4. **Build 3D viewer** - Three.js/Babylon.js viewer for GLB meshes
5. **Test with large maps** - Verify performance with Azeroth/Kalimdor

### Future: 3D Viewer Implementation
- Load GLB meshes on-demand from `mesh_manifest.json`
- Render placement markers in 3D space
- Camera controls (orbit, pan, zoom)
- Reuse 2D viewer placement data
- Toggle between 2D and 3D views

## CRITICAL: WoW File Resolution Priority

**WoW reads files in this order:**
1. ✅ **Loose files in Data/ folders** (HIGHEST priority)
2. ✅ Patch MPQs (patch-3.MPQ > patch-2.MPQ > patch.MPQ)
3. ✅ Base MPQs

**Why This Matters:**
- Players exploited this for model swapping (giant campfire = escape geometry)
- `md5translate.txt` can exist in BOTH MPQ and `Data/textures/Minimap/md5translate.txt`
- **Implementation MUST check filesystem BEFORE MPQ**

**Existing Infrastructure:**
- `StormLibWrapper/MpqArchive.cs` - Open/read MPQ
- `StormLibWrapper/MPQReader.cs` - Extract files
- `StormLibWrapper/DirectoryReader.cs` - Auto-detect patch chain
- `MpqArchive.AddPatchArchives()` - Automatic patching

**What We Need:**
```csharp
interface IArchiveSource {
    bool FileExists(string path);
    Stream OpenFile(string path);
}

class PrioritizedArchiveSource : IArchiveSource {
    // Check loose files FIRST, then delegate to MpqArchive
}
```

## Git Status
- **Branch**: `wrb-poc3b`
- **Last Commit**: `1ecd378` - "Terrain MCNK layers refactor"
- **Test Data**: `test_data/development/World/Maps/development/`
- **Output**: `analysis_output/viewer/` (served at `http://localhost:8080`)

## Files Modified This Session (2025-10-14)
- `WoWRollback.AnalysisModule/AdtMpqTerrainExtractor.cs` - NEW! Extracts MCNK data from MPQs
- `WoWRollback.AnalysisModule/AdtMeshExtractor.cs` - NEW! Generates GLB terrain meshes
- `WoWRollback.AnalysisModule/WoWRollback.AnalysisModule.csproj` - Added WoWFormatLib + SharpGLTF
- `WoWRollback.AnalysisModule/AnalysisViewerAdapter.cs` - Added terrain/cluster/mesh generation
- `WoWRollback.Cli/Program.cs` - Added Step 5 (terrain) & Step 6 (mesh) to analysis pipeline
- `ViewerAssets/js/sedimentary-layers-csv.js` - Fixed performance (viewport culling) + shift-click selection

## What Works Now
✅ Objects appear in correct tiles on viewer  
✅ Clusters appear in correct tiles  
✅ Cross-tile objects deduplicated properly  
✅ Terrain extraction from MPQs (MCNK chunks)  
✅ 3D mesh extraction (GLB format)  
✅ Mesh files copied to viewer output  
✅ MPQ reading with loose file priority  
✅ Sedimentary Layers performance fixed (97% reduction!)  
✅ Shift-click range selection in layer checkboxes  

## What's Next
⏳ Build 3D viewer (Three.js/Babylon.js)  
⏳ Test with large maps (Azeroth/Kalimdor)  
⏳ OBJ export option (alternative to GLB)
