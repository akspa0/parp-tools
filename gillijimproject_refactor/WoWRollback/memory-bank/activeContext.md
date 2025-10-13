# Active Context - WoWRollback Viewer Coordinate System & Archive Reading

## Current Focus (2025-10-12)
**Overlay Coordinate System Fixes & Enhanced Archive Analysis Planning**

We successfully fixed critical coordinate system bugs in the viewer overlays where objects and clusters were appearing in wrong tiles. Now planning major enhancement to read from WoW client installations (MPQ + loose files) and generate comprehensive analysis for all maps.

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

## Current Blockers

### ❌ Terrain MCNK Extraction Returning 0 Chunks
- `AdtTerrainExtractor.ExtractTerrainForMap()` returns 0 chunks
- Likely `AdtFormatDetector.EnumerateMapTiles()` not finding files
- Added diagnostics logging but not tested yet
- Each ADT should have 256 MCNK chunks (16x16 grid)

### ❌ No MPQ Archive Reading
- Currently only works with extracted/loose files
- **GOOD NEWS**: `StormLibWrapper` already exists in `lib/WoWTools.Minimaps/`!
- Just need to wrap it with loose-file-priority layer

## Next Steps (For Fresh Session)

### Immediate Tasks
1. **Fix terrain extraction** - Add diagnostics to see why 0 tiles found
2. **Remove `terrain_complete` viewer code** - It's broken, will be replaced
3. **Create `IArchiveSource` abstraction** - Wrap existing `MpqArchive` with loose file priority

### Major Enhancement: WoW Client Archive Reading
See `plans/enhanced-archive-analysis.md` for full 5-phase plan:
- Phase 1: Archive reading (MPQ + loose files with priority)
- Phase 2: DBC export & map discovery
- Phase 3: WDT parsing (map types, tile detection)
- Phase 4: Detailed terrain analysis (full MCNK subchunks)
- Phase 5: CLI redesign (`analyze-archive` command)

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
- **Last Commit**: `f2ee2f8` - "Minor fixes to cluster overlays"
- **Test Data**: `test_data/development/World/Maps/development/`
- **Output**: `analysis_output/viewer/` (served at `http://localhost:8080`)

## Files Modified This Session
- `WoWRollback.Core/Services/Viewer/OverlayBuilder.cs` - Added `ComputeActualTile()`, early dummy filtering
- `WoWRollback.Core/Services/Viewer/ViewerReportWriter.cs` - Removed pre-filtering by CSV tile
- `WoWRollback.Core/Services/Viewer/ClusterOverlayBuilder.cs` - Group by computed tile
- `WoWRollback.AnalysisModule/AdtTerrainExtractor.cs` - Added diagnostics (not tested)
- `WoWRollback.Core/Services/Viewer/TerrainOverlayBuilder.cs` - Fixed CSV parsing column indices

## What Works Now
✅ Objects appear in correct tiles on viewer  
✅ Clusters appear in correct tiles  
✅ Cross-tile objects deduplicated properly  
✅ No more UID=0 spam in logs  
✅ Viewer serves at `http://localhost:8080`  

## What's Still Broken
❌ Terrain extraction (0 chunks)  
❌ terrain_complete overlay (needs removal)  
❌ No MPQ reading (only loose files work)
