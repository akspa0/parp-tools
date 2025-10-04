# Phase 0 Progress - 2D Foundation Completion

## ✅ Completed

### 1. Terrain CSV Path Fix
- **Issue**: Terrain CSVs weren't being found (404 errors for terrain_complete files)
- **Root Cause**: `rebuild-and-regenerate.ps1` was looking in wrong subdirectory
- **Fix**: Updated path to `csv/{MapName}/{MapName}_mcnk_terrain.csv`
- **Status**: ✅ WORKING (green checkmarks in rebuild script)

### 2. AreaTable Mapping Fix (IMPLEMENTED - NEEDS TESTING)
- **Issue**: Area boundaries show "Unknown Area 1234" instead of real names
- **Root Cause**: Alpha AreaIDs don't match 3.3.5 AreaTable.dbc structure
- **Solution**: Extract AreaIDs from converted LK ADTs (which have correct IDs)
- **Files Created/Modified**:
  - ✅ `AlphaWdtAnalyzer.Core/Terrain/LkAdtAreaReader.cs` (NEW)
  - ✅ `AlphaWdtAnalyzer.Core/Terrain/McnkTerrainExtractor.cs` (UPDATED)
  - ✅ `AlphaWdtAnalyzer.Core/AnalysisPipeline.cs` (UPDATED)
  - ✅ `AlphaWdtAnalyzer.Cli/Program.cs` (UPDATED)
- **Status**: ✅ COMPILED, ⏳ NEEDS TESTING

## ⏳ In Progress

### 3. Shadow Map Overlay
- **Issue**: Shadow map data extracted but no UI layer
- **Solution**: Implement MCSH decoder + viewer layer
- **Files Needed**:
  - `AlphaWdtAnalyzer.Core/Terrain/McshDecoder.cs` (NEW)
  - `WoWRollback.Core/Services/Viewer/McnkShadowOverlayBuilder.cs` (NEW)
  - `ViewerAssets/js/overlays/shadowMapLayer.js` (NEW)
  - Updates to `ViewerAssets/index.html` and `overlayManager.js`
- **Status**: ⏸️ PLANNED (waiting for AreaTable fix testing)

---

## How AreaTable Fix Works

### Data Flow

```
1. Alpha WDT file
   ↓
2. Convert to LK ADTs (AlphaWdtAnalyzer --export-adt)
   ↓ AlphaWdtAnalyzer applies Alpha→LK AreaID mapping during conversion
3. LK ADT files with correct AreaIDs
   ↓
4. LkAdtAreaReader extracts AreaIDs from LK ADTs
   ↓
5. McnkTerrainExtractor combines:
   - Flags, liquids, holes from Alpha WDT
   - AreaIDs from LK ADTs
   ↓
6. CSV with LK AreaIDs
   ↓
7. WoWRollback reads CSV
   ↓
8. AreaTableLookup finds area names
   ↓
9. Browser shows "Elwynn Forest" ✅ (not "Unknown Area 12" ❌)
```

### Key Components

#### LkAdtAreaReader
- Reads MCNK chunks from converted LK ADT files
- Extracts AreaID field (offset 56 in MCNK header)
- Returns list of (TileRow, TileCol, ChunkY, ChunkX, AreaId)

#### McnkTerrainExtractor.ExtractTerrainWithLkAreaIds()
- Extracts all terrain data from Alpha WDT (flags, liquids, holes)
- Calls LkAdtAreaReader to get LK AreaIDs
- Merges Alpha terrain data with LK AreaIDs
- Returns complete terrain entries with correct AreaIDs

#### AnalysisPipeline
- New option: `LkAdtDirectory` (optional)
- If provided: uses `ExtractTerrainWithLkAreaIds()`
- If not provided: uses `ExtractTerrain()` (Alpha AreaIDs, shows warnings)

#### Program.cs Execution Order
**BEFORE FIX**:
```
1. AnalysisPipeline.Run() → Extract terrain (Alpha AreaIDs)
2. AdtExportPipeline → Convert ADTs to LK
```

**AFTER FIX**:
```
1. AdtExportPipeline → Convert ADTs to LK
2. Detect LK ADT directory
3. AnalysisPipeline.Run(lkAdtDirectory) → Extract terrain (LK AreaIDs)
```

---

## Testing Instructions

### Test AreaTable Fix

```powershell
cd WoWRollback

# Clean previous outputs
Remove-Item cached_maps, rollback_outputs -Recurse -Force -ErrorAction SilentlyContinue

# Rebuild with small map for quick testing
.\rebuild-and-regenerate.ps1 `
  -Maps @("DeadminesInstance") `
  -Versions @("0.5.3.3368") `
  -AlphaRoot ..\test_data\ `
  -RefreshCache `
  -Serve
```

### Expected Output

**In console during rebuild**:
```
[cache] Building LK ADTs for 0.5.3.3368/DeadminesInstance
[area] Reading AreaIDs from X LK ADT files in DeadminesInstance
[area] Extracted AreaIDs for X chunks from X tiles
[McnkTerrainExtractor] Extracting terrain with LK AreaIDs from DeadminesInstance
[McnkTerrainExtractor] Replaced X AreaIDs with LK values, 0 not found in LK ADTs
[debug] ✓ Terrain CSV created: ...\DeadminesInstance_mcnk_terrain.csv
[cache] Copied terrain CSV to rollback_outputs
```

**In browser** (http://localhost:8080):
1. Open DeadminesInstance map
2. Enable "Area Boundaries" overlay
3. **Expected**: Real area names appear (e.g., "The Deadmines", "Ironclad Cove")
4. **NOT**: "Unknown Area 1581" or similar

### Validation Checklist

- [ ] Console shows `[area] Reading AreaIDs from X LK ADT files`
- [ ] Console shows `[McnkTerrainExtractor] Replaced X AreaIDs with LK values`
- [ ] No warnings about `using Alpha AreaIDs`
- [ ] Green checkmark for terrain CSV creation
- [ ] CSV was copied to rollback_outputs
- [ ] Browser: Area boundaries show real names
- [ ] Browser: No console errors
- [ ] Test with larger map (Azeroth) to verify zone names (Elwynn, Durotar, etc.)

---

## Known Issues / Limitations

### Pre-Existing Warnings (Not Related to Fix)
- `FileStream.Read` inexact read warnings (lines 207, 217 in McnkTerrainExtractor.cs)
  - Should use `ReadExactly()` but outside scope of this fix
  - Low priority, doesn't affect functionality

### Edge Cases Handled
1. **No LK ADTs available**: Falls back to Alpha AreaIDs with warning
2. **Missing chunks in LK ADTs**: Keeps Alpha AreaID for missing chunks
3. **Directory doesn't exist**: Graceful fallback with console warning

### Not Yet Implemented
- Shadow map overlay (Phase 0, lower priority)
- Performance optimization for large maps
- Additional overlay types (heightmap, vertex colors standalone)

---

## Next Steps

### Immediate (This Session)
1. ⏳ **TEST AreaTable fix** with DeadminesInstance
2. ⏳ Verify area names appear in browser
3. ⏳ Test with larger map (Azeroth) if time permits
4. ⏳ Document any issues found

### Next Session
1. ⏸️ Implement shadow map overlay (if AreaTable works)
2. ⏸️ Performance optimization (parallel processing, caching)
3. ⏸️ Full test suite for all maps
4. ⏸️ Begin Phase 1 (3D terrain foundation)

---

## Success Criteria for Phase 0

- [x] Terrain CSV path fix (DONE)
- [x] AreaTable mapping implementation (DONE - needs testing)
- [ ] AreaTable mapping verified (area names show correctly)
- [ ] Shadow map overlay (optional)
- [ ] No console errors in viewer
- [ ] All overlays working smoothly
- [ ] Documentation complete
- [ ] Ready to start Phase 1 (3D)

---

## Files Modified Summary

### AlphaWDTAnalysisTool
- `AlphaWdtAnalyzer.Core/Terrain/LkAdtAreaReader.cs` (NEW - 150 lines)
- `AlphaWdtAnalyzer.Core/Terrain/McnkTerrainExtractor.cs` (UPDATED - added ExtractTerrainWithLkAreaIds)
- `AlphaWdtAnalyzer.Core/AnalysisPipeline.cs` (UPDATED - added LkAdtDirectory option)
- `AlphaWdtAnalyzer.Cli/Program.cs` (UPDATED - reordered ADT export before analysis)

### WoWRollback
- `rebuild-and-regenerate.ps1` (UPDATED - fixed terrain CSV path, added debugging)
- `ViewerAssets/js/overlays/areaIdLayer.js` (PREVIOUSLY FIXED - hide() bug)

### Documentation
- `docs/planning/02-remaining-terrain-features.md` (NEW)
- `docs/planning/03-3d-viewer-vision.md` (NEW)
- `docs/planning/04-phase0-final-fixes.md` (NEW)
- `docs/planning/PHASE0_PROGRESS.md` (THIS FILE)
- `docs/architecture/TERRAIN_OVERLAY_FIX.md` (CREATED)
- `docs/architecture/TERRAIN_DEBUGGING.md` (CREATED)
- `docs/architecture/MCSH_SHADOW_MAP_FORMAT.md` (CREATED)

---

## Estimated Completion

- **AreaTable Testing**: 30 min - 1 hour
- **Shadow Maps**: 2-3 hours (if we proceed)
- **Total Phase 0**: ~95% complete! 🎉

Ready to test the AreaTable fix once the current rebuild completes!
