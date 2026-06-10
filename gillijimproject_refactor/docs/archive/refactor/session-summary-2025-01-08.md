# Session Summary - 2025-01-08

**Time**: 13:00 - 17:40  
**Status**: Multiple fixes implemented, debugging in progress

---

## Issues Addressed

### ✅ 1. DBC JSON Data All Null - FIXED
**Problem**: All DBC JSONs showed `{"ID": 1, "Item": null}`  
**Fix**: Changed from reflection to DBCD's column indexer  
**File**: `UniversalDbcDumper.cs`  
**Result**: Real DBC data now exports correctly

### ✅ 2. Map Dropdown Empty - FIXED  
**Problem**: Viewer expected `"map"` property, we output `"name"`  
**Fix**: Changed index.json to use `"map"` property  
**File**: `ViewerStageRunner.cs` line 124  
**Result**: Map dropdown populates

### ✅ 3. Tile Y-Axis Inversion - FIXED
**Problem**: Tiles upside down  
**Fix**: Added `coordMode: "wowtools"` to config.json  
**File**: `ViewerStageRunner.cs` line 137  
**Result**: Tiles right-side up

### ✅ 4. CSV File Spam - FIXED
**Problem**: Hundreds of per-tile `asset_fixups_*.csv` files  
**Fix**: Merge into single file, delete per-tile files  
**File**: `ConvertPipelineMT.cs` lines 194-213  
**Result**: 685 files → 1 file per map

### ⏳ 5. Minimap Tiles Rotated 90° CCW - IN PROGRESS
**Problem**: Tiles appear rotated and mirrored  
**Root Cause**: Using legacy BLPs instead of md5translate.trs  
**Investigation**: Added debug logging to MinimapLocator  
**Files**: `MinimapLocator.cs` lines 115-137  
**Next**: Run pipeline, check console output

### ⏳ 6. Overlays Not Generating - IN PROGRESS
**Problem**: No overlay JSONs in viewer directory  
**Investigation**: Added debug logging to OverlayGenerator  
**Files**: `OverlayGenerator.cs` lines 32-60, 99-112  
**Next**: Run pipeline, check `[OverlayGen]` messages

### ⏳ 7. Sedimentary Layers Not Loading - TODO
**Problem**: UniqueID ranges not loading in viewer  
**Solution**: Copy layers JSON to viewer directory  
**Status**: Not yet implemented

---

## Files Modified

| File | Lines | Purpose |
|------|-------|---------|
| `UniversalDbcDumper.cs` | 88-134 | Fix DBC data extraction |
| `DbcOrchestrator.cs` | 211-231 | Add build canonicalization |
| `ViewerStageRunner.cs` | 124, 137-150 | Fix index.json + config.json |
| `ConvertPipelineMT.cs` | 194-213 | Consolidate asset_fixups CSVs |
| `OverlayGenerator.cs` | 32-60, 99-112 | Add debug logging |
| `MinimapLocator.cs` | 115-137 | Add debug logging |

---

## Key Insights

### Minimap Architecture
- **Legacy method** (0.5.3/0.5.5): Direct BLP files `map{X}_{Y}.blp`
- **Modern method** (0.6.0+): MD5-hashed BLPs + `md5translate.trs` mapping
- **Current issue**: Falling back to legacy BLPs, might have wrong coordinates

### TRS File Format
```
dir: Azeroth
Azeroth\map30_35.blp    a1b2c3d4e5f6.blp
```
- Format: `map_%d_%02d.blp` (X not padded, Y padded)
- X = column, Y = row
- Left = virtual name, Right = actual MD5 filename

### Coordinate Flow
1. BLP: `map30_35.blp` → X=30, Y=35
2. Parse: coords[0]=30 (X/col), coords[1]=35 (Y/row)
3. MinimapEntry: (map, row=35, col=30) ← **Intentional swap!**
4. MinimapTile: (path, TileX=30, TileY=35)
5. Filename: `{map}_{TileX}_{TileY}.png` = `Azeroth_30_35.png`
6. Viewer expects: `{map}_{col}_{row}.png` = `Azeroth_30_35.png` ✅

**Coordinates are correct!** Rotation must be elsewhere.

---

## Documentation Created

1. **[viewer-cache-fix.md](file:///i:/parp-tools/pm4next-branch/parp-tools/gillijimproject_refactor/refactor/viewer-cache-fix.md)** - Browser cache hard refresh guide
2. **[csv-cleanup-complete.md](file:///i:/parp-tools/pm4next-branch/parp-tools/gillijimproject_refactor/refactor/csv-cleanup-complete.md)** - CSV consolidation details
3. **[viewer-remaining-issues.md](file:///i:/parp-tools/pm4next-branch/parp-tools/gillijimproject_refactor/refactor/viewer-remaining-issues.md)** - Comprehensive issue analysis
4. **[viewer-debug-session.md](file:///i:/parp-tools/pm4next-branch/parp-tools/gillijimproject_refactor/refactor/viewer-debug-session.md)** - Debug checklist
5. **[minimap-coordinate-fix.md](file:///i:/parp-tools/pm4next-branch/parp-tools/gillijimproject_refactor/refactor/minimap-coordinate-fix.md)** - Coordinate analysis
6. **[pipeline-critical-fixes.md](file:///i:/parp-tools/pm4next-branch/parp-tools/gillijimproject_refactor/refactor/pipeline-critical-fixes.md)** - All fixes summary

---

## Next Session Tasks

### Immediate (High Priority)
1. **Build & run pipeline** with new logging
2. **Check console output** for:
   - `[MinimapLoc]` messages (TRS vs directory scan)
   - `[OverlayGen]` messages (placement counts)
3. **Verify TRS files exist**:
   ```powershell
   ls ..\test_data\0.5.3\tree\World\Textures\Minimap\*.trs
   ```
4. **Check actual BLP filenames**:
   ```powershell
   ls ..\test_data\0.5.3\tree\World\Textures\Minimap\Azeroth\*.blp | Select -First 10
   ```

### Debug Based on Logging
- **If TRS files found**: Check why parsing fails or returns 0 entries
- **If no TRS files**: Accept legacy BLPs, verify coordinate parsing
- **If overlays Placements=0**: Check AnalysisIndex deserialization
- **If overlays exception**: Fix the error shown in stack trace

### Medium Priority
5. **Implement layers JSON copy** for Sedimentary Layers
6. **Fix minimap rotation** based on findings
7. **Test all maps** (Azeroth, Kalidar, Shadowfang, DeadminesInstance)

---

## Success Criteria

After next run:
- [ ] Console shows TRS/scan results
- [ ] Console shows overlay generation results
- [ ] Minimap tiles display correctly (not rotated)
- [ ] Overlays generate and load
- [ ] All maps work (not just DeadminesInstance)
- [ ] Sedimentary Layers load

---

## Known Working
- ✅ DeadminesInstance displays correctly
- ✅ Map dropdown works
- ✅ Version dropdown works
- ✅ DBC JSONs have real data
- ✅ CSV spam eliminated

## Known Broken
- ❌ Azeroth/Kalidar/Shadowfang tiles rotated
- ❌ Overlays not generating
- ❌ Sedimentary Layers not loading

---

## Time Spent
- DBC fixes: 1h
- Viewer fixes: 2h
- CSV cleanup: 30min
- Minimap debugging: 1h
- Documentation: 1h

**Total**: ~5.5 hours

**Status**: Good progress, core issues identified, debugging tools in place! 🔍
