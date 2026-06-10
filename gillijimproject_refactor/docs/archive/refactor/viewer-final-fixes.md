# Viewer Final Fixes - Map Dropdown & Overlays

**Date**: 2025-01-08 16:59  
**Status**: 1 fix implemented, overlay investigation needed

---

## Current Status from Screenshot

✅ **Working**:
- Minimap tiles ARE converting (PNGs exist)
- Grid/tile layout appears correctly
- Version dropdown works ("All Chunks")

❌ **Not Working**:
- Map dropdown empty (despite minimap data existing for 5 maps)
- Map appears rotated/sideways
- No overlays loading (terrain/objects)

---

## Fix #1: Map Dropdown - FIXED ✅

### Problem
index.json had property name mismatch:
```javascript
// Viewer expects (js/main.js:157):
option.value = mapData.map;  // Looking for .map property

// But we output:
{
  "name": "Azeroth",  // ❌ Wrong property name!
  "tiles": [...]
}
```

### Solution
Changed `ViewerStageRunner.cs` line 124:
```csharp
maps = session.Options.Maps.Select(mapName => new
{
    map = mapName,  // ✅ Changed from "name" to "map"
    tiles = mapTiles.ContainsKey(mapName) ? mapTiles[mapName].ToArray() : Array.Empty<TileInfo>()
}).ToArray()
```

**Expected index.json**:
```json
{
  "maps": [
    {
      "map": "Kalimdor",
      "tiles": [
        {"row": 30, "col": 30, "versions": ["0.5.3"]},
        ...
      ]
    }
  ]
}
```

---

## Issue #2: Map Rotation/Sideways Display

### Possible Causes
1. **Coordinate system mismatch**: wow.tools uses different coord system than our data
2. **Tile naming**: Filenames might have row/col swapped
3. **Viewer expectations**: Viewer might expect different tile coordinate origin

### Investigation Needed
- Check wow.tools actual tile URLs vs our generated filenames
- Verify coordinate transformation in MinimapLocator
- Compare with working wow.tools viewer

**Current Filenames** (verified correct format):
```
Shadowfang_25_30.png
Shadowfang_26_30.png
...
```

**Viewer expects** (js/state.js:90):
```javascript
`minimap/${version}/${mapName}/${mapName}_${col}_${row}.png`
```

**Matches!** So rotation issue is likely coordinate system, not filenames.

---

## Issue #3: No Overlays Generated

### Investigation Results

**What Exists**:
- ✅ AnalysisIndex files exist: `03_adts/0.5.3/analysis/{map}/index.json`
- ✅ Placement data exists in indices (verified Shadowfang has 80K lines)
- ✅ AnalysisOrchestrator calls OverlayGenerator with GenerateOverlays=true
- ✅ OverlayGenerator.GenerateFromIndex() should run

**What's Missing**:
- ❌ No overlay JSONs in: `05_viewer/overlays/0.5.3/{map}/objects_combined/`
- ❌ Only `metadata.json` exists in overlays directory

### Hypothesis
OverlayGenerator may be:
1. **Silently failing** - exception caught but not logged
2. **Writing to wrong path** - overlays go elsewhere
3. **Placements filter** - AnalysisIndex.Placements.Count == 0 check failing

### Next Steps
1. Check OverlayGenerator error handling
2. Verify AnalysisIndex deserialization includes Placements
3. Add logging to overlay generation
4. Check if overlays exist in alternate location

---

## Files Modified

| File | Change | Status |
|------|--------|--------|
| `ViewerStageRunner.cs` | Changed "name" → "map" in index.json | ✅ Fixed |

---

## Testing Plan

### After Build
```powershell
dotnet build
dotnet run --project WoWRollback.Orchestrator -- \
  --maps Shadowfang \
  --versions 0.5.3 \
  --alpha-root ..\test_data \
  --lk-dbc-dir ..\test_data\3.3.5\tree\DBFilesClient \
  --serve
```

### Verify Map Dropdown
1. Open viewer at http://localhost:8080
2. **Check**: Map dropdown shows "Shadowfang" (not empty)
3. **Check**: Can select map and it loads

### Verify Overlays (If Generated)
```powershell
ls parp_out\session_*\05_viewer\overlays\0.5.3\Shadowfang\objects_combined\*.json
# Should show: tile_r25_c30.json, tile_r25_c31.json, etc.
```

### Check Overlay Loading in Viewer
1. Open browser console (F12)
2. Look for overlay load errors
3. Check Network tab for 404s on overlay JSONs

---

## Remaining Work

### High Priority
1. ✅ Map dropdown fix - DONE
2. ⏳ Investigate why overlays aren't generating
3. ⏳ Fix map rotation/coordinate issue

### Medium Priority
4. Verify coordinate transforms match wow.tools
5. Add error logging to overlay generation
6. Test with multiple maps simultaneously

### Low Priority
7. Optimize overlay generation performance
8. Add overlay variant switching (M2 only, WMO only)

---

## Notes

### wow.tools Coordinate System
User confirmed: "the tiles should always match their ADT counterparts"
- Tile filenames ARE correct
- Rotation issue is likely Y-axis flip or coordinate origin difference
- Need to check viewer's tileBounds() function vs our coordinate system

### Directory Structure Verified
```
parp_out/session_20251008_164853/
├── 01_dbcs/              ✅ DBC JSONs exist (with real data now)
├── 02_crosswalks/        ✅ Crosswalk CSVs
├── 03_adts/              ✅ ADT output + analysis indices
│   └── 0.5.3/
│       └── analysis/
│           ├── Azeroth/index.json        (Placements data)
│           ├── Kalimdor/index.json       (Placements data)
│           └── Shadowfang/index.json     (80K lines, Placements data)
├── 04_analysis/          ❌ Empty objects/terrain folders
│   └── 0.5.3/
│       ├── master/       ✅ Has 10 items
│       ├── objects/      ❌ Empty (should have placements)
│       ├── terrain/      ❌ Empty
│       └── uniqueids/    ✅ Has 5 items
├── 05_viewer/            
│   ├── minimap/          ✅ 951 Kalimdor PNGs, 25 Shadowfang PNGs, etc.
│   ├── overlays/         ❌ Only metadata.json (no tile JSONs)
│   └── index.json        ✅ Will have correct format after rebuild
```

### Key Insight
The `04_analysis/{version}/objects/` and `terrain/` folders are empty, suggesting CSVs aren't being copied. But OverlayGenerator should work from AnalysisIndex directly, not CSVs.

**Action**: Need to trace OverlayGenerator execution path to see where it fails silently.

---

## Status Summary

- ✅ **DBC data**: Fixed (uses column indexer)
- ✅ **Minimap PNGs**: Fixed (uses MinimapLocator)
- ✅ **index.json format**: Fixed (lowercase properties)
- ✅ **Map dropdown**: Fixed (uses "map" property)
- ⏳ **Overlays**: Investigation needed
- ⏳ **Map rotation**: Investigation needed

**Next Session**: Debug overlay generation + fix coordinate system
