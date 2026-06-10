# Viewer Debug Session - Next Steps

**Date**: 2025-01-08 17:22  
**Status**: Logging added, ready to debug

---

## What We Fixed This Session

### ✅ Completed
1. **Map dropdown** - Changed "name" → "map" property
2. **Y-axis inversion** - Added `coordMode: "wowtools"` to config.json
3. **DeadminesInstance** - Now displays correctly!
4. **Debug logging** - Added comprehensive logging to OverlayGenerator

### ⏳ Still Broken
1. **Kalidar & Shadowfang** - Don't display (investigation needed)
2. **Overlays** - Not generating (will see with new logging)
3. **Sedimentary Layers** - UniqueID ranges not loading

---

## Next Run - What to Watch For

### Build & Run
```powershell
dotnet build

dotnet run --project WoWRollback.Orchestrator -- \
  --maps Shadowfang,Kalidar,DeadminesInstance \
  --versions 0.5.3 \
  --alpha-root ..\test_data \
  --lk-dbc-dir ..\test_data\3.3.5\tree\DBFilesClient \
  --community-listfile ..\test_data\community-listfile-withcapitals.csv \
  --lk-listfile "..\test_data\World of Warcraft 3x.txt" \
  --serve
```

### Console Output to Check

#### Overlay Generation
Look for these new log lines:
```
[OverlayGen] Generating overlays for Shadowfang (0.5.3)
[OverlayGen] Placements count: 3247
[OverlayGen] SUCCESS: Generated 25 overlay files for Shadowfang
[OverlayGen] Output directory: I:\...\05_viewer\overlays\0.5.3\Shadowfang\objects_combined
```

**If you see**:
- `Placements count: 0` → AnalysisIndex deserialization issue
- `ERROR: analysisIndex is null` → File not loading
- `ERROR: Placements property is null` → Model mismatch
- `EXCEPTION: ...` → Code error (will show stack trace)

#### Minimap Generation
```
✓ Generated 25 minimap tiles for Shadowfang
✓ Generated 56 minimap tiles for Kalidar
✓ Generated 36 minimap tiles for DeadminesInstance
```

**If missing**: MinimapLocator not finding BLP files

---

## Browser Console Checks

### Open Developer Tools
```
F12 → Console tab
```

### Look For

#### Tile Load Errors
```
Failed to load resource: net::ERR_FILE_NOT_FOUND
minimap/0.5.3/Shadowfang/Shadowfang_25_30.png
```
→ Minimap files not generated or wrong path

#### Overlay Load Errors
```
Failed to load overlay: overlays/0.5.3/Shadowfang/objects_combined/tile_r25_c30.json
```
→ Overlay files not generated

#### JavaScript Errors
```
Uncaught TypeError: Cannot read property 'tiles' of undefined
```
→ index.json format issue

---

## Verification Commands

### After Pipeline Runs

#### 1. Check Overlay Files Exist
```powershell
# Should show 25 files for Shadowfang
ls parp_out\session_*\05_viewer\overlays\0.5.3\Shadowfang\objects_combined\*.json | Measure-Object

# Should show 56 files for Kalidar
ls parp_out\session_*\05_viewer\overlays\0.5.3\Kalidar\objects_combined\*.json | Measure-Object

# Should show 36 files for DeadminesInstance
ls parp_out\session_*\05_viewer\overlays\0.5.3\DeadminesInstance\objects_combined\*.json | Measure-Object
```

#### 2. Check Minimap PNGs
```powershell
ls parp_out\session_*\05_viewer\minimap\0.5.3\Shadowfang\*.png | Measure-Object
ls parp_out\session_*\05_viewer\minimap\0.5.3\Kalidar\*.png | Measure-Object
```

#### 3. Check index.json Format
```powershell
cat parp_out\session_*\05_viewer\index.json | ConvertFrom-Json | Select -ExpandProperty maps | Select map, @{N='TileCount';E={$_.tiles.Count}}
```

Expected output:
```
map                  TileCount
---                  ---------
DeadminesInstance    36
Kalidar              56
Shadowfang           25
```

#### 4. Check config.json
```powershell
cat parp_out\session_*\05_viewer\config.json | ConvertFrom-Json | Select coordMode
```

Expected: `coordMode: wowtools`

---

## Likely Issues & Fixes

### Issue 1: Overlays Not Generating

**Symptom**: `[OverlayGen] Placements count: 0`

**Cause**: AnalysisIndex.Placements not deserializing

**Fix**: Check Models.cs for property name mismatch
```csharp
// Verify this matches JSON exactly (case-sensitive!)
public List<PlacementEntry> Placements { get; set; }
```

**Test**:
```powershell
$json = Get-Content "parp_out\session_*\03_adts\0.5.3\analysis\Shadowfang\index.json" | ConvertFrom-Json
$json.Placements.Count  # Should be > 0
```

---

### Issue 2: Kalidar/Shadowfang Not Displaying

**Symptom**: Tiles don't appear in viewer, but DeadminesInstance works

**Possible Causes**:
1. **Minimap BLPs not found** - MinimapLocator path issue
2. **Tile coordinate range** - Viewer bounds calculation wrong
3. **Map name mismatch** - Case sensitivity in MinimapLocator

**Debug**:
```powershell
# Check if BLP source files exist
ls ..\test_data\0.5.3\tree\World\Textures\Minimap\Shadowfang\*.blp
ls ..\test_data\0.5.3\tree\World\Textures\Minimap\Kalidar\*.blp

# Check if PNGs were generated
ls parp_out\session_*\05_viewer\minimap\0.5.3\Shadowfang\*.png
ls parp_out\session_*\05_viewer\minimap\0.5.3\Kalidar\*.png
```

**If BLPs exist but PNGs don't**: MinimapLocator not finding them
**If PNGs exist but viewer doesn't show**: Coordinate range or path issue

---

### Issue 3: Sedimentary Layers Not Loading

**Symptom**: "Load UniqueID Ranges" button doesn't work

**Cause**: Layers JSON not copied to viewer directory

**Fix**: Add to ViewerStageRunner (not yet implemented)
```csharp
// TODO: Add this method
private static void CopyLayersToViewer(SessionContext session, IReadOnlyList<AdtStageResult> adtResults)
{
    foreach (var result in adtResults.Where(r => r.Success))
    {
        var layersSourcePath = Path.Combine(
            session.Paths.AnalysisDir, 
            result.Version, 
            "uniqueids", 
            $"{result.Map}_uniqueid_layers.json");
            
        if (File.Exists(layersSourcePath))
        {
            var layersDestPath = Path.Combine(
                session.Paths.ViewerDir, 
                "layers", 
                result.Version, 
                $"{result.Map}_layers.json");
                
            Directory.CreateDirectory(Path.GetDirectoryName(layersDestPath)!);
            File.Copy(layersSourcePath, layersDestPath, overwrite: true);
        }
    }
}
```

---

## Success Criteria

After next run, we should see:

### Console Output ✅
```
[OverlayGen] Generating overlays for Shadowfang (0.5.3)
[OverlayGen] Placements count: 3247
[OverlayGen] SUCCESS: Generated 25 overlay files for Shadowfang

[OverlayGen] Generating overlays for Kalidar (0.5.3)
[OverlayGen] Placements count: 1856
[OverlayGen] SUCCESS: Generated 56 overlay files for Kalidar

✓ Generated 25 minimap tiles for Shadowfang
✓ Generated 56 minimap tiles for Kalidar
✓ Generated 36 minimap tiles for DeadminesInstance
```

### File Structure ✅
```
05_viewer/
├── minimap/0.5.3/
│   ├── Shadowfang/       (25 PNGs)
│   ├── Kalidar/          (56 PNGs)
│   └── DeadminesInstance/ (36 PNGs)
│
├── overlays/0.5.3/
│   ├── Shadowfang/objects_combined/       (25 JSONs)
│   ├── Kalidar/objects_combined/          (56 JSONs)
│   └── DeadminesInstance/objects_combined/ (36 JSONs)
│
├── index.json  (with "map" property, lowercase "row"/"col")
└── config.json (with "coordMode": "wowtools")
```

### Viewer Behavior ✅
- All 3 maps selectable in dropdown
- Tiles display correctly (not upside down)
- Objects appear as markers on map
- Clicking marker shows popup with details

---

## Files Modified This Session

| File | Change | Purpose |
|------|--------|---------|
| `ViewerStageRunner.cs` | Added `coordMode: "wowtools"` | Fix Y-axis inversion |
| `ViewerStageRunner.cs` | Changed "name" → "map" | Fix map dropdown |
| `OverlayGenerator.cs` | Added debug logging | Diagnose overlay issues |

---

## Next Session Tasks

1. **Run pipeline with logging** - See what overlay generation shows
2. **Fix overlay generation** - Based on log output
3. **Fix Kalidar/Shadowfang** - Debug minimap/tile issues
4. **Implement layers copy** - Enable Sedimentary Layers
5. **Test end-to-end** - Verify all features work

**Status**: Ready to debug with comprehensive logging! 🔍

---

## Quick Reference

### Run Command
```powershell
dotnet run --project WoWRollback.Orchestrator -- --maps Shadowfang,Kalidar,DeadminesInstance --versions 0.5.3 --alpha-root ..\test_data --lk-dbc-dir ..\test_data\3.3.5\tree\DBFilesClient --community-listfile ..\test_data\community-listfile-withcapitals.csv --lk-listfile "..\test_data\World of Warcraft 3x.txt" --serve
```

### Check Overlays
```powershell
ls parp_out\session_*\05_viewer\overlays\0.5.3\*\objects_combined\*.json -Recurse | Measure-Object
```

### Check Minimaps
```powershell
ls parp_out\session_*\05_viewer\minimap\0.5.3\*\*.png -Recurse | Measure-Object
```

### Browser Console
```
F12 → Console → Look for errors
```
