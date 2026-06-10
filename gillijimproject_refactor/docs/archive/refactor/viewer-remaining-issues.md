# Viewer Remaining Issues - Action Plan

**Date**: 2025-01-08 17:20  
**Status**: DeadminesInstance works ✅, other maps broken ❌

---

## Current State

### ✅ Working
- **DeadminesInstance**: Tiles display correctly with Y-axis inversion
- **Map dropdown**: Populates with all maps
- **Coordinate system**: Fixed with `coordMode: "wowtools"`

### ❌ Not Working
1. **Kalidar & Shadowfang**: Maps don't display (despite having data)
2. **Overlays**: Not generating or not loading
3. **Sedimentary Layers**: UniqueID ranges not loading

---

## Issue 1: Kalidar & Shadowfang Not Displaying

### Hypothesis
DeadminesInstance works because it's a small instance (6x6 tiles). Larger maps may have:
1. **Tile coordinate range issues** - Viewer bounds calculation wrong
2. **Missing minimap BLPs** - MinimapLocator not finding files
3. **Path resolution issues** - BLP paths incorrect for these maps

### Investigation Steps
```powershell
# Check if minimap PNGs exist
ls parp_out\session_*\05_viewer\minimap\0.5.3\Kalidar\*.png | Measure-Object
ls parp_out\session_*\05_viewer\minimap\0.5.3\Shadowfang\*.png | Measure-Object

# Check index.json tile ranges
cat parp_out\session_*\05_viewer\index.json | Select-String "Kalidar|Shadowfang" -Context 5

# Check browser console for errors
# F12 → Console → Look for 404s or tile load failures
```

### Likely Fix
Check MinimapLocator's map name matching - might be case-sensitive or alias issue.

---

## Issue 2: Overlays Not Generating

### Current Understanding
From code review:
- ✅ `AnalysisOrchestrator` calls `OverlayGenerator.GenerateFromIndex()`
- ✅ AnalysisIndex files exist with Placement data
- ❌ No overlay JSONs in `05_viewer/overlays/{version}/{map}/objects_combined/`

### Root Cause Analysis

**Path 1: OverlayGenerator.GenerateFromIndex()**
```csharp
// Line 32: Early exit if no placements
if (analysisIndex.Placements.Count == 0)
{
    return new OverlayGenerationResult(
        0, 0, 0, 0,
        Success: false,
        ErrorMessage: $"No placements found in analysis index for {mapName}");
}
```

**Hypothesis**: `analysisIndex.Placements` is null or empty despite JSON having data.

**Verification**:
```powershell
# Check if Placements exist in JSON
Select-String -Path "parp_out\session_*\03_adts\0.5.3\analysis\Shadowfang\index.json" -Pattern '"Placements"' -Context 2

# Check deserialization
# Add logging to OverlayGenerator line 32 to see Placements.Count
```

### Likely Fixes
1. **AnalysisIndex model mismatch** - Placements property name/type wrong
2. **JSON deserialization issue** - Case sensitivity or missing property
3. **Silent exception** - Try/catch swallowing errors

---

## Issue 3: Sedimentary Layers (UniqueID Ranges)

### What It Does
Displays color-coded layers based on UniqueID ranges:
- Each "sedimentary layer" = contiguous range of UniqueIDs
- Shows object placement chronology
- Allows filtering by ID range

### Current State
- ✅ UniqueID CSVs generated: `04_analysis/0.5.3/uniqueids/{map}_uniqueid_analysis.csv`
- ✅ Layers JSON generated: `04_analysis/0.5.3/uniqueids/{map}_uniqueid_layers.json`
- ❌ Viewer not loading these files

### Expected Viewer Behavior
1. Load `uniqueid_layers.json` from analysis output
2. Parse layer ranges (e.g., `{min: 1000, max: 2000, color: "#FF5733"}`)
3. Display in "Sedimentary Layers" panel
4. Filter objects by selected layer

### Fix Required
**Path issue**: Viewer expects layers JSON in viewer directory, but we generate in analysis directory.

**Solution**: Copy layers JSON to viewer during ViewerStageRunner:
```csharp
// In GenerateViewerDataFiles():
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
    Directory.CreateDirectory(Path.GetDirectoryName(layersDestPath));
    File.Copy(layersSourcePath, layersDestPath, overwrite: true);
}
```

---

## Issue 4: Object Overlays Architecture

### Current Approach (2D Points)
- Objects plotted as Leaflet markers at pixel coordinates
- Click → popup with object details
- Works for top-down view

### Desired Approach (3D Point Cloud)
User wants: "3D point cloud with labeled data points"

### Implementation Options

#### Option A: Keep 2D, Enhance Popups ✅ (Recommended for now)
- Continue using Leaflet markers
- Rich popups with:
  - UniqueID, AssetPath, Type (M2/WMO)
  - World coordinates (X, Y, Z)
  - Rotation, Scale
  - Tile location
- **Pros**: Already implemented, works with current viewer
- **Cons**: No Z-axis visualization

#### Option B: Add 3D Viewer (Future Enhancement)
- Integrate Three.js for 3D point cloud
- Toggle between 2D map and 3D view
- **Pros**: Full 3D visualization, can add low-res terrain mesh
- **Cons**: Significant development effort, new dependencies

### Recommendation
1. **Phase 1** (Now): Fix 2D overlays, ensure they load
2. **Phase 2** (Next): Enhance popup data (add all placement fields)
3. **Phase 3** (Future): Add 3D viewer toggle

---

## Action Plan - Priority Order

### 🔴 Critical (Fix Now)
1. **Debug why overlays don't generate**
   - Add logging to OverlayGenerator
   - Check AnalysisIndex.Placements deserialization
   - Verify output paths

2. **Fix Kalidar/Shadowfang display**
   - Check minimap PNG generation
   - Verify tile coordinate ranges
   - Check browser console for errors

### 🟡 High Priority (Next Session)
3. **Copy UniqueID layers to viewer**
   - Modify ViewerStageRunner
   - Ensure Sedimentary Layers panel loads data

4. **Enhance overlay popups**
   - Add all placement fields to JSON
   - Improve popup formatting

### 🟢 Medium Priority (Future)
5. **3D point cloud viewer**
   - Research Three.js integration
   - Design 2D/3D toggle UI

6. **Low-res terrain mesh**
   - Use existing terrain generation code
   - Integrate with 3D viewer

---

## Debugging Steps - Immediate

### Step 1: Check Overlay Generation
```powershell
# Run with verbose logging
dotnet run --project WoWRollback.Orchestrator -- \
  --maps Shadowfang \
  --versions 0.5.3 \
  --alpha-root ..\test_data \
  --lk-dbc-dir ..\test_data\3.3.5\tree\DBFilesClient \
  --verbose

# Look for overlay generation messages in console
```

### Step 2: Verify AnalysisIndex
```powershell
# Check Placements exist in JSON
$json = Get-Content "parp_out\session_*\03_adts\0.5.3\analysis\Shadowfang\index.json" | ConvertFrom-Json
$json.Placements.Count
# Should show: 3000+ (not 0)
```

### Step 3: Check Minimap Files
```powershell
# Verify Shadowfang minimaps exist
ls parp_out\session_*\05_viewer\minimap\0.5.3\Shadowfang\*.png | Measure-Object
# Should show: 25 files

# Check if viewer can access them
# Open http://localhost:8080/minimap/0.5.3/Shadowfang/Shadowfang_25_30.png
# Should display image, not 404
```

### Step 4: Browser Console
```
F12 → Console
Look for:
- "Failed to load overlay" errors
- 404 errors on overlay JSON paths
- Tile load failures
```

---

## Expected File Structure After Fixes

```
parp_out/session_XXXXXX/
├── 04_analysis/0.5.3/
│   └── uniqueids/
│       ├── Shadowfang_uniqueid_analysis.csv ✅
│       └── Shadowfang_uniqueid_layers.json ✅
│
├── 05_viewer/
│   ├── minimap/0.5.3/
│   │   ├── DeadminesInstance/  ✅ (6x6 = 36 tiles)
│   │   ├── Kalidar/            ❌ (should have tiles)
│   │   └── Shadowfang/         ❌ (should have 25 tiles)
│   │
│   ├── overlays/0.5.3/
│   │   ├── DeadminesInstance/objects_combined/
│   │   │   └── tile_r*.json    ❌ (should exist)
│   │   ├── Kalidar/objects_combined/
│   │   │   └── tile_r*.json    ❌ (should exist)
│   │   └── Shadowfang/objects_combined/
│   │       └── tile_r*.json    ❌ (should exist)
│   │
│   ├── layers/0.5.3/           ❌ (NEW - need to create)
│   │   ├── DeadminesInstance_layers.json
│   │   ├── Kalidar_layers.json
│   │   └── Shadowfang_layers.json
│   │
│   ├── index.json ✅
│   └── config.json ✅
```

---

## Code Changes Needed

### 1. Add Overlay Logging (OverlayGenerator.cs)
```csharp
public OverlayGenerationResult GenerateFromIndex(...)
{
    try
    {
        Console.WriteLine($"[OverlayGen] Generating for {mapName}");
        Console.WriteLine($"[OverlayGen] Placements count: {analysisIndex.Placements.Count}");
        
        if (analysisIndex.Placements.Count == 0)
        {
            Console.WriteLine($"[OverlayGen] ERROR: No placements!");
            return new OverlayGenerationResult(...);
        }
        
        // ... rest of method
        
        Console.WriteLine($"[OverlayGen] Generated {objectOverlays} overlay files");
    }
    catch (Exception ex)
    {
        Console.WriteLine($"[OverlayGen] EXCEPTION: {ex.Message}");
        Console.WriteLine($"[OverlayGen] Stack: {ex.StackTrace}");
        // ...
    }
}
```

### 2. Copy Layers JSON (ViewerStageRunner.cs)
```csharp
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
            
            ConsoleLogger.Success($"  ✓ Copied layers for {result.Map}");
        }
    }
}

// Call in Run():
CopyLayersToViewer(session, adtResults);
```

### 3. Check AnalysisIndex Model (Models.cs)
```csharp
// Verify property name matches JSON exactly
public class AnalysisIndex
{
    public string MapName { get; set; }
    public List<TileInfo> Tiles { get; set; }
    public List<string> Textures { get; set; }
    public List<PlacementEntry> Placements { get; set; }  // ← Check this!
}
```

---

## Success Criteria

- [ ] Shadowfang displays with 25 tiles correctly arranged
- [ ] Kalidar displays with tiles correctly arranged
- [ ] Object overlays generate for all maps
- [ ] Overlay JSONs exist in viewer directory
- [ ] Sedimentary Layers panel loads UniqueID ranges
- [ ] Clicking object marker shows popup with details
- [ ] Browser console shows no errors

---

## Next Session Goals

1. Debug overlay generation (add logging)
2. Fix Kalidar/Shadowfang display
3. Implement layers JSON copy
4. Test full pipeline end-to-end
5. Document overlay popup data structure

**Status**: Investigation plan complete, ready to debug! 🔍
