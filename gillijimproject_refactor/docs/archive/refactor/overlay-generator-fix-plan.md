# OverlayGenerator Fix Plan (2025-10-08)

## Executive Summary

**Current State**: Build fails due to 3 compilation errors
**Impact**: Viewer overlay generation blocked, CSV cleanup blocked
**Estimated Fix Time**: 30 minutes

**Root Causes**:
1. `PlacementOverlayJson` record incomplete (OverlayGenerator.cs:490)
2. Parameter mismatch in orchestrator calls (AnalysisOrchestrator.cs:122, 136)
3. Missing helper methods in OverlayGenerator

**Strategy**: Fix compilation errors → Remove dead code → Test end-to-end

---

## Compilation Error Analysis

### Primary Issue: Broken Record Definition (OverlayGenerator.cs Line 490)
**Location**: `PlacementOverlayJson` record definition (lines 480-490)
**Problem**: Record is incomplete - missing closing brace and remaining properties
**Cause**: Line 491 starts a new doc comment mid-definition, breaking C# syntax
**Fix**: Complete the record with missing properties (`Flags`, `DoodadSet`, `NameSet`) and close properly

### Critical Issue: Parameter Mismatch (AnalysisOrchestrator.cs Lines 122 & 136)
**Location**: Calls to `GenerateFromIndex` and `GenerateObjectsFromPlacementsCsv`
**Problem**: Orchestrator passes wrong number of parameters
**Details**:
```csharp
// LINE 122: Wrong - passes 4 params (viewerOutputDir, mapName, version)
overlayGenerator.GenerateFromIndex(analysisIndex, viewerOutputDir, mapName, version);

// EXPECTED: 5 params (analysisOutputDir, viewerDir, mapName, version)
overlayGenerator.GenerateFromIndex(analysisIndex, analysisOutputDir, viewerDir, mapName, version);

// LINE 136: Wrong - passes 3 params after csvPath
overlayGenerator.GenerateObjectsFromPlacementsCsv(csvPath, viewerOutputDir, mapName, version);

// EXPECTED: 4 params after csvPath
overlayGenerator.GenerateObjectsFromPlacementsCsv(csvPath, analysisOutputDir, viewerDir, mapName, version);
```
**Fix**: Update orchestrator calls to pass `analysisOutputDir` parameter

```csharp
// Current (BROKEN):
private sealed record PlacementOverlayJson
{
    public required string Kind { get; init; }
    // ...
    public float Scale { get; init; }
    
/// <summary>  // <-- BREAKS HERE
/// Generates overlay...

// Fixed:
private sealed record PlacementOverlayJson
{
    public required string Kind { get; init; }
    // ...
    public float Scale { get; init; }
    public ushort Flags { get; init; }
    public ushort DoodadSet { get; init; }
    public ushort NameSet { get; init; }
}  // <-- CLOSE RECORD

/// <summary>  // <-- Now in correct position
```

### Secondary Issues: Missing Helper Methods

**Referenced but not defined**:
1. `LoadOrCreateMasterIndex` (line 38)
2. `WritePlacementsFromMaster` (line 112)
3. `ReadCsvRows` (line 122)
4. `ParseInt` (referenced in terrain/shadow CSV parsing)
5. `ParseBool` (referenced in terrain/shadow CSV parsing)

**Missing types**:
1. `PlacementRecord` (line 571)
2. `AssetType` enum (line 579)

### Tertiary Issues: Dead Legacy Code

Methods that should be **removed** (no longer called, contain TODOs/stubs):
1. `Generate` (lines 499-568) - legacy direct-ADT-reading method
2. `GenerateTerrainOverlay` (lines 631-662) - stub with TODO
3. `GenerateObjectsOverlay` (lines 664-689) - stub with TODO
4. `GenerateShadowOverlay` (lines 691-704) - stub returns false
5. `GenerateObjectsOverlayFromPlacements` (lines 570-629) - unreferenced

## Fix Strategy

### Phase 1: Immediate Compilation Fix (5 min)
**Goal**: Get both files to compile

**File 1: OverlayGenerator.cs**

1. **Fix `PlacementOverlayJson` record** (line 490):
   - Add missing properties: `Flags`, `DoodadSet`, `NameSet`
   - Add closing brace after `NameSet`
   - Ensure doc comment for `Generate` is properly positioned

**File 2: AnalysisOrchestrator.cs**

2. **Fix `GenerateFromIndex` call** (line 122):
   ```csharp
   // Change from:
   var objResult = overlayGenerator.GenerateFromIndex(
       analysisIndex,
       viewerOutputDir,
       mapName,
       version);
   
   // To:
   var objResult = overlayGenerator.GenerateFromIndex(
       analysisIndex,
       analysisOutputDir,  // <-- Add this parameter
       viewerOutputDir,
       mapName,
       version);
   ```

3. **Fix `GenerateObjectsFromPlacementsCsv` call** (line 136):
   ```csharp
   // Change from:
   var objFromCsv = overlayGenerator.GenerateObjectsFromPlacementsCsv(
       copiedPlacementsCsv,
       viewerOutputDir,
       mapName,
       version);
   
   // To:
   var objFromCsv = overlayGenerator.GenerateObjectsFromPlacementsCsv(
       copiedPlacementsCsv,
       analysisOutputDir,  // <-- Add this parameter
       viewerOutputDir,
       mapName,
       version);
   ```

2. **Add missing helper methods** (minimal implementations):
   ```csharp
   private MapMasterIndexDocument LoadOrCreateMasterIndex(
       AnalysisIndex analysisIndex,
       string analysisOutputDir,
       string mapName,
       string version,
       out string masterPath)
   {
       var masterDir = Path.Combine(analysisOutputDir, "master");
       masterPath = Path.Combine(masterDir, $"{mapName}_master_index.json");
       
       if (File.Exists(masterPath))
       {
           return JsonSerializer.Deserialize<MapMasterIndexDocument>(
               File.ReadAllText(masterPath))!;
       }
       
       // Create from analysisIndex if missing
       return new MapMasterIndexDocument
       {
           Map = mapName,
           Version = version,
           GeneratedAtUtc = DateTime.UtcNow,
           Tiles = analysisIndex.Tiles.Select(t => new MapTileRecord
           {
               TileX = t.TileX,
               TileY = t.TileY,
               Placements = t.Placements
           }).ToList()
       };
   }
   
   private int WritePlacementsFromMaster(MapMasterIndexDocument master, string objectsDir)
   {
       int count = 0;
       foreach (var tile in master.Tiles)
       {
           if (tile.Placements.Count == 0) continue;
           
           var overlay = new TileOverlayJson
           {
               TileX = tile.TileX,
               TileY = tile.TileY,
               Placements = tile.Placements.Select(ToPlacementJson).ToList()
           };
           
           var jsonPath = Path.Combine(objectsDir, $"tile_{tile.TileX}_{tile.TileY}.json");
           File.WriteAllText(jsonPath, JsonSerializer.Serialize(overlay, 
               new JsonSerializerOptions { WriteIndented = true }));
           count++;
       }
       return count;
   }
   
   private List<string[]> ReadCsvRows(string csvPath)
   {
       var rows = new List<string[]>();
       using var reader = new StreamReader(csvPath);
       string? line;
       line = reader.ReadLine(); // skip header
       while ((line = reader.ReadLine()) != null)
       {
           if (string.IsNullOrWhiteSpace(line)) continue;
           rows.Add(SplitCsv(line));
       }
       return rows;
   }
   
   private static int ParseInt(string s) => 
       int.TryParse(s, out var v) ? v : 0;
   
   private static bool ParseBool(string s) =>
       bool.TryParse(s, out var v) && v;
   ```

**Result**: File compiles, methods work

### Phase 2: Remove Dead Code (10 min)
**Goal**: Clean up legacy/unreferenced methods

**Remove these methods**:
1. `Generate` (lines 499-568)
2. `GenerateObjectsOverlayFromPlacements` (lines 570-629)
3. `GenerateTerrainOverlay` (lines 631-662)
4. `GenerateObjectsOverlay` (lines 664-689)
5. `GenerateShadowOverlay` (lines 691-704)

**Keep only**:
- `GenerateFromIndex` (primary JSON-based method)
- `GenerateObjectsFromPlacementsCsv` (CSV fallback)
- `GenerateTerrainOverlaysFromCsv` (CSV-based terrain)
- `GenerateShadowOverlaysFromCsv` (CSV-based shadow)

**Result**: ~200 lines removed, cleaner class structure

### Phase 3: Verify Orchestrator Integration (5 min)
**Goal**: Ensure `AnalysisOrchestrator` calls the correct method

Check `AnalysisOrchestrator.RunAnalysis`:
1. Verify it calls `GenerateFromIndex` with correct parameters
2. Ensure `analysisOutputDir` is passed for master index lookup
3. Confirm viewer directory structure matches expectations

**Result**: End-to-end path validated

### Phase 4: Build & Test (10 min)
**Goal**: Verify overlay generation works

1. **Build**: `dotnet build WoWRollback.AnalysisModule`
2. **Test run**: Use orchestrator with small test map
3. **Verify outputs**:
   - `analysis/master/<map>_master_index.json` exists
   - `viewer/overlays/<version>/<map>/objects_combined/tile_*.json` generated
   - JSON structure matches viewer expectations

**Result**: Working overlay generation

## Acceptance Criteria

- [ ] `OverlayGenerator.cs` compiles with zero errors
- [ ] All helper methods defined (no missing references)
- [ ] Dead legacy methods removed
- [ ] `dotnet build WoWRollback.AnalysisModule` succeeds
- [ ] Test run generates overlay JSONs successfully
- [ ] Viewer can load and render placements from new overlays

## Estimated Time
- **Phase 1**: 5 minutes (immediate fix)
- **Phase 2**: 10 minutes (cleanup)
- **Phase 3**: 5 minutes (validation)
- **Phase 4**: 10 minutes (testing)
- **Total**: ~30 minutes

## Next Steps After Fix
1. Remove redundant placement CSVs (blocked until overlays work)
2. Consolidate fixup logs into master index
3. Update viewer plugin to prefer JSON overlays over CSV fallback
4. Add unit tests for overlay generation logic

---

## Summary

### Files to Modify
1. **OverlayGenerator.cs**: Fix broken record + add missing helpers + remove dead code
2. **AnalysisOrchestrator.cs**: Fix two method calls to pass correct parameters

### Expected Outcomes
- ✅ `dotnet build WoWRollback.AnalysisModule` succeeds
- ✅ Master index JSON read from `analysis/master/`
- ✅ Overlay JSONs written to `viewer/overlays/<version>/<map>/objects_combined/`
- ✅ Viewer can load and render placements
- ✅ CSV cleanup work can proceed

### Risk Assessment
**Low Risk**: Changes are isolated to two files with clear fixes
**No Breaking Changes**: Maintains existing JSON output format for viewer
**Backward Compatible**: CSV fallback paths preserved
