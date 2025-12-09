# OverlayGenerator Assessment (2025-10-08)

## Current State Summary

### Build Status
**FAILING** - 3 compilation errors blocking entire WoWRollback.AnalysisModule

### Affected Components
1. **OverlayGenerator.cs** - Broken record definition + missing helpers
2. **AnalysisOrchestrator.cs** - Parameter mismatch in method calls
3. **Viewer overlays** - Cannot be generated, blocking UI rendering
4. **CSV cleanup** - Blocked waiting for JSON-based overlay generation

---

## Architecture Analysis

### Data Flow
```
AlphaWdtAnalyzer
├── Reads Alpha WDT/ADT files
├── Extracts placements (flat list)
└── Writes analysis/index.json with AnalysisIndex

AnalysisOrchestrator
├── Loads AnalysisIndex from index.json
├── Calls MapMasterIndexWriter
│   └── Writes analysis/master/<map>_master_index.json (per-tile grouped)
└── Calls OverlayGenerator.GenerateFromIndex
    ├── Loads master index JSON
    └── Writes viewer/overlays/<version>/<map>/objects_combined/tile_*.json

Viewer
└── Loads overlay JSONs and renders placements
```

### Type Relationships

**AlphaWdtAnalyzer.Core.AnalysisIndex** (source):
- `List<PlacementRecord> Placements` (flat list across all tiles)
- `List<MapTile> Tiles` (just tile metadata: X, Y, AdtPath)

**OverlayGenerator.MapMasterIndexDocument** (intermediate):
- `List<MapTileRecord> Tiles` where each has:
  - `List<MapPlacement> Placements` (grouped by tile)

**OverlayGenerator.TileOverlayJson** (output):
- Per-tile JSON with placements array

### Current Implementation Gap

**Problem**: `LoadOrCreateMasterIndex` is **missing** but referenced by:
- `GenerateFromIndex` (line 38)

**Required Logic**:
1. Load existing `analysis/master/<map>_master_index.json` if exists
2. If missing, transform `AnalysisIndex` (flat placements) → `MapMasterIndexDocument` (per-tile grouped)
3. Group placements by `TileX`/`TileY`

---

## Compilation Errors Detail

### Error 1: Broken Record Definition
**File**: OverlayGenerator.cs  
**Line**: 490  
**Issue**: `PlacementOverlayJson` record incomplete

```csharp
// Current (line 480-490):
private sealed record PlacementOverlayJson
{
    public required string Kind { get; init; }
    public uint? UniqueId { get; init; }
    public string? AssetPath { get; init; }
    public required float[] World { get; init; }
    public required float[] TileOffset { get; init; }
    public required int[] Chunk { get; init; }
    public required float[] Rotation { get; init; }
    public float Scale { get; init; }
    // MISSING: Flags, DoodadSet, NameSet
    // MISSING: closing brace

/// <summary>  // <-- Line 491: Doc comment breaks record definition
```

**Fix**: Add missing properties and close properly

### Error 2: Parameter Mismatch (Call 1)
**File**: AnalysisOrchestrator.cs  
**Line**: 122-126  
**Issue**: Missing `analysisOutputDir` parameter

```csharp
// Current:
var objResult = overlayGenerator.GenerateFromIndex(
    analysisIndex,      // param 1
    viewerOutputDir,    // param 2 (should be param 3)
    mapName,            // param 3 (should be param 4)
    version);           // param 4 (should be param 5)

// Expected signature:
GenerateFromIndex(
    AnalysisIndex analysisIndex,      // param 1 ✓
    string analysisOutputDir,         // param 2 MISSING
    string viewerDir,                 // param 3
    string mapName,                   // param 4
    string version)                   // param 5
```

### Error 3: Parameter Mismatch (Call 2)
**File**: AnalysisOrchestrator.cs  
**Line**: 136-140  
**Issue**: Missing `analysisOutputDir` parameter

```csharp
// Current:
var objFromCsv = overlayGenerator.GenerateObjectsFromPlacementsCsv(
    copiedPlacementsCsv,  // param 1
    viewerOutputDir,      // param 2 (should be param 3)
    mapName,              // param 3 (should be param 4)
    version);             // param 4 (should be param 5)

// Expected signature:
GenerateObjectsFromPlacementsCsv(
    string placementsCsvPath,         // param 1 ✓
    string analysisOutputDir,         // param 2 MISSING
    string viewerDir,                 // param 3
    string mapName,                   // param 4
    string version)                   // param 5
```

---

## Missing Helper Methods

### 1. LoadOrCreateMasterIndex
**Referenced**: Line 38 in `GenerateFromIndex`  
**Purpose**: Load master index JSON or create from AnalysisIndex  
**Critical Logic**: Must group flat `AnalysisIndex.Placements` by tile

**Correct Implementation**:
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
    
    // Try load existing
    if (File.Exists(masterPath))
    {
        var json = File.ReadAllText(masterPath);
        return JsonSerializer.Deserialize<MapMasterIndexDocument>(json)!;
    }
    
    // Create from AnalysisIndex: group placements by tile
    var placementsByTile = analysisIndex.Placements
        .GroupBy(p => (p.TileX, p.TileY))
        .ToDictionary(g => g.Key, g => g.ToList());
    
    var tiles = analysisIndex.Tiles.Select(t =>
    {
        var key = (t.X, t.Y);
        var tilePlacements = placementsByTile.TryGetValue(key, out var list)
            ? list.Select(p => new MapPlacement
              {
                  Kind = p.Type.ToString(),
                  UniqueId = p.UniqueId.HasValue ? (uint)p.UniqueId.Value : null,
                  AssetPath = p.AssetPath,
                  RawNorth = 0f,  // Not in PlacementRecord
                  RawUp = 0f,
                  RawWest = 0f,
                  WorldNorth = p.WorldY,  // WoW coords: Y=North
                  WorldWest = p.WorldX,   // X=West
                  WorldUp = p.WorldZ,     // Z=Up
                  TileOffsetNorth = 0f,
                  TileOffsetWest = 0f,
                  ChunkX = 0,
                  ChunkY = 0,
                  RotationX = p.RotationX,
                  RotationY = p.RotationY,
                  RotationZ = p.RotationZ,
                  Scale = p.Scale,
                  Flags = p.Flags,
                  DoodadSet = p.DoodadSet,
                  NameSet = p.NameSet
              }).ToList()
            : new List<MapPlacement>();
        
        return new MapTileRecord
        {
            TileX = t.X,
            TileY = t.Y,
            Placements = tilePlacements
        };
    }).ToList();
    
    return new MapMasterIndexDocument
    {
        Map = mapName,
        Version = version,
        GeneratedAtUtc = DateTime.UtcNow,
        Tiles = tiles
    };
}
```

### 2. WritePlacementsFromMaster
**Referenced**: Line 112 in `GenerateObjectsFromPlacementsCsv`  
**Purpose**: Write per-tile overlay JSONs from master index

```csharp
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
```

### 3. ReadCsvRows
**Referenced**: Line 122 in `GenerateObjectsFromPlacementsCsv`  
**Purpose**: Parse CSV into rows

```csharp
private List<string[]> ReadCsvRows(string csvPath)
{
    var rows = new List<string[]>();
    using var reader = new StreamReader(csvPath);
    string? line = reader.ReadLine(); // skip header
    while ((line = reader.ReadLine()) != null)
    {
        if (string.IsNullOrWhiteSpace(line)) continue;
        rows.Add(SplitCsv(line));
    }
    return rows;
}
```

### 4. ParseInt / ParseBool
**Referenced**: Terrain/shadow overlay methods  
**Purpose**: Safe parsing

```csharp
private static int ParseInt(string s) => int.TryParse(s, out var v) ? v : 0;
private static bool ParseBool(string s) => bool.TryParse(s, out var v) && v;
```

---

## Dead Code to Remove

### Methods (No Longer Called)
1. `Generate` (lines 499-568) - Legacy ADT-reading approach
2. `GenerateObjectsOverlayFromPlacements` (lines 570-629) - Unreferenced
3. `GenerateTerrainOverlay` (lines 631-662) - Stub with TODO
4. `GenerateObjectsOverlay` (lines 664-689) - Stub with TODO
5. `GenerateShadowOverlay` (lines 691-704) - Stub returns false

**Impact**: ~200 lines can be removed after Phase 1 compiles

---

## Action Plan

### Phase 1: Fix Compilation (10 min)
1. Fix `PlacementOverlayJson` record in OverlayGenerator.cs
2. Add missing helper methods to OverlayGenerator.cs
3. Fix method calls in AnalysisOrchestrator.cs
4. Verify `dotnet build WoWRollback.AnalysisModule` succeeds

### Phase 2: Remove Dead Code (10 min)
1. Delete 5 legacy methods from OverlayGenerator.cs
2. Clean up unused using statements
3. Verify build still passes

### Phase 3: Integration Test (10 min)
1. Run orchestrator with test map
2. Verify master index JSON created
3. Verify overlay JSONs generated
4. Check JSON structure matches viewer expectations

**Total Estimated Time**: 30 minutes

---

## Risk Assessment

### Low Risk Changes
- ✅ Syntax fixes (broken record)
- ✅ Parameter additions (orchestrator calls)
- ✅ Adding missing helpers (well-defined behavior)

### Medium Risk Changes
- ⚠️ `LoadOrCreateMasterIndex` logic (grouping placements by tile)
  - **Mitigation**: Existing MapMasterIndexWriter already does this correctly
  - **Fallback**: Can always load from existing master JSON

### No Breaking Changes
- ✅ Output JSON format unchanged
- ✅ CSV fallback paths preserved
- ✅ Viewer integration unaffected

---

## Success Criteria

- [ ] `dotnet build WoWRollback.AnalysisModule` passes with zero errors
- [ ] Master index JSON loaded correctly from `analysis/master/`
- [ ] Overlay JSONs written to `viewer/overlays/<version>/<map>/objects_combined/`
- [ ] Viewer can load and render placements
- [ ] CSV cleanup work unblocked

---

## Next Steps After Fix

1. **Verify viewer integration** - Load test map in viewer UI
2. **Remove redundant CSVs** - Consolidate placement CSVs into master JSON only
3. **Add unit tests** - Test overlay generation with synthetic AnalysisIndex
4. **Documentation** - Update WoWRollback docs with new JSON-first workflow
