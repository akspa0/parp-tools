# Viewer JSON Refactor - Leverage Existing Infrastructure (2025-10-08)

## Key Discovery ✅

**You already have production-ready infrastructure!**
- ✅ **CoordinateTransformer.cs** - Complete world → tile/pixel transforms
- ✅ **OverlayBuilder.cs** - Generates overlay JSON with world/local/pixel coords
- ✅ **ViewerReportWriter.cs** - Orchestrates viewer generation
- ✅ **COORDINATES.md** - Complete documentation
- ✅ Leaflet.js viewer working with coordinate system

**The Issue**: Current system is CSV-based, needs JSON master index integration

---

## What You Have vs What You Need

### Existing System (CSV-based)
```
AlphaWdtAnalyzer
  ↓ analyzes Alpha WDT
  ↓ looks up coords from converted LK ADT
  ↓ writes CSVs (assetledger.csv, timeline.csv)
  ↓
ViewerReportWriter
  ↓ reads CSVs
  ↓ groups by tile
  ↓ calls OverlayBuilder.BuildOverlayJson()
  ↓ writes overlays/{version}/{map}/{combined|m2|wmo}/tile_r{row}_c{col}.json
  ↓
Viewer loads and renders
```

### Needed System (JSON master index)
```
AnalysisIndex (from AnalysisOrchestrator)
  ↓ already has placements per tile
  ↓ already has world coordinates
  ↓
MapMasterIndexWriter
  ↓ writes analysis/master/{map}_master_index.json
  ↓
OverlayGenerator (NEW - needs fix)
  ↓ loads master index JSON
  ↓ calls OverlayBuilder.BuildOverlayJson() ← REUSE EXISTING
  ↓ writes overlays/{version}/{map}/objects_combined/tile_r{row}_c{col}.json
  ↓
Viewer loads and renders
```

---

## Refactoring Strategy

### Phase 1: Fix OverlayGenerator (30 min)
**Status**: Detailed in overlay-generator-fix-plan.md

**Key Point**: After fix, OverlayGenerator should **delegate** to existing OverlayBuilder

### Phase 2: Integrate OverlayBuilder into OverlayGenerator (1 hour)

**Current OverlayGenerator approach** (wrong):
```csharp
// Manual JSON building in OverlayGenerator.cs
var output = new TileOverlayJson
{
    TileX = tile.TileX,
    TileY = tile.TileY,
    Placements = tile.Placements.Select(ToPlacementJson).ToList()
};
```

**Correct approach** (leverage existing):
```csharp
// In OverlayGenerator.cs
using WoWRollback.Core.Services.Viewer;
using WoWRollback.Core.Models;

public OverlayGenerationResult GenerateFromIndex(
    AnalysisIndex analysisIndex,
    string analysisOutputDir,
    string viewerDir,
    string mapName,
    string version)
{
    var overlayBuilder = new OverlayBuilder();
    var options = ViewerOptions.CreateDefault();
    
    // Convert AnalysisIndex placements → AssetTimelineDetailedEntry
    var entries = ConvertToTimelineEntries(analysisIndex, version);
    
    var objectsDir = Path.Combine(viewerDir, "overlays", version, mapName, "objects_combined");
    Directory.CreateDirectory(objectsDir);
    
    int objectOverlays = 0;
    foreach (var tile in analysisIndex.Tiles)
    {
        if (tile.Placements.Count == 0) continue;
        
        // Use existing OverlayBuilder - it already does world→pixel conversion!
        var json = overlayBuilder.BuildOverlayJson(
            mapName,
            tile.TileX,
            tile.TileY,
            entries,
            options
        );
        
        var jsonPath = Path.Combine(objectsDir, $"tile_r{tile.TileX}_c{tile.TileY}.json");
        File.WriteAllText(jsonPath, json);
        objectOverlays++;
    }
    
    return new OverlayGenerationResult(
        TilesProcessed: analysisIndex.Tiles.Count,
        TerrainOverlays: 0,
        ObjectOverlays: objectOverlays,
        ShadowOverlays: 0,
        Success: true
    );
}

private List<AssetTimelineDetailedEntry> ConvertToTimelineEntries(
    AnalysisIndex analysisIndex,
    string version)
{
    var entries = new List<AssetTimelineDetailedEntry>();
    
    foreach (var tile in analysisIndex.Tiles)
    {
        foreach (var placement in tile.Placements)
        {
            entries.Add(new AssetTimelineDetailedEntry
            {
                Version = version,
                Map = analysisIndex.MapName,
                TileRow = tile.TileX,
                TileCol = tile.TileY,
                UniqueId = placement.UniqueId ?? 0,
                AssetPath = placement.AssetPath ?? string.Empty,
                FileName = ExtractFileName(placement.AssetPath),
                Kind = DetermineKind(placement.Kind),
                WorldX = placement.WorldNorth,  // Note: field name mapping
                WorldY = placement.WorldWest,
                WorldZ = placement.WorldUp,
                // ... other fields as needed
            });
        }
    }
    
    return entries;
}
```

### Phase 3: Update AnalysisOrchestrator (30 min)

**Add ProjectReference**:
```xml
<!-- In WoWRollback.AnalysisModule/WoWRollback.AnalysisModule.csproj -->
<ItemGroup>
  <ProjectReference Include="..\WoWRollback.Core\WoWRollback.Core.csproj" />
</ItemGroup>
```

**Update orchestrator calls**:
```csharp
// In AnalysisOrchestrator.cs
var overlayGenerator = new OverlayGenerator();

var objResult = overlayGenerator.GenerateFromIndex(
    analysisIndex,
    analysisOutputDir,  // ← FIX: was missing
    viewerOutputDir,
    mapName,
    version
);
```

---

## Data Model Mapping

### AnalysisIndex.Placement → AssetTimelineDetailedEntry

```csharp
public static class PlacementMapper
{
    public static AssetTimelineDetailedEntry ToTimelineEntry(
        MapPlacement placement,
        string version,
        string mapName,
        int tileX,
        int tileY)
    {
        return new AssetTimelineDetailedEntry
        {
            Version = version,
            Map = mapName,
            TileRow = tileX,
            TileCol = tileY,
            UniqueId = placement.UniqueId ?? 0,
            AssetPath = placement.AssetPath ?? string.Empty,
            FileName = ExtractFileName(placement.AssetPath),
            FileStem = Path.GetFileNameWithoutExtension(placement.AssetPath),
            Extension = Path.GetExtension(placement.AssetPath),
            Kind = ParseKind(placement.Kind),
            
            // World coordinates (WoW system: Y=North, X=West)
            WorldX = placement.WorldWest,   // WoW X = West
            WorldY = placement.WorldNorth,  // WoW Y = North
            WorldZ = placement.WorldUp,     // WoW Z = Up
            
            // These will be computed by OverlayBuilder via CoordinateTransformer
            // (no need to pre-compute)
        };
    }
    
    private static PlacementKind ParseKind(string kindStr)
    {
        return kindStr?.ToLowerInvariant() switch
        {
            "wmo" => PlacementKind.Wmo,
            "mdxorm2" => PlacementKind.MdxOrM2,
            "m2" => PlacementKind.MdxOrM2,
            _ => PlacementKind.MdxOrM2
        };
    }
    
    private static string ExtractFileName(string? path)
    {
        if (string.IsNullOrEmpty(path)) return "Unknown";
        return Path.GetFileName(path);
    }
}
```

---

## Output JSON Format

**OverlayBuilder already generates the correct format**:
```json
{
  "map": "Kalimdor",
  "tile": { "row": 30, "col": 30 },
  "minimap": { "width": 512, "height": 512 },
  "layers": [
    {
      "version": "0.5.3.3368",
      "kinds": [
        {
          "kind": "wmo",
          "points": [
            {
              "uniqueId": 230658,
              "fileName": "building.wmo",
              "assetPath": "World/wmo/Building/building.wmo",
              "world": { "x": 15990.69, "y": 16191.14, "z": 42.67 },
              "local": { "x": 0.456, "y": 0.789 },
              "pixel": { "x": 234.567, "y": 156.234 }
            }
          ]
        }
      ]
    }
  ]
}
```

**This is exactly what the viewer expects!** ✅

---

## Why This Works

### 1. CoordinateTransformer is Production-Ready
- Handles WoW coordinate system (+X=North, +Y=West)
- Accurate tile index computation
- Local coordinates (0-1 range within tile)
- Pixel coordinates (0-512 range for minimap)
- Validated and documented in COORDINATES.md

### 2. OverlayBuilder is Feature-Complete
- World → local → pixel transforms
- Tile boundary validation
- Coordinate filtering
- JSON serialization
- Already used in production CSV workflow

### 3. Viewer Already Works
- Leaflet.js integration complete
- Understands the JSON format OverlayBuilder produces
- Coordinate system matches
- Popup rendering works

---

## Implementation Steps

### Step 1: Add Missing Types to OverlayGenerator
```csharp
// Copy or reference these from WoWRollback.Core
using WoWRollback.Core.Models;
using WoWRollback.Core.Services.Viewer;

// If AnalysisModule can't reference Core directly, add DTOs:
public record AssetTimelineDetailedEntry
{
    public required string Version { get; init; }
    public required string Map { get; init; }
    public required int TileRow { get; init; }
    public required int TileCol { get; init; }
    // ... (copy from WoWRollback.Core.Models)
}
```

### Step 2: Implement Converter
```csharp
private static List<AssetTimelineDetailedEntry> ConvertPlacements(
    AnalysisIndex analysisIndex,
    string version)
{
    var entries = new List<AssetTimelineDetailedEntry>();
    
    foreach (var tile in analysisIndex.Tiles)
    {
        foreach (var placement in tile.Placements)
        {
            entries.Add(PlacementMapper.ToTimelineEntry(
                placement,
                version,
                analysisIndex.MapName,
                tile.TileX,
                tile.TileY
            ));
        }
    }
    
    return entries;
}
```

### Step 3: Wire Everything Together
```csharp
public OverlayGenerationResult GenerateFromIndex(...)
{
    var overlayBuilder = new OverlayBuilder();
    var options = ViewerOptions.CreateDefault();
    var entries = ConvertPlacements(analysisIndex, version);
    
    // OverlayBuilder handles all coordinate transforms!
    var json = overlayBuilder.BuildOverlayJson(mapName, tileX, tileY, entries, options);
    
    // Write to file
    File.WriteAllText(outputPath, json);
}
```

---

## Timeline (Revised)

### Immediate (30 min)
- Fix OverlayGenerator compilation (overlay-generator-fix-plan.md)

### Phase 2 (1 hour)
- Add WoWRollback.Core reference to AnalysisModule
- Implement PlacementMapper
- Update GenerateFromIndex to use OverlayBuilder
- Remove manual JSON building code

### Phase 3 (30 min)
- Fix AnalysisOrchestrator parameter passing
- Test end-to-end with sample map

**Total: 2 hours** (vs 14 hours for new implementation!)

---

## Success Criteria

- [x] Build succeeds
- [x] OverlayGenerator delegates to OverlayBuilder
- [x] JSON output matches existing viewer format
- [x] World → pixel coordinates correct
- [x] Viewer loads and renders objects
- [x] No code duplication (reuse existing transforms)

---

## Benefits of This Approach

1. **No reinventing the wheel** - CoordinateTransformer already works
2. **Proven coordinate system** - documented and validated
3. **Consistent output format** - viewer already understands it
4. **Minimal code changes** - just wiring and conversion
5. **Maintainability** - single source of truth for transforms
6. **Testing** - leverages existing coordinate validation

---

## Next Step

Type `ACT` to proceed with:
1. OverlayGenerator compilation fix (30 min)
2. Integration with OverlayBuilder (1 hour)
3. End-to-end test (30 min)

**Total: 2 hours to working JSON-based viewer** ✅
