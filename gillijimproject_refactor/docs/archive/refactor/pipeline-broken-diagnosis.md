# Pipeline Broken - Root Cause Analysis (2025-10-08)

## Symptoms
1. ❌ No minimap PNG tiles generated
2. ❌ No per-tile overlay JSONs (only metadata.json)
3. ❌ Viewer shows `[Object object]` in dropdowns
4. ✅ DBCs ARE being extracted (AreaTable CSVs exist)

---

## Root Causes

### Problem 1: No Minimap PNGs
**Location**: `ViewerStageRunner.cs` (lines 106-131)

**What it does**:
```csharp
private static int GenerateOverlayMetadata(SessionContext session, IReadOnlyList<AdtStageResult> adtResults)
{
    // Only writes metadata.json
    // Does NOT generate minimap PNGs!
    var metadataPath = Path.Combine(overlaysDir, "metadata.json");
    File.WriteAllText(metadataPath, JsonSerializer.Serialize(metadata, options));
    return 1;
}
```

**What's missing**: Minimap generation using `ViewerReportWriter` from WoWRollback.Core

**Expected behavior**:
```
05_viewer/
├── minimap/
│   └── 0.5.3/
│       └── Kalimdor/
│           ├── Kalimdor_30_30.png
│           ├── Kalimdor_30_31.png
│           └── ...
```

**Current behavior**: Directory doesn't exist, 0 PNG files

---

### Problem 2: Overlay JSONs Not Visible
**Location**: `AnalysisOrchestrator.cs` (lines 122-131)

**What it does**:
```csharp
var objResult = overlayGenerator.GenerateFromIndex(
    analysisIndex,
    analysisOutputDir,
    viewerOutputDir,  // = 05_viewer/
    mapName,
    version);
```

**Where it writes**:
```
05_viewer/overlays/0.5.3/Kalimdor/objects_combined/tile_r30_c30.json
```

**Problem**: Let me check if files are actually there...

---

### Problem 3: Viewer UI Shows [Object object]
**Location**: `index.json` format issue

**Current index.json**:
```json
{
  "maps": ["Azeroth", "Kalidar", "Shadowfang", ...],
  "versions": [
    { "version": "0.5.3", "alias": "0.5.3" }
  ],
  "tiles": {
    "Azeroth": [{ "version": "0.5.3", "tiles": 685 }]
  }
}
```

**Expected format** (for viewer's state.js):
```json
{
  "comparisonKey": "0.5.3",
  "versions": ["0.5.3"],
  "maps": [
    {
      "name": "Kalimdor",
      "tiles": [
        { "row": 30, "col": 30, "versions": ["0.5.3"] }
      ]
    }
  ]
}
```

**Problem**: Format mismatch - viewer expects different structure

---

## Missing Component: ViewerReportWriter Integration

The `ViewerReportWriter` class in WoWRollback.Core (ViewerReportWriter.cs:14-554) is **NOT BEING CALLED** anywhere in the orchestrator!

**What it does**:
1. Generates minimap PNG tiles from ADTs
2. Uses OverlayBuilder to create overlay JSONs with proper coords
3. Creates proper directory structure
4. Generates terrain/object/shadow overlays

**Where it should be called**: In `ViewerStageRunner` or `AnalysisStageRunner`

**Current flow**:
```
AnalysisStageRunner
  → AnalysisOrchestrator
    → OverlayGenerator
      → Writes JSON overlays (works ✅)

ViewerStageRunner
  → Copies HTML/JS/CSS assets (works ✅)
  → Generates index.json (wrong format ❌)
  → Generates metadata.json (works ✅)
  → Does NOT generate minimaps ❌
```

**Expected flow**:
```
AnalysisStageRunner or ViewerStageRunner
  → ViewerReportWriter.Generate()
    → MinimapComposer.ComposeAsync() → PNG tiles
    → OverlayBuilder.BuildOverlayJson() → overlay JSONs
    → Creates proper index structure
```

---

## Fix Strategy

### Option A: Wire ViewerReportWriter into Orchestrator (Proper Fix)
**Pros**: Reuses production-tested code, generates minimaps + overlays correctly
**Cons**: Need to adapt VersionComparisonResult → single-version workflow
**Time**: 2-3 hours

### Option B: Add Minimap Generation to ViewerStageRunner (Quick Fix)
**Pros**: Minimal changes, keeps existing architecture
**Cons**: Duplicates minimap code, doesn't use ViewerReportWriter
**Time**: 1 hour

### Option C: Hybrid Approach (Recommended)
1. Add MinimapComposer calls to ViewerStageRunner for PNG generation (30 min)
2. Keep existing OverlayGenerator for overlay JSONs (already works)
3. Fix index.json format to match viewer expectations (30 min)
4. **Total**: 1 hour

---

## Implementation Plan (Option C - Hybrid)

### Step 1: Add Minimap Generation (30 min)

**File**: `ViewerStageRunner.cs`

```csharp
private static void GenerateMinimapTiles(SessionContext session, IReadOnlyList<AdtStageResult> adtResults)
{
    var composer = new MinimapComposer();
    var options = ViewerOptions.CreateDefault();
    
    foreach (var result in adtResults.Where(r => r.Success))
    {
        var adtMapDir = Path.Combine(session.Paths.AdtDir, result.Version, "World", "Maps", result.Map);
        var minimapOutDir = Path.Combine(session.Paths.ViewerDir, "minimap", result.Version, result.Map);
        Directory.CreateDirectory(minimapOutDir);
        
        // Find all ADT files
        var adtFiles = Directory.GetFiles(adtMapDir, "*.adt", SearchOption.TopDirectoryOnly);
        foreach (var adtFile in adtFiles)
        {
            var fileName = Path.GetFileName(adtFile); // e.g., Kalimdor_30_30.adt
            var pngName = fileName.Replace(".adt", ".png");
            var pngPath = Path.Combine(minimapOutDir, pngName);
            
            try
            {
                using var stream = File.OpenRead(adtFile);
                composer.ComposeAsync(stream, pngPath, options).Wait();
            }
            catch (Exception ex)
            {
                ConsoleLogger.Warn($"Failed to generate minimap for {fileName}: {ex.Message}");
            }
        }
    }
}
```

**Call in Run()**:
```csharp
public ViewerStageResult Run(SessionContext session, IReadOnlyList<AdtStageResult> adtResults)
{
    // ... existing code ...
    CopyViewerAssets(session);
    GenerateViewerDataFiles(session, adtResults);
    GenerateMinimapTiles(session, adtResults); // ADD THIS
    var overlayCount = GenerateOverlayMetadata(session, adtResults);
    // ...
}
```

### Step 2: Fix index.json Format (30 min)

**File**: `ViewerStageRunner.cs`

**Replace GenerateViewerDataFiles()** with:

```csharp
private static void GenerateViewerDataFiles(SessionContext session, IReadOnlyList<AdtStageResult> adtResults)
{
    // Load actual tile data from analysis indices
    var mapTiles = new Dictionary<string, List<TileInfo>>();
    
    foreach (var result in adtResults.Where(r => r.Success))
    {
        var analysisIndexPath = Path.Combine(
            session.Paths.AdtDir, 
            result.Version, 
            "analysis", 
            "index.json");
            
        if (File.Exists(analysisIndexPath))
        {
            var indexJson = File.ReadAllText(analysisIndexPath);
            var analysisIndex = JsonSerializer.Deserialize<AnalysisIndex>(indexJson);
            
            if (!mapTiles.ContainsKey(result.Map))
            {
                mapTiles[result.Map] = new List<TileInfo>();
            }
            
            foreach (var tile in analysisIndex.Tiles)
            {
                mapTiles[result.Map].Add(new TileInfo
                {
                    Row = tile.TileX,
                    Col = tile.TileY,
                    Versions = new[] { result.Version }
                });
            }
        }
    }
    
    // Generate index.json in viewer-expected format
    var indexData = new
    {
        comparisonKey = session.Options.Versions.FirstOrDefault() ?? "0.5.3",
        versions = session.Options.Versions.ToArray(),
        maps = session.Options.Maps.Select(mapName => new
        {
            name = mapName,
            tiles = mapTiles.ContainsKey(mapName) ? mapTiles[mapName] : new List<TileInfo>()
        }).ToArray()
    };

    var indexPath = Path.Combine(session.Paths.ViewerDir, "index.json");
    File.WriteAllText(indexPath, JsonSerializer.Serialize(indexData, new JsonSerializerOptions { WriteIndented = true }));

    // config.json stays the same
    var configData = new
    {
        default_version = session.Options.Versions.FirstOrDefault() ?? "0.5.3",
        default_map = session.Options.Maps.FirstOrDefault() ?? "Kalimdor",
        tile_size = 512
    };

    var configPath = Path.Combine(session.Paths.ViewerDir, "config.json");
    File.WriteAllText(configPath, JsonSerializer.Serialize(configData, new JsonSerializerOptions { WriteIndented = true }));
}

private class TileInfo
{
    public int Row { get; set; }
    public int Col { get; set; }
    public string[] Versions { get; set; } = Array.Empty<string>();
}
```

---

## Expected Results After Fix

### Directory Structure:
```
05_viewer/
├── minimap/
│   └── 0.5.3/
│       └── Kalimdor/
│           ├── Kalimdor_30_30.png ✅
│           ├── Kalimdor_30_31.png ✅
│           └── ... (951 tiles)
├── overlays/
│   └── 0.5.3/
│       └── Kalimdor/
│           └── objects_combined/
│               ├── tile_r30_c30.json ✅
│               └── ... (951 tiles)
├── index.json ✅ (correct format)
├── config.json ✅
└── index.html ✅
```

### Viewer Behavior:
- ✅ Dropdowns show "0.5.3" and map names (not [Object object])
- ✅ Map tiles load as PNG images
- ✅ Objects appear at correct positions (using OverlayBuilder coords)

---

## Timeline

1. **Implement minimap generation**: 30 min
2. **Fix index.json format**: 30 min
3. **Test with sample map**: 15 min
4. **Total**: ~1 hour 15 min

**Status**: Ready to implement - type `ACT` to proceed
