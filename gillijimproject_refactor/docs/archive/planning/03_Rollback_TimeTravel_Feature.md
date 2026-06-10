# Phase 0: WoWRollback Core Feature - Time-Travel Visualization ⏮️

**Priority**: **BEFORE Phase 1** - This is the namesake feature!

**Goal**: Implement chronological uniqueID filtering to visualize map development over time.

---

## 📋 Feature Overview

### The Core Concept
**UniqueIDs are chronological** - they represent the order work was done on maps. By filtering ID ranges, we can literally "roll back" to earlier development states.

### Discovery
Analysis of Dun Morogh showed:
- ✅ **Definitive layers** of work per tile (visible in uniqueID distributions)
- ✅ **Asset reuse patterns** (older assets borrowed by newer zones)
- ✅ **Filename changes** (assets renamed but still used in original locations)
- ✅ **Chronological progression** (IDs increase as development proceeds)

### User Experience
```
[Viewer with Timeline Slider]
                    
2003 ←─────────●─────────→ 2004
      "Dun Morogh v1"    "Dun Morogh Final"
      
Map shows only assets with UniqueID ≤ selected range
Watch the zone build itself over time! ⏱️
```

---

## 🎯 Feature Requirements

### 1. UniqueID Analysis (CSV Generation)
**Output**: Per-map CSV files showing ID distributions

```csv
# Dun_Morogh_uniqueID_analysis.csv
MapName,TileX,TileY,AssetType,UniqueIDMin,UniqueIDMax,Count,Layer
Azeroth,29,40,M2,1234,2456,45,Layer_1
Azeroth,29,40,M2,5678,7890,32,Layer_2
Azeroth,29,40,WMO,1000,1500,12,Layer_1
Azeroth,29,41,M2,1300,2500,52,Layer_1
...
```

### 2. Layer Detection
**Algorithm**: Identify distinct "work sessions" per tile

```
Tile (29, 40) UniqueIDs:
  M2:  [1234─2456] ──gap── [5678─7890] ──gap── [9000─9500]
       └─ Layer 1 ─┘        └─ Layer 2 ─┘        └─ Layer 3 ─┘
       
Gap threshold: 100+ IDs = new layer
```

### 3. Viewer Integration
**UI**: Timeline slider + ID range display

```
┌─────────────────────────────────────────────────┐
│ WoWRollback Viewer - Time Travel Mode 🕰️       │
├─────────────────────────────────────────────────┤
│ Map: Dun Morogh (Azeroth 29,40 → 32,48)        │
│                                                  │
│ Timeline Slider:                                │
│ ├─────────●──────────────────────────────┤      │
│ 1234      5678                     12000         │
│ (Layer 1) (Layer 2)              (Layer 5)      │
│                                                  │
│ Current Filter: UniqueID ≤ 5678                 │
│ - Showing: 2,456 objects (Layers 1-2)          │
│ - Hidden:  8,344 objects (Layers 3-5)          │
│                                                  │
│ Mode: ○ Global (all tiles)  ● Per-Tile         │
│                                                  │
│ [Export Filtered ADTs] [Save Snapshot]         │
└─────────────────────────────────────────────────┘
```

### 4. Asset Filtering
**Two Modes**:

#### Mode A: Global Filter (Map-Wide)
```
User selects: UniqueID ≤ 5000
→ ALL tiles filter to ID ≤ 5000
→ Consistent snapshot across entire map
```

#### Mode B: Per-Tile Filter (Advanced)
```
User defines per-tile ranges:
  Tile (29,40): UniqueID ≤ 3000  (early work)
  Tile (29,41): UniqueID ≤ 6000  (mid work)
  Tile (30,40): No filter        (all work)
  
→ Custom rollback per region
→ Visualize "work spreading" across map
```

---

## 🏗️ Implementation Architecture

### New Namespace
```
WoWRollback.Core/Analysis/
├── UniqueIdAnalyzer.cs          // Generate CSV reports
├── LayerDetector.cs             // Detect work layers
├── AssetFilter.cs               // Filter assets by ID
└── RollbackExporter.cs          // Export filtered ADTs

WoWRollback.Viewer/
└── TimelineController.cs        // Slider + visualization
```

### Data Structures

```csharp
// UniqueID Distribution per tile
public record TileIdDistribution
{
    public string MapName { get; init; }
    public int TileX { get; init; }
    public int TileY { get; init; }
    public AssetType Type { get; init; }  // M2, WMO
    public List<IdRange> Ranges { get; init; }
}

public record IdRange
{
    public uint MinId { get; init; }
    public uint MaxId { get; init; }
    public int Count { get; init; }
    public int Layer { get; init; }
}

// Layer detection result
public record LayerInfo
{
    public int LayerNumber { get; init; }
    public uint IdRangeStart { get; init; }
    public uint IdRangeEnd { get; init; }
    public int ObjectCount { get; init; }
    public DateTime EstimatedDate { get; init; }  // If we can correlate
}

// Filter configuration
public record RollbackFilter
{
    public FilterMode Mode { get; init; }
    public uint? GlobalMaxId { get; init; }          // Mode A
    public Dictionary<(int, int), uint> PerTileMaxIds { get; init; }  // Mode B
}

public enum FilterMode
{
    Global,      // Single ID cutoff for all tiles
    PerTile      // Custom cutoff per tile
}
```

---

## 📦 Implementation Tasks

### Task 1: UniqueID Analysis
**New File**: `WoWRollback.Core/Analysis/UniqueIdAnalyzer.cs`

```csharp
namespace WoWRollback.Core.Analysis;

public class UniqueIdAnalyzer
{
    public async Task<MapIdAnalysis> AnalyzeMapAsync(
        string mapRoot,
        string mapName,
        CancellationToken ct = default)
    {
        var analysis = new MapIdAnalysis { MapName = mapName };
        
        // Find all LK ADT files for this map
        var adtFiles = Directory.GetFiles(
            Path.Combine(mapRoot, mapName), 
            "*.adt", 
            SearchOption.AllDirectories);
        
        await Parallel.ForEachAsync(adtFiles, 
            new ParallelOptions { CancellationToken = ct },
            async (adtPath, token) =>
        {
            var tileInfo = await AnalyzeTileAsync(adtPath, token);
            lock (analysis.Tiles)
            {
                analysis.Tiles.Add(tileInfo);
            }
        });
        
        // Detect layers across entire map
        analysis.GlobalLayers = DetectGlobalLayers(analysis.Tiles);
        
        return analysis;
    }
    
    private async Task<TileIdDistribution> AnalyzeTileAsync(
        string adtPath, 
        CancellationToken ct)
    {
        // Read LK ADT
        var adt = LkAdtReader.Read(adtPath);
        
        var tile = new TileIdDistribution
        {
            MapName = Path.GetFileName(Path.GetDirectoryName(adtPath)),
            TileX = adt.TileX,
            TileY = adt.TileY
        };
        
        // Analyze M2 placements (MDDF chunk)
        var m2Ids = adt.M2Placements
            .Select(p => p.UniqueId)
            .OrderBy(id => id)
            .ToList();
        
        if (m2Ids.Any())
        {
            tile.M2Distribution = new IdDistribution
            {
                MinId = m2Ids.First(),
                MaxId = m2Ids.Last(),
                Count = m2Ids.Count,
                Histogram = BuildHistogram(m2Ids)
            };
        }
        
        // Analyze WMO placements (MODF chunk)
        var wmoIds = adt.WmoPlacemen.Select(p => p.UniqueId)
            .OrderBy(id => id)
            .ToList();
        
        if (wmoIds.Any())
        {
            tile.WmoDistribution = new IdDistribution
            {
                MinId = wmoIds.First(),
                MaxId = wmoIds.Last(),
                Count = wmoIds.Count,
                Histogram = BuildHistogram(wmoIds)
            };
        }
        
        return tile;
    }
    
    private Dictionary<uint, int> BuildHistogram(List<uint> ids)
    {
        // Group IDs into buckets (e.g., every 100 IDs)
        const int bucketSize = 100;
        return ids
            .GroupBy(id => (id / bucketSize) * bucketSize)
            .ToDictionary(g => g.Key, g => g.Count());
    }
    
    public async Task ExportAnalysisAsync(
        MapIdAnalysis analysis,
        string outputPath)
    {
        var csv = new StringBuilder();
        csv.AppendLine("MapName,TileX,TileY,AssetType,MinId,MaxId,Count,Layer");
        
        foreach (var tile in analysis.Tiles.OrderBy(t => t.TileY).ThenBy(t => t.TileX))
        {
            if (tile.M2Distribution != null)
            {
                csv.AppendLine($"{tile.MapName},{tile.TileX},{tile.TileY}," +
                    $"M2,{tile.M2Distribution.MinId},{tile.M2Distribution.MaxId}," +
                    $"{tile.M2Distribution.Count},{tile.M2Distribution.Layer}");
            }
            
            if (tile.WmoDistribution != null)
            {
                csv.AppendLine($"{tile.MapName},{tile.TileX},{tile.TileY}," +
                    $"WMO,{tile.WmoDistribution.MinId},{tile.WmoDistribution.MaxId}," +
                    $"{tile.WmoDistribution.Count},{tile.WmoDistribution.Layer}");
            }
        }
        
        await File.WriteAllTextAsync(outputPath, csv.ToString());
    }
}
```

---

### Task 2: Layer Detection
**New File**: `WoWRollback.Core/Analysis/LayerDetector.cs`

```csharp
public class LayerDetector
{
    private const int GapThreshold = 100;  // IDs gap to consider new layer
    
    public List<LayerInfo> DetectLayers(List<uint> sortedIds)
    {
        if (!sortedIds.Any()) return new List<LayerInfo>();
        
        var layers = new List<LayerInfo>();
        int currentLayer = 1;
        uint layerStart = sortedIds[0];
        uint lastId = sortedIds[0];
        var currentLayerIds = new List<uint> { sortedIds[0] };
        
        for (int i = 1; i < sortedIds.Count; i++)
        {
            var currentId = sortedIds[i];
            var gap = currentId - lastId;
            
            if (gap > GapThreshold)
            {
                // End current layer
                layers.Add(new LayerInfo
                {
                    LayerNumber = currentLayer,
                    IdRangeStart = layerStart,
                    IdRangeEnd = lastId,
                    ObjectCount = currentLayerIds.Count
                });
                
                // Start new layer
                currentLayer++;
                layerStart = currentId;
                currentLayerIds.Clear();
            }
            
            currentLayerIds.Add(currentId);
            lastId = currentId;
        }
        
        // Add final layer
        layers.Add(new LayerInfo
        {
            LayerNumber = currentLayer,
            IdRangeStart = layerStart,
            IdRangeEnd = lastId,
            ObjectCount = currentLayerIds.Count
        });
        
        return layers;
    }
    
    public List<LayerInfo> DetectGlobalLayers(List<TileIdDistribution> tiles)
    {
        // Collect all IDs across all tiles
        var allIds = tiles
            .SelectMany(t => new[] { t.M2Distribution, t.WmoDistribution })
            .Where(d => d != null)
            .SelectMany(d => Enumerable.Range((int)d.MinId, (int)(d.MaxId - d.MinId + 1))
                .Select(id => (uint)id))
            .Distinct()
            .OrderBy(id => id)
            .ToList();
        
        return DetectLayers(allIds);
    }
}
```

---

### Task 3: Viewer Timeline Controller
**New File**: `WoWRollback.Viewer/TimelineController.js`

```javascript
class TimelineController {
    constructor(mapViewer) {
        this.mapViewer = mapViewer;
        this.analysis = null;
        this.currentMaxId = null;
        this.mode = 'global';  // 'global' or 'per-tile'
        this.perTileFilters = new Map();
        
        this.initializeUI();
    }
    
    async loadAnalysis(mapName) {
        const response = await fetch(`analysis/${mapName}_uniqueID_analysis.json`);
        this.analysis = await response.json();
        
        // Setup slider range
        this.minId = this.analysis.globalLayers[0].idRangeStart;
        this.maxId = this.analysis.globalLayers[this.analysis.globalLayers.length - 1].idRangeEnd;
        
        this.updateSlider();
    }
    
    initializeUI() {
        // Create timeline slider
        this.slider = document.createElement('input');
        this.slider.type = 'range';
        this.slider.min = 0;
        this.slider.max = 100;
        this.slider.value = 100;
        
        this.slider.addEventListener('input', (e) => {
            const percent = e.target.value / 100;
            this.currentMaxId = this.minId + (this.maxId - this.minId) * percent;
            this.applyFilter();
        });
        
        // Layer markers
        this.layerLabels = document.createElement('div');
        this.layerLabels.className = 'timeline-layers';
        
        // Mode toggle
        this.modeToggle = document.createElement('select');
        this.modeToggle.innerHTML = `
            <option value="global">Global Filter</option>
            <option value="per-tile">Per-Tile Filter</option>
        `;
        this.modeToggle.addEventListener('change', (e) => {
            this.mode = e.target.value;
            this.updateUI();
        });
        
        // Object count display
        this.stats = document.createElement('div');
        this.stats.className = 'timeline-stats';
        
        // Add to viewer
        const timeline = document.createElement('div');
        timeline.className = 'timeline-controller';
        timeline.appendChild(this.modeToggle);
        timeline.appendChild(this.slider);
        timeline.appendChild(this.layerLabels);
        timeline.appendChild(this.stats);
        
        document.querySelector('.map-viewer').appendChild(timeline);
    }
    
    updateSlider() {
        // Add layer markers to slider
        this.layerLabels.innerHTML = '';
        
        this.analysis.globalLayers.forEach((layer, index) => {
            const percent = (layer.idRangeStart - this.minId) / (this.maxId - this.minId) * 100;
            
            const marker = document.createElement('div');
            marker.className = 'layer-marker';
            marker.style.left = `${percent}%`;
            marker.textContent = `L${layer.layerNumber}`;
            marker.title = `Layer ${layer.layerNumber}: ${layer.objectCount} objects (ID ${layer.idRangeStart}-${layer.idRangeEnd})`;
            
            this.layerLabels.appendChild(marker);
        });
    }
    
    applyFilter() {
        if (this.mode === 'global') {
            this.applyGlobalFilter(this.currentMaxId);
        } else {
            this.applyPerTileFilter();
        }
        
        this.updateStats();
    }
    
    applyGlobalFilter(maxId) {
        // Tell map viewer to filter objects
        this.mapViewer.setAssetFilter({
            mode: 'global',
            maxId: maxId
        });
    }
    
    applyPerTileFilter() {
        // Use per-tile filters from user configuration
        this.mapViewer.setAssetFilter({
            mode: 'per-tile',
            perTileFilters: this.perTileFilters
        });
    }
    
    updateStats() {
        const visible = this.countVisibleObjects();
        const hidden = this.countHiddenObjects();
        const total = visible + hidden;
        
        this.stats.innerHTML = `
            <div>Current Filter: UniqueID ≤ ${Math.floor(this.currentMaxId)}</div>
            <div>Showing: ${visible.toLocaleString()} objects (${((visible/total)*100).toFixed(1)}%)</div>
            <div>Hidden: ${hidden.toLocaleString()} objects</div>
        `;
    }
}
```

---

### Task 4: Asset Filtering in Viewer
**Update**: `WoWRollback.Viewer/map-viewer.js`

```javascript
class MapViewer {
    constructor() {
        // ... existing code ...
        this.assetFilter = null;
    }
    
    setAssetFilter(filter) {
        this.assetFilter = filter;
        this.refreshVisibleObjects();
    }
    
    refreshVisibleObjects() {
        // Reprocess all loaded tiles with current filter
        for (const [tileKey, tileData] of this.loadedTiles.entries()) {
            this.applyFilterToTile(tileKey, tileData);
        }
    }
    
    applyFilterToTile(tileKey, tileData) {
        const [tileX, tileY] = this.parseTileKey(tileKey);
        
        // Filter M2 placements
        tileData.m2Placements.forEach(placement => {
            const isVisible = this.shouldShowAsset(
                placement.uniqueId,
                tileX,
                tileY
            );
            
            this.setObjectVisibility(placement.instanceId, isVisible);
        });
        
        // Filter WMO placements
        tileData.wmoplacements.forEach(placement => {
            const isVisible = this.shouldShowAsset(
                placement.uniqueId,
                tileX,
                tileY
            );
            
            this.setObjectVisibility(placement.instanceId, isVisible);
        });
    }
    
    shouldShowAsset(uniqueId, tileX, tileY) {
        if (!this.assetFilter) return true;
        
        if (this.assetFilter.mode === 'global') {
            return uniqueId <= this.assetFilter.maxId;
        } else {
            // Per-tile mode
            const tileFilter = this.assetFilter.perTileFilters.get(`${tileX},${tileY}`);
            if (!tileFilter) return true;  // No filter = show all
            return uniqueId <= tileFilter;
        }
    }
    
    setObjectVisibility(instanceId, visible) {
        const object = this.scene.getObjectByProperty('instanceId', instanceId);
        if (object) {
            object.visible = visible;
        }
    }
}
```

---

### Task 5: Export Filtered ADTs
**New File**: `WoWRollback.Core/Analysis/RollbackExporter.cs`

```csharp
public class RollbackExporter
{
    public async Task ExportFilteredAdtAsync(
        string sourceAdtPath,
        string outputAdtPath,
        RollbackFilter filter,
        CancellationToken ct = default)
    {
        // Read source ADT
        var adt = LkAdtReader.Read(sourceAdtPath);
        
        // Apply filter
        var filteredAdt = ApplyFilter(adt, filter);
        
        // Write filtered ADT
        await LkAdtWriter.WriteAsync(outputAdtPath, filteredAdt, ct);
    }
    
    private LkAdtData ApplyFilter(LkAdtData adt, RollbackFilter filter)
    {
        uint maxId = filter.Mode == FilterMode.Global
            ? filter.GlobalMaxId.Value
            : filter.PerTileMaxIds.GetValueOrDefault((adt.TileX, adt.TileY), uint.MaxValue);
        
        // Filter M2 placements
        adt.M2Placements = adt.M2Placements
            .Where(p => p.UniqueId <= maxId)
            .ToList();
        
        // Filter WMO placements
        adt.WmoPlac = adt.WmoPlac.Where(p => p.UniqueId <= maxId)
            .ToList();
        
        // TODO: Decide nulling strategy
        // Option A: Remove entries (changes chunk sizes)
        // Option B: Remap to invisible model (preserves structure)
        
        return adt;
    }
}
```

---

## 🎨 CLI Commands

```powershell
# Analyze map uniqueIDs
wowrollback analyze-uniqueids \
  --lk-root ../wow_3.3.5/Data \
  --map Azeroth \
  --output analysis/Azeroth_uniqueID_analysis.csv

# Generate layer report
wowrollback detect-layers \
  --analysis analysis/Azeroth_uniqueID_analysis.csv \
  --output analysis/Azeroth_layers.json

# Export filtered ADTs
wowrollback export-rollback \
  --lk-root ../wow_3.3.5/Data \
  --map Azeroth \
  --max-uniqueid 5000 \
  --output rollback/Azeroth_layer2

# Per-tile filtering
wowrollback export-rollback \
  --lk-root ../wow_3.3.5/Data \
  --map Azeroth \
  --tile-filters "29,40:3000;29,41:6000" \
  --output rollback/Azeroth_custom
```

---

## 📊 Example Analysis Output

### CSV: `Dun_Morogh_uniqueID_analysis.csv`
```csv
MapName,TileX,TileY,AssetType,MinId,MaxId,Count,Layer
Azeroth,29,40,M2,1234,2456,45,1
Azeroth,29,40,M2,5678,7890,32,2
Azeroth,29,40,M2,9000,9500,28,3
Azeroth,29,40,WMO,1000,1500,12,1
Azeroth,29,40,WMO,6000,7000,8,2
```

### JSON: `Dun_Morogh_layers.json`
```json
{
  "mapName": "Azeroth_DunMorogh",
  "analyzedTiles": 48,
  "globalLayers": [
    {
      "layerNumber": 1,
      "idRangeStart": 1000,
      "idRangeEnd": 3500,
      "objectCount": 1243,
      "estimatedDate": "2003-Q2"
    },
    {
      "layerNumber": 2,
      "idRangeStart": 5500,
      "idRangeEnd": 8000,
      "objectCount": 856,
      "estimatedDate": "2003-Q3"
    }
  ]
}
```

---

## ✅ Success Criteria

- [ ] Generate uniqueID CSV for Dun Morogh with distinct layers
- [ ] Viewer timeline slider filters assets correctly
- [ ] Global filter works (all tiles use same cutoff)
- [ ] Per-tile filter works (custom per tile)
- [ ] Export filtered LK ADTs successfully
- [ ] Filtered ADTs load in 3.3.5 client
- [ ] Visual progression shows zone "building over time"

---

## 🎯 Future Enhancements (Phase 0.5)

### Alpha WDT Filtering
```csharp
// Null out placements in Alpha WDT
// Challenge: Preserve chunk offsets
public async Task FilterAlphaWdtAsync(
    string alphaWdtPath,
    RollbackFilter filter)
{
    // Read Alpha WDT
    // Apply filter to MMDX/MMID/MDDF chunks
    // Option: Replace with invisible model reference
    // Write back without changing offsets
}
```

### LK → Alpha Backporting
```csharp
// Convert filtered LK ADT → Alpha WDT format
// This hardens our format understanding!
public async Task BackportToAlphaAsync(
    LkAdtData lkAdt,
    string outputAlphaWdtPath)
{
    // Convert structures
    // Write Alpha format
    // Validate round-trip
}
```

---

## 🔧 Prerequisite Fix

**CRITICAL**: Fix AreaTable source before Phase 0!

See `docs/planning/FIX_AREATABLE_SOURCE.md`

**Problem**: Currently reading area IDs from Alpha WDT CSV files.  
**Solution**: Read area IDs directly from cached LK ADT files (authoritative source).

**Tasks**:
- [ ] Add `ReadMcnkChunks()` to `LkAdtReader.cs`
- [ ] Create `LkAdtTerrainReader.cs`
- [ ] Update `McnkTerrainOverlayBuilder.cs`
- [ ] Test area ID accuracy

**Benefit**: Authoritative area IDs + prepares for rollback feature (needs LK ADT reading).

---

## 📅 Implementation Timeline

### Week 1: Analysis & Layer Detection
- [ ] UniqueIdAnalyzer implementation
- [ ] LayerDetector algorithm
- [ ] CSV export
- [ ] Test on Dun Morogh

### Week 2: Viewer Integration
- [ ] TimelineController UI
- [ ] Slider functionality
- [ ] Asset filtering in viewer
- [ ] Visual testing

### Week 3: Export Functionality
- [ ] RollbackExporter (LK ADTs)
- [ ] Global filter mode
- [ ] Per-tile filter mode
- [ ] In-game testing

### Week 4: Polish & Documentation
- [ ] CLI commands
- [ ] User documentation
- [ ] Example analyses
- [ ] Video demo

**Total: 4 weeks** (prerequisite for Phase 1)

---

**This is THE core feature - implement this FIRST to prove the concept!** ⏮️🎯
