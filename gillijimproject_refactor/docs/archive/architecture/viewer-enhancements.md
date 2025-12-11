# Viewer Enhancements Design

## Purpose

Document UI/UX improvements for the WoWRollback viewer including asset counters, statistics, and quality-of-life features.

---

## Feature 1: Asset Counter / Statistics Panel

### Requirements

Display real-time statistics about currently visible assets in the viewport:
- Total objects visible
- Breakdown by type (M2, WMO)
- Breakdown by version
- Tile coverage information

### UI Design

#### Location: Fixed Panel (Top-Right)

```html
<div class="stats-panel">
    <div class="stats-header">
        <h3>Viewport Statistics</h3>
        <button class="collapse-btn">−</button>
    </div>
    <div class="stats-content">
        <div class="stat-group">
            <label>Objects Visible:</label>
            <span id="totalObjects" class="stat-value">0</span>
        </div>
        <div class="stat-group indent">
            <label>M2 Models:</label>
            <span id="m2Count" class="stat-value">0</span>
        </div>
        <div class="stat-group indent">
            <label>WMO Objects:</label>
            <span id="wmoCount" class="stat-value">0</span>
        </div>
        <div class="stat-divider"></div>
        <div class="stat-group">
            <label>Tiles Loaded:</label>
            <span id="tilesLoaded" class="stat-value">0</span>
        </div>
        <div class="stat-group">
            <label>Current Version:</label>
            <span id="currentVersion" class="stat-value">-</span>
        </div>
        <div class="stat-group">
            <label>Current Map:</label>
            <span id="currentMap" class="stat-value">-</span>
        </div>
        <div class="stat-divider"></div>
        <div class="stat-group">
            <label>Zoom Level:</label>
            <span id="zoomLevel" class="stat-value">2.0</span>
        </div>
        <div class="stat-group">
            <label>Center:</label>
            <span id="centerCoords" class="stat-value">32, 32</span>
        </div>
    </div>
</div>
```

#### CSS Styling

```css
.stats-panel {
    position: fixed;
    top: 20px;
    right: 20px;
    width: 280px;
    background: rgba(42, 42, 42, 0.95);
    border: 1px solid #444;
    border-radius: 8px;
    padding: 12px;
    z-index: 1001;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
}

.stats-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid #555;
}

.stats-header h3 {
    margin: 0;
    font-size: 14px;
    color: #4CAF50;
    font-weight: bold;
}

.collapse-btn {
    background: transparent;
    border: none;
    color: #999;
    font-size: 18px;
    cursor: pointer;
    padding: 0;
    width: 20px;
    height: 20px;
    line-height: 18px;
}

.collapse-btn:hover {
    color: #4CAF50;
}

.stats-content {
    display: flex;
    flex-direction: column;
    gap: 6px;
}

.stat-group {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 12px;
}

.stat-group label {
    color: #ccc;
}

.stat-value {
    color: #fff;
    font-weight: bold;
    font-family: 'Consolas', 'Monaco', monospace;
}

.stat-group.indent {
    padding-left: 16px;
    font-size: 11px;
}

.stat-divider {
    height: 1px;
    background: #555;
    margin: 4px 0;
}

.stats-panel.collapsed .stats-content {
    display: none;
}

.stats-panel.collapsed .collapse-btn::before {
    content: '+';
}
```

---

### Implementation: `statsPanel.js`

```javascript
export class StatsPanel {
    constructor(map, state) {
        this.map = map;
        this.state = state;
        this.markers = new Map(); // Track all visible markers
        this.updateInterval = null;
    }
    
    init() {
        this.createPanel();
        this.attachEventListeners();
        this.startAutoUpdate();
    }
    
    createPanel() {
        const panel = document.createElement('div');
        panel.className = 'stats-panel';
        panel.innerHTML = `
            <div class="stats-header">
                <h3>Viewport Statistics</h3>
                <button class="collapse-btn">−</button>
            </div>
            <div class="stats-content">
                <div class="stat-group">
                    <label>Objects Visible:</label>
                    <span id="totalObjects" class="stat-value">0</span>
                </div>
                <div class="stat-group indent">
                    <label>M2 Models:</label>
                    <span id="m2Count" class="stat-value">0</span>
                </div>
                <div class="stat-group indent">
                    <label>WMO Objects:</label>
                    <span id="wmoCount" class="stat-value">0</span>
                </div>
                <div class="stat-divider"></div>
                <div class="stat-group">
                    <label>Tiles Loaded:</label>
                    <span id="tilesLoaded" class="stat-value">0</span>
                </div>
                <div class="stat-group">
                    <label>Current Version:</label>
                    <span id="currentVersion" class="stat-value">-</span>
                </div>
                <div class="stat-group">
                    <label>Current Map:</label>
                    <span id="currentMap" class="stat-value">-</span>
                </div>
                <div class="stat-divider"></div>
                <div class="stat-group">
                    <label>Zoom Level:</label>
                    <span id="zoomLevel" class="stat-value">2.0</span>
                </div>
                <div class="stat-group">
                    <label>Center:</label>
                    <span id="centerCoords" class="stat-value">32, 32</span>
                </div>
            </div>
        `;
        document.body.appendChild(panel);
        
        // Collapse button
        panel.querySelector('.collapse-btn').addEventListener('click', () => {
            panel.classList.toggle('collapsed');
        });
    }
    
    attachEventListeners() {
        // Update on map move/zoom
        this.map.on('moveend zoomend', () => this.update());
        
        // Update when version/map changes
        this.state.subscribe(() => this.update());
    }
    
    startAutoUpdate() {
        // Update every 2 seconds to catch marker changes
        this.updateInterval = setInterval(() => this.update(), 2000);
    }
    
    update() {
        const stats = this.calculateStats();
        this.render(stats);
    }
    
    calculateStats() {
        const bounds = this.map.getBounds();
        const zoom = this.map.getZoom();
        const center = this.map.getCenter();
        
        // Count visible markers
        let totalObjects = 0;
        let m2Count = 0;
        let wmoCount = 0;
        
        // Iterate through object markers layer
        if (window.objectMarkers) {
            window.objectMarkers.eachLayer(marker => {
                if (bounds.contains(marker.getLatLng())) {
                    totalObjects++;
                    
                    // Check marker type from popup content
                    const popup = marker.getPopup();
                    if (popup) {
                        const content = popup.getContent();
                        if (content.includes('WMO') || content.includes('wmo')) {
                            wmoCount++;
                        } else {
                            m2Count++;
                        }
                    }
                }
            });
        }
        
        // Count loaded tiles
        const tiles = this.getVisibleTiles(bounds);
        
        return {
            totalObjects,
            m2Count,
            wmoCount,
            tilesLoaded: tiles.length,
            currentVersion: this.state.selectedVersion || '-',
            currentMap: this.state.selectedMap || '-',
            zoomLevel: zoom.toFixed(1),
            centerCoords: `${Math.round(center.lat)}, ${Math.round(center.lng)}`
        };
    }
    
    getVisibleTiles(bounds) {
        const tiles = [];
        const latS = bounds.getSouth();
        const latN = bounds.getNorth();
        const west = bounds.getWest();
        const east = bounds.getEast();
        
        // Convert to tile coordinates (assuming wow.tools system)
        const rowNorth = this.latToRow(latN);
        const rowSouth = this.latToRow(latS);
        const minRow = Math.floor(Math.min(rowNorth, rowSouth));
        const maxRow = Math.ceil(Math.max(rowNorth, rowSouth));
        const minCol = Math.floor(west);
        const maxCol = Math.ceil(east);
        
        for (let r = minRow; r <= maxRow; r++) {
            for (let c = minCol; c <= maxCol; c++) {
                tiles.push({row: r, col: c});
            }
        }
        
        return tiles;
    }
    
    latToRow(lat) {
        return 63 - lat;
    }
    
    render(stats) {
        document.getElementById('totalObjects').textContent = stats.totalObjects.toLocaleString();
        document.getElementById('m2Count').textContent = stats.m2Count.toLocaleString();
        document.getElementById('wmoCount').textContent = stats.wmoCount.toLocaleString();
        document.getElementById('tilesLoaded').textContent = stats.tilesLoaded;
        document.getElementById('currentVersion').textContent = stats.currentVersion;
        document.getElementById('currentMap').textContent = stats.currentMap;
        document.getElementById('zoomLevel').textContent = stats.zoomLevel;
        document.getElementById('centerCoords').textContent = stats.centerCoords;
    }
    
    destroy() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }
    }
}
```

---

### Integration in `main.js`

```javascript
import { StatsPanel } from './statsPanel.js';

// Initialize stats panel
let statsPanel;

export async function init() {
    // ... existing init code ...
    
    // Create stats panel
    statsPanel = new StatsPanel(map, state);
    statsPanel.init();
}
```

---

## Feature 2: Complete Map Processing

### Requirements

Process **all** maps from Map.dbc that have corresponding minimap data, not just a hardcoded subset.

### Current Limitation

Tool may only process specific maps like:
- Azeroth
- Kalimdor
- Development (test maps)

### Solution: Map Discovery & Auto-Processing

#### Step 1: Map.dbc Parsing

**File**: `AlphaWdtAnalyzer.Core/MapDiscovery.cs` (NEW)

```csharp
using DBCD;
using DBCD.Providers;

namespace AlphaWdtAnalyzer.Core;

public sealed class MapDiscovery
{
    private readonly string dbcPath;
    private readonly string wdtRootPath;
    
    public MapDiscovery(string dbcPath, string wdtRootPath)
    {
        this.dbcPath = dbcPath;
        this.wdtRootPath = wdtRootPath;
    }
    
    public List<MapInfo> DiscoverMaps()
    {
        var maps = new List<MapInfo>();
        
        // Parse Map.dbc
        using var dbcd = new DBCD.DBCD(new DBCProvider(), new DBDProvider());
        var mapDbc = dbcd.Load("Map", "Alpha 0.5.3");
        
        foreach (var row in mapDbc.Values)
        {
            int mapId = row["ID"];
            string internalName = row["Directory"];
            string mapName = row["MapName_enUS"];
            int instanceType = row["InstanceType"];
            
            // Check if WDT exists
            string wdtPath = Path.Combine(wdtRootPath, "World", "Maps", internalName, $"{internalName}.wdt");
            
            if (File.Exists(wdtPath))
            {
                maps.Add(new MapInfo(
                    MapId: mapId,
                    InternalName: internalName,
                    DisplayName: mapName,
                    InstanceType: instanceType,
                    WdtPath: wdtPath
                ));
                
                Console.WriteLine($"Discovered map: {internalName} (ID {mapId}) - {mapName}");
            }
            else
            {
                Console.WriteLine($"Skipping map {internalName} (no WDT found)");
            }
        }
        
        return maps;
    }
}

public record MapInfo(
    int MapId,
    string InternalName,
    string DisplayName,
    int InstanceType,
    string WdtPath
);
```

---

#### Step 2: Batch Processing

**File**: `AlphaWdtAnalyzer.Cli/Program.cs` (MODIFY)

Add new command:

```csharp
--process-all-maps    Process all maps found in Map.dbc
```

Implementation:

```csharp
private static async Task ProcessAllMapsAsync(string dbcPath, string wdtRoot, string outputRoot)
{
    var discovery = new MapDiscovery(dbcPath, wdtRoot);
    var maps = discovery.DiscoverMaps();
    
    Console.WriteLine($"Found {maps.Count} maps with WDT files");
    
    int processed = 0;
    int failed = 0;
    
    foreach (var map in maps)
    {
        try
        {
            Console.WriteLine($"\n[{processed + 1}/{maps.Count}] Processing {map.InternalName}...");
            
            // Run full extraction pipeline
            await ProcessMapAsync(map.WdtPath, outputRoot);
            
            processed++;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"ERROR: Failed to process {map.InternalName}: {ex.Message}");
            failed++;
        }
    }
    
    Console.WriteLine($"\n=== Processing Complete ===");
    Console.WriteLine($"Processed: {processed}/{maps.Count}");
    Console.WriteLine($"Failed: {failed}");
}

private static async Task ProcessMapAsync(string wdtPath, string outputRoot)
{
    // Extract terrain data
    var terrain = McnkTerrainExtractor.ExtractTerrain(wdtPath);
    var shadows = McnkShadowExtractor.ExtractShadows(wdtPath);
    
    // Write CSVs
    // ... (existing extraction logic)
}
```

---

#### Step 3: Minimap Validation

**Issue**: Not all maps may have minimap tiles even if they have WDTs.

**Solution**: Add minimap check to discovery:

```csharp
private bool HasMinimapTiles(string internalName)
{
    string minimapDir = Path.Combine(minimapRootPath, internalName);
    
    if (!Directory.Exists(minimapDir))
        return false;
    
    // Check if at least one .blp or .png file exists
    var files = Directory.GetFiles(minimapDir, "*.blp", SearchOption.TopDirectoryOnly)
        .Concat(Directory.GetFiles(minimapDir, "*.png", SearchOption.TopDirectoryOnly));
    
    return files.Any();
}
```

---

#### Step 4: Parallel Processing (Optional)

For faster processing of many maps:

```csharp
private static async Task ProcessAllMapsParallelAsync(List<MapInfo> maps, string outputRoot)
{
    var options = new ParallelOptions
    {
        MaxDegreeOfParallelism = Environment.ProcessorCount / 2  // Use half of CPU cores
    };
    
    await Parallel.ForEachAsync(maps, options, async (map, ct) =>
    {
        try
        {
            Console.WriteLine($"Processing {map.InternalName}...");
            await ProcessMapAsync(map.WdtPath, outputRoot);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"ERROR: {map.InternalName}: {ex.Message}");
        }
    });
}
```

---

### CLI Usage

```powershell
# Process all maps from Map.dbc
dotnet run --project AlphaWDTAnalysisTool/AlphaWdtAnalyzer.Cli -- \
  --process-all-maps \
  --dbc-path "test_data/0.5.3/DBFilesClient/Map.dbc" \
  --wdt-root "test_data/0.5.3" \
  --minimap-root "test_data/0.5.3/Interface/WorldMap" \
  --out "output/all_maps" \
  --extract-mcnk-terrain \
  --extract-mcnk-shadows
```

---

### Map Filtering Options

Add filters for specific map types:

```powershell
--map-type <type>         Filter by instance type:
                          0 = Normal (continents)
                          1 = Dungeon
                          2 = Raid
                          3 = PvP Battleground
                          4 = Arena
                          
--exclude-instances       Skip dungeons/raids/battlegrounds
--continents-only         Only process Azeroth/Kalimdor
```

---

### Progress Reporting

Enhanced progress output:

```
=== Map Discovery ===
Found 42 maps in Map.dbc
  - 2 continents
  - 15 dungeons
  - 3 raids
  - 22 other instances

=== Processing Maps ===
[1/42] Azeroth........................ OK (234 tiles, 45s)
[2/42] Kalimdor....................... OK (198 tiles, 38s)
[3/42] Development.................... OK (12 tiles, 3s)
[4/42] MonasteryInstances............. SKIP (no WDT)
[5/42] PVPZone01...................... OK (4 tiles, 1s)
...

=== Summary ===
Processed: 38/42 maps
Skipped: 4 (no WDT or minimap)
Total time: 12m 34s
Output: output/all_maps/
```

---

### Integration with WoWRollback

Update `VersionComparisonService` to auto-discover maps:

```csharp
public static void CompareVersions(
    string versionRoot,
    string outputDir,
    bool autoDiscoverMaps = true)
{
    List<string> maps;
    
    if (autoDiscoverMaps)
    {
        // Scan for all CSV directories
        var csvDir = Path.Combine(versionRoot, "csv");
        maps = Directory.GetDirectories(csvDir)
            .Select(d => Path.GetFileName(d))
            .ToList();
        
        Console.WriteLine($"Auto-discovered {maps.Count} maps from CSV data");
    }
    else
    {
        // Use provided map list
        maps = defaultMaps;
    }
    
    // Process each map...
}
```

---

## Testing Checklist

### Asset Counter
- [ ] Panel displays correctly
- [ ] Counts update on pan/zoom
- [ ] M2/WMO breakdown correct
- [ ] Tile count accurate
- [ ] Collapse/expand works
- [ ] No performance impact

### All Maps Processing
- [ ] Map.dbc parsing works
- [ ] WDT discovery finds all maps
- [ ] Minimap validation filters correctly
- [ ] CSV output for all maps
- [ ] Parallel processing stable
- [ ] Progress reporting accurate
- [ ] WoWRollback auto-discovery works
- [ ] Viewer index.json includes all maps

---

## Documentation Updates

### AlphaWDTAnalysisTool/README.md

Add to "Features":
```markdown
- **Auto-discovery**: Process all maps from Map.dbc automatically
- **Batch processing**: Process multiple maps in parallel
- **Minimap validation**: Skip maps without minimap data
```

### WoWRollback/README.md

Add to "Features":
```markdown
- **Statistics panel**: Real-time viewport object counts
- **All maps supported**: Auto-discovers all processed maps
```

---

## Future Enhancements

1. **Extended Statistics**
   - Objects by UID range
   - Heatmap of object density
   - Historical object count changes

2. **Export Statistics**
   - CSV export of object counts per tile
   - JSON export for analysis tools

3. **Performance Metrics**
   - Overlay load times
   - Render FPS
   - Memory usage
