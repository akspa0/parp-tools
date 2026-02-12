# AreaID Boundary Overlay Design

## Purpose

Visualize AreaTable.dbc zone boundaries by detecting AreaID transitions between adjacent MCNK chunks and rendering boundary lines with area labels.

---

## Background

### What is AreaID?

Each MCNK chunk has an `areaid` field (uint32 at offset 0x34) that references an entry in `AreaTable.dbc`. This defines the zone/subzone the chunk belongs to.

**Examples**:
- AreaID 1519 = "Stormwind City"
- AreaID 12 = "Elwynn Forest"
- AreaID 1 = "Dun Morogh"

### Alpha vs. LK AreaIDs

**Alpha AreaID Encoding**:
```
areaid = (zone << 16) | sub
```

**Example**:
- Alpha AreaID `360960` = zone 5, sub 3840
- Mapped to LK AreaID `1519` (Stormwind City) via DBCTool CSV

**AlphaWDTAnalysisTool** already handles this mapping and writes verified AreaIDs in verbose mode:
- `areaid_verify_<x>_<y>.csv` contains Alpha→LK mappings
- Output ADTs contain patched LK AreaIDs

---

## Data Sources

### 1. MCNK Chunk Data (Primary)

Extracted from `<map>_mcnk_terrain.csv`:

```csv
map,tile_row,tile_col,chunk_row,chunk_col,...,areaid,...
Azeroth,31,34,0,0,...,1519,...
Azeroth,31,34,0,1,...,1519,...
Azeroth,31,34,0,2,...,12,...  ← Boundary detected (1519 → 12)
```

### 2. AreaID Verify CSV (Optional Metadata)

From `areaid_verify_<x>_<y>.csv`:

```csv
tile,chunk_y,chunk_x,alpha_areaid,alpha_zone,alpha_sub,chosen_lk_area,reason
31_34,0,0,360960,5,3840,1519,"strict CSV numeric match"
31_34,0,1,360960,5,3840,1519,"strict CSV numeric match"
31_34,0,2,12,0,12,12,"strict CSV numeric match"
```

### 3. AreaTable Lookup (From AlphaWDTAnalysisTool)

**AlphaWDTAnalysisTool already generates AreaTable CSVs**:

**Output Files**:
- `<output>/AreaTable_Alpha.csv` - Alpha 0.5.3/0.5.5 area names
- `<output>/AreaTable_335.csv` - LK 3.3.5 area names

**CSV Format**:
```csv
id,name
1519,Stormwind City
12,Elwynn Forest
1,Dun Morogh
```

**Source Class**: `AlphaWdtAnalyzer.Core/Dbc/AreaTableDbcExporter.cs`

**Usage**: WoWRollback should consume these existing CSVs rather than re-extracting from DBC.

---

## Boundary Detection Algorithm

### Input: 16×16 Grid of Chunks

```
    0   1   2   3   4   5  ...
  ┌───┬───┬───┬───┬───┬───┐
0 │1519│1519│ 12│ 12│ 12│ 12│
  ├───┼───┼───┼───┼───┼───┤
1 │1519│1519│ 12│ 12│ 12│ 12│
  ├───┼───┼───┼───┼───┼───┤
2 │1519│ 12│ 12│ 12│ 12│ 12│
  └───┴───┴───┴───┴───┴───┘
```

### Edge Detection

For each chunk at `(row, col)`, check four neighbors:
- North: `(row-1, col)`
- South: `(row+1, col)`
- East: `(row, col+1)`
- West: `(row, col-1)`

If `chunk.areaid ≠ neighbor.areaid`, draw boundary line between them.

### Implementation (C#)

```csharp
public sealed class AreaBoundaryDetector
{
    public static List<AreaBoundary> DetectBoundaries(List<McnkTerrainEntry> chunks)
    {
        var boundaries = new List<AreaBoundary>();
        var chunkDict = chunks.ToDictionary(c => (c.ChunkRow, c.ChunkCol), c => c.AreaId);
        
        foreach (var chunk in chunks)
        {
            var row = chunk.ChunkRow;
            var col = chunk.ChunkCol;
            var areaId = chunk.AreaId;
            
            // Check North
            if (row > 0 && chunkDict.TryGetValue((row - 1, col), out var northId) && northId != areaId)
            {
                boundaries.Add(new AreaBoundary(
                    FromArea: areaId,
                    ToArea: northId,
                    ChunkRow: row,
                    ChunkCol: col,
                    Edge: "north"
                ));
            }
            
            // Check East
            if (col < 15 && chunkDict.TryGetValue((row, col + 1), out var eastId) && eastId != areaId)
            {
                boundaries.Add(new AreaBoundary(
                    FromArea: areaId,
                    ToArea: eastId,
                    ChunkRow: row,
                    ChunkCol: col,
                    Edge: "east"
                ));
            }
            
            // South and West are redundant (already covered by neighbor's North/East)
        }
        
        return boundaries;
    }
}

public record AreaBoundary(
    int FromArea,
    int ToArea,
    int ChunkRow,
    int ChunkCol,
    string Edge  // "north", "east", "south", "west"
);
```

---

## Overlay JSON Schema

### Per-Tile AreaID Overlay

```json
{
  "map": "Azeroth",
  "tile": {"row": 31, "col": 34},
  "chunk_size": 32,
  "layers": [
    {
      "version": "0.5.3",
      "area_ids": {
        "chunks": [
          {
            "row": 0,
            "col": 0,
            "area_id": 1519,
            "area_name": "Stormwind City",
            "alpha_zone": 5,
            "alpha_sub": 3840
          },
          {
            "row": 0,
            "col": 2,
            "area_id": 12,
            "area_name": "Elwynn Forest",
            "alpha_zone": 0,
            "alpha_sub": 12
          }
        ],
        "boundaries": [
          {
            "from_area": 1519,
            "from_name": "Stormwind City",
            "to_area": 12,
            "to_name": "Elwynn Forest",
            "chunk_row": 0,
            "chunk_col": 2,
            "edge": "east"
          }
        ],
        "unique_areas": [
          {"area_id": 1519, "name": "Stormwind City", "color": "#FF5733"},
          {"area_id": 12, "name": "Elwynn Forest", "color": "#33FF57"}
        ]
      }
    }
  ]
}
```

---

## Visualization (ViewerAssets)

### Module: `areaIdLayer.js`

```javascript
export function renderAreaIds(map, tileRow, tileCol, data, options) {
    const areaLayer = L.layerGroup();
    
    if (!data.layers || data.layers.length === 0) return areaLayer;
    
    const versionData = data.layers[0];
    const areaData = versionData.area_ids;
    
    if (!areaData) return areaLayer;
    
    const chunkSize = data.chunk_size || 32;
    const showFill = options.colorByArea || false;
    const showBoundaries = options.showBoundaries !== false;
    const showLabels = options.showAreaNames || false;
    
    // Render area fill (optional)
    if (showFill) {
        areaData.chunks.forEach(chunk => {
            const bounds = getChunkBounds(tileRow, tileCol, chunk.row, chunk.col, chunkSize);
            const color = chunk.color || hashColor(chunk.area_id);
            
            const rect = L.rectangle(bounds, {
                color: 'transparent',
                weight: 0,
                fillColor: color,
                fillOpacity: 0.15,
                interactive: false
            });
            areaLayer.addLayer(rect);
        });
    }
    
    // Render boundaries
    if (showBoundaries && areaData.boundaries) {
        areaData.boundaries.forEach(boundary => {
            const line = createBoundaryLine(
                tileRow, tileCol,
                boundary.chunk_row, boundary.chunk_col,
                boundary.edge, chunkSize
            );
            
            line.setStyle({
                color: '#FFD700',  // Gold
                weight: 3,
                opacity: 0.9
            });
            
            line.bindPopup(`
                <strong>Area Boundary</strong><br>
                From: ${boundary.from_name} (${boundary.from_area})<br>
                To: ${boundary.to_name} (${boundary.to_area})
            `);
            
            areaLayer.addLayer(line);
        });
    }
    
    // Render area labels (at chunk centers)
    if (showLabels) {
        const labeledAreas = new Set();
        
        areaData.chunks.forEach(chunk => {
            // Only label each area once per tile
            const key = `${chunk.area_id}`;
            if (labeledAreas.has(key)) return;
            labeledAreas.add(key);
            
            const center = getChunkCenter(tileRow, tileCol, chunk.row, chunk.col, chunkSize);
            const label = L.marker(center, {
                icon: L.divIcon({
                    className: 'area-label',
                    html: `<div class="area-label-text">${chunk.area_name}</div>`,
                    iconSize: [120, 20]
                })
            });
            
            areaLayer.addLayer(label);
        });
    }
    
    return areaLayer;
}

function createBoundaryLine(tileRow, tileCol, chunkRow, chunkCol, edge, chunkSize) {
    const pixelX = chunkCol * chunkSize;
    const pixelY = chunkRow * chunkSize;
    
    let p1, p2;
    
    switch (edge) {
        case 'north':
            p1 = pixelToLatLng(tileRow, tileCol, pixelX, pixelY, 512, 512);
            p2 = pixelToLatLng(tileRow, tileCol, pixelX + chunkSize, pixelY, 512, 512);
            break;
        case 'east':
            p1 = pixelToLatLng(tileRow, tileCol, pixelX + chunkSize, pixelY, 512, 512);
            p2 = pixelToLatLng(tileRow, tileCol, pixelX + chunkSize, pixelY + chunkSize, 512, 512);
            break;
        case 'south':
            p1 = pixelToLatLng(tileRow, tileCol, pixelX, pixelY + chunkSize, 512, 512);
            p2 = pixelToLatLng(tileRow, tileCol, pixelX + chunkSize, pixelY + chunkSize, 512, 512);
            break;
        case 'west':
            p1 = pixelToLatLng(tileRow, tileCol, pixelX, pixelY, 512, 512);
            p2 = pixelToLatLng(tileRow, tileCol, pixelX, pixelY + chunkSize, 512, 512);
            break;
    }
    
    return L.polyline([[p1.lat, p1.lng], [p2.lat, p2.lng]]);
}

function hashColor(areaId) {
    // Simple hash to consistent color
    const hue = (areaId * 137) % 360;
    return `hsl(${hue}, 70%, 50%)`;
}
```

---

## CSS Styling

```css
/* Area labels */
.area-label {
    background: transparent;
    border: none;
}

.area-label-text {
    background: rgba(0, 0, 0, 0.7);
    color: #FFD700;
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: bold;
    text-align: center;
    white-space: nowrap;
    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.8);
    pointer-events: none;
}

/* Boundary popups */
.leaflet-popup-content {
    font-family: 'Segoe UI', sans-serif;
}
```

---

## UI Controls

```html
<div class="overlay-group">
    <label>
        <input type="checkbox" id="showAreaBoundaries">
        Area Boundaries
    </label>
    <div class="indent" id="areaOptions">
        <label>
            <input type="checkbox" id="showAreaNames" checked>
            Show Area Names
        </label>
        <label>
            <input type="checkbox" id="colorByArea">
            Color Chunks by Area
        </label>
        <label>
            <input type="checkbox" id="showBoundaryLines" checked>
            Show Boundary Lines
        </label>
    </div>
</div>
```

---

## AreaTable Integration

### WoWRollback: Read Existing CSVs

```csharp
namespace WoWRollback.Core.Services;

public sealed class AreaTableReader
{
    public static Dictionary<int, string> ReadAreaTableCsv(string csvPath)
    {
        var areas = new Dictionary<int, string>();
        
        if (!File.Exists(csvPath))
        {
            Console.WriteLine($"Warning: AreaTable CSV not found: {csvPath}");
            return areas;
        }
        
        using var reader = new StreamReader(csvPath);
        reader.ReadLine(); // Skip header
        
        string line;
        while ((line = reader.ReadLine()) != null)
        {
            var parts = line.Split(',', 2); // Split on first comma only
            if (parts.Length < 2) continue;
            
            if (int.TryParse(parts[0], out int id))
            {
                string name = parts[1].Trim('"'); // Remove CSV quotes if present
                areas[id] = name;
            }
        }
        
        Console.WriteLine($"Loaded {areas.Count} area names from {Path.GetFileName(csvPath)}");
        return areas;
    }
    
    public static AreaTableLookup LoadForVersion(string versionRoot, string version)
    {
        // Try to load both Alpha and LK AreaTables
        var alphaPath = Path.Combine(versionRoot, "AreaTable_Alpha.csv");
        var lkPath = Path.Combine(versionRoot, "AreaTable_335.csv");
        
        var alphaAreas = ReadAreaTableCsv(alphaPath);
        var lkAreas = ReadAreaTableCsv(lkPath);
        
        return new AreaTableLookup(alphaAreas, lkAreas);
    }
}

public class AreaTableLookup
{
    private readonly Dictionary<int, string> alphaAreas;
    private readonly Dictionary<int, string> lkAreas;
    
    public AreaTableLookup(Dictionary<int, string> alphaAreas, Dictionary<int, string> lkAreas)
    {
        this.alphaAreas = alphaAreas;
        this.lkAreas = lkAreas;
    }
    
    public string GetName(int areaId, bool preferAlpha = true)
    {
        if (preferAlpha && alphaAreas.TryGetValue(areaId, out var alphaName))
            return alphaName;
        
        if (lkAreas.TryGetValue(areaId, out var lkName))
            return lkName;
        
        return $"Unknown Area {areaId}";
    }
    
    public (string alphaName, string lkName) GetBothNames(int areaId)
    {
        var alphaName = alphaAreas.TryGetValue(areaId, out var a) ? a : null;
        var lkName = lkAreas.TryGetValue(areaId, out var l) ? l : null;
        
        return (alphaName, lkName);
    }
}
```

**Usage in AreaIdOverlayBuilder**:

```csharp
var areaLookup = AreaTableReader.LoadForVersion(versionRoot, version);

foreach (var chunk in chunks)
{
    var areaName = areaLookup.GetName(chunk.AreaId, preferAlpha: true);
    // Use areaName in overlay JSON...
}
```

---

## Integration with Existing Viewer

### 1. Generate Viewer AreaTable JSON

WoWRollback should generate a combined `areatables.json` from the CSVs:

```csharp
// In VersionComparisonService
private static void GenerateAreaTableJson(
    Dictionary<string, string> versionRoots,
    string outputDir)
{
    var combined = new Dictionary<string, Dictionary<int, AreaNames>>();
    
    foreach (var (version, root) in versionRoots)
    {
        var lookup = AreaTableReader.LoadForVersion(root, version);
        var versionData = new Dictionary<int, AreaNames>();
        
        // Get all unique area IDs from both tables
        var allIds = lookup.GetAllAreaIds();
        
        foreach (var areaId in allIds)
        {
            var (alphaName, lkName) = lookup.GetBothNames(areaId);
            versionData[areaId] = new AreaNames
            {
                Alpha = alphaName,
                LK = lkName,
                Display = alphaName ?? lkName ?? $"Unknown {areaId}"
            };
        }
        
        combined[version] = versionData;
    }
    
    var json = JsonSerializer.Serialize(combined, new JsonSerializerOptions { WriteIndented = true });
    var outPath = Path.Combine(outputDir, "viewer", "areatables.json");
    File.WriteAllText(outPath, json);
}
```

**Output**: `viewer/areatables.json`

```json
{
  "0.5.3": {
    "1519": {
      "alpha": "Stormwind",
      "lk": "Stormwind City",
      "display": "Stormwind"
    },
    "12": {
      "alpha": "Elwynn Forest",
      "lk": "Elwynn Forest",
      "display": "Elwynn Forest"
    }
  }
}
```

### 2. Load in Viewer

```javascript
// state.js
async loadAreaTable() {
    try {
        const response = await fetch('areatables.json');
        if (response.ok) {
            this.areaTable = await response.json();
        }
    } catch (e) {
        console.warn('AreaTable not found');
        this.areaTable = {};
    }
}

getAreaName(areaId) {
    const versionData = this.areaTable[this.selectedVersion];
    if (!versionData) return `Area ${areaId}`;
    
    const area = versionData[areaId];
    if (!area) return `Area ${areaId}`;
    
    return area.display;
}
```

### 2. Add to Main Viewer

```javascript
// main.js
import { renderAreaIds } from './overlays/areaIdLayer.js';

let areaIdLayer = L.layerGroup();
areaIdLayer.addTo(map);

document.getElementById('showAreaBoundaries').addEventListener('change', (e) => {
    if (e.target.checked) {
        updateAreaIdLayer();
    } else {
        areaIdLayer.clearLayers();
    }
});

async function updateAreaIdLayer() {
    areaIdLayer.clearLayers();
    
    const bounds = map.getBounds();
    const tiles = getVisibleTiles(bounds);
    
    for (const tile of tiles) {
        const path = `overlays/${state.selectedVersion}/${state.selectedMap}/area_ids/tile_r${tile.row}_c${tile.col}.json`;
        try {
            const data = await loadOverlay(path);
            const layer = renderAreaIds(state.selectedMap, tile.row, tile.col, data, {
                colorByArea: document.getElementById('colorByArea').checked,
                showAreaNames: document.getElementById('showAreaNames').checked,
                showBoundaries: document.getElementById('showBoundaryLines').checked
            });
            layer.getLayers().forEach(l => areaIdLayer.addLayer(l));
        } catch (e) {
            // No area data for this tile
        }
    }
}
```

---

## Testing Checklist

- [ ] AreaID extracted correctly from MCNK
- [ ] Boundary detection works for all edges
- [ ] JSON schema validates
- [ ] AreaTable config loads successfully
- [ ] Boundary lines render at correct positions
- [ ] Area names display correctly
- [ ] Color-by-area works
- [ ] Popups show correct area info
- [ ] UI toggles work independently
- [ ] Performance < 50ms per tile

---

## Future Enhancements

1. **Cross-Tile Boundaries**: Detect boundaries between adjacent tiles
2. **Smoothing**: Merge adjacent boundary segments into continuous lines
3. **Area Statistics**: Show area coverage percentages
4. **Subzone Support**: Display both zone and subzone names
5. **Historical Boundaries**: Compare AreaID changes between versions
6. **Heatmap**: Visualize AreaID change frequency
