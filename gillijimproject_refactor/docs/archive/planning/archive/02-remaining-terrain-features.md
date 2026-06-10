# Remaining Terrain Features

## Status: In Progress

Two terrain-related features need implementation:

---

## 1. Shadow Map Overlay (MCSH Data)

### Current State
- ✅ **Data extraction working** - `--extract-mcnk-shadows` flag exists
- ✅ **CSV generation working** - `McnkShadowExtractor.cs` and `McnkShadowCsvWriter.cs` exist
- ❌ **UI layer missing** - No overlay in viewer
- ❌ **JSON transformation missing** - No builder to convert CSV → JSON

### CSV Structure
```
MapName,TileRow,TileCol,ChunkY,ChunkX,ShadowBitmap
Azeroth,30,30,0,0,0000000000000000111111111111111122222222...
```

The `ShadowBitmap` is a 4096-character string representing a 64×64 shadow map where each character is a shadow intensity (0-9, A-F for 0-15).

### Implementation Plan

#### Step 1: Create Shadow Overlay Builder
Create `WoWRollback.Core/Services/Viewer/McnkShadowOverlayBuilder.cs`:

```csharp
public static class McnkShadowOverlayBuilder
{
    public static void BuildOverlaysForMap(
        string mapName,
        string csvDir,
        string outputDir,
        string version)
    {
        // Read {mapName}_mcnk_shadows.csv
        var shadowCsvPath = Path.Combine(csvDir, $"{mapName}_mcnk_shadows.csv");
        if (!File.Exists(shadowCsvPath)) return;
        
        var shadows = McnkShadowCsvReader.ReadCsv(shadowCsvPath);
        
        // Group by tile
        var byTile = shadows.GroupBy(s => (s.TileRow, s.TileCol));
        
        foreach (var tileGroup in byTile)
        {
            var (tileRow, tileCol) = tileGroup.Key;
            var chunks = tileGroup.ToList();
            
            // Convert to JSON overlay format
            var overlay = new
            {
                type = "shadow_map",
                version = version,
                chunks = chunks.Select(c => new
                {
                    chunkY = c.ChunkY,
                    chunkX = c.ChunkX,
                    // Parse 64×64 bitmap string into intensity values
                    shadowData = ParseShadowBitmap(c.ShadowBitmap)
                })
            };
            
            var json = JsonSerializer.Serialize(overlay);
            var overlayDir = Path.Combine(outputDir, mapName, "shadow_map");
            Directory.CreateDirectory(overlayDir);
            
            var outPath = Path.Combine(overlayDir, $"tile_r{tileRow}_c{tileCol}.json");
            File.WriteAllText(outPath, json);
        }
    }
    
    private static int[][] ParseShadowBitmap(string bitmap)
    {
        // Convert 4096-char string to 64×64 array
        var result = new int[64][];
        for (int y = 0; y < 64; y++)
        {
            result[y] = new int[64];
            for (int x = 0; x < 64; x++)
            {
                int index = y * 64 + x;
                char c = bitmap[index];
                result[y][x] = c >= '0' && c <= '9' ? c - '0' : c - 'A' + 10;
            }
        }
        return result;
    }
}
```

#### Step 2: Integrate into ViewerReportWriter
In `ViewerReportWriter.GenerateTerrainOverlays()`, add:

```csharp
// Generate shadow map overlays if available
McnkShadowOverlayBuilder.BuildOverlaysForMap(mapName, csvMapDir, overlayVersionRoot, version);
```

#### Step 3: Create CSV Reader
Create `WoWRollback.Core/Models/McnkShadowEntry.cs`:

```csharp
public record McnkShadowEntry(
    string MapName,
    int TileRow,
    int TileCol,
    int ChunkY,
    int ChunkX,
    string ShadowBitmap
);
```

Create `WoWRollback.Core/Services/McnkShadowCsvReader.cs`:

```csharp
public static class McnkShadowCsvReader
{
    public static List<McnkShadowEntry> ReadCsv(string path)
    {
        var entries = new List<McnkShadowEntry>();
        using var reader = new StreamReader(path);
        
        // Skip header
        reader.ReadLine();
        
        while (!reader.EndOfStream)
        {
            var line = reader.ReadLine();
            if (string.IsNullOrWhiteSpace(line)) continue;
            
            var parts = line.Split(',');
            if (parts.Length < 6) continue;
            
            entries.Add(new McnkShadowEntry(
                parts[0],
                int.Parse(parts[1]),
                int.Parse(parts[2]),
                int.Parse(parts[3]),
                int.Parse(parts[4]),
                parts[5]
            ));
        }
        
        return entries;
    }
}
```

#### Step 4: Create UI Layer
Create `ViewerAssets/js/overlays/shadowMapLayer.js`:

```javascript
export class ShadowMapLayer {
    constructor(map, mapName, version) {
        this.map = map;
        this.mapName = mapName;
        this.version = version;
        this.canvases = new Map();
        this.visible = false;
    }

    async loadTile(row, col) {
        const url = `/overlays/${this.version}/${this.mapName}/shadow_map/tile_r${row}_c${col}.json`;
        const response = await fetch(url);
        if (!response.ok) return null;
        
        const data = await response.json();
        
        // Create canvas for this tile
        const canvas = document.createElement('canvas');
        canvas.width = 64 * 16;  // 64 chunks × 16 pixels per chunk
        canvas.height = 64 * 16;
        const ctx = canvas.getContext('2d');
        
        // Draw shadow map
        data.chunks.forEach(chunk => {
            const chunkPixelX = chunk.chunkX * 16;
            const chunkPixelY = chunk.chunkY * 16;
            
            // Downsample 64×64 shadow data to 16×16 for visualization
            for (let y = 0; y < 16; y++) {
                for (let x = 0; x < 16; x++) {
                    const shadowY = Math.floor(y * 4);
                    const shadowX = Math.floor(x * 4);
                    const intensity = chunk.shadowData[shadowY][shadowX];
                    
                    // Map 0-15 to grayscale (0 = black, 15 = transparent)
                    const alpha = intensity / 15;
                    ctx.fillStyle = `rgba(0, 0, 0, ${1 - alpha})`;
                    ctx.fillRect(chunkPixelX + x, chunkPixelY + y, 1, 1);
                }
            }
        });
        
        return canvas;
    }

    show() {
        this.visible = true;
        // Show all loaded canvases as ImageOverlay
    }

    hide() {
        this.visible = false;
        // Hide all canvases
    }
}
```

#### Step 5: Add UI Controls
In `index.html`, add checkbox:

```html
<label>
    <input type="checkbox" id="overlay-shadow-map">
    Shadow Map
</label>
```

#### Step 6: Wire Up in overlayManager.js

```javascript
import { ShadowMapLayer } from './overlays/shadowMapLayer.js';

// In constructor
this.shadowMapLayer = new ShadowMapLayer(map, mapName, version);

// Add toggle handler
document.getElementById('overlay-shadow-map').addEventListener('change', (e) => {
    if (e.target.checked) {
        this.shadowMapLayer.show();
    } else {
        this.shadowMapLayer.hide();
    }
});
```

### Reference Implementation

From **noggit-red** (`lib/noggit-red/src/noggit/MapChunk.cpp`):
```cpp
// MCSH decoding - unpacks 512 bytes to 64×64 shadow map
uint8_t* p = _shadow_map;
uint8_t* c = compressed_shadow_map;
for (int i = 0; i < 64 * 8; ++i)  // 512 compressed bytes
{
    for (int b = 0x01; b != 0x100; b <<= 1)  // 8 bits per byte
    {
        *p++ = ((*c) & b) ? 85 : 0;  // Unpack: bit set = 85, else 0
    }
    c++;
}
```

**Key Insight**: Each bit expands to a shadow value (0 or 85). This is trivial to implement!

See **`docs/architecture/MCSH_SHADOW_MAP_FORMAT.md`** for complete C# implementation.

### Timeline
- **Complexity**: Low (with reference implementation)
- **Estimated Time**: 2-3 hours
- **Priority**: Low (nice-to-have visualization feature)

---

## 2. AreaTable Mapping Fix (Use LK AreaIDs)

### Current Problem
- Alpha MCNK.AreaID values don't match 3.3.5 AreaTable.dbc structure
- Shows "Unknown Area 1234" instead of real names
- AlphaWdtAnalyzer conversion DOES map Alpha → LK AreaIDs correctly

### Current Flow (Broken)
```
Alpha WDT → Extract MCNK.AreaID → Use Alpha AreaID → ❌ No match in AreaTable
```

### Proposed Flow (Working)
```
Alpha WDT → Convert to LK ADT → Extract LK MCNK.AreaID → ✅ Matches AreaTable
```

### Implementation Plan

#### Option A: Extract AreaID from LK ADTs (Recommended)

The converted LK ADTs already have correct AreaIDs! We just need to extract them.

**Step 1**: Read AreaID from converted LK ADTs in `McnkTerrainExtractor.cs`:

```csharp
// Current: Reading from Alpha WDT
public static List<McnkTerrainEntry> ExtractTerrain(WdtAlphaScanner wdt)
{
    // Extracts Alpha AreaID which doesn't map to 3.3.5 AreaTable
}

// Proposed: Read from LK ADTs after conversion
public static List<McnkTerrainEntry> ExtractTerrainFromLkAdts(string lkAdtDirectory)
{
    var entries = new List<McnkTerrainEntry>();
    
    foreach (var adtPath in Directory.GetFiles(lkAdtDirectory, "*.adt"))
    {
        // Parse LK ADT
        using var fs = File.OpenRead(adtPath);
        using var br = new BinaryReader(fs);
        
        // Read MCNK chunks and extract AreaID from LK format
        // LK AreaID will correctly map to 3.3.5 AreaTable
        
        var (row, col) = ParseTileCoordinates(Path.GetFileName(adtPath));
        
        // Extract all MCNK data with LK AreaIDs
        var mcnkData = ExtractMcnkFromLkAdt(br, row, col);
        entries.AddRange(mcnkData);
    }
    
    return entries;
}
```

**Step 2**: Update `rebuild-and-regenerate.ps1` to extract from LK ADTs:

```powershell
# After ADT conversion completes
$lkAdtDir = Join-Path $tempExportDir 'World\Maps\$Map'

# Extract terrain from LK ADTs (not Alpha WDT)
$terrainCsv = Join-Path $rollbackMapCsvDir ($Map + '_mcnk_terrain.csv')
& dotnet run --project $alphaToolProject --configuration Release -- `
    extract-terrain-from-lk `
    --input $lkAdtDir `
    --out $terrainCsv
```

**Step 3**: Use existing AreaTable.dbc (3.3.5) which will now match:

```csharp
// This will now work because LK AreaIDs match 3.3.5 AreaTable
var areaName = areaLookup.GetAreaName(mcnkEntry.AreaId);
// Returns "Elwynn Forest" instead of "Unknown Area 1234"
```

#### Option B: Create Alpha→LK AreaID Mapping (Fallback)

If Option A is too complex, create a hardcoded mapping:

```csharp
// AlphaToLkAreaIdMap.cs
public static class AlphaToLkAreaIdMap
{
    private static readonly Dictionary<int, int> AlphaToLk = new()
    {
        { 1234, 12 },  // Alpha Elwynn → LK Elwynn
        { 5678, 14 },  // Alpha Durotar → LK Durotar
        // ... etc
    };
    
    public static int MapToLk(int alphaAreaId)
    {
        return AlphaToLk.TryGetValue(alphaAreaId, out var lkId) ? lkId : alphaAreaId;
    }
}
```

Then in area overlay builder:
```csharp
var lkAreaId = AlphaToLkAreaIdMap.MapToLk(mcnkEntry.AreaId);
var areaName = areaLookup.GetAreaName(lkAreaId);
```

### Recommended Approach

**Option A (Extract from LK ADTs)** is better because:
1. ✅ No hardcoded mapping needed
2. ✅ AlphaWdtAnalyzer already does the conversion correctly
3. ✅ More maintainable (single source of truth)
4. ✅ Handles all maps automatically

### Timeline
- **Complexity**: Medium-High (requires LK ADT parsing)
- **Estimated Time**: 6-8 hours
- **Priority**: High (broken feature that users expect to work)

---

## Implementation Order

### Phase 1 (Immediate - This Session)
1. ✅ Fix terrain CSV path mismatch
2. ✅ Fix area boundaries hide() bug
3. ✅ Update README with troubleshooting

### Phase 2 (Next Session - AreaTable Fix)
1. Extract AreaID from converted LK ADTs
2. Update terrain CSV generation to use LK AreaIDs
3. Verify area boundary names appear correctly
4. Update documentation

### Phase 3 (Future - Shadow Maps)
1. Create McnkShadowOverlayBuilder
2. Create shadow map UI layer
3. Add shadow map toggle to viewer
4. Test with real data

---

## Notes

### Why Area Names Don't Work Currently

**Alpha AreaTable.dbc structure** is completely different from **3.3.5 AreaTable.dbc**:
- Different IDs
- Different parent/child relationships  
- Different field offsets
- Different record counts

**AlphaWdtAnalyzer's conversion** handles this by:
1. Reading Alpha AreaID from MCNK
2. Looking up Alpha AreaTable entry
3. Finding equivalent LK AreaTable entry
4. Writing LK AreaID to converted MCNK

We just need to use that **already-converted LK AreaID** instead of the original Alpha one!

### Shadow Map Visualization Ideas

Shadow maps could be rendered as:
1. **Semi-transparent black overlay** (simplest - shows baked shadows)
2. **Colored heatmap** (red = dark, blue = light)
3. **Contour lines** (like topographic maps)
4. **Adjustable opacity slider**

Recommend starting with option 1 (semi-transparent black).

---

## Testing Checklist

### AreaTable Fix Testing
- [ ] Generate terrain overlays for Elwynn Forest
- [ ] Enable Area Boundaries overlay
- [ ] Verify area names show "Elwynn Forest", "Goldshire", etc. (not "Unknown Area XXX")
- [ ] Test with multiple maps (Durotar, Mulgore, etc.)
- [ ] Verify parent/child zone relationships work correctly

### Shadow Map Testing
- [ ] Generate shadow map overlay for small map (DeadminesInstance)
- [ ] Enable Shadow Map overlay in viewer
- [ ] Verify dark areas appear in correct locations
- [ ] Test opacity slider (if implemented)
- [ ] Performance test on large map (Azeroth)

---

## Related Files

### AreaTable Fix
- `AlphaWdtAnalyzer.Core/Terrain/McnkTerrainExtractor.cs` - Extract AreaID
- `WoWRollback.Core/Services/Viewer/McnkTerrainOverlayBuilder.cs` - Use AreaID for names
- `rebuild-and-regenerate.ps1` - Orchestration

### Shadow Maps
- `AlphaWdtAnalyzer.Core/Terrain/McnkShadowExtractor.cs` - Already exists
- `WoWRollback.Core/Services/Viewer/McnkShadowOverlayBuilder.cs` - To create
- `ViewerAssets/js/overlays/shadowMapLayer.js` - To create
