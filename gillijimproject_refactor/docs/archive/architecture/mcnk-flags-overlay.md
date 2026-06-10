# MCNK Flags Overlay Design

## Purpose

Extract and visualize terrain chunk flags from ADT files, focusing on **impassible** areas and **holes** for navigation analysis.

---

## ADT Structure Reference

### MCNK Chunk (Map Chunk)

From `reference_data/wowdev.wiki/ADT_v18.md`:

```c
struct SMChunk {
  struct {
    uint32_t has_mcsh : 1;           // bit 0
    uint32_t impass : 1;             // bit 1 ← TARGET
    uint32_t lq_river : 1;           // bit 2
    uint32_t lq_ocean : 1;           // bit 3
    uint32_t lq_magma : 1;           // bit 4
    uint32_t lq_slime : 1;           // bit 5
    uint32_t has_mccv : 1;           // bit 6
    uint32_t unknown_0x80 : 1;       // bit 7
    uint32_t : 7;                    // bits 8-14
    uint32_t do_not_fix_alpha_map : 1;  // bit 15
    uint32_t high_res_holes : 1;     // bit 16 ← HOLE INDICATOR
    uint32_t : 15;                   // bits 17-31
  } flags;
  
  uint32_t IndexX;                  // Chunk column (0-15)
  uint32_t IndexY;                  // Chunk row (0-15)
  // ... (other fields)
  uint16_t holes_low_res;           // Offset 0x03C ← LOW-RES HOLES
  // ... (other fields)
  uint64_t holes_high_res;          // Offset 0x014 (if high_res_holes flag set)
  // ... (other fields)
};
```

**Key Fields:**
- **`flags.impass`** (bit 1): Marks chunk as impassible terrain
- **`flags.high_res_holes`** (bit 16): If set, use 64-bit hole map; else use 16-bit
- **`holes_low_res`** (uint16): 4×4 grid of hole flags (16 bits)
- **`holes_high_res`** (uint64): 8×8 grid of hole flags (64 bits, only if `high_res_holes` flag set)

### Hole Bitmap Layout

#### Low-Resolution (16-bit, 4×4 grid):
```
Bit layout (each bit = one sub-chunk):
  0x1    0x2    0x4    0x8
  0x10   0x20   0x40   0x80
  0x100  0x200  0x400  0x800
  0x1000 0x2000 0x4000 0x8000
```

#### High-Resolution (64-bit, 8×8 grid):
```
Read as byte array: Holes[row] >> col) & 1
If read as uint64_t, invert rows due to endianness
```

---

## Data Pipeline Design

### Stage 1: Extraction (AlphaWDTAnalysisTool)

#### New CSV Schema: `<map>_mcnk_flags.csv`

```csv
map,tile_row,tile_col,chunk_row,chunk_col,flags_raw,impassible,has_holes,hole_type,hole_bitmap_hex,hole_count
Azeroth,31,34,0,0,0x00000002,true,false,none,0x0000,0
Azeroth,31,34,0,1,0x00010002,true,true,low_res,0x00F0,4
Azeroth,31,34,15,15,0x00000000,false,false,none,0x0000,0
```

#### Column Definitions:

| Column | Type | Description |
|--------|------|-------------|
| `map` | string | Map name (e.g., "Azeroth") |
| `tile_row` | int | ADT tile row (0-63) |
| `tile_col` | int | ADT tile column (0-63) |
| `chunk_row` | int | MCNK chunk row within tile (0-15) |
| `chunk_col` | int | MCNK chunk column within tile (0-15) |
| `flags_raw` | hex string | Full 32-bit flags value (for debugging) |
| `impassible` | bool | True if `flags.impass` bit is set |
| `has_holes` | bool | True if any hole bits are set |
| `hole_type` | enum | "none", "low_res", "high_res" |
| `hole_bitmap_hex` | hex string | Hex representation of hole bitmap |
| `hole_count` | int | Number of hole bits set (0-16 or 0-64) |

#### Implementation: New Class `McnkFlagsExtractor.cs`

```csharp
namespace AlphaWdtAnalyzer.Core;

public sealed class McnkFlagsExtractor
{
    public static List<McnkFlagEntry> ExtractFlags(WdtAlphaScanner wdt)
    {
        var results = new List<McnkFlagEntry>();
        
        foreach (var adtNum in wdt.AdtNumbers)
        {
            var adt = new AdtAlpha(wdt.WdtPath, wdt.AdtMhdrOffsets[adtNum], adtNum);
            var tileX = adt.GetXCoord();
            var tileY = adt.GetYCoord();
            
            // Each ADT has 16×16 MCNK chunks
            for (int chunkRow = 0; chunkRow < 16; chunkRow++)
            {
                for (int chunkCol = 0; chunkCol < 16; chunkCol++)
                {
                    var mcnk = adt.GetMcnkChunk(chunkRow, chunkCol);
                    
                    // Parse flags (first 4 bytes of MCNK)
                    uint flags = BitConverter.ToUInt32(mcnk, 0);
                    bool impassible = (flags & 0x2) != 0;
                    bool highResHoles = (flags & 0x10000) != 0;
                    
                    // Parse holes
                    string holeType = "none";
                    string holeBitmap = "0x0000";
                    int holeCount = 0;
                    
                    if (highResHoles)
                    {
                        // Read 64-bit holes at offset 0x14
                        ulong holes = BitConverter.ToUInt64(mcnk, 0x14);
                        holeType = "high_res";
                        holeBitmap = $"0x{holes:X16}";
                        holeCount = CountBits(holes);
                    }
                    else
                    {
                        // Read 16-bit holes at offset 0x3C
                        ushort holes = BitConverter.ToUInt16(mcnk, 0x3C);
                        if (holes != 0)
                        {
                            holeType = "low_res";
                            holeBitmap = $"0x{holes:X4}";
                            holeCount = CountBits(holes);
                        }
                    }
                    
                    results.Add(new McnkFlagEntry(
                        wdt.MapName,
                        tileY, tileX,
                        chunkRow, chunkCol,
                        $"0x{flags:X8}",
                        impassible,
                        holeCount > 0,
                        holeType,
                        holeBitmap,
                        holeCount
                    ));
                }
            }
        }
        
        return results;
    }
    
    private static int CountBits(ulong value)
    {
        int count = 0;
        while (value != 0)
        {
            count += (int)(value & 1);
            value >>= 1;
        }
        return count;
    }
}

public record McnkFlagEntry(
    string Map,
    int TileRow,
    int TileCol,
    int ChunkRow,
    int ChunkCol,
    string FlagsRaw,
    bool Impassible,
    bool HasHoles,
    string HoleType,
    string HoleBitmapHex,
    int HoleCount
);
```

---

### Stage 2: Transformation (WoWRollback.Core)

#### Overlay JSON Schema

```json
{
  "map": "Azeroth",
  "tile": {"row": 31, "col": 34},
  "minimap": {"width": 512, "height": 512},
  "chunk_size": 32,
  "layers": [
    {
      "version": "0.5.3",
      "terrain_flags": {
        "impassible_chunks": [
          {"row": 0, "col": 0},
          {"row": 0, "col": 1}
        ],
        "hole_chunks": [
          {
            "row": 0,
            "col": 1,
            "type": "low_res",
            "holes": [0, 1, 4, 5]
          }
        ]
      }
    }
  ]
}
```

#### Implementation: `McnkFlagsOverlayBuilder.cs`

```csharp
namespace WoWRollback.Core.Services.Viewer;

public sealed class McnkFlagsOverlayBuilder
{
    public object BuildOverlayJson(
        string map,
        int tileRow,
        int tileCol,
        IEnumerable<McnkFlagEntry> entries,
        string version)
    {
        var versionEntries = entries
            .Where(e => e.Map == map && e.TileRow == tileRow && e.TileCol == tileCol)
            .ToList();
        
        var impassibleChunks = versionEntries
            .Where(e => e.Impassible)
            .Select(e => new { row = e.ChunkRow, col = e.ChunkCol })
            .ToList();
        
        var holeChunks = versionEntries
            .Where(e => e.HasHoles)
            .Select(e => new
            {
                row = e.ChunkRow,
                col = e.ChunkCol,
                type = e.HoleType,
                holes = DecodeHoleBitmap(e.HoleBitmapHex, e.HoleType)
            })
            .ToList();
        
        return new
        {
            map,
            tile = new { row = tileRow, col = tileCol },
            minimap = new { width = 512, height = 512 },
            chunk_size = 32, // pixels per chunk (512 / 16)
            layers = new[]
            {
                new
                {
                    version,
                    terrain_flags = new
                    {
                        impassible_chunks = impassibleChunks,
                        hole_chunks = holeChunks
                    }
                }
            }
        };
    }
    
    private static int[] DecodeHoleBitmap(string hexBitmap, string holeType)
    {
        var holes = new List<int>();
        
        if (holeType == "low_res")
        {
            ushort bitmap = Convert.ToUInt16(hexBitmap, 16);
            for (int i = 0; i < 16; i++)
            {
                if ((bitmap & (1 << i)) != 0)
                    holes.Add(i);
            }
        }
        else if (holeType == "high_res")
        {
            ulong bitmap = Convert.ToUInt64(hexBitmap, 16);
            for (int i = 0; i < 64; i++)
            {
                if ((bitmap & (1UL << i)) != 0)
                    holes.Add(i);
            }
        }
        
        return holes.ToArray();
    }
}
```

---

### Stage 3: Visualization (ViewerAssets)

#### New Module: `terrainFlagsLayer.js`

```javascript
// Render impassible chunks as red semi-transparent overlays
export function renderTerrainFlags(map, tileRow, tileCol, data, options) {
    const terrainLayer = L.layerGroup();
    
    if (!data.layers || data.layers.length === 0) return terrainLayer;
    
    const versionData = data.layers[0];
    const flags = versionData.terrain_flags;
    
    if (!flags) return terrainLayer;
    
    const chunkSize = data.chunk_size || 32; // pixels per chunk
    
    // Render impassible chunks
    if (flags.impassible_chunks) {
        flags.impassible_chunks.forEach(chunk => {
            const bounds = getChunkBounds(tileRow, tileCol, chunk.row, chunk.col, chunkSize);
            const rect = L.rectangle(bounds, {
                color: '#FF0000',
                weight: 1,
                fillColor: '#FF0000',
                fillOpacity: 0.3,
                interactive: true
            });
            rect.bindPopup(`<strong>Impassible Terrain</strong><br>Chunk: [${chunk.row},${chunk.col}]`);
            terrainLayer.addLayer(rect);
        });
    }
    
    // Render holes
    if (flags.hole_chunks) {
        flags.hole_chunks.forEach(chunk => {
            const gridSize = chunk.type === 'high_res' ? 8 : 4;
            const subChunkSize = chunkSize / gridSize;
            
            chunk.holes.forEach(holeIndex => {
                const subRow = Math.floor(holeIndex / gridSize);
                const subCol = holeIndex % gridSize;
                const bounds = getSubChunkBounds(
                    tileRow, tileCol,
                    chunk.row, chunk.col,
                    subRow, subCol,
                    chunkSize, gridSize
                );
                
                const rect = L.rectangle(bounds, {
                    color: '#000000',
                    weight: 1,
                    fillColor: '#000000',
                    fillOpacity: 0.7,
                    interactive: true
                });
                rect.bindPopup(`<strong>Terrain Hole</strong><br>Chunk: [${chunk.row},${chunk.col}]<br>Type: ${chunk.type}`);
                terrainLayer.addLayer(rect);
            });
        });
    }
    
    return terrainLayer;
}

function getChunkBounds(tileRow, tileCol, chunkRow, chunkCol, chunkSizePx) {
    // Convert chunk position to pixel offset within tile
    const pixelX = chunkCol * chunkSizePx;
    const pixelY = chunkRow * chunkSizePx;
    
    // Convert to Leaflet lat/lng coordinates
    const { lat: lat1, lng: lng1 } = pixelToLatLng(tileRow, tileCol, pixelX, pixelY, 512, 512);
    const { lat: lat2, lng: lng2 } = pixelToLatLng(tileRow, tileCol, pixelX + chunkSizePx, pixelY + chunkSizePx, 512, 512);
    
    return [[lat1, lng1], [lat2, lng2]];
}
```

#### Integration in `main.js`

```javascript
import { renderTerrainFlags } from './terrainFlagsLayer.js';

// Add terrain flags layer
let terrainFlagsLayer = L.layerGroup();
terrainFlagsLayer.addTo(map);

// Toggle control
const terrainFlagsToggle = document.getElementById('showTerrainFlags');
terrainFlagsToggle.addEventListener('change', (e) => {
    if (e.target.checked) {
        updateTerrainFlags();
    } else {
        terrainFlagsLayer.clearLayers();
    }
});

async function updateTerrainFlags() {
    terrainFlagsLayer.clearLayers();
    
    const bounds = map.getBounds();
    const tiles = getVisibleTiles(bounds);
    
    for (const tile of tiles) {
        const path = state.getTerrainFlagsPath(state.selectedMap, tile.row, tile.col, state.selectedVersion);
        try {
            const data = await loadOverlay(path);
            const layer = renderTerrainFlags(state.selectedMap, tile.row, tile.col, data, state.config);
            layer.getLayers().forEach(l => terrainFlagsLayer.addLayer(l));
        } catch (e) {
            // Tile has no terrain flags data
        }
    }
}
```

---

## UI Controls

Add to `index.html` sidebar:

```html
<div class="control-group">
    <h3>Terrain Overlays</h3>
    <label>
        <input type="checkbox" id="showTerrainFlags">
        Show Impassible & Holes
    </label>
    <label>
        <input type="checkbox" id="showHolesOnly">
        Holes Only
    </label>
    <label>
        <input type="checkbox" id="showImpassibleOnly">
        Impassible Only
    </label>
</div>
```

---

## Performance Considerations

1. **Chunking**: 16×16 chunks per tile = 256 rectangles max per tile
2. **Lazy loading**: Only load flags for visible tiles
3. **Caching**: Cache flag data alongside placement overlays
4. **Filtering**: Allow toggling individual flag types to reduce visual clutter

---

## Testing Checklist

- [ ] Extract flags from Alpha ADT successfully
- [ ] CSV output validates against schema
- [ ] JSON overlay builds correctly
- [ ] Impassible chunks render as red overlay
- [ ] Holes render as black overlay
- [ ] Low-res holes (4×4) display correctly
- [ ] High-res holes (8×8) display correctly
- [ ] Popup shows chunk coordinates
- [ ] Layer toggles work
- [ ] Performance acceptable (< 100ms render time per tile)

---

## Future Enhancements

1. **Liquid flags**: Visualize `lq_river`, `lq_ocean`, `lq_magma`, `lq_slime`
2. **Shadow map**: Render `has_mcsh` chunks differently
3. **Vertex coloring**: Visualize `has_mccv` chunks
4. **Area boundaries**: Show AreaID transitions
5. **Height map**: Overlay MCVT height data as contour lines
