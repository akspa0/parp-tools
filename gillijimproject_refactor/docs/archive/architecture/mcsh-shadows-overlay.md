# MCSH Shadow Map Overlay Design

## Purpose

Extract and visualize **baked shadow maps** from MCSH subchunks in ADT files as a semi-transparent grayscale overlay.

---

## ADT Structure Reference

### MCSH Sub-chunk (Map Chunk Shadow)

From `reference_data/wowdev.wiki/ADT_v18.md`:

```c
// MCSH is a subchunk within MCNK
struct {
  uint1_t shadow_map[64][64];  // 64×64 bitmap, 1 bit per pixel
  // or 63×63 with auto-fill if do_not_fix_alpha_map flag not set
} mcsh;
```

**Key Points**:
- Shadow map is **64×64 bits** = **512 bytes** per chunk
- Each bit represents a shadow value: `0` = shadowed (dark), `1` = lit (bright)
- Stored **LSB first** (least significant bit first)
- Only present if MCNK `flags.has_mcsh` (bit 0) is set
- Offset: `MCNK.ofsShadow` (offset 0x2C), size: `MCNK.sizeShadow` (offset 0x30)

### Data Layout

```
Byte 0:  [bit0, bit1, bit2, bit3, bit4, bit5, bit6, bit7]  ← Row 0, pixels 0-7
Byte 1:  [bit0, bit1, bit2, bit3, bit4, bit5, bit6, bit7]  ← Row 0, pixels 8-15
...
Byte 7:  [bit0, bit1, bit2, bit3, bit4, bit5, bit6, bit7]  ← Row 0, pixels 56-63
Byte 8:  [bit0, bit1, bit2, bit3, bit4, bit5, bit6, bit7]  ← Row 1, pixels 0-7
...
Byte 511: [bit0, bit1, bit2, bit3, bit4, bit5, bit6, bit7] ← Row 63, pixels 56-63
```

**Total**: 64 rows × 8 bytes = 512 bytes

---

## Data Pipeline Design

### Stage 1: Extraction (AlphaWDTAnalysisTool)

#### CSV Schema: `<map>_mcnk_shadows.csv`

```csv
map,tile_row,tile_col,chunk_row,chunk_col,has_shadow,shadow_size,shadow_bitmap_base64
Azeroth,31,34,0,0,true,512,AAAA//8AAAD//wAA...
Azeroth,31,34,0,1,false,0,
Azeroth,31,34,15,15,true,512,/////wAAAAD//wAA...
```

#### Column Definitions:

| Column | Type | Description |
|--------|------|-------------|
| `map` | string | Map name |
| `tile_row` | int | ADT tile row (0-63) |
| `tile_col` | int | ADT tile column (0-63) |
| `chunk_row` | int | MCNK chunk row within tile (0-15) |
| `chunk_col` | int | MCNK chunk column within tile (0-15) |
| `has_shadow` | bool | True if MCSH data exists |
| `shadow_size` | int | Size in bytes (usually 512) |
| `shadow_bitmap_base64` | string | Base64-encoded shadow bitmap |

---

#### Implementation: `McnkShadowExtractor.cs`

```csharp
namespace AlphaWdtAnalyzer.Core;

public sealed class McnkShadowExtractor
{
    public static List<McnkShadowEntry> ExtractShadows(WdtAlphaScanner wdt)
    {
        var results = new List<McnkShadowEntry>();
        
        foreach (var adtNum in wdt.AdtNumbers)
        {
            var adt = new AdtAlpha(wdt.WdtPath, wdt.AdtMhdrOffsets[adtNum], adtNum);
            var tileX = adt.GetXCoord();
            var tileY = adt.GetYCoord();
            
            for (int chunkRow = 0; chunkRow < 16; chunkRow++)
            {
                for (int chunkCol = 0; chunkCol < 16; chunkCol++)
                {
                    var mcnk = adt.GetMcnkChunk(chunkRow, chunkCol);
                    
                    // Check if chunk has shadow map
                    uint flags = BitConverter.ToUInt32(mcnk, 0x00);
                    bool hasMcsh = (flags & 0x1) != 0;
                    
                    if (!hasMcsh)
                    {
                        results.Add(new McnkShadowEntry(
                            Map: wdt.MapName,
                            TileRow: tileY,
                            TileCol: tileX,
                            ChunkRow: chunkRow,
                            ChunkCol: chunkCol,
                            HasShadow: false,
                            ShadowSize: 0,
                            ShadowBitmapBase64: string.Empty
                        ));
                        continue;
                    }
                    
                    // Read shadow map offset and size
                    uint ofsShadow = BitConverter.ToUInt32(mcnk, 0x2C);
                    uint sizeShadow = BitConverter.ToUInt32(mcnk, 0x30);
                    
                    if (sizeShadow == 0 || ofsShadow == 0)
                    {
                        results.Add(new McnkShadowEntry(
                            Map: wdt.MapName,
                            TileRow: tileY,
                            TileCol: tileX,
                            ChunkRow: chunkRow,
                            ChunkCol: chunkCol,
                            HasShadow: false,
                            ShadowSize: 0,
                            ShadowBitmapBase64: string.Empty
                        ));
                        continue;
                    }
                    
                    // Extract shadow bitmap
                    byte[] shadowData = new byte[sizeShadow];
                    Array.Copy(mcnk, (int)ofsShadow, shadowData, 0, (int)sizeShadow);
                    
                    // Encode as base64
                    string base64 = Convert.ToBase64String(shadowData);
                    
                    results.Add(new McnkShadowEntry(
                        Map: wdt.MapName,
                        TileRow: tileY,
                        TileCol: tileX,
                        ChunkRow: chunkRow,
                        ChunkCol: chunkCol,
                        HasShadow: true,
                        ShadowSize: (int)sizeShadow,
                        ShadowBitmapBase64: base64
                    ));
                }
            }
        }
        
        return results;
    }
}

public record McnkShadowEntry(
    string Map,
    int TileRow,
    int TileCol,
    int ChunkRow,
    int ChunkCol,
    bool HasShadow,
    int ShadowSize,
    string ShadowBitmapBase64
);
```

---

### Stage 2: Transformation (WoWRollback.Core)

#### Overlay JSON Schema

```json
{
  "map": "Azeroth",
  "tile": {"row": 31, "col": 34},
  "shadow_map": {
    "width": 1024,
    "height": 1024,
    "format": "png",
    "data_url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABA..."
  }
}
```

**Strategy**: Composite all 16×16 chunk shadow maps into a single 1024×1024 image:
- Each chunk shadow = 64×64 pixels
- 16 chunks × 64 = 1024 pixels per dimension
- Output as PNG data URL for direct rendering in viewer

---

#### Implementation: `ShadowMapCompositor.cs`

```csharp
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace WoWRollback.Core.Services.Viewer;

public sealed class ShadowMapCompositor
{
    public static string BuildShadowMapOverlay(
        string map,
        int tileRow,
        int tileCol,
        List<McnkShadowEntry> shadows)
    {
        const int chunkSize = 64;
        const int tilesPerDim = 16;
        const int imageSize = chunkSize * tilesPerDim; // 1024
        
        // Create 1024×1024 image (grayscale)
        using var image = new Image<Rgba32>(imageSize, imageSize);
        
        foreach (var shadow in shadows)
        {
            if (!shadow.HasShadow || string.IsNullOrEmpty(shadow.ShadowBitmapBase64))
                continue;
            
            // Decode shadow bitmap
            byte[] bitmap = Convert.FromBase64String(shadow.ShadowBitmapBase64);
            
            // Convert bit array to 64×64 pixel array
            var pixels = DecodeShadowBitmap(bitmap);
            
            // Composite into tile image at correct position
            int offsetX = shadow.ChunkCol * chunkSize;
            int offsetY = shadow.ChunkRow * chunkSize;
            
            for (int y = 0; y < chunkSize; y++)
            {
                for (int x = 0; x < chunkSize; x++)
                {
                    bool isLit = pixels[y, x];
                    byte value = isLit ? (byte)255 : (byte)0;
                    
                    // Render as semi-transparent dark overlay
                    // Lit = transparent, Shadowed = dark
                    byte alpha = isLit ? (byte)0 : (byte)128;
                    
                    image[offsetX + x, offsetY + y] = new Rgba32(0, 0, 0, alpha);
                }
            }
        }
        
        // Convert to PNG data URL
        using var ms = new MemoryStream();
        image.SaveAsPng(ms);
        byte[] imageBytes = ms.ToArray();
        string base64 = Convert.ToBase64String(imageBytes);
        
        return $"data:image/png;base64,{base64}";
    }
    
    private static bool[,] DecodeShadowBitmap(byte[] bitmap)
    {
        const int size = 64;
        var pixels = new bool[size, size];
        
        int bitIndex = 0;
        for (int y = 0; y < size; y++)
        {
            for (int x = 0; x < size; x++)
            {
                int byteIndex = bitIndex / 8;
                int bitOffset = bitIndex % 8;
                
                // Extract bit (LSB first)
                bool isLit = (bitmap[byteIndex] & (1 << bitOffset)) != 0;
                pixels[y, x] = isLit;
                
                bitIndex++;
            }
        }
        
        return pixels;
    }
}
```

---

### Stage 3: Visualization (ViewerAssets)

#### Module: `shadowMapLayer.js`

```javascript
export function renderShadowMap(map, tileRow, tileCol, data, options) {
    if (!data.shadow_map || !data.shadow_map.data_url) {
        return null;
    }
    
    const bounds = getTileBounds(tileRow, tileCol);
    
    // Create image overlay
    const shadowLayer = L.imageOverlay(
        data.shadow_map.data_url,
        bounds,
        {
            opacity: options.shadowOpacity || 0.5,
            interactive: false,
            className: 'shadow-map-overlay'
        }
    );
    
    return shadowLayer;
}

function getTileBounds(tileRow, tileCol) {
    // Convert tile indices to Leaflet bounds
    // Assuming wow.tools coordinate system
    const north = 63 - tileRow;
    const south = north - 1;
    const west = tileCol;
    const east = tileCol + 1;
    
    return [[south, west], [north, east]];
}
```

---

#### Integration in `main.js`

```javascript
import { renderShadowMap } from './overlays/shadowMapLayer.js';

let shadowMapLayer = L.layerGroup();
shadowMapLayer.addTo(map);

// UI control
document.getElementById('showShadows').addEventListener('change', (e) => {
    if (e.target.checked) {
        updateShadowMaps();
    } else {
        shadowMapLayer.clearLayers();
    }
});

// Opacity slider
document.getElementById('shadowOpacity').addEventListener('input', (e) => {
    const opacity = parseFloat(e.target.value);
    shadowMapLayer.eachLayer(layer => {
        if (layer.setOpacity) {
            layer.setOpacity(opacity);
        }
    });
});

async function updateShadowMaps() {
    shadowMapLayer.clearLayers();
    
    const bounds = map.getBounds();
    const tiles = getVisibleTiles(bounds);
    
    for (const tile of tiles) {
        const path = `overlays/${state.selectedVersion}/${state.selectedMap}/shadows/tile_r${tile.row}_c${tile.col}.json`;
        try {
            const data = await loadOverlay(path);
            const layer = renderShadowMap(state.selectedMap, tile.row, tile.col, data, {
                shadowOpacity: parseFloat(document.getElementById('shadowOpacity').value)
            });
            if (layer) {
                shadowMapLayer.addLayer(layer);
            }
        } catch (e) {
            // No shadow data for this tile
        }
    }
}
```

---

## UI Controls

```html
<div class="overlay-group">
    <label>
        <input type="checkbox" id="showShadows">
        Baked Shadows (MCSH)
    </label>
    <div class="indent" id="shadowOptions">
        <label>
            Shadow Opacity:
            <input type="range" id="shadowOpacity" min="0" max="1" step="0.1" value="0.5">
            <span id="shadowOpacityValue">50%</span>
        </label>
    </div>
</div>
```

---

## Performance Considerations

### Option 1: Composite PNG (Recommended)
**Pros**:
- Single image per tile = 1 HTTP request
- Browser-native rendering (very fast)
- Automatic scaling/interpolation

**Cons**:
- Requires image processing library (ImageSharp)
- PNG encoding takes ~50-100ms per tile

### Option 2: Canvas Rendering
**Pros**:
- No server-side image processing
- Can render directly from base64 bitmap

**Cons**:
- 256 draw calls per tile (one per chunk)
- More complex client-side code
- Potential performance issues on slower devices

**Recommendation**: Use Option 1 (composite PNG) for best performance.

---

## Alternative: Client-Side Canvas Rendering

If server-side image processing is not desired:

```javascript
function renderShadowMapCanvas(shadows) {
    const canvas = document.createElement('canvas');
    canvas.width = 1024;
    canvas.height = 1024;
    const ctx = canvas.getContext('2d');
    
    shadows.forEach(shadow => {
        if (!shadow.has_shadow) return;
        
        const bitmap = atob(shadow.shadow_bitmap_base64);
        const offsetX = shadow.chunk_col * 64;
        const offsetY = shadow.chunk_row * 64;
        
        // Decode and render 64×64 shadow bitmap
        for (let y = 0; y < 64; y++) {
            for (let x = 0; x < 64; x++) {
                const bitIndex = y * 64 + x;
                const byteIndex = Math.floor(bitIndex / 8);
                const bitOffset = bitIndex % 8;
                const isLit = (bitmap.charCodeAt(byteIndex) & (1 << bitOffset)) !== 0;
                
                if (!isLit) {
                    ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
                    ctx.fillRect(offsetX + x, offsetY + y, 1, 1);
                }
            }
        }
    });
    
    return canvas.toDataURL('image/png');
}
```

---

## Testing Checklist

- [ ] MCSH data extracts correctly when `has_mcsh` flag is set
- [ ] Empty entries created when no shadow data
- [ ] Base64 encoding/decoding works
- [ ] Shadow bitmap decodes to correct 64×64 grid
- [ ] Composite 1024×1024 image renders correctly
- [ ] PNG data URL loads in viewer
- [ ] Image overlay displays at correct tile position
- [ ] Opacity slider works
- [ ] Performance acceptable (< 200ms composite time per tile)
- [ ] Shadow detail visible at various zoom levels

---

## Example Output

### Shadow Map Visualization

```
┌─────────┬─────────┬─────────┬─────────┐
│ ░░░░░░░ │ ████████│ ░░░░░░░ │ ████████│  ← Dark = shadowed
│ ░░░░░░░ │ ████████│ ░░░░░░░ │ ████████│  ← Light = lit
│ ░░░░░░░ │ ████████│ ░░░░░░░ │ ████████│
├─────────┼─────────┼─────────┼─────────┤
│ ████████│ ░░░░░░░ │ ████████│ ░░░░░░░ │
│ ████████│ ░░░░░░░ │ ████████│ ░░░░░░░ │
│ ████████│ ░░░░░░░ │ ████████│ ░░░░░░░ │
└─────────┴─────────┴─────────┴─────────┘
      16×16 chunks = 1024×1024 pixels
```

---

## Future Enhancements

1. **Adaptive Resolution**: Render lower resolution at distant zoom levels
2. **Compression**: Use WebP instead of PNG for smaller file sizes
3. **Caching**: Cache composite images on server to avoid re-generation
4. **Blending**: Blend shadows with terrain textures for more realistic look
5. **Animation**: Fade shadows in/out smoothly when toggling
