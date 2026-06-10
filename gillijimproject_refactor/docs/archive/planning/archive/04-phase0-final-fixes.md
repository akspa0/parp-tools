# Phase 0 Final Fixes - AreaTable & Shadow Maps

## Overview

Two critical fixes to complete the 2D foundation before 3D work begins:

1. **AreaTable Mapping Fix** - Show real area names instead of "Unknown Area 1234"
2. **Shadow Map Overlay** - Visualize baked terrain shadows

---

## Fix 1: AreaTable Mapping (High Priority)

### Problem
Area boundaries show "Unknown Area 1234" instead of real names like "Elwynn Forest" or "Goldshire".

**Root Cause**: We're using Alpha AreaIDs which don't match 3.3.5 AreaTable.dbc structure.

### Solution
Extract AreaID from **converted LK ADTs** (which already have correct IDs) instead of Alpha WDT.

### Implementation Steps

#### Step 1: Read AreaID from LK ADTs

**Create**: `AlphaWdtAnalyzer.Core/Terrain/LkAdtAreaReader.cs`

```csharp
using System;
using System.Collections.Generic;
using System.IO;
using Warcraft.NET.Files.ADT.Chunks;
using Warcraft.NET.Files.ADT;

namespace AlphaWdtAnalyzer.Core.Terrain;

/// <summary>
/// Reads AreaID values from converted LK ADT files.
/// </summary>
public static class LkAdtAreaReader
{
    public record ChunkAreaInfo(
        string MapName,
        int TileRow,
        int TileCol,
        int ChunkY,
        int ChunkX,
        int AreaId
    );

    /// <summary>
    /// Extracts AreaID from all MCNK chunks in a LK ADT directory.
    /// </summary>
    public static List<ChunkAreaInfo> ExtractAreaIds(string lkAdtDirectory, string mapName)
    {
        var results = new List<ChunkAreaInfo>();
        
        if (!Directory.Exists(lkAdtDirectory))
        {
            Console.WriteLine($"[warn] LK ADT directory not found: {lkAdtDirectory}");
            return results;
        }

        var adtFiles = Directory.GetFiles(lkAdtDirectory, "*.adt", SearchOption.TopDirectoryOnly);
        Console.WriteLine($"[area] Reading AreaIDs from {adtFiles.Length} LK ADT files");

        foreach (var adtPath in adtFiles)
        {
            try
            {
                var (row, col) = ParseTileCoordinates(Path.GetFileNameWithoutExtension(adtPath));
                var chunkAreas = ReadAdtAreaIds(adtPath, mapName, row, col);
                results.AddRange(chunkAreas);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[warn] Failed to read {Path.GetFileName(adtPath)}: {ex.Message}");
            }
        }

        Console.WriteLine($"[area] Extracted AreaIDs for {results.Count} chunks");
        return results;
    }

    private static List<ChunkAreaInfo> ReadAdtAreaIds(string adtPath, string mapName, int tileRow, int tileCol)
    {
        var results = new List<ChunkAreaInfo>();

        using var fs = File.OpenRead(adtPath);
        
        // Use Warcraft.NET to parse ADT
        var adt = new ADT(fs.ReadBytes((int)fs.Length));
        
        // Read all MCNK chunks
        for (int chunkY = 0; chunkY < 16; chunkY++)
        {
            for (int chunkX = 0; chunkX < 16; chunkX++)
            {
                int chunkIndex = chunkY * 16 + chunkX;
                var mcnk = adt.Chunks[chunkIndex];
                
                results.Add(new ChunkAreaInfo(
                    mapName,
                    tileRow,
                    tileCol,
                    chunkY,
                    chunkX,
                    mcnk.AreaID
                ));
            }
        }

        return results;
    }

    private static (int row, int col) ParseTileCoordinates(string filename)
    {
        // Parse "MapName_32_45" → (32, 45)
        var parts = filename.Split('_');
        if (parts.Length >= 3)
        {
            if (int.TryParse(parts[^2], out int row) && int.TryParse(parts[^1], out int col))
            {
                return (row, col);
            }
        }
        throw new FormatException($"Cannot parse tile coordinates from: {filename}");
    }
}
```

#### Step 2: Update McnkTerrainExtractor

**Modify**: `AlphaWdtAnalyzer.Core/Terrain/McnkTerrainExtractor.cs`

Add overload that reads from LK ADTs:

```csharp
/// <summary>
/// Extracts terrain data from converted LK ADTs (has correct AreaIDs).
/// </summary>
public static List<McnkTerrainEntry> ExtractTerrainFromLkAdts(
    string lkAdtDirectory, 
    string mapName,
    WdtAlphaScanner alphaWdt)
{
    var entries = new List<McnkTerrainEntry>();
    
    // Extract AreaIDs from LK ADTs
    var areaData = LkAdtAreaReader.ExtractAreaIds(lkAdtDirectory, mapName);
    var areaLookup = areaData.ToDictionary(
        a => (a.TileRow, a.TileCol, a.ChunkY, a.ChunkX),
        a => a.AreaId
    );
    
    // Extract other terrain data from Alpha WDT (flags, liquids, etc.)
    foreach (var adtPath in alphaWdt.AdtPaths)
    {
        var (row, col) = ParseTileCoordinates(adtPath);
        
        using var fs = File.OpenRead(adtPath);
        using var br = new BinaryReader(fs);
        
        // ... existing MCNK parsing code ...
        
        for (int chunkY = 0; chunkY < 16; chunkY++)
        {
            for (int chunkX = 0; chunkX < 16; chunkX++)
            {
                // Extract flags, liquids, etc. from Alpha
                var flags = ExtractMcnkFlags(br, chunkY, chunkX);
                var liquidType = ExtractLiquidType(br, chunkY, chunkX);
                // ... etc ...
                
                // Get LK AreaID from lookup
                var key = (row, col, chunkY, chunkX);
                var areaId = areaLookup.TryGetValue(key, out var id) ? id : 0;
                
                entries.Add(new McnkTerrainEntry(
                    mapName,
                    row,
                    col,
                    chunkY,
                    chunkX,
                    flags,
                    liquidType,
                    areaId,  // LK AreaID that maps to 3.3.5 AreaTable
                    // ... other fields ...
                ));
            }
        }
    }
    
    return entries;
}
```

#### Step 3: Update CLI to Use LK AreaIDs

**Modify**: `AlphaWdtAnalyzer.Cli/Program.cs`

When `--export-adt` is used:

```csharp
// After ADT conversion completes
if (exportAdt)
{
    AdtExportPipeline.ExportSingle(new AdtExportPipeline.Options
    {
        SingleWdtPath = wdt!,
        CommunityListfilePath = listfile!,
        LkListfilePath = lkListfile,
        ExportDir = exportDir!,
        // ... other options ...
    });
    
    // Extract terrain data with LK AreaIDs
    if (extractMcnkTerrain)
    {
        var lkAdtDir = Path.Combine(exportDir!, "World", "Maps", mapName);
        var terrainEntries = McnkTerrainExtractor.ExtractTerrainFromLkAdts(
            lkAdtDir, 
            mapName,
            wdt
        );
        
        var terrainCsvPath = Path.Combine(csvDir, mapName, $"{mapName}_mcnk_terrain.csv");
        McnkTerrainCsvWriter.WriteCsv(terrainEntries, terrainCsvPath);
        Console.WriteLine($"[terrain] Extracted {terrainEntries.Count} chunks with LK AreaIDs");
    }
}
```

#### Step 4: Verify in Viewer

Once regenerated, area boundaries should show real names!

**Test Cases**:
- Elwynn Forest → "Elwynn Forest" ✅
- Goldshire → "Goldshire" ✅
- Northshire Valley → "Northshire Valley" ✅
- Unknown areas → Still shows "Unknown Area X" (expected for unmapped zones)

### Testing

```powershell
# Regenerate small map for quick testing
.\rebuild-and-regenerate.ps1 `
  -Maps @("DeadminesInstance") `
  -Versions @("0.5.3.3368") `
  -AlphaRoot ..\test_data\ `
  -RefreshCache `
  -Serve

# Check output
# Should see: [terrain] Extracted XXX chunks with LK AreaIDs
# In browser: Enable Area Boundaries, verify names appear
```

---

## Fix 2: Shadow Map Overlay (Low Priority)

### Problem
Shadow map data is extracted but no UI layer to visualize it.

### Solution
Implement decoder from noggit reference + create viewer layer.

### Implementation Steps

#### Step 1: Create MCSH Decoder

**Create**: `AlphaWdtAnalyzer.Core/Terrain/McshDecoder.cs`

```csharp
namespace AlphaWdtAnalyzer.Core.Terrain;

/// <summary>
/// Decodes MCSH (MapChunk Shadow) compressed shadow maps.
/// Reference: noggit-red/src/noggit/MapChunk.cpp lines 235-264
/// </summary>
public static class McshDecoder
{
    /// <summary>
    /// Decodes 512-byte compressed MCSH to 4096-byte shadow map (64×64).
    /// </summary>
    /// <param name="compressed">512-byte compressed shadow data</param>
    /// <returns>4096-byte uncompressed shadow map (0=shadowed, 85=lit)</returns>
    public static byte[] Decode(byte[] compressed)
    {
        if (compressed.Length != 512)
            throw new ArgumentException("MCSH data must be 512 bytes", nameof(compressed));
        
        var uncompressed = new byte[4096];
        int outputIndex = 0;
        
        // Each byte unpacks to 8 shadow values
        foreach (byte compressedByte in compressed)
        {
            // Process each bit (LSB to MSB)
            for (int bit = 0; bit < 8; bit++)
            {
                int mask = 1 << bit;
                // Bit set = 85 (lit), bit clear = 0 (shadowed)
                uncompressed[outputIndex++] = (compressedByte & mask) != 0 ? (byte)85 : (byte)0;
            }
        }
        
        return uncompressed;
    }
    
    /// <summary>
    /// Decodes and applies edge fixing (duplicates row/col 62 to 63).
    /// </summary>
    public static byte[] DecodeWithEdgeFix(byte[] compressed)
    {
        var uncompressed = Decode(compressed);
        
        // Fix last row and column by copying second-to-last
        // This prevents rendering artifacts at chunk edges
        for (int i = 0; i < 64; i++)
        {
            uncompressed[i * 64 + 63] = uncompressed[i * 64 + 62];  // Right edge
            uncompressed[63 * 64 + i] = uncompressed[62 * 64 + i];  // Bottom edge
        }
        uncompressed[63 * 64 + 63] = uncompressed[62 * 64 + 62];  // Corner
        
        return uncompressed;
    }
    
    /// <summary>
    /// Encodes shadow map as intensity digits (0-5) for CSV.
    /// </summary>
    public static string EncodeAsDigits(byte[] shadowMap)
    {
        // 0 → '0', 85 → '5'
        var chars = new char[shadowMap.Length];
        for (int i = 0; i < shadowMap.Length; i++)
        {
            chars[i] = shadowMap[i] == 0 ? '0' : '5';
        }
        return new string(chars);
    }
    
    /// <summary>
    /// Decodes intensity digits back to shadow map.
    /// </summary>
    public static byte[] DecodeFromDigits(string digits)
    {
        if (digits.Length != 4096)
            throw new ArgumentException("Shadow digit string must be 4096 chars", nameof(digits));
        
        var shadowMap = new byte[4096];
        for (int i = 0; i < 4096; i++)
        {
            shadowMap[i] = digits[i] == '0' ? (byte)0 : (byte)85;
        }
        return shadowMap;
    }
}
```

#### Step 2: Update Shadow Extractor

**Modify**: `AlphaWdtAnalyzer.Core/Terrain/McnkShadowExtractor.cs`

Use decoder to output intensity digits:

```csharp
private static string ExtractShadowData(BinaryReader br, long mcshOffset, int mcshSize)
{
    if (mcshSize != 512)
    {
        Console.WriteLine($"[warn] MCSH size should be 512, got {mcshSize}");
        return new string('5', 4096);  // All lit as fallback
    }
    
    br.BaseStream.Position = mcshOffset;
    var compressed = br.ReadBytes(512);
    
    // Decode to 64×64 shadow map
    var uncompressed = McshDecoder.DecodeWithEdgeFix(compressed);
    
    // Encode as compact digit string (4096 chars: '0' or '5')
    return McshDecoder.EncodeAsDigits(uncompressed);
}
```

CSV output format:
```csv
MapName,TileRow,TileCol,ChunkY,ChunkX,ShadowMap
Azeroth,30,30,0,0,0000000000555555555555000000555555000055550000...
```

#### Step 3: Create Shadow Overlay Builder

**Create**: `WoWRollback.Core/Services/Viewer/McnkShadowOverlayBuilder.cs`

```csharp
using System.Text.Json;

namespace WoWRollback.Core.Services.Viewer;

public static class McnkShadowOverlayBuilder
{
    public static void BuildOverlaysForMap(
        string mapName,
        string csvDir,
        string outputDir,
        string version)
    {
        var shadowCsvPath = Path.Combine(csvDir, $"{mapName}_mcnk_shadows.csv");
        if (!File.Exists(shadowCsvPath))
        {
            Console.WriteLine($"[shadow] No shadow CSV found for {mapName}, skipping");
            return;
        }
        
        var shadows = McnkShadowCsvReader.ReadCsv(shadowCsvPath);
        var byTile = shadows.GroupBy(s => (s.TileRow, s.TileCol));
        
        int tileCount = 0;
        foreach (var tileGroup in byTile)
        {
            var (tileRow, tileCol) = tileGroup.Key;
            var chunks = tileGroup.ToList();
            
            // Build JSON overlay
            var overlay = new
            {
                type = "shadow_map",
                version = version,
                map = mapName,
                chunks = chunks.Select(c => new
                {
                    y = c.ChunkY,
                    x = c.ChunkX,
                    // Convert digit string to 64×64 array
                    shadow = ParseShadowDigits(c.ShadowMap)
                })
            };
            
            var json = JsonSerializer.Serialize(overlay, new JsonSerializerOptions 
            { 
                WriteIndented = false  // Compact for smaller files
            });
            
            var overlayDir = Path.Combine(outputDir, mapName, "shadow_map");
            Directory.CreateDirectory(overlayDir);
            
            var outPath = Path.Combine(overlayDir, $"tile_r{tileRow}_c{tileCol}.json");
            File.WriteAllText(outPath, json);
            tileCount++;
        }
        
        Console.WriteLine($"[shadow] Built {tileCount} shadow overlay tiles for {mapName} ({version})");
    }
    
    private static int[][] ParseShadowDigits(string digits)
    {
        var result = new int[64][];
        for (int y = 0; y < 64; y++)
        {
            result[y] = new int[64];
            for (int x = 0; x < 64; x++)
            {
                int index = y * 64 + x;
                result[y][x] = digits[index] - '0';  // '0'→0, '5'→5
            }
        }
        return result;
    }
}
```

#### Step 4: Integrate into Viewer Generation

**Modify**: `WoWRollback.Core/Services/Viewer/ViewerReportWriter.cs`

In `GenerateTerrainOverlays()`:

```csharp
private void GenerateTerrainOverlays(
    string rootDirectory,
    string version,
    string mapName,
    string overlayVersionRoot,
    string safeMapName)
{
    // ... existing terrain overlay code ...
    
    // Generate shadow map overlays if available
    McnkShadowOverlayBuilder.BuildOverlaysForMap(
        mapName, 
        csvMapDir, 
        overlayVersionRoot, 
        version
    );
}
```

#### Step 5: Create Viewer Layer

**Create**: `ViewerAssets/js/overlays/shadowMapLayer.js`

```javascript
export class ShadowMapLayer {
    constructor(map, mapName, version) {
        this.map = map;
        this.mapName = mapName;
        this.version = version;
        this.overlays = new Map();
        this.visible = false;
    }

    async loadTile(row, col) {
        const key = `${row}_${col}`;
        if (this.overlays.has(key)) {
            return this.overlays.get(key);
        }

        const url = `/overlays/${this.version}/${this.mapName}/shadow_map/tile_r${row}_c${col}.json`;
        
        try {
            const response = await fetch(url);
            if (!response.ok) return null;
            
            const data = await response.json();
            
            // Create canvas for shadow rendering
            const canvas = this.renderShadowMap(data);
            
            // Convert to Leaflet ImageOverlay
            const bounds = this.calculateTileBounds(row, col);
            const overlay = L.imageOverlay(canvas.toDataURL(), bounds, {
                opacity: 0.5,
                interactive: false,
                className: 'shadow-overlay'
            });
            
            this.overlays.set(key, overlay);
            
            if (this.visible) {
                overlay.addTo(this.map);
            }
            
            return overlay;
        } catch (err) {
            console.warn(`Failed to load shadow tile ${row},${col}:`, err);
            return null;
        }
    }

    renderShadowMap(data) {
        // Create 1024×1024 canvas (16 chunks × 64 pixels each)
        const canvas = document.createElement('canvas');
        canvas.width = 1024;
        canvas.height = 1024;
        const ctx = canvas.getContext('2d');
        
        // Draw each chunk's shadow map
        data.chunks.forEach(chunk => {
            const chunkPixelX = chunk.x * 64;
            const chunkPixelY = chunk.y * 64;
            
            // Draw 64×64 shadow pixels
            const imageData = ctx.createImageData(64, 64);
            for (let y = 0; y < 64; y++) {
                for (let x = 0; x < 64; x++) {
                    const index = (y * 64 + x) * 4;
                    const shadow = chunk.shadow[y][x];  // 0-5
                    
                    // Black with varying alpha
                    // 0 = opaque black (full shadow)
                    // 5 = transparent (no shadow)
                    const alpha = 255 - (shadow * 51);  // 0→255, 5→0
                    
                    imageData.data[index + 0] = 0;      // R
                    imageData.data[index + 1] = 0;      // G
                    imageData.data[index + 2] = 0;      // B
                    imageData.data[index + 3] = alpha;  // A
                }
            }
            
            ctx.putImageData(imageData, chunkPixelX, chunkPixelY);
        });
        
        return canvas;
    }

    calculateTileBounds(row, col) {
        // Calculate lat/lng bounds for this ADT tile
        // (Matches existing minimap bounds)
        const tileSize = 533.33333;
        const minLat = -(row * tileSize);
        const maxLat = -((row + 1) * tileSize);
        const minLng = col * tileSize;
        const maxLng = (col + 1) * tileSize;
        
        return [[maxLat, minLng], [minLat, maxLng]];
    }

    show() {
        this.visible = true;
        this.overlays.forEach(overlay => overlay.addTo(this.map));
    }

    hide() {
        this.visible = false;
        this.overlays.forEach(overlay => overlay.remove());
    }

    setOpacity(opacity) {
        this.overlays.forEach(overlay => overlay.setOpacity(opacity));
    }
}
```

#### Step 6: Add UI Controls

**Modify**: `ViewerAssets/index.html`

Add shadow map checkbox:

```html
<div class="overlay-controls">
    <!-- ... existing overlays ... -->
    
    <label class="overlay-option">
        <input type="checkbox" id="overlay-shadow-map">
        <span>Shadow Maps</span>
    </label>
    
    <div class="overlay-sub-options" id="shadow-options" style="display: none;">
        <label>
            <input type="range" id="shadow-opacity" min="0" max="100" value="50">
            <span>Opacity: <span id="shadow-opacity-value">50%</span></span>
        </label>
    </div>
</div>
```

**Modify**: `ViewerAssets/js/overlays/overlayManager.js`

```javascript
import { ShadowMapLayer } from './shadowMapLayer.js';

export class OverlayManager {
    constructor(map, mapName, version) {
        // ... existing code ...
        this.shadowMapLayer = new ShadowMapLayer(map, mapName, version);
        this.initShadowControls();
    }

    initShadowControls() {
        const checkbox = document.getElementById('overlay-shadow-map');
        const options = document.getElementById('shadow-options');
        const opacitySlider = document.getElementById('shadow-opacity');
        const opacityValue = document.getElementById('shadow-opacity-value');
        
        checkbox.addEventListener('change', (e) => {
            if (e.target.checked) {
                this.shadowMapLayer.show();
                options.style.display = 'block';
            } else {
                this.shadowMapLayer.hide();
                options.style.display = 'none';
            }
        });
        
        opacitySlider.addEventListener('input', (e) => {
            const opacity = e.target.value / 100;
            this.shadowMapLayer.setOpacity(opacity);
            opacityValue.textContent = `${e.target.value}%`;
        });
    }

    async loadTile(row, col) {
        // ... existing terrain overlay loading ...
        
        // Load shadow map if enabled
        if (document.getElementById('overlay-shadow-map').checked) {
            await this.shadowMapLayer.loadTile(row, col);
        }
    }
}
```

### Testing

```powershell
# Regenerate with shadow extraction
.\rebuild-and-regenerate.ps1 `
  -Maps @("Azeroth") `
  -Versions @("0.5.3.3368") `
  -AlphaRoot ..\test_data\ `
  -RefreshCache `
  -Serve

# In browser:
# 1. Enable "Shadow Maps" checkbox
# 2. Adjust opacity slider
# 3. Verify dark shadows appear under trees, buildings
# 4. Light areas in open fields should be transparent
```

---

## Implementation Order

### This Session (if time permits)
1. ✅ AreaTable fix (high priority, user-facing)
   - Implement LkAdtAreaReader
   - Update McnkTerrainExtractor
   - Test with DeadminesInstance

### Next Session
2. ⏸️ Shadow maps (low priority, nice-to-have)
   - Implement McshDecoder
   - Create viewer layer
   - Test with Azeroth

### Future
3. ⏸️ Performance optimization
4. ⏸️ Additional overlay types
5. ⏸️ Begin Phase 1 (3D terrain)

---

## Success Criteria

### AreaTable Fix ✅
- [ ] Area boundaries show "Elwynn Forest" not "Unknown Area 12"
- [ ] Subzones show correctly (e.g., "Goldshire")
- [ ] Console log shows: `[terrain] Extracted XXX chunks with LK AreaIDs`
- [ ] No "Unknown Area" for well-known zones

### Shadow Maps ✅
- [ ] Shadow overlay checkbox appears in UI
- [ ] Dark shadows visible under buildings/trees
- [ ] Open fields are light/transparent
- [ ] Opacity slider works
- [ ] No performance degradation

---

## Estimated Time

- **AreaTable Fix**: 3-4 hours
  - LkAdtAreaReader: 1 hour
  - Integration: 1 hour
  - Testing: 1 hour
  - Debug edge cases: 1 hour

- **Shadow Maps**: 2-3 hours
  - Decoder: 30 min (trivial with reference)
  - Overlay builder: 1 hour
  - Viewer layer: 1 hour
  - UI integration: 30 min

**Total Phase 0 Completion**: 5-7 hours

---

## Next Steps

1. **Wait for current rebuild to finish** (Azeroth + Kalimdor will take a while)
2. **Verify current terrain overlays work** (green checkmarks earlier were good sign!)
3. **Implement AreaTable fix** (highest priority)
4. **Test with one small map** (DeadminesInstance)
5. **Implement shadow maps** (if time permits)
6. **Full regeneration** with all fixes
7. **Phase 0 complete!** 🎉

Then we can start Phase 1 (3D terrain) with confidence! 🚀
