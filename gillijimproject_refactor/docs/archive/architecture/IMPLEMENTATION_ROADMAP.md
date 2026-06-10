# Implementation Roadmap: Complete MCNK & AreaID Overlays

## Goal

Add comprehensive visualization of **all** MCNK terrain data and AreaID boundaries to the WoWRollback viewer.

---

## Phase 1: Design & Documentation ✅

**Status**: COMPLETE

**Deliverables**:
- [x] `overlay-system-architecture.md` - Overall system design
- [x] `mcnk-flags-overlay.md` - Original MCNK implementation design (impassible + holes)
- [x] `mcnk-complete-overlay.md` - **Complete MCNK extraction (all flags + liquids + metadata)**
- [x] `areaid-overlay.md` - **AreaID boundary visualization**
- [x] This roadmap document

---

## Phase 2: Extraction (AlphaWDTAnalysisTool)

**Goal**: Parse MCNK chunks and extract **all** terrain data to CSV

### Tasks

#### 2.1: Add Complete MCNK Parser to AlphaWdtAnalyzer.Core

**File**: `AlphaWdtAnalyzer.Core/McnkTerrainExtractor.cs` (NEW)

```csharp
// Extract ALL MCNK data: flags, liquids, holes, AreaID, positions
public sealed class McnkTerrainExtractor
{
    public static List<McnkTerrainEntry> ExtractTerrain(WdtAlphaScanner wdt) { ... }
}

public record McnkTerrainEntry(
    string Map,
    int TileRow, int TileCol,
    int ChunkRow, int ChunkCol,
    string FlagsRaw,
    bool HasMcsh, bool Impassible,
    bool LqRiver, bool LqOcean, bool LqMagma, bool LqSlime,
    bool HasMccv, bool HighResHoles,
    int AreaId, int NumLayers,
    bool HasHoles, string HoleType, string HoleBitmapHex, int HoleCount,
    float PositionX, float PositionY, float PositionZ
);
```

**Implementation Details**:
- Parse flags at offset 0x00 (32 bits)
- Extract ALL flag bits (impassible, liquids, shadows, vertex colors)
- Read AreaID at offset 0x34
- Read chunk position at offset 0x68 (3 floats)
- Parse holes (low-res at 0x3C, high-res at 0x14 if flag set)
- Read nLayers at offset 0x0C

**Dependencies**:
- Understand `AdtAlpha` class API
- Locate MCNK chunk access method
- Reference `mcnk-complete-overlay.md` for complete structure

**Verification**:
- Unit test with known ADT file
- Validate all flags match expected values
- Cross-check AreaID with existing `areaid_verify_*.csv` if available

---

#### 2.2: Add CSV Writer

**File**: `AlphaWdtAnalyzer.Cli/Program.cs` (MODIFY)

Add new command-line option:
```csharp
--extract-mcnk-terrain    // Extract complete MCNK terrain data
```

Add CSV writer:
```csharp
private static void WriteMcnkTerrainCsv(List<McnkTerrainEntry> entries, string outputPath)
{
    using var writer = new StreamWriter(outputPath);
    
    // Write header (23 columns)
    writer.WriteLine("map,tile_row,tile_col,chunk_row,chunk_col," +
                     "flags_raw,has_mcsh,impassible,lq_river,lq_ocean,lq_magma,lq_slime," +
                     "has_mccv,high_res_holes,areaid,num_layers," +
                     "has_holes,hole_type,hole_bitmap_hex,hole_count," +
                     "position_x,position_y,position_z");
    
    foreach (var entry in entries)
    {
        writer.WriteLine($"{entry.Map},{entry.TileRow},{entry.TileCol}," +
                        $"{entry.ChunkRow},{entry.ChunkCol}," +
                        $"{entry.FlagsRaw},{entry.HasMcsh},{entry.Impassible}," +
                        $"{entry.LqRiver},{entry.LqOcean},{entry.LqMagma},{entry.LqSlime}," +
                        $"{entry.HasMccv},{entry.HighResHoles},{entry.AreaId},{entry.NumLayers}," +
                        $"{entry.HasHoles},{entry.HoleType},{entry.HoleBitmapHex},{entry.HoleCount}," +
                        $"{entry.PositionX},{entry.PositionY},{entry.PositionZ}");
    }
}
```

**Output Location**:
```
<output-dir>/csv/<map>/<map>_mcnk_terrain.csv
```

**Verification**:
- Run extraction on test map (e.g., Azeroth)
- Verify CSV schema matches `mcnk-complete-overlay.md` design
- Spot-check: impassible areas, liquid flags, AreaIDs
- Confirm AreaID values match existing `areaid_verify_*.csv` data

---

#### 2.3: Update README

**File**: `AlphaWDTAnalysisTool/README.md` (MODIFY)

Add to "Features" section:
```markdown
- Complete MCNK chunk extraction (all flags, liquids, holes, AreaID, positions)
```

Add to "CLI Usage":
```markdown
--extract-mcnk-terrain     Extract complete MCNK terrain data to CSV
                           (includes: impassible, liquids, holes, AreaID, vertex colors, positions)
```

Add to "Output Layout":
```markdown
csv/
  <Map>/
    <Map>_mcnk_terrain.csv # Complete per-chunk terrain data (23 columns)
```

---

## Phase 3: Transformation (WoWRollback.Core)

**Goal**: Transform CSV data into viewer-ready overlay JSONs (multiple overlay types)

### Tasks

#### 3.1: Add Model Classes

**File**: `WoWRollback.Core/Models/McnkModels.cs` (NEW)

```csharp
public record McnkTerrainEntry(
    string Map,
    int TileRow, int TileCol,
    int ChunkRow, int ChunkCol,
    string FlagsRaw,
    bool HasMcsh, bool Impassible,
    bool LqRiver, bool LqOcean, bool LqMagma, bool LqSlime,
    bool HasMccv, bool HighResHoles,
    int AreaId, int NumLayers,
    bool HasHoles, string HoleType, string HoleBitmapHex, int HoleCount,
    float PositionX, float PositionY, float PositionZ
);

public record AreaBoundary(
    int FromArea, int ToArea,
    int ChunkRow, int ChunkCol,
    string Edge  // "north", "east", "south", "west"
);
```

---

#### 3.2: Add CSV Reader

**File**: `WoWRollback.Core/Services/McnkTerrainCsvReader.cs` (NEW)

```csharp
public sealed class McnkTerrainCsvReader
{
    public static List<McnkTerrainEntry> ReadCsv(string filePath)
    {
        var entries = new List<McnkTerrainEntry>();
        
        using var reader = new StreamReader(filePath);
        reader.ReadLine(); // Skip header
        
        string line;
        while ((line = reader.ReadLine()) != null)
        {
            var parts = line.Split(',');
            entries.Add(new McnkTerrainEntry(
                Map: parts[0],
                TileRow: int.Parse(parts[1]),
                TileCol: int.Parse(parts[2]),
                ChunkRow: int.Parse(parts[3]),
                ChunkCol: int.Parse(parts[4]),
                FlagsRaw: parts[5],
                HasMcsh: bool.Parse(parts[6]),
                Impassible: bool.Parse(parts[7]),
                LqRiver: bool.Parse(parts[8]),
                LqOcean: bool.Parse(parts[9]),
                LqMagma: bool.Parse(parts[10]),
                LqSlime: bool.Parse(parts[11]),
                HasMccv: bool.Parse(parts[12]),
                HighResHoles: bool.Parse(parts[13]),
                AreaId: int.Parse(parts[14]),
                NumLayers: int.Parse(parts[15]),
                HasHoles: bool.Parse(parts[16]),
                HoleType: parts[17],
                HoleBitmapHex: parts[18],
                HoleCount: int.Parse(parts[19]),
                PositionX: float.Parse(parts[20]),
                PositionY: float.Parse(parts[21]),
                PositionZ: float.Parse(parts[22])
            ));
        }
        
        return entries;
    }
}
```

---

#### 3.3: Add Overlay Builders

**Files** (NEW):
- `WoWRollback.Core/Services/Viewer/TerrainPropertiesOverlayBuilder.cs`
- `WoWRollback.Core/Services/Viewer/LiquidsOverlayBuilder.cs`
- `WoWRollback.Core/Services/Viewer/HolesOverlayBuilder.cs`
- `WoWRollback.Core/Services/Viewer/AreaIdOverlayBuilder.cs`

See `mcnk-complete-overlay.md` and `areaid-overlay.md` for full implementations.

**Key Responsibilities**:
1. **TerrainPropertiesOverlayBuilder**: Group chunks by impassible, has_mcsh, has_mccv
2. **LiquidsOverlayBuilder**: Group chunks by liquid type (river, ocean, magma, slime)
3. **HolesOverlayBuilder**: Decode hole bitmaps and generate hole indices
4. **AreaIdOverlayBuilder**: Detect boundaries between different AreaIDs, include area names

---

#### 3.4: Integrate into VersionComparisonService

**File**: `WoWRollback.Core/Services/VersionComparisonService.cs` (MODIFY)

Add to comparison pipeline:
```csharp
private static void BuildMcnkTerrainOverlays(
    string outputDir,
    Dictionary<string, string> versionRoots,
    IEnumerable<string> maps,
    Dictionary<int, AreaInfo> areaTable)
{
    foreach (var map in maps)
    {
        foreach (var (version, root) in versionRoots)
        {
            var csvPath = Path.Combine(root, "csv", map, $"{map}_mcnk_terrain.csv");
            if (!File.Exists(csvPath)) continue;
            
            var entries = McnkTerrainCsvReader.ReadCsv(csvPath);
            
            // Group by tile
            var byTile = entries.GroupBy(e => (e.TileRow, e.TileCol));
            
            foreach (var tileGroup in byTile)
            {
                var chunks = tileGroup.ToList();
                
                // Build all overlay types
                var terrainProps = TerrainPropertiesOverlayBuilder.Build(chunks, version);
                var liquids = LiquidsOverlayBuilder.Build(chunks, version);
                var holes = HolesOverlayBuilder.Build(chunks, version);
                var areaIds = AreaIdOverlayBuilder.Build(chunks, version, areaTable);
                
                // Combine into single JSON (or separate files per overlay type)
                var combined = new {
                    map,
                    tile = new { row = tileGroup.Key.TileRow, col = tileGroup.Key.TileCol },
                    chunk_size = 32,
                    layers = new[] {
                        new {
                            version,
                            terrain_properties = terrainProps,
                            liquids,
                            holes,
                            area_ids = areaIds
                        }
                    }
                };
                
                var outPath = Path.Combine(outputDir, "viewer", "overlays", version, map,
                    "terrain_complete", $"tile_r{tileGroup.Key.TileRow}_c{tileGroup.Key.TileCol}.json");
                
                Directory.CreateDirectory(Path.GetDirectoryName(outPath)!);
                File.WriteAllText(outPath, JsonSerializer.Serialize(combined, 
                    new JsonSerializerOptions { WriteIndented = true }));
            }
        }
    }
}
```

---

## Phase 4: Visualization (ViewerAssets)

**Goal**: Render terrain flags as interactive map overlays

### Tasks

#### 4.1: Add Terrain Flags Layer Module

**File**: `ViewerAssets/js/terrainFlagsLayer.js` (NEW)

See `mcnk-flags-overlay.md` for full implementation.

---

#### 4.2: Add UI Controls

**File**: `ViewerAssets/index.html` (MODIFY)

Add to sidebar:
```html
<div class="control-group">
    <h3>Terrain Overlays</h3>
    <label>
        <input type="checkbox" id="showTerrainFlags">
        Show Impassible & Holes
    </label>
    <div class="indent">
        <label>
            <input type="checkbox" id="showHolesOnly">
            Holes Only
        </label>
        <label>
            <input type="checkbox" id="showImpassibleOnly">
            Impassible Only
        </label>
    </div>
</div>
```

---

#### 4.3: Add CSS Styling

**File**: `ViewerAssets/styles.css` (MODIFY)

```css
/* Terrain flags controls */
.control-group .indent {
    margin-left: 20px;
    margin-top: 5px;
}

.control-group h3 {
    margin-bottom: 10px;
    color: #4CAF50;
    font-size: 14px;
}
```

---

#### 4.4: Integrate into Main Viewer

**File**: `ViewerAssets/js/main.js` (MODIFY)

```javascript
import { renderTerrainFlags } from './terrainFlagsLayer.js';

let terrainFlagsLayer = L.layerGroup();
let showTerrainFlags = false;

// Add event listeners
document.getElementById('showTerrainFlags').addEventListener('change', (e) => {
    showTerrainFlags = e.target.checked;
    updateTerrainFlagsLayer();
});

async function updateTerrainFlagsLayer() {
    terrainFlagsLayer.clearLayers();
    if (!showTerrainFlags) return;
    
    // Load and render for visible tiles
    const bounds = map.getBounds();
    const tiles = getVisibleTiles(bounds);
    
    for (const tile of tiles) {
        try {
            const path = `overlays/${state.selectedVersion}/${state.selectedMap}/terrain_flags/tile_r${tile.row}_c${tile.col}.json`;
            const data = await loadOverlay(path);
            const layer = renderTerrainFlags(state.selectedMap, tile.row, tile.col, data, state.config);
            layer.getLayers().forEach(l => terrainFlagsLayer.addLayer(l));
        } catch (e) {
            // No terrain flags for this tile
        }
    }
}

// Add to map update pipeline
map.on('moveend zoomend', () => {
    updateTerrainFlagsLayer();
});
```

---

## Phase 5: Testing & Documentation

### Tasks

#### 5.1: End-to-End Test

1. Extract MCNK flags from test map
2. Generate comparison with terrain flags overlay
3. Load viewer and verify rendering
4. Test all toggle controls
5. Verify performance (< 100ms per tile)

---

#### 5.2: Update Main README

**File**: `WoWRollback/README.md` (MODIFY)

Add to overlay variants:
```markdown
Overlay variants:
- `combined` – all placements for the selected version
- `m2` – MDX/M2 doodads only
- `wmo` – WMO placements only
- `terrain_flags` – MCNK impassible areas and holes
```

---

#### 5.3: Add Usage Examples

**File**: `docs/examples/terrain-flags-usage.md` (NEW)

```markdown
# Terrain Flags Usage Example

## Step 1: Extract Flags

dotnet run --project AlphaWDTAnalysisTool/AlphaWdtAnalyzer.Cli -- \
  --input "test_data/0.5.3/World/Maps/Azeroth/Azeroth.wdt" \
  --extract-mcnk-flags \
  --out "output/0.5.3"

## Step 2: Generate Comparison

dotnet run --project WoWRollback/WoWRollback.Cli -- \
  compare-versions \
  --versions 0.5.3 \
  --root output \
  --maps Azeroth \
  --viewer-report

## Step 3: View Results

Open: rollback_outputs/comparisons/<key>/viewer/index.html
Enable: "Show Impassible & Holes" checkbox
```

---

## Success Criteria

- [ ] MCNK flags extract correctly from Alpha ADTs
- [ ] CSV output validates against schema
- [ ] JSON overlays build successfully
- [ ] Impassible chunks render as red overlay
- [ ] Holes render as black overlay
- [ ] UI toggles work correctly
- [ ] Performance < 100ms per tile
- [ ] Documentation updated
- [ ] Examples provided

---

## Timeline Estimate

- **Phase 1**: 2 hours (Complete)
- **Phase 2**: 4 hours (Extraction)
- **Phase 3**: 3 hours (Transformation)
- **Phase 4**: 4 hours (Visualization)
- **Phase 5**: 2 hours (Testing/Docs)

**Total**: ~15 hours

---

## Next Steps

Type **ACT** to begin Phase 2 implementation.
