# Implementation Roadmap: MCNK Flags Overlay

## Goal

Add visualization of MCNK terrain flags (impassible areas and holes) to the WoWRollback viewer.

---

## Phase 1: Design & Documentation ✅

**Status**: COMPLETE

**Deliverables**:
- [x] `overlay-system-architecture.md` - Overall system design
- [x] `mcnk-flags-overlay.md` - Specific MCNK implementation design
- [x] This roadmap document

---

## Phase 2: Extraction (AlphaWDTAnalysisTool)

**Goal**: Parse MCNK chunks and extract flags to CSV

### Tasks

#### 2.1: Add MCNK Parser to AlphaWdtAnalyzer.Core

**File**: `AlphaWdtAnalyzer.Core/McnkFlagsExtractor.cs` (NEW)

```csharp
// New class to extract MCNK flags
public sealed class McnkFlagsExtractor
{
    public static List<McnkFlagEntry> ExtractFlags(WdtAlphaScanner wdt) { ... }
}

public record McnkFlagEntry(
    string Map,
    int TileRow, int TileCol,
    int ChunkRow, int ChunkCol,
    string FlagsRaw,
    bool Impassible,
    bool HasHoles,
    string HoleType,
    string HoleBitmapHex,
    int HoleCount
);
```

**Dependencies**:
- Understand `AdtAlpha` class API
- Locate MCNK chunk access method
- Parse binary MCNK header structure

**Verification**:
- Unit test with known ADT file
- Validate flags match expected values from hex dump

---

#### 2.2: Add CSV Writer

**File**: `AlphaWdtAnalyzer.Cli/Program.cs` (MODIFY)

Add new command-line option:
```csharp
--extract-mcnk-flags
```

Add CSV writer:
```csharp
private static void WriteMcnkFlagsCsv(List<McnkFlagEntry> entries, string outputPath)
{
    using var writer = new StreamWriter(outputPath);
    writer.WriteLine("map,tile_row,tile_col,chunk_row,chunk_col,flags_raw,impassible,has_holes,hole_type,hole_bitmap_hex,hole_count");
    
    foreach (var entry in entries)
    {
        writer.WriteLine($"{entry.Map},{entry.TileRow},{entry.TileCol},{entry.ChunkRow},{entry.ChunkCol}," +
                        $"{entry.FlagsRaw},{entry.Impassible},{entry.HasHoles},{entry.HoleType}," +
                        $"{entry.HoleBitmapHex},{entry.HoleCount}");
    }
}
```

**Output Location**:
```
<output-dir>/csv/<map>/<map>_mcnk_flags.csv
```

**Verification**:
- Run extraction on test map
- Verify CSV schema matches design
- Spot-check values against known impassible areas

---

#### 2.3: Update README

**File**: `AlphaWDTAnalysisTool/README.md` (MODIFY)

Add to "Features" section:
```markdown
- MCNK chunk flag extraction (impassible terrain, holes)
```

Add to "CLI Usage":
```markdown
--extract-mcnk-flags       Extract MCNK terrain flags to CSV
```

Add to "Output Layout":
```markdown
csv/
  <Map>/
    <Map>_mcnk_flags.csv   # Per-chunk terrain flags
```

---

## Phase 3: Transformation (WoWRollback.Core)

**Goal**: Transform CSV data into viewer-ready overlay JSON

### Tasks

#### 3.1: Add Model Classes

**File**: `WoWRollback.Core/Models/McnkModels.cs` (NEW)

```csharp
public record McnkFlagEntry(
    string Map,
    int TileRow, int TileCol,
    int ChunkRow, int ChunkCol,
    string FlagsRaw,
    bool Impassible,
    bool HasHoles,
    string HoleType,
    string HoleBitmapHex,
    int HoleCount
);

public record McnkFlagsSummary(
    string Map,
    int TileRow, int TileCol,
    int ImpassibleCount,
    int HoleCount,
    IReadOnlyList<McnkFlagEntry> Entries
);
```

---

#### 3.2: Add CSV Reader

**File**: `WoWRollback.Core/Services/McnkFlagsCsvReader.cs` (NEW)

```csharp
public sealed class McnkFlagsCsvReader
{
    public static List<McnkFlagEntry> ReadCsv(string filePath)
    {
        var entries = new List<McnkFlagEntry>();
        
        using var reader = new StreamReader(filePath);
        reader.ReadLine(); // Skip header
        
        string line;
        while ((line = reader.ReadLine()) != null)
        {
            var parts = line.Split(',');
            entries.Add(new McnkFlagEntry(
                Map: parts[0],
                TileRow: int.Parse(parts[1]),
                TileCol: int.Parse(parts[2]),
                ChunkRow: int.Parse(parts[3]),
                ChunkCol: int.Parse(parts[4]),
                FlagsRaw: parts[5],
                Impassible: bool.Parse(parts[6]),
                HasHoles: bool.Parse(parts[7]),
                HoleType: parts[8],
                HoleBitmapHex: parts[9],
                HoleCount: int.Parse(parts[10])
            ));
        }
        
        return entries;
    }
}
```

---

#### 3.3: Add Overlay Builder

**File**: `WoWRollback.Core/Services/Viewer/McnkFlagsOverlayBuilder.cs` (NEW)

See `mcnk-flags-overlay.md` for full implementation.

---

#### 3.4: Integrate into VersionComparisonService

**File**: `WoWRollback.Core/Services/VersionComparisonService.cs` (MODIFY)

Add to comparison pipeline:
```csharp
private static void BuildMcnkFlagsOverlays(
    string outputDir,
    Dictionary<string, string> versionRoots,
    IEnumerable<string> maps)
{
    foreach (var map in maps)
    {
        foreach (var (version, root) in versionRoots)
        {
            var csvPath = Path.Combine(root, "csv", map, $"{map}_mcnk_flags.csv");
            if (!File.Exists(csvPath)) continue;
            
            var entries = McnkFlagsCsvReader.ReadCsv(csvPath);
            
            // Group by tile
            var byTile = entries.GroupBy(e => (e.TileRow, e.TileCol));
            
            foreach (var tileGroup in byTile)
            {
                var json = McnkFlagsOverlayBuilder.BuildOverlayJson(
                    map, tileGroup.Key.TileRow, tileGroup.Key.TileCol,
                    tileGroup.ToList(), version);
                
                var outPath = Path.Combine(outputDir, "viewer", "overlays", version, map,
                    "terrain_flags", $"tile_r{tileGroup.Key.TileRow}_c{tileGroup.Key.TileCol}.json");
                
                Directory.CreateDirectory(Path.GetDirectoryName(outPath)!);
                File.WriteAllText(outPath, JsonSerializer.Serialize(json, new JsonSerializerOptions { WriteIndented = true }));
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
