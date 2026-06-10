# WoWDataPlot - Layer-Based UniqueID Filtering Workflow

## Overview

This workflow allows you to visualize UniqueID distributions as "layers" and selectively filter content based on development timeline. Perfect for **digital archaeology** and **selective content rollback**.

## Concept

Each **layer** represents a range of UniqueIDs that were added during a specific development period. By visualizing these layers:

1. **Identify content evolution** - See which areas were built when
2. **Selective removal** - Remove only specific UniqueID ranges (e.g., remove all content added in Alpha 0.7.0+)
3. **Per-tile control** - Different tiles may have different layer compositions

## Workflow

### Step 1: Analyze Layers

Extract all placements and group them into UniqueID-based layers:

```bash
dotnet run --project WoWDataPlot -- analyze-layers \
  --wdt path/to/Kalidar.wdt \
  --output kalidar_layers.json \
  --layer-size 1000
```

**Output:** `kalidar_layers.json` containing:
- Global UniqueID range (min/max)
- Layer definitions (UniqueID ranges)
- Per-tile layer breakdown
- Placement counts per layer

**Example JSON:**
```json
{
  "wdtName": "Kalidar",
  "totalPlacements": 125430,
  "minUniqueId": 0,
  "maxUniqueId": 5432,
  "globalLayers": [
    {
      "name": "Layer 0-999",
      "minUniqueId": 0,
      "maxUniqueId": 999,
      "placementCount": 45231,
      "color": "#0000FF"
    },
    {
      "name": "Layer 1000-1999",
      "minUniqueId": 1000,
      "maxUniqueId": 1999,
      "placementCount": 32104,
      "color": "#00FF00"
    }
  ],
  "tiles": [
    {
      "tileX": 32,
      "tileY": 28,
      "totalPlacements": 1543,
      "layers": [
        {
          "name": "Layer 0-999",
          "minUniqueId": 0,
          "maxUniqueId": 999,
          "placementCount": 987,
          "color": "#0000FF"
        },
        {
          "name": "Layer 1000-1999",
          "minUniqueId": 1000,
          "maxUniqueId": 1999,
          "placementCount": 556,
          "color": "#00FF00"
        }
      ],
      "imagePath": "tile_28_32_layers.png"
    }
  ]
}
```

### Step 2: Generate Per-Tile Layer Images

Create visual representations of each tile's layers:

```bash
dotnet run --project WoWDataPlot -- generate-tile-layers \
  --wdt path/to/Kalidar.wdt \
  --layers kalidar_layers.json \
  --output-dir kalidar_tile_layers \
  --size 512
```

**Output:** One PNG per tile showing:
- Different colored points for each layer
- Legend showing layer names and counts
- Spatial distribution of UniqueID ranges

**File naming:** `tile_YY_XX_layers.png` (e.g., `tile_28_32_layers.png`)

### Step 3: Review Layers (Manual)

1. Open tile layer images in a viewer
2. Identify which layers contain unwanted content
3. Note the UniqueID ranges to exclude

**Example analysis:**
```
Tile (28, 32):
- Layer 0-999 (blue): Original terrain objects ✅ KEEP
- Layer 1000-1999 (green): Alpha 0.6.0 additions ✅ KEEP
- Layer 2000-2999 (red): Alpha 0.7.0 additions ❌ REMOVE
- Layer 3000+ (yellow): Experimental content ❌ REMOVE
```

### Step 4: Apply Filters (Integration with WoWRollback)

Use the layer analysis JSON to filter content in your rollback tool:

**Option A: Filter by UniqueID range**
```csharp
// Remove all placements with UniqueID >= 2000
var keepLayers = analysis.GlobalLayers
    .Where(l => l.MaxUniqueId < 2000)
    .ToList();
```

**Option B: Per-tile filtering**
```csharp
// Remove specific layers from specific tiles
foreach (var tile in analysis.Tiles)
{
    if (tile.TileX == 32 && tile.TileY == 28)
    {
        // Keep only layers 0-1999 for this tile
        var keepLayers = tile.Layers
            .Where(l => l.MaxUniqueId < 2000)
            .ToList();
    }
}
```

**Option C: Interactive web UI** (future enhancement)
```
1. Serve static tile images
2. User clicks layers to toggle on/off
3. Generate filter list from selections
4. Apply to rollback operation
```

## Use Cases

### Use Case 1: Remove All Content After Alpha 0.6.0

**Goal:** Rollback to Alpha 0.6.0 by removing all UniqueIDs added after that version

**Steps:**
1. Analyze layers with `--layer-size 500`
2. Identify which layers correspond to Alpha 0.7.0+ (e.g., UniqueID >= 3000)
3. Filter placements: `WHERE UniqueID < 3000`
4. Regenerate ADT files without excluded placements

### Use Case 2: Remove Experimental Content

**Goal:** Remove singleton/outlier objects that were experimental tests

**Steps:**
1. Analyze layers to find sparse regions
2. Review tile images for isolated placements
3. Note specific UniqueID ranges of experimental objects
4. Apply surgical exclusions

### Use Case 3: Clean Up Specific Zones

**Goal:** Remove newer content from one zone, keep it in others

**Steps:**
1. Generate tile layer images
2. Identify tiles belonging to target zone
3. Apply per-tile UniqueID filtering
4. Leave other zones unchanged

## Advanced: Automated Layer Detection

**Future enhancement:** Auto-detect logical layers based on:
- **Temporal clustering:** Large gaps in UniqueID sequences
- **Spatial clustering:** Placements grouped by area
- **Density analysis:** Sudden increases in placement density

**Example:**
```bash
# Auto-detect natural layer boundaries
dotnet run --project WoWDataPlot -- auto-detect-layers \
  --wdt path/to/Kalidar.wdt \
  --output kalidar_auto_layers.json \
  --min-gap 100
```

## Integration Points

### With WoWRollback.Cli

```bash
# Use layer JSON to filter rollback operation
dotnet run --project WoWRollback.Cli -- rollback \
  --input Kalidar.wdt \
  --output Kalidar_053.wdt \
  --filter-layers kalidar_layers.json \
  --exclude-layers "2000-2999,3000+"
```

### With Web Viewer

```bash
# Serve tile layer images for interactive selection
python -m http.server 8000 -d kalidar_tile_layers
# User selects layers via web UI
# Export selection as JSON filter
# Apply filter to rollback operation
```

## File Structure

```
project/
├── kalidar_layers.json              # Layer analysis metadata
├── kalidar_tile_layers/             # Per-tile visualizations
│   ├── tile_00_00_layers.png
│   ├── tile_00_01_layers.png
│   ├── ...
│   └── tile_63_63_layers.png
├── kalidar_filter.json              # User-selected layer filter
└── kalidar_rollback_053.wdt        # Output with filtered content
```

## Benefits

✅ **Visual confirmation** - See exactly what gets removed  
✅ **Granular control** - Per-tile, per-layer filtering  
✅ **Reversible** - Keep original WDT, generate filtered copies  
✅ **Auditable** - JSON files document what was removed and why  
✅ **Flexible** - Combine with other filtering criteria (AreaID, object type, etc.)

## Next Steps

1. ✅ **Basic visualization** - Fixed Y-axis, removed lines
2. ✅ **Layer analysis** - UniqueID range grouping
3. ✅ **Tile-by-tile images** - Per-tile layer breakdown
4. ⏳ **Interactive web UI** - Click layers to toggle
5. ⏳ **Auto-detection** - Identify natural layer boundaries
6. ⏳ **Filter application** - Integrate with WoWRollback.Cli
7. ⏳ **Batch processing** - Process entire map collections

---

**You're now equipped to perform surgical, layer-based content rollback with full visual confirmation!**
