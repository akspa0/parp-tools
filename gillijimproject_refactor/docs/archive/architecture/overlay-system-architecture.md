# Overlay System Architecture

## Purpose

This document defines the architecture for extracting, transforming, and visualizing ADT/WDT data as interactive overlays in the WoWRollback viewer.

## Overview

The overlay system is a **three-stage pipeline**:

```
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   ADT/WDT    │      │     CSV      │      │    Viewer    │
│  Raw Data    │ ───> │  Normalized  │ ───> │   Overlay    │
│  (Binary)    │      │    Data      │      │    JSON      │
└──────────────┘      └──────────────┘      └──────────────┘
      │                     │                      │
   Extract              Transform              Visualize
 (Alpha Tool)          (WoWRollback)           (Viewer)
```

### Stage 1: Extraction (AlphaWDTAnalysisTool)

**Purpose**: Read binary ADT/WDT files and extract structured data into CSV format.

**Input**: 
- Alpha WDT/ADT files (binary)
- Reference documentation (`reference_data/wowdev.wiki/ADT_v18.md`)

**Output**:
- CSV files with normalized, human-readable data
- One CSV type per data category (placements, flags, liquids, etc.)

**Key Responsibilities**:
- Binary chunk parsing (MCNK, MDDF, MODF, MH2O, etc.)
- Coordinate conversion (Alpha → WoW world coordinates)
- Flag bit extraction and labeling
- Data validation and error reporting

### Stage 2: Transformation (WoWRollback.Core)

**Purpose**: Aggregate CSV data across versions and compute overlay-ready datasets.

**Input**:
- CSV files from multiple versions
- Version comparison metadata

**Output**:
- Per-tile overlay JSON files
- Index metadata (available tiles, versions, maps)
- Diff data (version-to-version changes)

**Key Responsibilities**:
- Multi-version aggregation
- Coordinate transformation (world → tile-relative → pixel)
- Data filtering and categorization
- JSON schema generation

### Stage 3: Visualization (ViewerAssets)

**Purpose**: Render overlay data as interactive map layers using Leaflet.

**Input**:
- Overlay JSON files
- Minimap tile images (PNG)
- Index and config JSON

**Output**:
- Interactive web viewer with:
  - Selectable overlay layers
  - Version comparison
  - Asset inspection popups
  - Tile-by-tile navigation

**Key Responsibilities**:
- Dynamic overlay loading/unloading
- Coordinate mapping (JSON → Leaflet lat/lng)
- User interaction (clicks, popups, filtering)
- Performance optimization (caching, debouncing)

---

## Data Flow: Placement Overlays (Current Implementation)

### 1. Extraction: AlphaWdtAnalyzer.cs

Reads MDDF/MODF chunks from ADT files:

```csharp
// AdtScanner.cs - Parse MDDF (M2 placements)
for (int start = 0; start + 36 <= mddf.Length; start += 36)
{
    int nameIndex = BitConverter.ToInt32(mddf, start + 0);
    uint uniqueId = BitConverter.ToUInt32(mddf, start + 4);
    float worldX = BitConverter.ToSingle(mddf, start + 8);   // Alpha corner-relative
    float worldZ = BitConverter.ToSingle(mddf, start + 12);
    float worldY = BitConverter.ToSingle(mddf, start + 16);
    // ... rotations, scale, flags
}
```

Converts coordinates:

```csharp
// AlphaWdtAnalyzer.cs - Convert to WoW world coords
const double MAP_HALF_SIZE = 17066.66656;
float worldX = (float)(MAP_HALF_SIZE - placement.WorldX);
float worldY = (float)(MAP_HALF_SIZE - placement.WorldY);
```

Outputs CSV:

```csv
Version,Map,TileRow,TileCol,Kind,UniqueId,AssetPath,WorldX,WorldY,WorldZ,...
0.5.3,Azeroth,31,34,M2,192375,World\Doodad\Barn.m2,-1470.834,206.87,150.2,...
```

### 2. Transformation: VersionComparisonService.cs

Aggregates placements across versions:

```csharp
// Build per-version asset timelines
var assetTimelineDetailed = BuildAssetTimelineDetailed(assetsByVersion, versionOrder);

// Transform to tile-relative coordinates
var (localX, localY) = CoordinateTransformer.ComputeLocalCoordinates(
    entry.WorldX, entry.WorldY, tileRow, tileCol);
var (pixelX, pixelY) = CoordinateTransformer.ToPixels(
    localX, localY, minimapWidth, minimapHeight);
```

Generates overlay JSON:

```json
{
  "map": "Azeroth",
  "tile": {"row": 31, "col": 34},
  "minimap": {"width": 512, "height": 512},
  "layers": [
    {
      "version": "0.5.3",
      "kinds": [
        {
          "kind": "M2",
          "points": [
            {
              "uniqueId": 192375,
              "fileName": "Barn.m2",
              "world": {"x": -1470.834, "y": 206.87, "z": 150.2},
              "pixel": {"x": 256.4, "y": 128.7}
            }
          ]
        }
      ]
    }
  ]
}
```

### 3. Visualization: main.js

Loads and renders overlays:

```javascript
// Load overlay JSON
const data = await loadOverlay(overlayPath);

// Render markers on map
const circle = L.circleMarker([lat, lng], {
    radius: getScaledRadius(4),
    fillColor: '#2196F3',
    fillOpacity: 0.9
}).bindPopup(popupHtml);
objectMarkers.addLayer(circle);
```

---

## Design Principles

### 1. **Separation of Concerns**
- **Extraction** only reads binary formats and normalizes to CSV
- **Transformation** only aggregates and converts coordinates
- **Visualization** only renders JSON to interactive UI

### 2. **CSV as Interchange Format**
- Human-readable for debugging
- Versionable in Git
- Tool-agnostic (can be consumed by other tools)
- Schema can evolve independently

### 3. **Tile-Based Architecture**
- All data is tile-relative (row/col from 0-63)
- Enables lazy loading (only load visible tiles)
- Reduces memory footprint
- Parallelize processing per-tile

### 4. **Version-First Design**
- All data tagged with version identifier
- Enables temporal analysis and comparison
- Supports multi-version overlays

### 5. **Coordinate System Consistency**
```
WoW World Coordinates:
  - Origin: Map center (0,0)
  - X-axis: North (+) to South (-)
  - Y-axis: West (+) to East (-)
  - Range: ±17066.66656 yards
  - Tile size: 533.33333 yards (1600 feet)

Tile Indices:
  - Range: 0-63 (64×64 grid)
  - Tile [0,0] = NW corner (+17066, +17066)
  - Tile [63,63] = SE corner (-17066, -17066)

Tile-Relative Coordinates:
  - Range: 0.0-1.0 (normalized within tile)
  - (0,0) = NW corner of tile
  - (1,1) = SE corner of tile

Pixel Coordinates:
  - Range: 0 to (width-1), 0 to (height-1)
  - Maps tile-relative to actual pixel position
```

---

## Adding New Overlay Types

Follow this checklist when adding new overlay data (e.g., MCNK flags, liquids):

### Phase 1: Design
1. ☐ Document the ADT chunk structure (from `ADT_v18.md`)
2. ☐ Define the CSV schema (columns, data types, semantics)
3. ☐ Design the overlay JSON format
4. ☐ Create a design document in `docs/architecture/<overlay-name>.md`

### Phase 2: Extraction (AlphaWDTAnalysisTool)
1. ☐ Add binary parsing code in `AlphaWdtAnalyzer.Core`
2. ☐ Add CSV output in `AlphaWdtAnalyzer.Cli`
3. ☐ Add validation and error handling
4. ☐ Update README with new CSV output

### Phase 3: Transformation (WoWRollback)
1. ☐ Add CSV reader in `WoWRollback.Core/Models`
2. ☐ Add overlay builder in `WoWRollback.Core/Services/Viewer`
3. ☐ Add JSON schema to overlay output
4. ☐ Update `VersionComparisonService` to include new data

### Phase 4: Visualization (ViewerAssets)
1. ☐ Add overlay loader in `overlayLoader.js` (if needed)
2. ☐ Add rendering logic in `main.js` or new module
3. ☐ Add UI controls (layer toggles, filters)
4. ☐ Add CSS styling for new overlay elements
5. ☐ Test performance (lazy loading, caching)

### Phase 5: Documentation
1. ☐ Update main README with new overlay type
2. ☐ Add usage examples
3. ☐ Document coordinate transformations
4. ☐ Add troubleshooting notes

---

## Performance Guidelines

### Extraction
- **Batch processing**: Process entire map at once, not tile-by-tile
- **Memory efficiency**: Stream large chunks, don't load entire file
- **Parallelization**: Process tiles independently where possible

### Transformation
- **Incremental updates**: Only reprocess changed versions
- **Caching**: Cache expensive computations (coordinate transforms)
- **Filtering**: Filter early (at CSV read), not late (at JSON write)

### Visualization
- **Lazy loading**: Only load overlays for visible tiles
- **Debouncing**: Wait 500ms before loading on pan/zoom
- **Aggressive unloading**: Unload tiles >2 tiles from viewport
- **Smart caching**: Cache until version/map change, not per-request

---

## File Naming Conventions

### CSV Files (Extraction Output)
```
<map>_<category>_<tile_row>_<tile_col>.csv
Example: Azeroth_mcnk_flags_31_34.csv
```

### Overlay JSON (Transformation Output)
```
tile_r<row>_c<col>.json
Example: tile_r31_c34.json
```

### Viewer Paths
```
overlays/<version>/<map>/<variant>/tile_r<row>_c<col>.json
Example: overlays/0.5.3/Azeroth/combined/tile_r31_c34.json
```

---

## Next Steps

See specific design documents for implementation details:
- `mcnk-flags-overlay.md` - MCNK chunk flags (impassible, holes, etc.)
- `liquid-overlay.md` - MH2O liquid data visualization
- `area-overlay.md` - AreaID boundary visualization
