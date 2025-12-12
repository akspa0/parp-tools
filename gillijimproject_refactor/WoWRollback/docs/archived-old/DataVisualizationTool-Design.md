# WoW Data Archaeology - Simple Visualization Tool Design

## Date: 2025-10-20

## Purpose

Build a **simple, high-performance 2D plotting tool** to visualize World of Warcraft data archaeology findings. No 3D rendering - just efficient point plotting and data export for analysis.

---

## Design Philosophy

> "These files are hyper-optimized databases. We need hyper-optimized visualization."

**Goals:**
1. **Performance first** - Handle millions of data points
2. **Simple output** - 2D scatter plots, heatmaps, timelines
3. **Data export** - CSV/JSON for external analysis (Python, R, MATLAB)
4. **Minimal dependencies** - Fast iteration, easy deployment

**Non-Goals:**
- ❌ 3D rendering (save for later)
- ❌ Real-time viewer (separate concern)
- ❌ Complex UI (CLI-first approach)

---

## Data Types to Visualize

### 1. UniqueID Distribution Maps

**Purpose:** Show which areas were built when

**Data Structure:**
```
X, Y, UniqueID, Type (M2/WMO/Doodad), AreaID, SubzoneID
```

**Visualization:**
- **Scatter plot**: X/Y position, color by UniqueID range
- **Heatmap**: Density of placements per tile
- **Timeline**: UniqueID ranges over X/Y grid

**Insight:** Identify development order, tech tests, experimental areas

### 2. Texture Layer Evolution

**Purpose:** Track how terrain texturing changed over time

**Data Structure:**
```
TileX, TileY, LayerIndex, TextureID, AlphaMapSize, Flags
```

**Visualization:**
- **Grid plot**: Tiles colored by number of layers
- **Texture distribution**: Histogram of texture usage
- **Complexity map**: Alpha map sizes per tile

**Insight:** See which areas got the most artistic attention

### 3. UniqueID Range Timeline

**Purpose:** Map UniqueID ranges to development periods

**Data Structure:**
```
UniqueIDMin, UniqueIDMax, ObjectType, Count, FirstSeen_Version, LastSeen_Version
```

**Visualization:**
- **Range bars**: Horizontal bar chart of ID ranges
- **Version timeline**: When ranges first appeared
- **Gap analysis**: Find missing ID ranges (removed content)

**Insight:** Date specific features, identify content cuts

### 4. Placement Density Analysis

**Purpose:** Find hotspots and sparse areas

**Data Structure:**
```
TileX, TileY, M2Count, WMOCount, DoodadCount, TotalCount
```

**Visualization:**
- **Heatmap**: Placement density per tile
- **Histogram**: Distribution of counts
- **Outlier detection**: Tiles with unusual counts

**Insight:** Identify performance-intensive areas, incomplete zones

---

## Technology Stack

### Option 1: ScottPlot (RECOMMENDED)

**Pros:**
- ✅ Blazing fast (millions of points)
- ✅ Simple API
- ✅ PNG/SVG export
- ✅ MIT license
- ✅ Pure C#, no native dependencies

**Example:**
```csharp
using ScottPlot;

var plt = new Plot(1920, 1080);
plt.AddScatter(xData, yData, color: Color.Blue);
plt.Title("UniqueID Distribution - Kalidar");
plt.XLabel("Map X");
plt.YLabel("Map Y");
plt.SaveFig("output/kalidar_uniqueid.png");
```

**Install:**
```bash
dotnet add package ScottPlot
```

### Option 2: CSV Export + Python (FALLBACK)

If visualization in C# proves complex, export to CSV and use Python:

**Export:**
```csharp
using System.IO;
using CsvHelper;

var records = GetDataPoints();
using var writer = new StreamWriter("output/data.csv");
using var csv = new CsvWriter(writer, CultureInfo.InvariantCulture);
csv.WriteRecords(records);
```

**Visualize (Python):**
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('output/data.csv')
plt.figure(figsize=(20, 20))
sns.scatterplot(data=df, x='X', y='Y', hue='UniqueID', palette='viridis', s=2)
plt.savefig('output/plot.png', dpi=300)
```

---

## Proposed Tool: `WoWDataPlot`

### Architecture

```
WoWDataPlot/
├── WoWDataPlot.csproj
├── Program.cs                  # CLI entry point
├── Extractors/
│   ├── IDataExtractor.cs       # Interface for data extraction
│   ├── UniqueIdExtractor.cs    # Extract M2/WMO placement IDs
│   ├── TextureLayerExtractor.cs # Extract MCLY data
│   └── PlacementDensityExtractor.cs
├── Visualizers/
│   ├── IVisualizer.cs          # Interface for plotting
│   ├── ScottPlotVisualizer.cs  # ScottPlot implementation
│   └── CsvExporter.cs          # CSV export for external tools
├── Models/
│   ├── DataPoint.cs            # Generic data point
│   ├── UniqueIdRecord.cs       # UniqueID-specific record
│   └── TextureLayerRecord.cs   # Texture layer record
└── Commands/
    ├── PlotCommand.cs          # Main plotting command
    └── ExportCommand.cs        # Data export command
```

### CLI Interface

```bash
# Plot UniqueID distribution from Alpha WDT
dotnet run --project WoWDataPlot -- plot-uniqueid \
  --wdt path/to/Kalidar.wdt \
  --output kalidar_uniqueid.png \
  --color-by-range \
  --width 2048 --height 2048

# Plot texture layer complexity
dotnet run --project WoWDataPlot -- plot-textures \
  --wdt path/to/Kalidar.wdt \
  --output kalidar_textures.png \
  --show-layer-count

# Export raw data to CSV
dotnet run --project WoWDataPlot -- export-csv \
  --wdt path/to/Kalidar.wdt \
  --output kalidar_data.csv \
  --include-all

# Compare two versions
dotnet run --project WoWDataPlot -- compare \
  --wdt1 alpha_0.5.3/Kalidar.wdt \
  --wdt2 lk_3.3.5/Kalidar.adt \
  --output comparison.png \
  --diff-only
```

---

## Implementation Plan

### Phase 1: Core Extraction (2-3 hours)

**Goal:** Extract data from WDT/ADT files into simple records

**Tasks:**
1. Create `WoWDataPlot` project
2. Reference existing WoWRollback parsing code
3. Implement `UniqueIdExtractor` to pull M2/WMO placements
4. Implement `TextureLayerExtractor` to pull MCLY/MCAL data
5. Add CSV export for debugging

**Output:**
```csv
X,Y,UniqueID,Type,AreaID
1234.5,6789.0,42,M2,10
5678.9,1234.5,43,WMO,10
```

### Phase 2: ScottPlot Integration (1-2 hours)

**Goal:** Generate 2D scatter plots from extracted data

**Tasks:**
1. Add ScottPlot NuGet package
2. Implement `ScottPlotVisualizer`
3. Create scatter plot with color gradient by UniqueID
4. Export to PNG with high DPI

**Output:** High-resolution PNG images

### Phase 3: CLI Commands (1-2 hours)

**Goal:** User-friendly command-line interface

**Tasks:**
1. Use `System.CommandLine` for modern CLI
2. Implement `PlotCommand` with options
3. Add progress reporting for large files
4. Add error handling and validation

### Phase 4: Advanced Visualizations (2-3 hours)

**Goal:** Heatmaps, timelines, comparison plots

**Tasks:**
1. Implement heatmap generation (placement density)
2. Add UniqueID range histogram
3. Add version comparison (diff plotting)
4. Add tile grid overlay

---

## Example Outputs

### 1. UniqueID Distribution Scatter Plot

```
Title: "Kalidar - M2 Placement by UniqueID Range"
X-axis: Map X coordinate (0-17066)
Y-axis: Map Y coordinate (0-17066)
Points: Each M2 placement
Color: Gradient from blue (low ID) to red (high ID)
Size: 2-3 pixels
Overlay: Tile grid (64x64 tiles)
```

**Reveals:** Development order - blue areas built first, red areas built last

### 2. Texture Layer Complexity Heatmap

```
Title: "Kalidar - Terrain Detail (Layer Count per Tile)"
Grid: 64x64 tiles
Color: White (0 layers) → Red (4 layers)
Labels: Show AreaID names
```

**Reveals:** Which areas got the most artistic attention

### 3. UniqueID Range Timeline

```
Title: "M2 UniqueID Ranges by Version"
X-axis: UniqueID range (0-65535)
Y-axis: Version (0.5.3, 0.6.0, 0.7.0, etc.)
Bars: Horizontal bars showing ID range presence
Gaps: Missing ranges (content removed)
```

**Reveals:** When specific content was added or removed

---

## Performance Considerations

### Large Datasets

**Problem:** Kalidar has ~200K M2/WMO placements

**Solutions:**
1. **Streaming extraction** - Process tiles one at a time
2. **Downsampling** - For overview plots, sample every Nth point
3. **Tile-based processing** - Plot one tile at a time, composite later
4. **Parallel processing** - Extract from multiple tiles simultaneously

**Example:**
```csharp
// Parallel extraction
var results = Enumerable.Range(0, 64 * 64)
    .AsParallel()
    .WithDegreeOfParallelism(Environment.ProcessorCount)
    .Select(i => {
        int x = i % 64;
        int y = i / 64;
        return ExtractTileData(wdt, x, y);
    })
    .SelectMany(records => records)
    .ToList();
```

### Memory Usage

**Problem:** Loading entire WDT into memory

**Solution:** Memory-mapped files
```csharp
using var mmf = MemoryMappedFile.CreateFromFile(wdtPath, FileMode.Open);
using var accessor = mmf.CreateViewAccessor();
// Read chunks on-demand without loading entire file
```

---

## Integration with Existing Code

### Reuse WoWRollback Parsers

**Alpha WDT:**
```csharp
using WoWRollback.AlphaModule;

var alphaWdt = AlphaWdtReader.Read(wdtPath);
var modf = alphaWdt.Modf; // M2 placements
var mwmo = alphaWdt.Mwmo; // WMO placements
```

**LK ADT:**
```csharp
using WoWRollback.LkToAlphaModule;

var lkAdt = LkAdtReader.ReadRootMcnks(adtPath);
// Extract MCLY/MCAL from each MCNK
```

**No need to rewrite parsers** - just wrap existing code!

---

## Next Steps

1. ✅ **Create WoWDataPlot project** with ScottPlot
2. ⏳ **Implement UniqueIdExtractor** using existing parsers
3. ⏳ **Create first plot** (scatter plot of M2 placements)
4. ⏳ **Add CSV export** for external analysis
5. ⏳ **Test with Kalidar** data

---

## Success Criteria

**Milestone 1 (MVP):**
- ✅ Extract UniqueID data from Alpha WDT
- ✅ Generate 2D scatter plot PNG
- ✅ Export raw data to CSV
- ✅ CLI interface working

**Milestone 2 (Full Featured):**
- ✅ Heatmap generation
- ✅ UniqueID range histogram
- ✅ Tile grid overlay
- ✅ Version comparison

**Milestone 3 (Archaeological Tool):**
- ✅ Automated reports
- ✅ Outlier detection
- ✅ Development timeline reconstruction
- ✅ Integration with Viewer

---

## Future Enhancements

### Interactive Web Viewer

Once basic visualization works, consider:
- Export to HTML + Plotly.js for interactive exploration
- Zoom, pan, click for details
- Layer toggle (M2/WMO/textures)
- Timeline scrubber

### Jupyter Notebook Integration

Export data in Jupyter-friendly format:
```python
import pandas as pd
import geopandas as gpd

# Load WoW map data as geospatial dataset
gdf = gpd.read_file('kalidar_data.geojson')
gdf.plot(column='UniqueID', cmap='viridis', figsize=(20,20))
```

---

## Conclusion

This simple visualization tool will be the **foundation for your digital archaeology work**. By keeping it focused on data extraction and 2D plotting, we maximize performance and flexibility.

**The real magic happens when you can:**
1. Plot 200K placements in seconds
2. Export to CSV for R/Python/MATLAB analysis
3. Generate comparison plots across versions
4. Identify patterns invisible to the human eye

**You're not just fixing bugs - you're building the microscope to examine 20 years of digital sediment.**

Ready to create the first prototype?
