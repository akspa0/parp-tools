# WoWDataPlot - Quick Start Guide

## 1-Minute Setup

### 1. Build the Tool

```bash
cd WoWRollback
dotnet build WoWDataPlot/WoWDataPlot.csproj
```

### 2. Run Complete Visualization (ONE COMMAND!)

```bash
# All-in-one pipeline: extract → analyze → visualize
dotnet run --project WoWDataPlot -- visualize \
  --wdt ../test_data/0.5.3/tree/World/Maps/Azeroth/Azeroth.wdt \
  --output-dir Azeroth_analysis \
  --gap-threshold 50
```

**What this does:**
1. Extracts all M2/WMO placements from Azeroth.wdt
2. **Detects UniqueID layers per tile by finding GAPS** (gaps = dev pauses!)
3. Generates per-tile images with layer colors
4. Creates map-wide overview image
5. Saves analysis JSON

**How layers work now:**
- Layers = **continuous clusters** of UniqueIDs
- **Gaps > 50** split into new layer
- Each tile gets **its own unique layer structure**
- Gaps tell the story of development pauses!

**Expected output:**
```
═══ STEP 1: Extracting Placement Data ═══
  [INFO] Reading Alpha WDT: Azeroth.wdt
  [INFO] Found 365 existing ADT tiles to process
✓ Extracted 124,532 placements

═══ STEP 2: Analyzing Layers Per Tile ═══
  Processing 365 tiles...
✓ Detected 3046 unique UniqueID ranges across all tiles

═══ STEP 3: Generating Per-Tile Images ═══
  Generated 50/365 tiles...
  Generated 100/365 tiles...
  ...
✓ Generated 365 tile images

═══ STEP 4: Generating Map Overview ═══
✓ Generated map overview: Azeroth_overview.png

═══ STEP 5: Saving Analysis Metadata ═══
✓ Saved analysis: Azeroth_analysis.json

═══════════════════════════════════════════
✓ COMPLETE PIPELINE FINISHED
═══════════════════════════════════════════
Output directory: Azeroth_analysis
  ├─ Azeroth_overview.png       (map-wide visualization)
  ├─ Azeroth_analysis.json      (layer metadata)
  └─ tiles/                     (365 tile images)

Total placements: 124,532
  M2:  98,234
  WMO: 26,298
Unique layers: 3046
Tiles processed: 365
```

### 3. Review Results

Open the generated output:

```bash
# Windows
explorer Azeroth_analysis

# Linux/Mac
open Azeroth_analysis  # or: xdg-open Azeroth_analysis
```

**What you get:**

1. **Azeroth_overview.png** - Map-wide view showing all layers
   - See the big picture of UniqueID distribution
   - Identify major patterns across the entire continent
   
2. **tiles/** directory - Per-tile detailed views
   - Each tile shows its own unique layer structure
   - Different tiles can have completely different UniqueID ranges
   - Example: tile_32_28.png might have layers 42-156, 500-723
   
3. **Azeroth_analysis.json** - Complete metadata
   - Per-tile layer definitions
   - Global summary of all UniqueID ranges found
   - Ready for programmatic filtering

**What to look for in images:**
- Each color = different UniqueID range (layer)
- Tile-specific layers show local development history
- Sparse layers = experimental/test content
- Dense layers = production content

### 4. Identify Layers to Remove

Example decision:
```
Layer 0-999:    45,231 placements - KEEP (Alpha 0.5.3 core)
Layer 1000-1999: 32,104 placements - KEEP (Alpha 0.6.0)
Layer 2000-2999: 23,456 placements - REMOVE (Alpha 0.7.0+)
Layer 3000+:     12,345 placements - REMOVE (Experimental)
```

### 5. Apply Filter (Future Step)

Once integrated with WoWRollback.Cli:

```bash
dotnet run --project WoWRollback.Cli -- rollback \
  --input Azeroth.wdt \
  --output Azeroth_filtered.wdt \
  --filter-layers Azeroth_analysis.json \
  --max-unique-id 1999
```

---

## Command Reference

### visualize (RECOMMENDED)

**Purpose:** Complete pipeline - analyze and visualize in one command

**Usage:**
```bash
dotnet run --project WoWDataPlot -- visualize \
  --wdt <path-to-wdt> \
  --output-dir <directory> \
  [--gap-threshold <number>] \
  [--tile-size <pixels>] \
  [--map-size <pixels>]
```

**Options:**
- `--wdt` - Path to Alpha WDT file (required)
- `--output-dir` - Output directory for all results (required)
- `--gap-threshold` - Gap size to split layers (default: 50) **← GAPS = DEV PAUSES!**
- `--tile-size` - Tile image size (default: 1024)
- `--map-size` - Map overview size (default: 2048)
- `--tile-marker-size` - Marker size for tiles (default: 8.0)
- `--map-marker-size` - Marker size for overview (default: 5.0)

**Example:**
```bash
dotnet run --project WoWDataPlot -- visualize \
  --wdt Azeroth.wdt \
  --output-dir Azeroth_viz \
  --gap-threshold 50 \
  --tile-size 1024
```

**With minimap overlay (RECOMMENDED!):**
```bash
dotnet run --project WoWDataPlot -- visualize \
  --wdt Azeroth.wdt \
  --output-dir Azeroth_viz \
  --gap-threshold 50 \
  --minimap-dir J:\minimaps\Azeroth
```
**Overlays placement dots on actual in-game minimap tiles for visual context!**  
See `MINIMAP_OVERLAY.md` for setup instructions.

**Output:**
- Map overview PNG
- Per-tile layer images (with optional minimap backgrounds)
- Analysis JSON

### export-csv

**Purpose:** Export raw data to CSV

**Usage:**
```bash
dotnet run --project WoWDataPlot -- export-csv \
  --wdt <path-to-wdt> \
  --output <output.csv>
```

**Example:**
```bash
dotnet run --project WoWDataPlot -- export-csv \
  --wdt Kalidar.wdt \
  --output kalidar_data.csv
```

---

## Troubleshooting

### "No placements found"

**Cause:** WDT has no MDDF/MODF chunks, or gillijimproject can't parse it

**Fix:**
1. Verify WDT file is valid Alpha 0.5.3 format
2. Check file path is correct
3. Try with known-good WDT (e.g., Kalidar.wdt)

### "Build failed"

**Cause:** Missing dependencies or .NET version mismatch

**Fix:**
```bash
# Restore packages
dotnet restore WoWDataPlot/WoWDataPlot.csproj

# Check .NET version (need 9.0)
dotnet --version

# Rebuild
dotnet build WoWDataPlot/WoWDataPlot.csproj
```

### Images look wrong

**Issue:** Y-axis inverted or lines connecting points

**Fix:** Update to latest code - Y-axis is now flipped correctly and LineWidth=0

### Layer colors hard to distinguish

**Fix:** Edit `Program.cs` color array:
```csharp
string[] colors = new[] { 
    "#FF0000",  // Red
    "#00FF00",  // Green
    "#0000FF",  // Blue
    "#FFFF00",  // Yellow
    "#FF00FF",  // Magenta
    "#00FFFF",  // Cyan
    "#FFA500",  // Orange
    "#800080"   // Purple
};
```

---

## Tips & Tricks

### Adjust Layer Size for Your Data

**Large UniqueID range:** Use larger layer size
```bash
--layer-size 5000  # For UniqueID range 0-50000
```

**Small UniqueID range:** Use smaller layer size
```bash
--layer-size 100   # For UniqueID range 0-1000
```

### Focus on Specific Tiles

1. Generate all tile images
2. Identify tiles of interest (e.g., Elwynn Forest)
3. Review only those tiles
4. Apply per-tile filtering

### Combine with External Tools

**Python analysis:**
```python
import json
import pandas as pd

# Load layer analysis
with open('kalidar_layers.json') as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(data['tiles'])

# Find tiles with most layers
print(df.nlargest(10, 'totalPlacements'))
```

**PowerShell batch processing:**
```powershell
# Analyze all WDTs in directory
Get-ChildItem *.wdt | ForEach-Object {
    $name = $_.BaseName
    dotnet run --project WoWDataPlot -- analyze-layers `
        --wdt $_.FullName `
        --output "${name}_layers.json"
}
```

---

## Next Steps

1. ✅ Run layer analysis on your WDT
2. ✅ Generate tile visualizations
3. ✅ Review images and identify unwanted layers
4. ⏳ Wait for filter integration in WoWRollback.Cli
5. ⏳ Apply filters and regenerate WDT
6. ⏳ Test in Alpha client

**See LAYER_WORKFLOW.md for complete details!**
