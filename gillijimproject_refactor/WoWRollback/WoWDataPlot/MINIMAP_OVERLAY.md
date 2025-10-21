# Minimap Overlay Feature

## Overview

Overlay placement dots on top of actual in-game minimap tiles to provide **visual context** for decision-making. See placements in relation to cities, terrain, roads, and other landmarks!

## Why This Matters

**Without minimap context:**
```
Tile [32, 28]: Layer 42-87 (Blue)
```
**Question:** Is this near a city? A road? Empty wilderness?
**Problem:** Can't make informed decisions about what to keep/remove!

**With minimap context:**
```
Tile [32, 28] with minimap background:
  - Blue dots concentrated near Stormwind city walls
  - Green dots along the road to Goldshire
  - Red dots scattered in Elwynn Forest
```
**Benefit:** Now you can see EXACTLY where content is and make informed decisions!

## How to Use

### Step 1: Extract Minimap Tiles from Alpha Client

**Option A: Use BLP Converter (Recommended)**

If you have Alpha WoW client files:

```bash
# Minimaps are usually at:
# <WoW Install>/Data/World/Minimaps/<MapName>/

# Example for Azeroth (Eastern Kingdoms):
<WoW Install>/Data/World/Minimaps/Azeroth/map0_0.blp
<WoW Install>/Data/World/Minimaps/Azeroth/map0_1.blp
...
<WoW Install>/Data/World/Minimaps/Azeroth/map63_63.blp
```

Convert BLP files to PNG using a BLP converter tool:
1. [BLPConverter](https://www.wowinterface.com/downloads/info14110-BLPConverter.html)
2. [Warcraft BLP Image Converter](https://github.com/WoW-Tools/WowImgConverter)

**Option B: Extract from MPQ Archives**

If minimaps are in MPQ files:

```bash
# Use MPQEditor or similar tool to extract:
# World/Minimaps/Azeroth/*.blp
# Then convert to PNG
```

**Option C: Use Pre-Extracted PNG Tiles**

If you already have PNG minimap tiles, just point to the directory!

### Step 2: Organize Minimap Files

Create a directory structure:

```
minimaps/
├── Azeroth/          ← Eastern Kingdoms
│   ├── map0_0.png
│   ├── map0_1.png
│   ...
│   └── map63_63.png
├── Kalidar/          ← Kalimdor  
│   ├── map0_0.png
│   ├── map0_1.png
│   ...
│   └── map63_63.png
```

**Supported naming patterns:**
- `map<Y>_<X>.png` (e.g., `map32_28.png`)
- `map<YY>_<XX>.png` (e.g., `map32_28.png` with zero-padding)
- `<MapName>_<Y>_<X>.png` (e.g., `Azeroth_32_28.png`)
- `tile_<Y>_<X>.png`
- `<Y>_<X>.png`

### Step 3: Run with Minimap Overlay

```bash
dotnet run --project WoWDataPlot -- visualize \
  --wdt ..\test_data\0.5.3\tree\World\Maps\Azeroth\Azeroth.wdt \
  --output-dir .\Azeroth_with_minimaps \
  --minimap-dir J:\minimaps\Azeroth \
  --gap-threshold 50
```

**Output:**
```
═══ STEP 3: Generating Per-Tile Images ═══
  Generated 50/365 tiles...
  Generated 100/365 tiles...
  ...
✓ Generated 365 tile images
  → Overlaid 347/365 minimap backgrounds  ← Success!
```

**If minimaps not found:**
```
✓ Generated 365 tile images
  → Overlaid 0/365 minimap backgrounds  ← No minimaps found, check paths!
```

## What You Get

### Before (No Minimap)
```
Tile [32, 28]
- Layer 42-87:    Blue dots (46 placements)
- Layer 156-234:  Green dots (79 placements)
- Layer 890-923:  Red dots (34 placements)
```
**Problem:** No idea WHERE these are!

### After (With Minimap)
```
Tile [32, 28] with minimap showing Stormwind
- Blue dots:   Near city walls (early city development)
- Green dots:  Inside city buildings (mid development)
- Red dots:    Scattered outside walls (late additions)
```
**Benefit:** Visual context makes decisions obvious!

## Use Cases

### Use Case 1: Remove City Expansions

**Scenario:** You want to keep Alpha 0.5.3 Stormwind but remove later expansions

**With minimap overlay:**
1. See minimap showing Stormwind layout
2. Identify blue layer = original city
3. Identify green/red layers = expansions
4. Decision: Keep blue, remove green/red
5. Result: Pure Alpha 0.5.3 Stormwind!

### Use Case 2: Clean Up Test Content

**Scenario:** Remove experimental objects scattered in wilderness

**With minimap overlay:**
1. See minimap showing terrain
2. Notice sparse red dots in middle of nowhere
3. Recognize as test content (no nearby landmarks)
4. Decision: Remove red layer
5. Result: Clean wilderness!

### Use Case 3: Preserve Quest Hubs

**Scenario:** Keep all content near quest hubs, remove random spawns

**With minimap overlay:**
1. See minimap with roads and buildings
2. Blue dots near Goldshire = quest NPCs (keep)
3. Red dots in random forest = test spawns (remove)
4. Decision: Keep blue, remove red
5. Result: Quest content preserved!

## Troubleshooting

### No Minimaps Overlaid (0/365)

**Cause:** Minimap files not found

**Fix:**
1. Check directory path is correct
2. Verify files named correctly (e.g., `map0_0.png`)
3. Check files are PNG format (not BLP)
4. Ensure files are in `<minimap-dir>/<MapName>/` subdirectory

**Debug command:**
```bash
# List files in minimap directory
ls J:\minimaps\Azeroth\

# Should see:
# map0_0.png, map0_1.png, ..., map63_63.png
```

### Some Minimaps Missing (e.g., 200/365)

**Cause:** Some tiles don't have minimap images

**Reason:** Normal! Not all 64×64 tiles have data in Alpha WoW

**Solution:** No fix needed - tiles without minimaps will just show scatter plot

### Minimap Doesn't Align with Dots

**Cause:** Coordinate system mismatch

**Fix:** Check that:
1. Minimap is for correct map (Azeroth vs Kalidar)
2. Minimap tile coordinates match WoW coordinate system
3. Minimap image is not rotated/flipped

## Advanced: Batch Convert BLP to PNG

If you have many BLP files to convert:

**PowerShell script:**
```powershell
# Using ImageMagick or BLPConverter
$blpDir = "C:\WoW\Data\World\Minimaps\Azeroth"
$pngDir = "J:\minimaps\Azeroth"

Get-ChildItem "$blpDir\*.blp" | ForEach-Object {
    $pngName = $_.BaseName + ".png"
    $pngPath = Join-Path $pngDir $pngName
    
    # Example using BLPConverter CLI
    & "C:\Tools\BLPConverter.exe" $_.FullName $pngPath
}
```

**Python script (using pillow_blp):**
```python
from PIL import Image
from pillow_blp import BlpImagePlugin
import os

blp_dir = "C:/WoW/Data/World/Minimaps/Azeroth"
png_dir = "J:/minimaps/Azeroth"

for blp_file in os.listdir(blp_dir):
    if blp_file.endswith('.blp'):
        blp_path = os.path.join(blp_dir, blp_file)
        png_path = os.path.join(png_dir, blp_file.replace('.blp', '.png'))
        
        img = Image.open(blp_path)
        img.save(png_path, 'PNG')
```

## Expected Directory Structure

```
J:\minimaps\
├── Azeroth\
│   ├── map0_0.png    ← Northwest corner
│   ├── map0_1.png
│   ...
│   ├── map32_32.png  ← Map center
│   ...
│   └── map63_63.png  ← Southeast corner
│
├── Kalidar\
│   ├── map0_0.png
│   ...
│   └── map63_63.png
│
└── Development\
    ├── map0_0.png
    ...
```

## Full Example

**Complete workflow with minimaps:**

```bash
# 1. Extract minimaps from Alpha client
# (Use BLPConverter or MPQEditor)

# 2. Organize into directory
mkdir J:\minimaps\Azeroth
# Copy converted PNG files to J:\minimaps\Azeroth\

# 3. Run visualization with overlay
dotnet run --project WoWDataPlot -- visualize \
  --wdt ..\test_data\0.5.3\tree\World\Maps\Azeroth\Azeroth.wdt \
  --output-dir .\Azeroth_FULL \
  --minimap-dir J:\minimaps\Azeroth \
  --gap-threshold 50 \
  --tile-size 1024 \
  --tile-marker-size 10

# 4. Review tiles/ directory
# Each PNG now has minimap background with colored dots overlaid!

# 5. Make informed decisions about layer removal
# - See visual context
# - Identify content by location
# - Remove unwanted layers with confidence
```

## Benefits Summary

✅ **Visual Context** - See placements in relation to terrain  
✅ **Informed Decisions** - Know exactly what you're removing  
✅ **Spatial Understanding** - Identify cities, roads, landmarks  
✅ **Quality Control** - Verify content is in correct locations  
✅ **Documentation** - Visual record of where content was placed  

**Transform abstract UniqueID analysis into concrete spatial archaeology!** 🗺️
