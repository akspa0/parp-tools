# Minimap Coordinate Fix - Complete Analysis

**Date**: 2025-01-08 17:39  
**Issue**: Tiles rotated 90° CCW + mirrored

---

## Root Cause Identified

### The Real Problem
We're using **legacy minimap BLPs** (0.5.3/0.5.5 only) instead of **md5translate.trs** method.

**Why this matters**:
- Legacy BLPs: `map{X}_{Y}.blp` directly in map folders
- Modern method: MD5-hashed BLPs + `md5translate.trs` mapping file
- **We're falling back to legacy because TRS files don't exist or aren't being found**

---

## TRS File Format (from wowdev.wiki)

```
dir: Azeroth
Azeroth\map30_35.blp    a1b2c3d4e5f6.blp
Azeroth\map31_35.blp    f6e5d4c3b2a1.blp
...
```

**Format**:
- `map_%d_%02d.blp` = `map{X}_{Y}.blp`
- X = NOT zero-padded (30, 31, 32...)
- Y = zero-padded to 2 digits (35, 04, 00...)
- Left side: Virtual tile name
- Right side: Actual MD5-hashed BLP filename

---

## Current Code Flow

### MinimapLocator.LoadVersion()
```csharp
1. Find minimap root directory
2. Look for md5translate.trs files
3. Parse TRS → get tile mappings
4. IF TRS parsing fails or returns 0 entries:
   → Fall back to ScanMinimapDirectory() (legacy BLPs)
```

### Why We're Using Legacy BLPs
**Line 127-130**: `if (allEntries.Count == 0) { allEntries.AddRange(ScanMinimapDirectory(minimapRoot)); }`

**Possible reasons**:
1. **TRS files don't exist** in 0.5.3 test data
2. **TRS files exist but parsing fails** (exception swallowed)
3. **TRS files in wrong location** (GetCandidateTrsFiles not finding them)

---

## The Coordinate Swap Bug

### In ParseTrsFile() (Line 302)
```csharp
// BLP filename: map30_35.blp
var coords = stem.Substring(3).Split('_');  // ["30", "35"]
var tileX = coords[0];  // 30
var tileY = coords[1];  // 35

// TRS format: map{X}_{Y}.blp where X=col, Y=row
entries.Add(new MinimapEntry(currentMap, tileY, tileX, ...));
```

**This creates**: `MinimapEntry(map, row=35, col=30, ...)`

### In ScanMinimapDirectory() (Line 326)
```csharp
// Same logic - also swaps!
entries.Add(new MinimapEntry(mapName, tileY, tileX, blp, relative));
```

### Then in LoadVersion() (Line 160)
```csharp
tiles[key] = new MinimapTile(resolvedPath, entry.TileCol, entry.TileRow, ...);
```

**This creates**: `MinimapTile(path, TileX=30, TileY=35, ...)`

### Finally BuildFileName() (Line 446)
```csharp
return $"{mapName}_{TileX}_{TileY}.png";
// Returns: "Azeroth_30_35.png"
```

### Viewer Expects (state.js line 90)
```javascript
`minimap/${version}/${mapName}/${mapName}_${col}_${row}.png`
// Expects: "Azeroth_30_35.png" where col=30, row=35
```

**They match!** So the coordinate swap is actually **intentional** to match the viewer's expectations!

---

## The Real Bug: Tile Positioning

### The viewer calls (main.js line 516)
```javascript
const b = tileBounds(r, c);  // r=row, c=col
const overlay = L.imageOverlay(url, b, ...);
```

### tileBounds() returns (line 1002)
```javascript
return [[south, col], [north, col + 1]];
// Leaflet format: [[lat, lng], [lat, lng]]
// Where lat=row (Y), lng=col (X)
```

**This is correct!** Leaflet's `[[lat, lng]]` = `[[Y, X]]` = `[[row, col]]`

---

## So What's Actually Wrong?

### Hypothesis 1: TRS Parsing Fails
If TRS files exist but parsing fails, we fall back to legacy BLPs which might have different coordinate conventions.

### Hypothesis 2: No TRS Files
0.5.3 might genuinely use legacy BLPs, but they're named differently than we expect.

### Hypothesis 3: Viewer Coordinate System
The viewer's `coordMode: "wowtools"` might need different tile positioning logic.

---

## Investigation Steps

### Step 1: Check if TRS Files Exist
```powershell
# Look for md5translate files
ls ..\test_data\0.5.3\tree\World\Textures\Minimap\*.trs
ls ..\test_data\0.5.3\tree\World\Textures\Minimap\*.txt

# Check if they're being found
# Add logging to GetCandidateTrsFiles()
```

### Step 2: Check Actual BLP Filenames
```powershell
# See what BLP files actually exist
ls ..\test_data\0.5.3\tree\World\Textures\Minimap\Azeroth\*.blp | Select -First 10 Name

# Check if they're map{X}_{Y}.blp or MD5 hashes
```

### Step 3: Verify Generated PNG Names
```powershell
# Check what PNGs we generated
ls parp_out\session_*\05_viewer\minimap\0.5.3\Azeroth\*.png | Select -First 10 Name

# Should match: Azeroth_{col}_{row}.png
```

### Step 4: Check Viewer's Tile Requests
```
F12 → Network tab → Filter: PNG
Look at actual URLs being requested
Compare with files that exist
```

---

## Likely Fix

### If TRS Files Don't Exist (0.5.3 is too old)
**Accept that 0.5.3 uses legacy BLPs** and ensure coordinate parsing is correct.

**The current swap might be wrong!** Let me re-examine:

```csharp
// BLP: map30_35.blp
// TRS format says: map{X}_{Y}.blp where X is NOT padded, Y IS padded
// So: X=30 (col), Y=35 (row)

// Current code:
var tileX = coords[0];  // 30 = should be COL
var tileY = coords[1];  // 35 = should be ROW

// But we do:
entries.Add(new MinimapEntry(currentMap, tileY, tileX, ...));
//                                        ^^^^  ^^^^
//                                        row   col
// This puts: row=35, col=30 ✅ CORRECT!
```

**Wait, that's right!** We're correctly swapping because:
- coords[0] = X = col
- coords[1] = Y = row
- MinimapEntry expects (map, row, col, ...)
- So we pass (map, coords[1], coords[0], ...) = (map, Y, X, ...) = (map, row, col, ...) ✅

---

## The ACTUAL Bug

If coordinates are correct, the rotation must be in **how Leaflet positions the image**.

### Check: Are we using the right corner coordinates?

Leaflet ImageOverlay: `L.imageOverlay(url, [[south, west], [north, east]])`

Current code:
```javascript
return [[south, col], [north, col + 1]];
//       [south, west], [north, east]
```

**This assumes**:
- south < north (Y increases upward)
- west < east (X increases rightward)

**But WoW coordinates**:
- Row 0 is at TOP (north)
- Row 63 is at BOTTOM (south)
- Col 0 is at LEFT (west)
- Col 63 is at RIGHT (east)

**With `coordMode: "wowtools"`**:
- `rowToLat(row)` = `63 - row` (inverts Y)
- So row=0 → lat=63 (top)
- row=63 → lat=0 (bottom)

**For tile at row=30, col=35**:
- latTop = 63 - 30 = 33
- latBottom = 63 - 31 = 32
- north = max(33, 32) = 33
- south = min(33, 32) = 32
- bounds = [[32, 35], [33, 36]]

**This should work!** Unless... the image itself needs to be flipped?

---

## Nuclear Option: Check Image Orientation

The BLP → PNG conversion might be flipping the image!

### In MinimapComposer.cs
```csharp
using var blp = new BLPFile(blpStream);
var pixels = blp.GetPixels(0, out var width, out var height);
using var bgraImage = Image.LoadPixelData<Bgra32>(pixels, width, height);
image = bgraImage.CloneAs<Rgba32>();
```

**BLPSharp might return pixels in a different orientation!**

---

## Immediate Action

### 1. Add Debug Logging
```csharp
// In MinimapLocator.LoadVersion()
Console.WriteLine($"[MinimapLoc] Found {allEntries.Count} entries from TRS");
if (allEntries.Count == 0)
{
    Console.WriteLine($"[MinimapLoc] Falling back to directory scan");
    allEntries.AddRange(ScanMinimapDirectory(minimapRoot));
    Console.WriteLine($"[MinimapLoc] Found {allEntries.Count} entries from scan");
}
```

### 2. Verify TRS Files
```powershell
# Check if TRS exists
Test-Path "..\test_data\0.5.3\tree\World\Textures\Minimap\md5translate.trs"
```

### 3. Test Coordinate Fix
My edit to swap tileY/tileX might have **broken** it if it was already correct!

**REVERT MY CHANGE** and test again!

---

## Status

**Current**: Tiles rotated 90° CCW + mirrored  
**Root Cause**: Unknown - need to verify TRS files and coordinate flow  
**Next**: Add logging, check TRS files, possibly revert coordinate swap

**Critical**: Don't assume the swap is wrong - it might be correct and something else is broken!
