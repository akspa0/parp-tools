# Viewer Grid Alignment Diagnostic

**Date**: 2025-01-08 19:56  
**Issue**: Grid overlay misaligned despite coordinate fixes

---

## What the Screenshot Shows

### ✅ Working
- Minimap tiles display correctly (Azeroth visible, right-side up)
- Tiles are in center of viewport
- Colors and terrain look correct

### ❌ Not Working
- White grid lines extend far beyond minimap
- Grid appears rotated/misaligned
- Grid doesn't match tile boundaries

---

## Possible Causes

### 1. Config.json Missing coordMode
**Check**:
```powershell
$session = Get-ChildItem parp_out -Directory | Sort LastWriteTime -Desc | Select -First 1
cat "$($session.FullName)\05_viewer\config.json"
```

**Should contain**:
```json
{
  "coordMode": "wowtools",
  ...
}
```

**If missing**: Viewer uses default coordinate system (no Y-axis inversion)

---

### 2. Browser Cache Still Serving Old Files
**Symptoms**:
- Config looks correct in file
- But browser console shows old config

**Fix**:
```
1. F12 → Application tab
2. Clear storage → Check "Cache" and "Local storage"
3. Click "Clear site data"
4. Hard refresh: Ctrl + Shift + R
```

---

### 3. Tile Bounds Calculation Wrong
**Current logic** (`tileBounds()` function):
```javascript
function tileBounds(row, col) {
    if (isWowTools()) {
        const latTop = rowToLat(row);      // 63 - row
        const latBottom = rowToLat(row + 1); // 63 - (row+1)
        const north = Math.max(latTop, latBottom);
        const south = Math.min(latTop, latBottom);
        return [[south, col], [north, col + 1]];
    }
    return [[row, col], [row + 1, col + 1]];
}
```

**Example for row=32, col=32**:
- latTop = 63 - 32 = 31
- latBottom = 63 - 33 = 30
- north = 31, south = 30
- bounds = [[30, 32], [31, 33]]

**This should be correct!**

---

### 4. Index.json Tile Data Wrong
**Check**:
```powershell
$session = Get-ChildItem parp_out -Directory | Sort LastWriteTime -Desc | Select -First 1
$json = cat "$($session.FullName)\05_viewer\index.json" | ConvertFrom-Json
$json.maps | Where { $_.map -eq "Azeroth" } | Select -ExpandProperty tiles | Select -First 5
```

**Should show**:
```
row col versions
--- --- --------
30  30  {0.5.3}
30  31  {0.5.3}
...
```

**If row/col are swapped or wrong**: Tile data generation issue

---

### 5. Minimap Filenames Don't Match Expected Pattern
**Check**:
```powershell
$session = Get-ChildItem parp_out -Directory | Sort LastWriteTime -Desc | Select -First 1
ls "$($session.FullName)\05_viewer\minimap\0.5.3\Azeroth\*.png" | Select -First 5 Name
```

**Should be**: `Azeroth_30_30.png`, `Azeroth_30_31.png`, etc.

**Viewer expects** (from `state.js` line 90):
```javascript
`minimap/${version}/${mapName}/${mapName}_${col}_${row}.png`
```

**If pattern doesn't match**: Files won't load, grid will be empty

---

## Debugging Steps

### Step 1: Check Browser Console
```
F12 → Console tab
```

**Look for**:
- `isWowTools()` function exists and returns `true`
- `config.coordMode` is `"wowtools"`
- No 404 errors on minimap PNG files
- No JavaScript errors

**Test in console**:
```javascript
// Should return true
console.log(isWowTools());

// Should return "wowtools"
console.log(state.config.coordMode);

// Should return 0
console.log(latToRow(63));

// Should return 63
console.log(latToRow(0));
```

---

### Step 2: Check Network Tab
```
F12 → Network tab → Filter: PNG
```

**Look for**:
- Minimap PNG requests
- Check if they're 200 OK or 404 Not Found
- Verify filenames match pattern: `Azeroth_30_30.png`

---

### Step 3: Inspect Tile Overlay Elements
```
F12 → Elements tab
Find: <img class="leaflet-tile">
```

**Check**:
- `src` attribute points to correct PNG
- `style` attribute has correct `transform` values
- Multiple tiles visible (not just one)

---

### Step 4: Check Tile Bounds in Console
```javascript
// In browser console:
const bounds = tileBounds(32, 32);
console.log(bounds);
// Should show: [[30, 32], [31, 33]]
```

---

## Likely Root Causes (Ranked)

### 1. Browser Cache (90% likely) 🔥
**Symptoms**: Everything looks correct in files, but viewer still broken  
**Fix**: Clear all browser data, hard refresh

### 2. Config.json Missing coordMode (5% likely)
**Symptoms**: Grid uses wrong coordinate system  
**Fix**: Verify config.json has `"coordMode": "wowtools"`

### 3. Index.json Tile Data Wrong (3% likely)
**Symptoms**: Tiles load but at wrong positions  
**Fix**: Check tile row/col values in index.json

### 4. Minimap Files Missing/Wrong Names (2% likely)
**Symptoms**: Grid shows but no tiles  
**Fix**: Verify PNG filenames match expected pattern

---

## Quick Diagnostic Commands

### Check Config
```powershell
$s = Get-ChildItem parp_out -Directory | Sort LastWriteTime -Desc | Select -First 1
Select-String -Path "$($s.FullName)\05_viewer\config.json" -Pattern "coordMode"
```

### Check Index Tiles
```powershell
$s = Get-ChildItem parp_out -Directory | Sort LastWriteTime -Desc | Select -First 1
$json = Get-Content "$($s.FullName)\05_viewer\index.json" | ConvertFrom-Json
$json.maps[0].tiles | Select -First 3
```

### Check Minimap Files
```powershell
$s = Get-ChildItem parp_out -Directory | Sort LastWriteTime -Desc | Select -First 1
ls "$($s.FullName)\05_viewer\minimap\0.5.3\Azeroth\*.png" | Measure-Object
```

### Check main.js Has Fix
```powershell
$s = Get-ChildItem parp_out -Directory | Sort LastWriteTime -Desc | Select -First 1
Select-String -Path "$($s.FullName)\05_viewer\js\main.js" -Pattern "latToRow\(lat\)"
```

---

## If Still Broken After All Checks

### Nuclear Option: Delete Everything and Rebuild
```powershell
# Stop any running servers
# Then:
rm -Recurse -Force parp_out
rm -Recurse -Force WoWRollback\bin
rm -Recurse -Force WoWRollback\obj
rm -Recurse -Force WoWRollback.Orchestrator\bin
rm -Recurse -Force WoWRollback.Orchestrator\obj

dotnet clean
dotnet build

dotnet run --project WoWRollback.Orchestrator -- \
  --maps Azeroth \
  --versions 0.5.3 \
  --alpha-root ..\test_data \
  --lk-dbc-dir ..\test_data\3.3.5\tree\DBFilesClient \
  --serve
```

---

**Status**: Need diagnostic info to identify root cause! 🔍
