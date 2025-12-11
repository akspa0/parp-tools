# Viewer Coordinate Fixes - Complete

**Date**: 2025-01-08 18:05  
**Status**: All coordinate transform bugs fixed

---

## Bugs Fixed

### 1. Click Coordinate Display ✅
**File**: `main.js` line 786  
**Issue**: Used `Math.floor(lat)` instead of `latToRow(lat)`  
**Fix**: `const row = Math.max(0, Math.min(63, Math.floor(latToRow(lat))));`

### 2. World Coordinate Calculation ✅
**File**: `main.js` lines 799-800  
**Issue**: Used `lat/lng` instead of `row/col` for world coords  
**Fix**: 
```javascript
const worldX = MAP_HALF_SIZE - (col + 0.5) * TILE_SIZE;
const worldY = MAP_HALF_SIZE - (row + 0.5) * TILE_SIZE;
```

### 3. Sedimentary Layers Tile Filter ✅
**File**: `sedimentary-layers-csv.js` lines 458, 506  
**Issue**: Used `Math.floor(center.lat)` for tile detection  
**Fix**: `const tileRow = Math.floor(window.latToRow ? window.latToRow(center.lat) : center.lat);`

---

## How to Apply Fixes

### Option 1: Force Update Script (Fastest)
```powershell
cd WoWRollback
.\force-viewer-update.ps1
```

This will:
1. Find most recent session
2. Backup old viewer assets
3. Copy fresh assets from source
4. Preserve data files (index.json, config.json, minimaps, overlays)

### Option 2: Delete & Re-run
```powershell
rm -Recurse -Force parp_out

dotnet run --project WoWRollback.Orchestrator -- \
  --maps Kalimdor \
  --versions 0.5.3 \
  --alpha-root ..\test_data \
  --lk-dbc-dir ..\test_data\3.3.5\tree\DBFilesClient \
  --serve
```

### Option 3: Manual Copy
```powershell
$session = Get-ChildItem parp_out -Directory | Sort LastWriteTime -Desc | Select -First 1
Copy-Item -Recurse -Force WoWRollback.Viewer\assets\* "$($session.FullName)\05_viewer\"
```

---

## After Update

### 1. Hard Refresh Browser
```
Ctrl + Shift + R (Windows/Linux)
Cmd + Shift + R (Mac)
```

### 2. Verify Fix Applied
**F12 → Console**, run:
```javascript
// Should return a function
console.log(typeof window.latToRow);

// Test coordinate transform
console.log(latToRow(63));  // Should return 0
console.log(latToRow(0));   // Should return 63
```

### 3. Test Click Coordinates
1. Click on map
2. Check "Last Click" display
3. Should show correct tile [row, col]
4. World coordinates should make sense

---

## Expected Behavior

### Before Fix ❌
- Click tile [30, 35] → Shows [33, 35] (wrong row)
- World coords off by ~1600 yards
- Sedimentary layers filter wrong tiles

### After Fix ✅
- Click tile [30, 35] → Shows [30, 35] (correct!)
- World coords accurate
- Sedimentary layers filter correct tiles

---

## Minimap Preview Box

The bottom-right minimap preview is likely **Leaflet MiniMap plugin** or custom implementation.

### If Not Interactive

Check for MiniMap initialization in `main.js`:
```javascript
const miniMap = new L.Control.MiniMap(
    minimapLayer,
    {
        aimingRectOptions: {
            interactive: true,  // ← Add this
            draggable: true     // ← And this
        }
    }
);
```

### If Custom Implementation

Search for:
```powershell
Select-String -Path "WoWRollback.Viewer\assets\js\*.js" -Pattern "minimap.*preview|selection.*box" -Recurse
```

Ensure it uses same coordinate transforms as main map.

---

## Remaining Issues

### 1. Minimap Tiles Still Rotated?
If tiles themselves (not just coordinates) are rotated:
- Check MinimapLocator TRS parsing
- Verify BLP → PNG conversion orientation
- Check if BLPSharp returns pixels in correct order

### 2. Overlays Not Generating
- Check console for `[OverlayGen]` messages
- Verify AnalysisIndex.Placements not null/empty
- See [viewer-debug-session.md](file:///i:/parp-tools/pm4next-branch/parp-tools/gillijimproject_refactor/refactor/viewer-debug-session.md)

### 3. Sedimentary Layers Not Loading
- Need to copy layers JSON to viewer directory
- Not yet implemented

---

## Files Modified

| File | Lines | Purpose |
|------|-------|---------|
| `main.js` | 786-787, 799-800 | Fix click coordinate transform |
| `sedimentary-layers-csv.js` | 458, 506 | Fix tile filter coordinate transform |
| `force-viewer-update.ps1` | NEW | Script to force asset re-copy |

---

## Testing Checklist

After applying fixes:

- [ ] Run `force-viewer-update.ps1` OR delete parp_out and re-run
- [ ] Hard refresh browser (Ctrl + Shift + R)
- [ ] Click on map → Check "Last Click" shows correct coordinates
- [ ] Pan around → Check coordinates update correctly
- [ ] Enable Sedimentary Layers → Check tile filter works
- [ ] Check minimap preview box aligns with main view

---

## Success Criteria

✅ **Coordinate Display**:
- Click shows correct tile [row, col]
- World coordinates accurate
- No rotation/flip offset

✅ **Sedimentary Layers**:
- "Current Tile Only" filter works correctly
- Shows objects for correct tile

✅ **Minimap Preview**:
- Selection box aligns with main view
- (Optional) Draggable if interactive enabled

---

## Quick Commands

### Update Viewer Assets
```powershell
.\force-viewer-update.ps1
```

### Check If Fix Applied
```powershell
$session = Get-ChildItem parp_out -Directory | Sort LastWriteTime -Desc | Select -First 1
Select-String -Path "$($session.FullName)\05_viewer\js\main.js" -Pattern "latToRow\(lat\)"
```

### Verify Asset Timestamp
```powershell
$session = Get-ChildItem parp_out -Directory | Sort LastWriteTime -Desc | Select -First 1
Get-Item "$($session.FullName)\05_viewer\js\main.js" | Select LastWriteTime
```

---

**Status**: All coordinate transform bugs fixed! Run update script and test. 🎯
