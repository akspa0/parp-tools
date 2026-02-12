# Viewer Coordinate Fix - Final Solution

**Date**: 2025-01-08 18:01  
**Issue**: Coordinates rotated/flipped, minimap preview not interactive

---

## Root Cause

**Viewer assets not re-copying after build!**

When you run `dotnet build`, it compiles C# code but **doesn't trigger viewer asset copy**.  
The old JavaScript with broken coordinates is still in the output directory.

---

## The Fix Applied

### File: `main.js` lines 781-800

**Before** (broken):
```javascript
function handleMapClick(e) {
    const lat = e.latlng.lat;
    const lng = e.latlng.lng;
    
    // WRONG: Direct floor() doesn't account for coordMode transform
    const row = Math.max(0, Math.min(63, Math.floor(lat)));
    const col = Math.max(0, Math.min(63, Math.floor(lng)));
    
    // WRONG: Using lat/lng directly for world coords
    const worldX = MAP_HALF_SIZE - (lng + 0.5) * TILE_SIZE;
    const worldY = MAP_HALF_SIZE - (lat + 0.5) * TILE_SIZE;
```

**After** (fixed):
```javascript
function handleMapClick(e) {
    const lat = e.latlng.lat;
    const lng = e.latlng.lng;
    
    // CORRECT: Use latToRow() to handle coordMode transform
    const row = Math.max(0, Math.min(63, Math.floor(latToRow(lat))));
    const col = Math.max(0, Math.min(63, Math.floor(lng)));
    
    // CORRECT: Use row/col (not lat/lng) for world coords
    const worldX = MAP_HALF_SIZE - (col + 0.5) * TILE_SIZE;
    const worldY = MAP_HALF_SIZE - (row + 0.5) * TILE_SIZE;
```

**Why this matters**:
- With `coordMode: "wowtools"`, `lat` is already transformed (63 - row)
- Must use `latToRow()` to convert back: `latToRow(lat)` = `63 - lat` = `row`
- World coordinates must use actual row/col, not transformed lat/lng

---

## How to Force Asset Re-copy

### Option 1: Delete Output Directory (Recommended)
```powershell
# Delete the entire parp_out directory
rm -Recurse -Force parp_out

# Re-run pipeline (will copy fresh viewer assets)
dotnet run --project WoWRollback.Orchestrator -- \
  --maps Kalimdor \
  --versions 0.5.3 \
  --alpha-root ..\test_data \
  --lk-dbc-dir ..\test_data\3.3.5\tree\DBFilesClient \
  --serve
```

### Option 2: Delete Only Viewer Directory
```powershell
# Delete just the viewer output
rm -Recurse -Force parp_out\session_*\05_viewer

# Re-run pipeline
dotnet run --project WoWRollback.Orchestrator -- \
  --maps Kalimdor \
  --versions 0.5.3 \
  --alpha-root ..\test_data \
  --lk-dbc-dir ..\test_data\3.3.5\tree\DBFilesClient \
  --serve
```

### Option 3: Manually Copy Viewer Assets
```powershell
# Copy viewer assets to existing session
$session = Get-ChildItem parp_out -Directory | Sort LastWriteTime -Desc | Select -First 1
Copy-Item -Recurse -Force WoWRollback.Viewer\assets\* "$($session.FullName)\05_viewer\"
```

---

## Minimap Preview Box Issue

The bottom-right minimap preview with selection box is **Leaflet's built-in MiniMap plugin**.

### Current Issues
1. **Not interactive** - Can't drag selection box
2. **Coordinates wrong** - Same transform issue as main map

### Fix Required

Check if MiniMap plugin is configured correctly:

```javascript
// In main.js, look for minimap initialization
const miniMap = new L.Control.MiniMap(
    minimapLayer,
    {
        toggleDisplay: true,
        minimized: false,
        position: 'bottomright',
        // ADD THIS:
        aimingRectOptions: {
            color: '#ff7800',
            weight: 3,
            interactive: true  // ← Enable dragging
        },
        shadowRectOptions: {
            color: '#000000',
            weight: 1,
            interactive: false
        }
    }
);
```

**If MiniMap plugin not found**: It might be a custom implementation that needs the same coordinate fix.

---

## Verification Steps

### After Re-running Pipeline

#### 1. Check Viewer Assets Were Copied
```powershell
# Check timestamp of main.js
$session = Get-ChildItem parp_out -Directory | Sort LastWriteTime -Desc | Select -First 1
Get-Item "$($session.FullName)\05_viewer\js\main.js" | Select LastWriteTime

# Should be RECENT (within last few minutes)
```

#### 2. Check main.js Has Fix
```powershell
$session = Get-ChildItem parp_out -Directory | Sort LastWriteTime -Desc | Select -First 1
Select-String -Path "$($session.FullName)\05_viewer\js\main.js" -Pattern "latToRow\(lat\)"

# Should show line 786: const row = Math.max(0, Math.min(63, Math.floor(latToRow(lat))));
```

#### 3. Test in Browser
1. **Hard refresh**: `Ctrl + Shift + R`
2. **Click on map**
3. **Check "Last Click" display**:
   - Should show correct tile coordinates
   - Should show correct world coordinates
4. **Check minimap preview**:
   - Selection box should match main view
   - Should be draggable (if interactive enabled)

---

## Expected Behavior After Fix

### Coordinate Display ✅
- Click on tile → Shows correct `Tile [row,col]`
- World coordinates match WoW coordinate system
- No rotation/flip offset

### Minimap Preview ✅
- Selection box aligns with main view
- Dragging box pans main map (if interactive)
- Coordinates consistent between main and preview

### Grid Overlay ✅
- White grid lines align with tile boundaries
- Grid matches minimap tile layout
- No rotation/flip

---

## If Still Broken After Re-copy

### Check Browser Cache
```
1. F12 → Application tab → Clear storage
2. Check "Cache" and "Local storage"
3. Click "Clear site data"
4. Hard refresh (Ctrl + Shift + R)
```

### Check Viewer Source
```
1. F12 → Sources tab
2. Navigate to js/main.js
3. Search for "handleMapClick"
4. Verify line 786 has: latToRow(lat)
```

### Check Console Errors
```
F12 → Console tab
Look for:
- "latToRow is not defined" → Function not in scope
- "coordMode is undefined" → Config not loading
- Any other JavaScript errors
```

---

## Additional Fixes Needed

### 1. Minimap Preview Interactivity

**If using Leaflet MiniMap plugin**, add to initialization:
```javascript
aimingRectOptions: {
    interactive: true,
    draggable: true
}
```

**If custom implementation**, ensure:
- Mouse events bound to selection box
- Drag updates main map view
- Coordinates transformed correctly

### 2. All Click Handlers

Search for other places that convert lat/lng to row/col:
```powershell
Select-String -Path "WoWRollback.Viewer\assets\js\*.js" -Pattern "Math\.floor\(.*lat\)" -Recurse
```

Each should use `latToRow(lat)` instead of direct `Math.floor(lat)`.

### 3. Overlay Coordinate Transforms

Check overlay layers (terrain, liquids, etc.) use correct transforms:
```javascript
// In overlay layers, when calculating bounds:
const row = latToRow(lat);  // Not: Math.floor(lat)
```

---

## Summary

**Problem**: Viewer assets not re-copying after C# code changes  
**Solution**: Delete output directory and re-run pipeline  
**Fix Applied**: `handleMapClick()` now uses `latToRow()` for coordinate transform  
**Next**: Verify assets copied, test in browser with hard refresh  

**Status**: Fix implemented, awaiting asset re-copy and test! 🎯
