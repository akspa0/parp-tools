# Viewer Cache Issue - Quick Fix

**Date**: 2025-01-08 17:30  
**Issue**: Ghost tiles, misaligned grid, old viewer code cached

---

## Immediate Fix - Hard Refresh Browser

### Windows/Linux
1. **Chrome/Edge**: Press `Ctrl + Shift + R` or `Ctrl + F5`
2. **Firefox**: Press `Ctrl + Shift + R` or `Ctrl + F5`

### Mac
1. **Chrome/Edge**: Press `Cmd + Shift + R`
2. **Firefox**: Press `Cmd + Shift + R`

### Nuclear Option
```
1. Press F12 (open DevTools)
2. Go to Network tab
3. Check "Disable cache" checkbox
4. Refresh page (F5)
```

---

## What the Screenshot Shows

### ❌ Problems
1. **Ghost tiles** (white grid squares) - Viewer expecting tiles at wrong coordinates
2. **Misaligned grid** - Old coordinate system cached
3. **Partial minimap** - Some tiles load, others don't
4. **No overlays** - Separate issue (need to debug after cache clear)

### ✅ Good Signs
- Azeroth map selected correctly
- Some minimap tiles DO load (colored terrain visible)
- Map dropdown works

---

## Root Cause

**Browser cached old viewer JavaScript** that:
- Uses old coordinate system (no Y-axis inversion)
- Expects tiles at wrong grid positions
- Has outdated config.json format

**Our fixes**:
- ✅ Added `coordMode: "wowtools"` to config.json
- ✅ Fixed index.json format
- ✅ Viewer assets copied with `overwrite: true`

**But**: Browser still serving cached JS/config from memory!

---

## After Hard Refresh - Expected Behavior

### Should See
1. **No ghost tiles** - Only actual map tiles display
2. **Grid aligned** with minimap perfectly
3. **Tiles right-side up** - Y-axis inversion working
4. **Smooth panning** - No coordinate jumps

### Should NOT See
- White grid squares where no tiles exist
- Misaligned minimap vs grid
- Tiles appearing in wrong positions

---

## If Hard Refresh Doesn't Work

### Step 1: Clear All Browser Data
```
Chrome/Edge:
1. Ctrl + Shift + Delete
2. Select "Cached images and files"
3. Time range: "All time"
4. Click "Clear data"

Firefox:
1. Ctrl + Shift + Delete
2. Select "Cache"
3. Time range: "Everything"
4. Click "Clear Now"
```

### Step 2: Verify Viewer Files Are Current
```powershell
# Check config.json has coordMode
cat parp_out\session_*\05_viewer\config.json | Select-String "coordMode"
# Should show: "coordMode": "wowtools"

# Check index.json has "map" property
cat parp_out\session_*\05_viewer\index.json | Select-String '"map":'
# Should show: "map": "Azeroth", etc.
```

### Step 3: Force Re-copy Viewer Assets
```powershell
# Delete viewer directory
rm -Recurse -Force parp_out\session_*\05_viewer

# Re-run pipeline
dotnet run --project WoWRollback.Orchestrator -- \
  --maps Azeroth \
  --versions 0.5.3 \
  --alpha-root ..\test_data \
  --lk-dbc-dir ..\test_data\3.3.5\tree\DBFilesClient \
  --serve
```

---

## Overlays Still Not Working

**Separate issue** - After cache is cleared, we need to:

1. **Check console output** for `[OverlayGen]` messages
2. **Verify overlay files exist**:
   ```powershell
   ls parp_out\session_*\05_viewer\overlays\0.5.3\Azeroth\objects_combined\*.json
   ```
3. **Check browser console** (F12) for overlay load errors

---

## Quick Verification Checklist

After hard refresh:

- [ ] No ghost tiles (white grid squares gone)
- [ ] Grid aligns with minimap
- [ ] Tiles display right-side up
- [ ] Panning is smooth
- [ ] Browser console shows no errors (F12 → Console)

If all ✅: Cache issue fixed, proceed to debug overlays  
If still ❌: Try nuclear option (clear all browser data)

---

## Status

**Current**: Browser cache issue blocking viewer  
**Fix**: Hard refresh (Ctrl+Shift+R)  
**Next**: Debug overlays after cache cleared
