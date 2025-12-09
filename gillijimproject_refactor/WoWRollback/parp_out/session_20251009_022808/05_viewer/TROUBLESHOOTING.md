# Troubleshooting Guide

## Issue: "No data loaded" / Overlays not appearing

### Symptoms
- Status bar shows "❌ No data loaded"
- Plugins are checked but nothing appears on map
- Only minimap tiles and grid are visible

### Common Causes & Solutions

#### 1. **Not Running via Web Server**
**Problem:** Browser blocks `fetch()` requests when opening HTML files directly (`file://` protocol).

**Solution:** Run a local web server:
```bash
# Python 3
cd WoWRollback.Viewer/assets
python -m http.server 8000

# Node.js (http-server)
cd WoWRollback.Viewer/assets
npx http-server -p 8000

# VS Code Live Server extension
Right-click test-plugin-system.html → "Open with Live Server"
```

Then open: `http://localhost:8000/test-plugin-system.html`

#### 2. **Data Not Loaded Yet**
**Problem:** Plugins won't show anything until you check their boxes AND data is loaded.

**Solution:**
1. Wait for status bar to show "✅ Loaded X objects"
2. Check plugin checkboxes (Chunk Grid, Density, M2, WMO)
3. Plugins should appear immediately now

If still not appearing, press F12 and check console for errors.

#### 3. **Wrong Data Path**
**Problem:** Data files not at expected location.

**Current Path:** `../../parp_out/session_20251008_231621/04_analysis`

**Check:**
- Open browser console (F12)
- Look for messages like `[DataAdapter] Master index not found: ...`
- Verify the URL shown matches your actual file location

**Fix:** Update the data path in `test-plugin-system.html` around line 234:
```javascript
const dataPath = '../../parp_out/YOUR_SESSION_HERE/04_analysis';
```

#### 4. **Minimap Tiles Not Loaded**
**Problem:** Background tiles aren't loaded yet.

**Solution:**
Click one of these buttons:
- **"Simulate Tile Images"** - Loads colored test tiles
- **"Load Real Kalimdor Tiles"** - Loads actual minimap images (requires minimap folder)

Overlays will render on top of tiles.

#### 5. **Z-Index Issues**
**Problem:** Overlays rendering behind tiles.

**Status:** ✅ Fixed as of 2025-10-09

If overlays still behind tiles, verify `tilesPane` creation:
```javascript
// Should be in init() function
map.createPane('tilesPane');
map.getPane('tilesPane').style.zIndex = 100;
```

## Debugging Steps

### 1. Check Browser Console (F12)
Look for these messages:

**Good:**
```
[Test] ✅ DataAdapter loaded: {...}
[Test] Loaded X tiles
[ChunkGridPlugin] Rendered grid for tile...
[DensityHeatmapPlugin] Rendered heatmap for tile...
```

**Bad:**
```
[DataAdapter] Master index not found: ...
Failed to fetch
CORS error
```

### 2. Verify Data Files Exist
Check these files exist:
```
parp_out/session_20251008_231621/04_analysis/
  └── 0.5.3/
      └── master/
          ├── Azeroth_master_index.json (3.47MB)
          └── Azeroth_id_ranges_by_tile.json (230KB)
```

### 3. Test Data Loading Manually
Open browser console and run:
```javascript
// Check if data adapter exists
console.log(window.dataAdapter);

// Check if data loaded
console.log(window.dataAdapter?.masterIndex);

// Try loading manually
await window.dataAdapter.loadMap('0.5.3', 'Azeroth', '../../parp_out/session_20251008_231621/04_analysis');
```

### 4. Test Plugin Visibility
```javascript
// Check plugin states
console.log('Chunk Grid enabled:', window.chunkGridPlugin.enabled);
console.log('Chunk Grid visible:', window.chunkGridPlugin.visible);
console.log('Chunk Grid layers:', window.chunkGridPlugin.layers.length);

// Manually trigger load
const bounds = window.map.getBounds();
const zoom = window.map.getZoom();
window.chunkGridPlugin.loadVisibleData(bounds, zoom);
```

## Expected Behavior

When working correctly:

1. **On Page Load:**
   - Status: "✅ Plugin system initialized successfully!"
   - Console: "[Test] ✅ DataAdapter loaded: ..."
   - ADT Grid visible (white lines)

2. **After Clicking "Simulate Tile Images":**
   - Colored test tiles appear
   - Grid overlays on top of tiles
   - Minimap visible in bottom-right

3. **After Checking "Show Chunk Grid":**
   - Cyan grid appears (16×16 per tile)
   - Only on visible tiles
   - Console: "[Test] Chunk grid enabled"

4. **After Checking "Show Density Heatmap":**
   - Green/yellow/orange/red chunks appear
   - Shows where objects are clustered
   - Click chunks for details

5. **After Checking "Show M2/WMO Objects":**
   - Purple circles (M2) and orange squares (WMO) appear
   - Click for object details
   - Only where data exists

## Quick Test Sequence

1. **Refresh page**
2. **Check console** - Should see "✅ DataAdapter loaded"
3. **Click "Simulate Tile Images"**
4. **Check "Show Chunk Grid"** - Cyan grid should appear
5. **Check "Show Density Heatmap"** - Colors should appear
6. **Pan/zoom map** - New tiles load automatically

If any step fails, check the corresponding section above.

---

**Last Updated:** 2025-10-09 00:29
