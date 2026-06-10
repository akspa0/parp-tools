# Coordinate System Fix - Using Historical Solution

**Date**: 2025-01-08 20:12  
**Source**: Old docs from proof-of-concept viewer

---

## The Discovery

Found the EXACT same coordinate issue documented in `VIEWER_FIX_PLAN_FINAL.md` from September 2025!

### The Problem (from old docs):

```
ADT Coordinate System:
     0,0 -----> X (East)
      |
      v Y (South)
   
   Northwest = (0,0)
   Southeast = (63,63)

Leaflet CRS.Simple:
   Southwest = (0,0)
   Northeast = (maxX, maxY)
   Y increases UPWARD (north)

THE PROBLEM:
- ADT: Row 0 = North edge, Row 63 = South edge
- Leaflet: Y=0 = South edge, Y=max = North edge
- They're inverted!
```

---

## The Solution (from old docs)

**Use custom CRS with Y-axis flip transformation:**

```javascript
const WoWCRS = L.extend({}, L.CRS.Simple, {
    transformation: new L.Transformation(1, 0, -1, 64)
});

map = L.map('map', { crs: WoWCRS, ... });
```

**Parameters**:
- `scaleX=1`: X unchanged
- `offsetX=0`: No X offset
- `scaleY=-1`: Y flipped
- `offsetY=64`: Shift by 64 to keep in positive range

---

## What I Implemented

### 1. Custom CRS in initializeMap() ✅
**File**: `main.js` lines 60-80

```javascript
function initializeMap() {
    // WoW ADT coordinate system: Row 0 = North, Row 63 = South (Y increases downward)
    // Leaflet CRS.Simple: Y=0 = South, Y=max = North (Y increases upward)
    // Solution: Use custom CRS with Y-axis flip transformation
    const WoWCRS = L.extend({}, L.CRS.Simple, {
        transformation: new L.Transformation(1, 0, -1, 64)
    });
    
    map = L.map('map', {
        crs: WoWCRS,
        ...
    });
}
```

### 2. Simplified Coordinate Functions ✅
**File**: `main.js` lines 1001-1023

**Removed the `coordMode: "wowtools"` transforms** since CRS handles it:

```javascript
function rowToLat(row) {
    // With custom CRS transformation, row maps directly to lat
    return row;
}

function latToRow(lat) {
    // With custom CRS transformation, lat maps directly to row
    return lat;
}

function tileBounds(row, col) {
    // With custom CRS, coordinates map directly
    return [[row, col], [row + 1, col + 1]];
}

function pixelToLatLng(row, col, px, py, w, h) {
    // With custom CRS, coordinates map directly
    const lat = row + (py / h);
    const lng = col + (px / w);
    return { lat, lng };
}
```

### 3. Removed 404 Spam ✅
**Files**: `overlayLoader.js`, `overlays/overlayManager.js`

Caches 404 responses to prevent repeated requests.

---

## Why This Works

### Before (coordMode approach) ❌
- Used `L.CRS.Simple` (no transformation)
- Tried to handle Y-flip with `rowToLat()` = `63 - row`
- **Problem**: This created confusion and double-flips

### After (custom CRS) ✅
- Uses custom CRS with built-in Y-flip transformation
- Coordinates map directly: row → lat, col → lng
- **CRS handles the flip internally** - no manual transforms needed

---

## How Leaflet Transformation Works

```
L.Transformation(a, b, c, d)

Transforms coordinates:
  x' = a * x + b
  y' = c * y + d

Our transformation (1, 0, -1, 64):
  x' = 1 * x + 0 = x        (unchanged)
  y' = -1 * y + 64 = 64 - y  (flipped and shifted)

Example:
  Row 0 (North) → y' = 64 - 0 = 64 (top of map)
  Row 63 (South) → y' = 64 - 63 = 1 (bottom of map)
```

---

## Testing

### Build & Run
```powershell
cd WoWRollback
dotnet build

dotnet run --project WoWRollback.Orchestrator -- \
  --maps Azeroth \
  --versions 0.5.3 \
  --alpha-root ..\test_data \
  --lk-dbc-dir ..\test_data\3.3.5\tree\DBFilesClient \
  --serve
```

### Expected Behavior
- ✅ Minimap tiles display right-side up
- ✅ Grid aligns with tiles
- ✅ Click coordinates correct
- ✅ No 404 spam in console
- ✅ Geography matches expected (Feralas on west side of Kalimdor, etc.)

---

## Files Modified

| File | Lines | Change |
|------|-------|--------|
| `main.js` | 60-80 | Added custom WoWCRS with Y-flip |
| `main.js` | 1001-1023 | Simplified coordinate functions |
| `overlayLoader.js` | 6-18 | Cache 404s |
| `overlays/overlayManager.js` | 95-111 | Cache failed tiles |

---

## Key Insight from History

**The old docs solved this exact problem in September 2025!**

The solution was already documented:
- Custom CRS with Y-flip transformation
- Direct coordinate mapping (no manual transforms)
- Let Leaflet handle the coordinate system internally

**We were overthinking it** by trying to handle the flip manually with `coordMode: "wowtools"` and `rowToLat()` transforms.

---

## Success Criteria

- [ ] Build succeeds
- [ ] Viewer opens in browser
- [ ] Minimap tiles display correctly (not upside down)
- [ ] Grid aligns with tiles
- [ ] Click coordinates accurate
- [ ] No 404 spam
- [ ] Console usable for debugging

---

**Status**: Implemented historical solution! Ready to test! 🎯
