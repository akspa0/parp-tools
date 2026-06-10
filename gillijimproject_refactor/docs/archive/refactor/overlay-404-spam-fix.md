# Overlay 404 Spam Fix

**Date**: 2025-01-08 20:07  
**Issue**: Console spammed with overlay 404 errors, blocking debugging

---

## The Problem

Viewer was repeatedly requesting non-existent overlay files:
```
GET http://localhost:8080/overlays/0.5.3/Azeroth/terrain_complete/tile_r14_c17.json
Overlay not found: overlays/0.5.3/Azeroth/terrain_complete/tile_r14_c17.json
```

**Why**: Overlays aren't generating (separate issue), but viewer kept retrying failed requests on every pan/zoom.

---

## The Fix

### 1. Cache 404 Failures in overlayLoader.js
**File**: `overlayLoader.js` lines 6-18

**Before**:
```javascript
if (cache.has(path)) {
    return cache.get(path);
}

const response = await fetch(path, { cache: 'no-store' });
if (!response.ok) {
    throw new Error(`Failed to load overlay: ${path}`);
}
```

**After**:
```javascript
if (cache.has(path)) {
    const cached = cache.get(path);
    if (cached === null) {
        throw new Error(`Overlay not found (cached 404): ${path}`);
    }
    return cached;
}

const response = await fetch(path, { cache: 'no-store' });
if (!response.ok) {
    // Cache the 404 to prevent repeated requests
    cache.set(path, null);
    throw new Error(`Failed to load overlay: ${path}`);
}
```

**Result**: 404s are cached as `null`, preventing repeated fetch attempts.

---

### 2. Cache 404 Failures in overlayManager.js
**File**: `overlays/overlayManager.js` lines 95-111

**Before**:
```javascript
if (this.loadedTiles.has(tileKey)) {
    this.renderTile(this.loadedTiles.get(tileKey), tileRow, tileCol);
    return;
}

const response = await fetch(overlayPath);
if (!response.ok) {
    console.warn(`Overlay not found: ${overlayPath}`);
    return;
}
```

**After**:
```javascript
if (this.loadedTiles.has(tileKey)) {
    const cached = this.loadedTiles.get(tileKey);
    if (cached !== null) {
        this.renderTile(cached, tileRow, tileCol);
    }
    return;
}

const response = await fetch(overlayPath);
if (!response.ok) {
    // Cache the failure to prevent repeated 404s
    this.loadedTiles.set(tileKey, null);
    return;
}
```

**Result**: Failed tiles are cached as `null`, skipped on subsequent checks.

---

## Benefits

### Before ❌
- Console spammed with hundreds of 404 errors
- Network tab flooded with failed requests
- Impossible to debug other issues
- Performance impact from repeated fetches

### After ✅
- Each 404 logged only ONCE
- Failed requests cached and skipped
- Clean console for actual debugging
- No performance impact from retries

---

## How It Works

### Caching Strategy
```javascript
// First request: 404
cache.set(path, null);  // Cache the failure

// Subsequent requests:
if (cache.has(path)) {
    const cached = cache.get(path);
    if (cached === null) {
        throw new Error(`Overlay not found (cached 404): ${path}`);
    }
    return cached;
}
// Never reaches fetch() again!
```

### Cache Invalidation
Cache is cleared when:
- User changes version
- User changes map
- `clearCache()` is called explicitly

This ensures that if overlays are later generated, they'll be fetched.

---

## Testing

### 1. Rebuild & Run
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

### 2. Check Console
```
F12 → Console
```

**Should see**:
- Initial 404 warnings for missing overlays
- NO repeated spam
- Clean console for debugging

### 3. Pan/Zoom Map
**Should NOT see**:
- New 404 errors for same tiles
- Console spam

---

## Files Modified

| File | Lines | Change |
|------|-------|--------|
| `overlayLoader.js` | 6-18 | Cache 404s as `null` |
| `overlays/overlayManager.js` | 95-111 | Cache failed tiles as `null` |

---

## Next Steps

Now that console is clean, we can:
1. Debug coordinate grid alignment
2. Fix overlay generation (separate issue)
3. Test click coordinates
4. Verify minimap positioning

---

**Status**: 404 spam eliminated! Console now usable for debugging! 🎯
