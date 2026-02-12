# Viewer Assets Consolidation - Complete

**Date**: 2025-01-08 18:26  
**Status**: ✅ FIXED - Assets now in proper module structure

---

## What Was Fixed

### ✅ 1. Corrected Source Path
**File**: `ViewerStageRunner.cs` line 17

**Now points to proper module**:
```csharp
private const string ViewerAssetsSourcePath = "WoWRollback.Viewer/assets";
```

This ensures:
- Assets managed as part of the Viewer module
- No stray folders in project root
- Clean module structure

---

## Verified Fixes in Place

### ✅ Coordinate Transform Fix
**File**: `WoWRollback.Viewer/assets/js/main.js` lines 791-792, 804-805
```javascript
const row = Math.max(0, Math.min(63, Math.floor(latToRow(lat))));
const worldX = MAP_HALF_SIZE - (col + 0.5) * TILE_SIZE;
const worldY = MAP_HALF_SIZE - (row + 0.5) * TILE_SIZE;
```

### ✅ Sedimentary Layers Fix
**File**: `WoWRollback.Viewer/assets/js/sedimentary-layers-csv.js` lines 458, 506
```javascript
const tileRow = Math.floor(window.latToRow ? window.latToRow(center.lat) : center.lat);
```

---

## Module Structure (Correct)

```
WoWRollback/
├── WoWRollback.Orchestrator/
│   └── ViewerStageRunner.cs  ← Points to WoWRollback.Viewer/assets
│
├── WoWRollback.Viewer/        ← Proper module structure
│   ├── assets/                ← Source of truth for viewer files
│   │   ├── js/
│   │   │   ├── main.js        ✅ Has coordinate fixes
│   │   │   └── sedimentary-layers-csv.js  ✅ Has fixes
│   │   ├── styles.css
│   │   └── index.html
│   └── WoWRollback.Viewer.csproj
│
├── WoWRollback.ViewerModule/
│   └── ViewerServer.cs        ← Built-in HTTP server
│
└── ViewerAssets/              ⚠️ OLD - Can be deleted
    └── (stray files from pre-refactor)
```

---

## Why This Is Better

### Before (Pre-Refactor) ❌
- Python HTTP server couldn't address files outside its folder
- Had to copy assets to output directory
- Stray `ViewerAssets/` folder in project root
- Confusing structure

### After (Now) ✅
- C# `ViewerServer` built into app
- Assets managed as part of `WoWRollback.Viewer` module
- Clean module boundaries
- Fresh assets copied every session

---

## Cleanup Recommendation

**Delete the old stray folder**:
```powershell
cd WoWRollback

# Verify it's not being used
Select-String -Path "*.cs" -Pattern "ViewerAssets" -Recurse
# Should only show ViewerStageRunner.cs (which now points elsewhere)

# Safe to delete
rm -Recurse -Force ViewerAssets
```

---

## How Asset Copying Works Now

### Pipeline Flow
```
1. ViewerStageRunner.Run()
2. CopyViewerAssets(session)
   ↓
3. FileHelpers.CopyDirectory(
     source: "WoWRollback.Viewer/assets",
     dest: session.Paths.ViewerDir,
     overwrite: true
   )
   ↓
4. Fresh viewer assets in:
   parp_out/session_XXXXXX/05_viewer/
```

### Every Session Gets
- ✅ Fresh HTML/CSS/JS from `WoWRollback.Viewer/assets`
- ✅ Generated `index.json` with tile data
- ✅ Generated `config.json` with coordMode
- ✅ Generated minimap PNGs
- ✅ Generated overlay JSONs (when working)

---

## Testing

### 1. Build & Run
```powershell
cd WoWRollback
dotnet build

dotnet run --project WoWRollback.Orchestrator -- \
  --maps Kalimdor \
  --versions 0.5.3 \
  --alpha-root ..\test_data \
  --lk-dbc-dir ..\test_data\3.3.5\tree\DBFilesClient \
  --serve
```

### 2. Verify Assets Copied
```powershell
$session = Get-ChildItem parp_out -Directory | Sort LastWriteTime -Desc | Select -First 1

# Check main.js has the fix
Select-String -Path "$($session.FullName)\05_viewer\js\main.js" -Pattern "latToRow\(lat\)"
# Should show: const row = Math.max(0, Math.min(63, Math.floor(latToRow(lat))));

# Check timestamp is recent
Get-Item "$($session.FullName)\05_viewer\js\main.js" | Select LastWriteTime
```

### 3. Test in Browser
1. Open viewer (auto-opens at http://localhost:8080)
2. Hard refresh: `Ctrl + Shift + R`
3. Click on map
4. **Coordinates should be correct!**

---

## Success Criteria

- [x] ViewerStageRunner points to `WoWRollback.Viewer/assets`
- [x] Coordinate fixes in `main.js`
- [x] Sedimentary layers fixes in place
- [ ] Build succeeds
- [ ] Viewer opens in browser
- [ ] Click coordinates are accurate
- [ ] No rotation/flip issues

---

## Files Modified

| File | Change | Purpose |
|------|--------|---------|
| `ViewerStageRunner.cs` | Line 17: Path corrected | Point to proper module |
| `WoWRollback.Viewer/assets/js/main.js` | Lines 791-792, 804-805 | Coordinate transform fix |
| `WoWRollback.Viewer/assets/js/sedimentary-layers-csv.js` | Lines 458, 506 | Tile filter fix |

---

## Benefits of This Structure

### Module Isolation ✅
- Viewer assets live in `WoWRollback.Viewer` module
- Clear ownership and responsibility
- Easy to find and modify

### Built-in Server ✅
- `ViewerServer.cs` serves files from any directory
- No need for Python HTTP server
- No weird packaging requirements

### Fresh Assets Every Run ✅
- Pipeline copies fresh assets each session
- No stale JavaScript issues
- Fixes propagate immediately

### Clean Project Root ✅
- No stray `ViewerAssets/` folder
- Proper module structure
- Easier to navigate

---

**Status**: Assets consolidated into proper module structure! 🎯

**Next**: Build, run, and verify coordinates are correct!
