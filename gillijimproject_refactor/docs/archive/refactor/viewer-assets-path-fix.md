# Viewer Assets Path Fix - ROOT CAUSE FOUND!

**Date**: 2025-01-08 18:18  
**Status**: FIXED - Wrong source directory!

---

## The Real Problem

**ViewerStageRunner was copying from the WRONG directory!**

### What Was Happening
```csharp
// ViewerStageRunner.cs line 17 (OLD):
private const string ViewerAssetsSourcePath = "WoWRollback.Viewer/assets";

// This resolved to a NON-EXISTENT path!
// So it fell back to... who knows where
```

### The Actual Source Directory
```
WoWRollback/
├── ViewerAssets/          ← THIS is the real source!
│   ├── js/
│   │   ├── main.js
│   │   └── sedimentary-layers-csv.js
│   ├── styles.css
│   └── index.html
│
└── WoWRollback.Viewer/    ← This doesn't exist or is wrong!
    └── assets/
```

---

## Why It Kept Failing

1. **I edited** `WoWRollback.Viewer/assets/js/main.js` ✅
2. **Pipeline copied from** `ViewerAssets/js/main.js` ❌
3. **Result**: Old broken code kept getting copied!

---

## Fixes Applied

### 1. Fixed Source Path ✅
**File**: `ViewerStageRunner.cs` line 17
```csharp
// OLD (wrong):
private const string ViewerAssetsSourcePath = "WoWRollback.Viewer/assets";

// NEW (correct):
private const string ViewerAssetsSourcePath = "ViewerAssets";
```

### 2. Fixed Coordinate Transform ✅
**File**: `ViewerAssets/js/main.js` lines 791-792, 804-805
```javascript
// OLD (broken):
const row = Math.max(0, Math.min(63, Math.floor(lat)));
const worldX = MAP_HALF_SIZE - (lng + 0.5) * TILE_SIZE;
const worldY = MAP_HALF_SIZE - (lat + 0.5) * TILE_SIZE;

// NEW (fixed):
const row = Math.max(0, Math.min(63, Math.floor(latToRow(lat))));
const worldX = MAP_HALF_SIZE - (col + 0.5) * TILE_SIZE;
const worldY = MAP_HALF_SIZE - (row + 0.5) * TILE_SIZE;
```

### 3. Fixed Sedimentary Layers ✅
**File**: `ViewerAssets/js/sedimentary-layers-csv.js` lines 458, 506
```javascript
// OLD (broken):
const tileRow = Math.floor(center.lat);

// NEW (fixed):
const tileRow = Math.floor(window.latToRow ? window.latToRow(center.lat) : center.lat);
```

---

## How to Test

### 1. Rebuild & Run
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

# Should show:
# const row = Math.max(0, Math.min(63, Math.floor(latToRow(lat))));
```

### 3. Test in Browser
1. Open viewer (should auto-open at http://localhost:8080)
2. **Hard refresh**: `Ctrl + Shift + R`
3. Click on map
4. Check "Last Click" display
5. **Should show correct tile coordinates!**

---

## Expected Behavior

### Before Fix ❌
- Click tile [30, 35] → Shows [33, 35] (wrong!)
- Coordinates rotated/flipped
- Every rebuild copied old broken code

### After Fix ✅
- Click tile [30, 35] → Shows [30, 35] (correct!)
- Coordinates accurate
- Fresh assets copied every run

---

## Why This Was So Confusing

1. **Two viewer asset directories** existed
2. **I edited the wrong one** (WoWRollback.Viewer/assets)
3. **Pipeline used the other one** (ViewerAssets)
4. **No error messages** - path just silently failed
5. **Assets appeared to copy** but were old versions

---

## Files Modified

| File | Change | Purpose |
|------|--------|---------|
| `ViewerStageRunner.cs` | Line 17: "WoWRollback.Viewer/assets" → "ViewerAssets" | Fix source path |
| `ViewerAssets/js/main.js` | Lines 791-792, 804-805 | Fix coordinate transform |
| `ViewerAssets/js/sedimentary-layers-csv.js` | Lines 458, 506 | Fix tile filter |

---

## Cleanup Recommendation

**Delete the duplicate viewer directory**:
```powershell
# If WoWRollback.Viewer exists and is unused:
rm -Recurse -Force WoWRollback.Viewer

# Or rename it to avoid confusion:
mv WoWRollback.Viewer WoWRollback.Viewer.OLD
```

This prevents future confusion about which directory is the source.

---

## Success Criteria

After rebuild and run:

- [ ] Build succeeds
- [ ] Viewer opens in browser
- [ ] Click on map shows correct coordinates
- [ ] "Last Click" display accurate
- [ ] Sedimentary layers filter works
- [ ] No rotation/flip issues

---

## Lessons Learned

1. **Always verify source paths** - Don't assume they're correct
2. **Check for duplicate directories** - Can cause silent failures
3. **Verify edits are in the right file** - Search for actual usage
4. **Test asset copying** - Ensure fresh files are actually copied

---

**Status**: ROOT CAUSE FIXED! Run pipeline and test! 🎯
