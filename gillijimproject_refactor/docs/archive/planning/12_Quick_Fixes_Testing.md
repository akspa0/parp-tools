# Quick UX Fixes - Testing Guide

**Date**: 2025-10-05  
**Status**: Ready for Testing  
**Changes**: Popup z-index management + Sedimentary Layers implementation

---

## What Was Changed

### ✅ Fix #1: Multi-Popup Support
**Status**: Already existed in code!
- `autoClose: false` already present
- `closeOnClick: false` already present
- No changes needed

### ✅ Fix #2: Popup Z-Index Management
**Files Modified**:
- `ViewerAssets/js/main.js`
  - Added `bringPopupToFront()` function (lines 120-137)
  - Updated all 3 marker creation locations to call it on click

**What It Does**:
- When you click a popup, it comes to the front
- Other popups go behind it
- Clear visual hierarchy

### ✅ Fix #3: Sedimentary Layers Implementation
**Files Created**:
- `WoWRollback.Viewer/assets/js/sedimentary-layers.js` (NEW)

**Files Modified**:
- `ViewerAssets/js/main.js`
  - Import SedimentaryLayersManager
  - Initialize on map creation
  - Register all markers with their UniqueID
- `ViewerAssets/styles.css`
  - Added styling for layers panel (~100 lines)

**What It Does**:
- Type UniqueID (e.g., "12345") - shows only that ID
- Type range (e.g., "1000-2000") - shows that range
- Three modes:
  - **Dim Others**: Fades non-matching objects to 10% opacity
  - **Hide Others**: Completely hides non-matching objects
  - **Show Only**: Same as hide (alias)
- Clear search box - shows all objects

---

## Testing Steps

### Step 1: Build the Solution ⏳

```powershell
cd WoWRollback
dotnet build WoWRollback.sln --configuration Release
```

**Expected**: ✅ Build succeeds with 0 errors

---

### Step 2: Run Rebuild Script ⏳

```powershell
.\rebuild-and-regenerate.ps1 -Maps DeadminesInstance -Versions "0.5.3.3368" -UseNewViewerAssets
```

**Note**: Using `-UseNewViewerAssets` to get the updated JavaScript files.

**Expected**: 
- ✅ Script completes
- ✅ Viewer generated with new assets

---

### Step 3: Test Multi-Popup (Already Working) ⏳

```powershell
cd rollback_outputs\comparisons\0.5.3.3368\viewer
python -m http.server 8080
# Open: http://localhost:8080/index.html
```

**Manual Test**:
1. Click on map to load some objects
2. Click 3-5 different M2 markers
3. All popups should stay open
4. Click map background - popups stay open
5. Only close button (X) closes individual popup

**Expected**:
- ✅ Multiple popups open simultaneously
- ✅ No glitching or flickering
- ✅ Smooth interaction

---

### Step 4: Test Popup Z-Index ⏳

**Manual Test**:
1. Open 3+ popups (click different markers)
2. Click on popup A (should come to front)
3. Click on popup B (should come to front, A goes behind)
4. Click on popup C (should come to front, others behind)
5. Verify visual stacking is correct

**Expected**:
- ✅ Clicked popup always on top
- ✅ Other popups visible behind it
- ✅ Clear z-ordering
- ✅ No visual glitches

---

### Step 5: Test Sedimentary Layers Panel ⏳

**Manual Test - Panel Visibility**:
1. Look for "Sedimentary Layers" panel on right side of screen
2. Should be below overview canvas
3. Click the toggle button (−) to collapse
4. Click again (+) to expand

**Expected**:
- ✅ Panel visible on right side
- ✅ Toggle works smoothly
- ✅ Panel styled correctly (dark theme)

---

### Step 6: Test UniqueID Filtering ⏳

**Test Case 1: Single UniqueID**
```
1. Type a specific UniqueID in search box (e.g., "12345")
2. Press Enter or wait for input
3. Check status message at bottom
4. Observe map
```

**Expected**:
- ✅ Status shows: "UniqueID 12345: X object(s)"
- ✅ Only objects with that UniqueID are bright
- ✅ All other objects dimmed to 10% opacity
- ✅ Can still see dimmed objects faintly

**Test Case 2: UniqueID Range**
```
1. Type a range in search box (e.g., "1000-2000")
2. Press Enter or wait for input
3. Check status message
4. Observe map
```

**Expected**:
- ✅ Status shows: "Range 1000-2000: showing X UniqueIDs"
- ✅ Only objects in range are bright
- ✅ Objects outside range dimmed
- ✅ Works in both directions: "2000-1000" same as "1000-2000"

**Test Case 3: Clear Filter**
```
1. Clear the search box (delete all text)
2. Status should update
3. All objects back to normal
```

**Expected**:
- ✅ Status shows: "Showing all: X UniqueIDs (Y objects)"
- ✅ All objects at full opacity
- ✅ All objects interactive

---

### Step 7: Test Filter Modes ⏳

**Test "Dim Others" Mode** (default):
```
1. Type "1000-2000" in search
2. Mode selector should show "Dim Others"
3. Non-matching objects at 10% opacity but still visible
```

**Test "Hide Others" Mode**:
```
1. Change mode to "Hide Others"
2. Non-matching objects completely invisible
3. Can't click on them
4. Matching objects at full opacity
```

**Test "Show Only" Mode**:
```
1. Change mode to "Show Only"
2. Should behave same as "Hide Others"
3. Only matching objects visible
```

**Expected**:
- ✅ Dim mode: objects faded but visible
- ✅ Hide mode: objects completely gone
- ✅ Show mode: same as hide
- ✅ Smooth transitions between modes

---

### Step 8: Test Edge Cases ⏳

**Edge Case 1: Invalid Input**
```
1. Type "abc" in search box
2. Should show error or ignore
```

**Expected**: 
- ✅ No crash
- ✅ Either shows error message or no action

**Edge Case 2: Non-Existent UniqueID**
```
1. Type "999999" (unlikely to exist)
2. Should show "not found" message
```

**Expected**:
- ✅ Status shows: "UniqueID 999999: not found"
- ✅ All objects dimmed/hidden (nothing matches)

**Edge Case 3: Empty Range**
```
1. Type "5000-4000" (reversed, no objects in range)
2. Should auto-correct to "4000-5000"
```

**Expected**:
- ✅ Range auto-swaps if reversed
- ✅ Shows correct count

---

### Step 9: Performance Test ⏳

**Test with Dense Object Area**:
```
1. Navigate to area with many M2s (you discovered in 0.5.5!)
2. Apply filter: "0-99999" (show all)
3. Change modes rapidly
4. Clear filter
```

**Expected**:
- ✅ No lag or stuttering
- ✅ Filter applies quickly (<100ms)
- ✅ Mode changes instant
- ✅ Smooth interaction

---

### Step 10: Console Check ⏳

**Check Browser Console**:
```
F12 → Console tab
Look for:
- [SedimentaryLayers] Initialized
- No errors
- No warnings
```

**Expected**:
- ✅ Initialization message present
- ✅ Zero errors
- ✅ Zero warnings about missing elements

---

## Success Criteria

### Multi-Popup ✅
- [x] Already working (no changes needed)
- [ ] 5+ popups can be open
- [ ] No glitching

### Popup Z-Index ⏳
- [ ] Clicked popup comes to front
- [ ] Visual stacking correct
- [ ] Smooth transitions

### Sedimentary Layers ⏳
- [ ] Panel visible and styled
- [ ] Toggle button works
- [ ] Search input functional
- [ ] Single UniqueID filtering works
- [ ] Range filtering works
- [ ] All 3 modes work (dim/hide/show)
- [ ] Clear filter works
- [ ] No console errors
- [ ] Performance good

---

## Known Limitations

1. **No UniqueID List**: 
   - Doesn't show checkboxes for each UniqueID
   - Future enhancement: populate list of all UniqueIDs

2. **No Persistence**: 
   - Filter resets when changing maps
   - Future enhancement: remember filters per map

3. **No Visual Feedback**: 
   - Doesn't highlight matching objects differently
   - Future enhancement: green border for matches

---

## Rollback Procedure

**If issues found**:

```powershell
# Revert to old viewer assets (without flag)
.\rebuild-and-regenerate.ps1 -Maps DeadminesInstance -Versions "0.5.3.3368"
# Uses ViewerAssets/ (old path) - no new features
```

**Or manual revert**:
```powershell
git checkout -- WoWRollback/ViewerAssets/
git checkout -- WoWRollback/WoWRollback.Viewer/assets/
```

---

## Troubleshooting

### Issue: "sedimentary-layers.js not found"

**Symptom**: Console error about missing module

**Fix**: 
```powershell
# Copy manually
Copy-Item WoWRollback.Viewer\assets\js\sedimentary-layers.js ViewerAssets\js\
dotnet build --configuration Release
```

---

### Issue: Panel not visible

**Cause**: HTML element missing or z-index too low

**Fix**: Check `index.html` has `<div id="layersPanel">` section

---

### Issue: Filter doesn't work

**Cause**: Markers not registered

**Debug**:
```javascript
// In browser console
sedimentaryLayers.getStats()
// Should show: { totalUniqueIds: X, totalMarkers: Y, ... }
```

---

## Next Steps After Testing

### If All Tests Pass ✅
1. Commit changes:
   ```powershell
   git add WoWRollback/ViewerAssets/
   git add WoWRollback/WoWRollback.Viewer/
   git add docs/planning/
   git commit -m "Quick UX fixes: popup z-index + Sedimentary Layers"
   ```

2. Update master plan - mark quick fixes complete

3. Resume Phase 3-5 plugin refactor (or celebrate! 🎉)

### If Tests Fail ❌
1. Document failures
2. Use rollback procedure
3. Debug and fix
4. Re-test

---

**Status**: ⏳ Ready for User Testing  
**Time Required**: ~15 minutes manual testing  
**Risk**: LOW (incremental changes, easy rollback)
