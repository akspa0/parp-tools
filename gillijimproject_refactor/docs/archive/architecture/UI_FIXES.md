# Terrain Overlay UI Fixes

## Issues Fixed

### 1. ✅ Area Boundaries Not Unloading
**Bug**: Unchecking "Area Boundaries" didn't remove the overlay from the map.

**Root Cause**: `hide()` method in `areaIdLayer.js` had inverted logic:
```javascript
// WRONG:
hide() {
    if (!this.visible) {  // Only runs when already hidden!
        this.map.removeLayer(this.layerGroup);
        this.visible = false;
    }
}

// FIXED:
hide() {
    if (this.visible) {  // Runs when visible, hides it
        this.map.removeLayer(this.layerGroup);
        this.visible = false;
    }
}
```

**Files Modified**:
- `ViewerAssets/js/overlays/areaIdLayer.js` (line 26)

**Status**: ✅ Fixed

---

### 2. ✅ AreaTable CSV Reading
**Bug**: Area labels showed "Unknown Area 1234" instead of real names.

**Root Cause**: `AreaTableReader.cs` was parsing wrong column format.

DBCTool.V2 outputs 5-column CSV:
```
row_key,id,parent,continentId,name
1,249,589824,0,Dreadmaul Rock
```

Reader was treating it like 2-column (id,name) format.

**Fix**: Updated to parse column 1 (id) and column 4+ (name, joining for comma-containing names)

**Files Modified**:
- `WoWRollback.Core/Services/AreaTableReader.cs`

**Status**: ✅ Fixed

---

### 3. ✅ Multi-Layer Tooltip
**Issue**: Users unclear what "Multi-Layer" means.

**Explanation**: 
- "Multi-Layer" shows chunks using multiple blended texture layers (NumLayers > 1)
- WoW terrain can have up to 4 texture layers blended via alpha maps
- Single-layer = simple/untextured areas
- Multi-layer = detailed, blended terrain
- Useful for finding areas that weren't properly textured

**Fix**: Added tooltip to HTML:
```html
<label title="Chunks using multiple texture layers (blended terrain)">
    <input type="checkbox" id="showMultiLayer" checked> Multi-Layer
</label>
```

**Files Modified**:
- `ViewerAssets/index.html`

**Status**: ✅ Fixed with tooltip

---

### 4. ✅ Sub-Option Changes Not Updating
**Verification**: Sub-options (showImpassible, showVertexColored, etc.) already trigger re-rendering correctly.

**How It Works**:
1. Checkbox change → `main.js` event handler
2. Calls `overlayManager.setLayerOption(layerName, optionKey, value)`
3. `overlayManager.js` updates layer option
4. Calls `this.renderVisibleTiles()` (line 208)
5. Layer re-renders with new options

**Status**: ✅ Working correctly (no fix needed)

---

## Additional Improvements

### Tooltips Added
All terrain property checkboxes now have explanatory tooltips:

- **Impassible**: "Chunks marked as impassable (collision flag set)"
- **Vertex Colored**: "Chunks using vertex colors (MCCV - colored lighting per vertex)"
- **Multi-Layer**: "Chunks using multiple texture layers (blended terrain)"

---

## Testing Checklist

### Before Committing

- [x] AreaIdLayer hide() fix implemented
- [x] AreaTableReader parsing 5-column CSV
- [x] Tooltips added to HTML
- [ ] Test in browser:
  - [ ] Check/uncheck "Area Boundaries" → should show/hide
  - [ ] Check/uncheck sub-options → should update display
  - [ ] Hover over checkboxes → should see tooltips
  - [ ] Verify area labels show real names (not "Unknown Area 1234")

### Browser Testing Steps

1. **Rebuild WoWRollback.Core**:
   ```bash
   cd WoWRollback
   dotnet build WoWRollback.Core
   ```

2. **Regenerate Viewer**:
   ```bash
   .\rebuild-and-regenerate.ps1 `
     -Maps @("Azeroth") `
     -Versions @("0.5.3.3368") `
     -AlphaRoot ..\test_data\ `
     -Serve
   ```

3. **Test Overlays**:
   - Enable "Area Boundaries"
   - Verify gold boundary lines appear
   - Verify area labels show real names
   - **Uncheck "Area Boundaries"** → should disappear
   - Uncheck "Boundary Lines" → should hide lines but keep labels
   - Uncheck "Area Labels" → should hide labels but keep lines

4. **Test Terrain Properties**:
   - Enable "Terrain Properties"
   - Verify colored chunks appear
   - Uncheck "Impassible" → red chunks should disappear
   - Uncheck "Multi-Layer" → blue chunks should disappear
   - Uncheck "Vertex Colored" → green chunks should disappear

5. **Test Tooltips**:
   - Hover over "Impassible" checkbox
   - Should see: "Chunks marked as impassable (collision flag set)"
   - Test all other tooltips

---

## Known Remaining Issues

### Path Management
- ⏳ AreaTable CSVs need to be copied from DBCTool.V2 outputs
- ⏳ rebuild-and-regenerate.ps1 handles this automatically now
- ⏳ Future: Consolidate into WoWRollback.Data library

### Multi-Map Support
- ⏳ Need to test switching between multiple maps
- ⏳ Verify no cross-map contamination
- ⏳ Future: Add validation in overlay manager

### Performance
- ⏳ Large maps may load slowly with all overlays enabled
- ⏳ Future: Implement tile batching, overlay caching

---

## Summary

**Fixed**:
1. ✅ Area Boundaries now unload properly
2. ✅ Area names show correctly (not "Unknown")
3. ✅ Tooltips explain what each overlay type means
4. ✅ Sub-options update display correctly

**Next Steps**:
1. Test all fixes in browser
2. Commit changes to git
3. Process additional maps (Kalimdor, etc.)
4. Begin consolidation plan (WoWRollback.Data library)

**Ready for**: User testing and git commit! 🎉
