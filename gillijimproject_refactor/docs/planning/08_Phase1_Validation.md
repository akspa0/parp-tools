# Phase 1 Validation - WoWRollback.Viewer Project

**Date**: 2025-10-05  
**Status**: Ready for Testing  
**Goal**: Verify old and new viewer assets paths work identically

---

## What Was Built

### ✅ Completed Changes

1. **Created `WoWRollback.Viewer/` project**
   - `.csproj` configured to copy all assets on build
   - Targets .NET 9.0
   - References WoWRollback.Core

2. **Copied `ViewerAssets/` → `WoWRollback.Viewer/assets/`**
   - All HTML, CSS, JS files preserved
   - Directory structure maintained

3. **Added to solution**
   - `WoWRollback.sln` updated
   - Project GUID: `{A7B3C9D4-2E8F-4F1A-9B5C-6D7E8F9A0B1C}`
   - All build configurations added

4. **Feature flag implemented**
   - Parameter: `-UseNewViewerAssets` in `rebuild-and-regenerate.ps1`
   - Default: OFF (uses old `ViewerAssets/`)
   - When ON: Uses `WoWRollback.Viewer/bin/Release/net9.0/assets/`

---

## Validation Steps

### Step 1: Build the Solution ⏳

```powershell
cd WoWRollback
dotnet build WoWRollback.sln --configuration Release
```

**Expected Result**:
- ✅ Build succeeds with 0 errors
- ✅ Assets copied to `WoWRollback.Viewer/bin/Release/net9.0/assets/`
- ✅ Contains: `index.html`, `js/`, `styles.css`, etc.

**Verify assets copied**:
```powershell
ls WoWRollback.Viewer\bin\Release\net9.0\assets\
# Should show: index.html, js/, styles.css, etc.
```

---

### Step 2: Test Old Path (Default Behavior) ⏳

```powershell
# Default: uses ViewerAssets/ (old path)
.\rebuild-and-regenerate.ps1 -Maps DeadminesInstance -Versions "0.5.3.3368"
```

**Expected Result**:
- ✅ Script completes successfully
- ✅ Output: `rollback_outputs/comparisons/.../viewer/`
- ✅ Assets copied from `ViewerAssets/`
- ✅ Console shows: "✓ Copied viewer assets from: ...\ViewerAssets"

**Verify viewer works**:
```powershell
cd rollback_outputs\comparisons\0.5.3.3368\viewer
python -m http.server 8080
# Open: http://localhost:8080/index.html
```

**Manual Check**:
- ✅ Map loads correctly
- ✅ Overlays display
- ✅ Objects render
- ✅ No console errors

---

### Step 3: Test New Path (Feature Flag) ⏳

```powershell
# Use WoWRollback.Viewer assets (new path)
.\rebuild-and-regenerate.ps1 -Maps DeadminesInstance -Versions "0.5.3.3368" -UseNewViewerAssets
```

**Expected Result**:
- ✅ Script completes successfully
- ✅ Console shows: "[PHASE 1] Using WoWRollback.Viewer assets from: ..."
- ✅ Assets copied from `WoWRollback.Viewer\bin\Release\net9.0\assets\`
- ✅ Console shows: "✓ Copied viewer assets from: ...\WoWRollback.Viewer\bin\Release\net9.0\assets"

**Verify viewer works**:
```powershell
cd rollback_outputs\comparisons\0.5.3.3368\viewer
python -m http.server 8081  # Use different port
# Open: http://localhost:8081/index.html
```

**Manual Check**:
- ✅ Map loads correctly
- ✅ Overlays display
- ✅ Objects render
- ✅ No console errors

---

### Step 4: SHA256 Comparison ⏳

Compare the viewer outputs to ensure they're identical:

```powershell
# Create old output
.\rebuild-and-regenerate.ps1 -Maps DeadminesInstance -Versions "0.5.3.3368"
Move-Item rollback_outputs rollback_outputs_old

# Create new output
.\rebuild-and-regenerate.ps1 -Maps DeadminesInstance -Versions "0.5.3.3368" -UseNewViewerAssets
Move-Item rollback_outputs rollback_outputs_new

# Compare viewer directory contents
$oldViewer = "rollback_outputs_old\comparisons\0.5.3.3368\viewer"
$newViewer = "rollback_outputs_new\comparisons\0.5.3.3368\viewer"

# Get file hashes (excluding generated overlay data, which should be identical anyway)
Get-ChildItem -Path $oldViewer -Recurse -File | 
    Where-Object { $_.Name -match '\.(html|js|css)$' } |
    Get-FileHash | 
    Select-Object Hash, @{Name="RelativePath"; Expression={$_.Path.Replace($oldViewer, "")}} |
    Sort-Object RelativePath |
    Out-File old_hashes.txt

Get-ChildItem -Path $newViewer -Recurse -File | 
    Where-Object { $_.Name -match '\.(html|js|css)$' } |
    Get-FileHash | 
    Select-Object Hash, @{Name="RelativePath"; Expression={$_.Path.Replace($newViewer, "")}} |
    Sort-Object RelativePath |
    Out-File new_hashes.txt

# Compare
Compare-Object (Get-Content old_hashes.txt) (Get-Content new_hashes.txt)
```

**Expected Result**:
- ✅ No differences (empty output)
- ✅ All HTML/JS/CSS files identical (SHA256 match)

---

### Step 5: Visual Regression Test ⏳

**Side-by-Side Comparison**:

Open both viewers in different tabs:
- Old: `http://localhost:8080/index.html`
- New: `http://localhost:8081/index.html`

**Checklist**:
- [ ] Map tiles load at same speed
- [ ] Terrain overlay identical
- [ ] Shadow overlay identical
- [ ] Area ID overlay identical
- [ ] Holes overlay identical
- [ ] Liquids overlay identical
- [ ] Object markers identical
- [ ] Object counts match
- [ ] Popup content identical
- [ ] No visual differences

**Take Screenshots**:
```powershell
# Save screenshots of key views for documentation
# - Main map view
# - Zoomed in tile
# - Object popup
# - Each overlay type
```

---

## Success Criteria

### Must Pass All:

1. ✅ **Build Success**
   - Solution compiles without errors
   - Assets copied to output directory

2. ✅ **Old Path Works**
   - Default behavior unchanged
   - Viewer loads and functions normally

3. ✅ **New Path Works**
   - Feature flag activates successfully
   - Viewer loads from new location
   - Identical functionality to old

4. ✅ **SHA256 Validation**
   - All viewer assets byte-identical
   - No file differences

5. ✅ **Visual Validation**
   - No visual differences
   - No console errors
   - Performance unchanged

---

## Rollback Procedure

**If validation fails**:

```powershell
# Immediate rollback (< 1 minute)
git checkout -- WoWRollback.sln
git checkout -- WoWRollback/WoWRollback.Viewer/
git checkout -- rebuild-and-regenerate.ps1

# Or just don't use the flag
.\rebuild-and-regenerate.ps1 -Maps DeadminesInstance
# Old path still works
```

---

## Common Issues

### Issue 1: Assets Not Found (New Path)

**Symptom**: `Warning: Viewer assets directory not found`

**Cause**: Viewer project not built

**Fix**:
```powershell
dotnet build WoWRollback.sln --configuration Release
```

---

### Issue 2: Viewer Shows 404 Errors

**Symptom**: Browser console shows missing files

**Cause**: Assets not copied correctly

**Fix**:
```powershell
# Verify assets exist
ls WoWRollback.Viewer\bin\Release\net9.0\assets\

# Rebuild if missing
dotnet clean
dotnet build --configuration Release
```

---

### Issue 3: Different Behavior Between Old/New

**Symptom**: New path behaves differently

**Cause**: Asset copy error or file modification

**Fix**:
```powershell
# Re-copy assets
xcopy /E /I /Y ViewerAssets WoWRollback.Viewer\assets
dotnet build --configuration Release
```

---

## Post-Validation Actions

### If All Tests Pass ✅

1. **Update Phase 1 status to complete**
2. **Document results** in this file
3. **Commit changes**:
   ```powershell
   git add WoWRollback/WoWRollback.Viewer/
   git add WoWRollback/WoWRollback.sln
   git add WoWRollback/rebuild-and-regenerate.ps1
   git add docs/planning/08_Phase1_Validation.md
   git commit -m "Phase 1: Add WoWRollback.Viewer project with feature flag"
   ```
4. **Proceed to Phase 2** (Manifest System)

### If Tests Fail ❌

1. **Document failure details**
2. **Use rollback procedure**
3. **Investigate root cause**
4. **Fix and re-test**
5. **Do NOT proceed until passing**

---

## Validation Log

**Date**: 2025-10-05  
**Tester**: _____________  

| Step | Status | Notes |
|------|--------|-------|
| 1. Build Solution | ⏳ Pending | |
| 2. Test Old Path | ⏳ Pending | |
| 3. Test New Path | ⏳ Pending | |
| 4. SHA256 Compare | ⏳ Pending | |
| 5. Visual Test | ⏳ Pending | |

**Overall Result**: ⏳ Pending

**Approver**: _____________  
**Date**: _____________

---

## Next Phase Preview

Once Phase 1 passes validation:

**Phase 2: Backend Manifest System**
- Create `OverlayManifestBuilder.cs`
- Generate `overlay_manifest.json`
- No viewer changes yet
- Estimated: 1 week

---

**Phase 1 Status**: ⏳ Ready for Validation  
**Blocking**: Requires build + testing  
**Time Required**: ~30 minutes manual testing
