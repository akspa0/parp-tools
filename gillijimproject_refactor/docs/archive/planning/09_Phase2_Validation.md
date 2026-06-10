# Phase 2 Validation - Backend Manifest System

**Date**: 2025-10-05  
**Status**: Ready for Testing  
**Goal**: Verify overlay_manifest.json files generated correctly

---

## What Was Built

### ✅ Completed Changes

1. **Created `OverlayManifestBuilder.cs`**
   - Generates `overlay_manifest.json` for each version/map
   - Detects available overlays automatically
   - Provides tile coverage information
   - Future: Will drive plugin discovery in viewer

2. **Integrated with `ViewerReportWriter.cs`**
   - Calls `GenerateOverlayManifests()` after overlays generated
   - Detects terrain/shadow data presence
   - Writes manifests to correct locations

3. **Manifest Schema**
   - Version and map metadata
   - List of available overlays with plugin IDs
   - Tile patterns and coverage info
   - Tile bounds and coordinates

---

## Manifest Schema Example

```json
{
  "version": "0.5.3.3368",
  "map": "DeadminesInstance",
  "generatedAt": "2025-10-05T06:48:00Z",
  "overlays": [
    {
      "id": "objects.combined",
      "plugin": "objects",
      "title": "All Objects (M2 + WMO)",
      "enabled": true,
      "tilePattern": "combined/tile_r{row}_c{col}.json",
      "tileCoverage": "complete"
    },
    {
      "id": "objects.m2",
      "plugin": "objects",
      "title": "M2 Models Only",
      "enabled": false,
      "tilePattern": "m2/tile_r{row}_c{col}.json",
      "tileCoverage": "complete"
    },
    {
      "id": "objects.wmo",
      "plugin": "objects",
      "title": "WMO Objects Only",
      "enabled": false,
      "tilePattern": "wmo/tile_r{row}_c{col}.json",
      "tileCoverage": "complete"
    },
    {
      "id": "terrain.properties",
      "plugin": "terrain",
      "title": "Terrain Properties",
      "enabled": true,
      "tilePattern": "terrain_complete/tile_{col}_{row}.json",
      "tileCoverage": "sparse",
      "description": "Height maps, flags, liquids, area IDs"
    }
  ],
  "tiles": {
    "count": 42,
    "bounds": {
      "minRow": 15,
      "maxRow": 22,
      "minCol": 28,
      "maxCol": 35
    },
    "tiles": [
      { "row": 15, "col": 28 },
      { "row": 15, "col": 29 },
      ...
    ]
  }
}
```

---

## Validation Steps

### Step 1: Build the Solution ⏳

```powershell
cd WoWRollback
dotnet build WoWRollback.sln --configuration Release
```

**Expected Result**:
- ✅ Build succeeds with 0 errors
- ✅ `OverlayManifestBuilder.cs` compiles
- ✅ `ViewerReportWriter.cs` updated successfully

---

### Step 2: Run Rebuild Script ⏳

```powershell
.\rebuild-and-regenerate.ps1 -Maps DeadminesInstance -Versions "0.5.3.3368"
```

**Expected Console Output**:
```
...
[Phase 2] Generated overlay_manifest.json for DeadminesInstance (0.5.3.3368)
...
```

---

### Step 3: Verify Manifest Files Generated ⏳

```powershell
# Check manifest exists
$manifestPath = "rollback_outputs\comparisons\0.5.3.3368\viewer\overlays\0.5.3.3368\DeadminesInstance\overlay_manifest.json"
Test-Path $manifestPath
# Should output: True

# View manifest content
Get-Content $manifestPath | ConvertFrom-Json | ConvertTo-Json -Depth 10
```

**Expected Structure**:
- ✅ `version` field present
- ✅ `map` field present
- ✅ `generatedAt` timestamp
- ✅ `overlays` array with 3-5 entries
- ✅ `tiles` object with bounds and tile list

---

### Step 4: Validate Manifest Content ⏳

```powershell
$manifest = Get-Content $manifestPath | ConvertFrom-Json

# Check required overlays present
$manifest.overlays | ForEach-Object { $_.id }
# Should output:
# objects.combined
# objects.m2
# objects.wmo
# terrain.properties (if terrain CSV exists)
# shadow.overview (if shadow data exists)

# Check tile patterns correct
$manifest.overlays | Where-Object { $_.id -eq 'objects.combined' } | Select-Object -ExpandProperty tilePattern
# Should output: combined/tile_r{row}_c{col}.json

# Check tile count
$manifest.tiles.count
# Should match number of tiles for DeadminesInstance
```

---

### Step 5: Verify Manifest Accuracy ⏳

**Check object overlay files**:
```powershell
$overlayDir = "rollback_outputs\comparisons\0.5.3.3368\viewer\overlays\0.5.3.3368\DeadminesInstance"

# Count combined overlay files
$combinedFiles = Get-ChildItem "$overlayDir\combined" -Filter "*.json" | Measure-Object
Write-Host "Combined overlay files: $($combinedFiles.Count)"

# Compare with manifest tile count
$manifest.tiles.count
# Should match
```

**Check terrain overlay detection**:
```powershell
$terrainDir = "$overlayDir\terrain_complete"
$hasTerrainFiles = (Test-Path $terrainDir) -and ((Get-ChildItem $terrainDir -Filter "*.json").Count -gt 0)

$terrainOverlay = $manifest.overlays | Where-Object { $_.plugin -eq 'terrain' }
if ($hasTerrainFiles) {
    if ($terrainOverlay) {
        Write-Host "✅ Terrain overlay correctly detected"
    } else {
        Write-Host "❌ Terrain files exist but not in manifest"
    }
} else {
    if (!$terrainOverlay) {
        Write-Host "✅ No terrain overlay (correct - no data)"
    } else {
        Write-Host "❌ Terrain overlay in manifest but no files"
    }
}
```

---

### Step 6: Test Backward Compatibility ⏳

**Ensure viewer still works without reading manifests**:

```powershell
cd rollback_outputs\comparisons\0.5.3.3368\viewer
python -m http.server 8080
# Open: http://localhost:8080/index.html
```

**Manual Check**:
- ✅ Viewer loads normally
- ✅ Overlays work as before
- ✅ Manifest file ignored by current viewer
- ✅ No console errors
- ✅ No visual differences

---

## Success Criteria

### Must Pass All:

1. ✅ **Build Success**
   - Solution compiles without errors
   - No breaking changes to existing code

2. ✅ **Manifests Generated**
   - `overlay_manifest.json` created for each version/map
   - Files written to correct locations
   - Console shows "[Phase 2]" messages

3. ✅ **Manifest Content Valid**
   - JSON well-formed
   - Required fields present
   - Overlay list correct
   - Tile patterns match actual files

4. ✅ **Detection Accurate**
   - Objects overlay always present
   - Terrain overlay only if data exists
   - Shadow overlay only if data exists
   - Tile count matches actual files

5. ✅ **Backward Compatible**
   - Viewer works without reading manifests
   - No regressions
   - Zero visual differences

---

## Expected File Structure

After successful Phase 2:

```
rollback_outputs/comparisons/0.5.3.3368/viewer/
├── overlays/
│   └── 0.5.3.3368/
│       └── DeadminesInstance/
│           ├── overlay_manifest.json ← NEW in Phase 2
│           ├── combined/
│           │   ├── tile_r15_c28.json
│           │   └── ...
│           ├── m2/
│           │   └── ...
│           ├── wmo/
│           │   └── ...
│           └── terrain_complete/ (if exists)
│               └── ...
├── index.json
├── config.json
└── ... (other viewer files)
```

---

## Common Issues

### Issue 1: Manifest Not Generated

**Symptom**: No overlay_manifest.json files

**Cause**: Build failed or integration error

**Fix**:
```powershell
# Check for compile errors
dotnet build WoWRollback.sln --configuration Release

# Check console output for Phase 2 messages
.\rebuild-and-regenerate.ps1 -Maps DeadminesInstance | Select-String "Phase 2"
```

---

### Issue 2: Incorrect Tile Count

**Symptom**: Manifest tile count doesn't match files

**Cause**: Tile collection logic error

**Fix**:
```powershell
# Compare manually
$overlayDir = "rollback_outputs\comparisons\0.5.3.3368\viewer\overlays\0.5.3.3368\DeadminesInstance"
$actualCount = (Get-ChildItem "$overlayDir\combined" -Filter "*.json").Count
$manifestCount = (Get-Content "$overlayDir\overlay_manifest.json" | ConvertFrom-Json).tiles.count
Write-Host "Actual: $actualCount, Manifest: $manifestCount"
```

---

### Issue 3: Wrong Overlay Detection

**Symptom**: Terrain overlay listed but no files exist

**Cause**: Detection logic error

**Fix**: Check directory existence and file count logic in `GenerateOverlayManifests()`

---

## Post-Validation Actions

### If All Tests Pass ✅

1. **Update Phase 2 status to complete**
2. **Commit changes**:
   ```powershell
   git add WoWRollback\WoWRollback.Core\Services\Viewer\OverlayManifestBuilder.cs
   git add WoWRollback\WoWRollback.Core\Services\Viewer\ViewerReportWriter.cs
   git add docs\planning\09_Phase2_Validation.md
   git commit -m "Phase 2: Add backend manifest system"
   ```
3. **Proceed to Phase 3** (Runtime Plugin Architecture)

### If Tests Fail ❌

1. **Document failure details**
2. **Debug manifest generation**
3. **Fix and re-test**
4. **Do NOT proceed until passing**

---

## Validation Log

**Date**: _____________  
**Tester**: _____________  

| Step | Status | Notes |
|------|--------|-------|
| 1. Build Solution | ⏳ Pending | |
| 2. Run Rebuild Script | ⏳ Pending | |
| 3. Verify Manifests Exist | ⏳ Pending | |
| 4. Validate Content | ⏳ Pending | |
| 5. Check Accuracy | ⏳ Pending | |
| 6. Test Backward Compat | ⏳ Pending | |

**Overall Result**: ⏳ Pending

**Approver**: _____________  
**Date**: _____________

---

## Next Phase Preview

Once Phase 2 passes validation:

**Phase 3: Runtime Plugin Architecture**
- Create `PluginRuntime.cs` in viewer
- Load manifests in JavaScript
- No visual changes yet (prep work)
- Estimated: 1 week

---

**Phase 2 Status**: ⏳ Ready for Validation  
**Blocking**: Requires build + testing  
**Time Required**: ~15 minutes validation
