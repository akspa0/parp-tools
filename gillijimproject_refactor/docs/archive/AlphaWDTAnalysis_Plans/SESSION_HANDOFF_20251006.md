# Session Handoff - 2025-10-06

**Session Type**: Map.dbc Integration + Refactoring Planning  
**Status**: Partial Implementation - Needs Continuation  
**Next Session Goal**: Complete MapIdResolver integration + Begin Phase 1A fixes  

---

## What We Accomplished Today

### ✅ Completed
1. **MapDbcReader.cs** - Created new reader for Map.dbc
   - Location: `DBCTool.V2/Core/MapDbcReader.cs`
   - Reads Map.dbc → JSON with MapID, Directory, Name, InstanceType
   - Integrated into CompareAreaV2Command
   - Successfully generated `dbctool_out/0.5.3/maps.json`

2. **MapIdResolver.cs** - Created resolver for AlphaWdtAnalyzer
   - Location: `AlphaWDTAnalysisTool/AlphaWdtAnalyzer.Core/Dbc/MapIdResolver.cs`
   - Loads maps.json, provides MapID lookup by directory name
   - Validates map existence

3. **maps.json for 0.5.3** - Successfully generated
   - Location: `DBCTool.V2/dbctool_out/0.5.3/maps.json`
   - Contains 22 maps including:
     - Azeroth (ID 0)
     - Kalimdor (ID 1)
     - Shadowfang (ID 33) ← This was the test case
     - DeadminesInstance (ID 36)

4. **AdtExportPipeline partial update**
   - Added MapIdResolver loading logic in ExportSingle (lines 122-132)
   - Added MapIdResolver loading logic in ExportBatch (TODO: verify)
   - Updated ResolveMapIdFromDbc signature to accept MapIdResolver
   - Updated one call site (line 448 in ExportBatch)

5. **Planning Documents** - Created comprehensive refactoring plan
   - `docs/AlphaWDTAnalysis_Plans/001_output_normalization_and_stabilization.md`
   - `docs/AlphaWDTAnalysis_Plans/README.md`
   - `docs/AlphaWDTAnalysis_Plans/SESSION_HANDOFF_20251006.md` (this file)

### ⚠️ Partially Completed
1. **AdtExportPipeline.cs MapIdResolver integration**
   - Function signature updated ✅
   - MapIdResolver loading added ✅
   - **One call site NOT updated** ❌ (line 238 in ExportSingle)
   - Reason: Banned from editing after 3 failed attempts

### ❌ Not Started
1. WMO-only map graceful skip
2. Output structure normalization
3. Validation framework
4. Viewer fixes

---

## Critical Issue: Editing Ban

**File**: `AlphaWDTAnalysisTool/AlphaWdtAnalyzer.Core/Export/AdtExportPipeline.cs`  
**Line**: 238  
**Issue**: Tool is banned from editing this file after 3 consecutive failed edit attempts

**Required Manual Fix**:
```csharp
// Current (line 238):
int currentMapId = ResolveMapIdFromDbc(wdt.MapName, opts.DbctoolLkDir, opts.Verbose);

// Should be:
int currentMapId = ResolveMapIdFromDbc(wdt.MapName, mapIdResolver, opts.DbctoolLkDir, opts.Verbose);
```

**Context**: 
- The `mapIdResolver` variable is already created earlier in the function (line 123)
- The function `ResolveMapIdFromDbc` expects 4 parameters now, but the call only passes 3
- This causes a compilation error

**Verification After Fix**:
```bash
cd AlphaWDTAnalysisTool
dotnet build --configuration Release
# Should compile without errors
```

---

## Test Cases Ready

### Test 1: DeadminesInstance (Should Work)
```powershell
cd WoWRollback
.\rebuild-and-regenerate.ps1 `
  -Maps @("DeadminesInstance") `
  -Versions @("0.5.3.3368") `
  -AlphaRoot ..\test_data\ `
  -RefreshCache

# Expected:
# [MapIdResolver] Loaded 0.5.3 with 22 maps
# [MapId] Resolved 'DeadminesInstance' -> 36 from 0.5.3 maps.json
# [PatchMap] ... by-tgt-map[36]=... (should have entries)
# Terrain CSV: area_ids_patched > 0
```

### Test 2: Shadowfang (Currently Failing)
```powershell
.\rebuild-and-regenerate.ps1 `
  -Maps @("Shadowfang") `
  -Versions @("0.5.3.3368") `
  -AlphaRoot ..\test_data\ `
  -RefreshCache

# Currently:
# [McnkTerrainExtractor] Replaced 0 AreaIDs with LK values

# After fix, expected:
# [MapId] Resolved 'Shadowfang' -> 33 from 0.5.3 maps.json
# Crosswalk loaded: Area_patch_crosswalk_map33_0.5.3_to_335.csv
# [McnkTerrainExtractor] Replaced >0 AreaIDs with LK values
```

### Test 3: WMO-Only Map (Should Skip Gracefully)
```powershell
.\rebuild-and-regenerate.ps1 `
  -Maps @("MonasteryInstances") `
  -Versions @("0.5.3.3368")

# Currently: Crashes with "Attempted to read past end of stream"
# After fix: Should skip with message "WMO-only map (no ADT tiles)"
```

---

## Files Modified This Session

### New Files Created
1. `DBCTool.V2/Core/MapDbcReader.cs`
2. `AlphaWDTAnalysisTool/AlphaWdtAnalyzer.Core/Dbc/MapIdResolver.cs`
3. `docs/AlphaWDTAnalysis_Plans/001_output_normalization_and_stabilization.md`
4. `docs/AlphaWDTAnalysis_Plans/README.md`
5. `docs/AlphaWDTAnalysis_Plans/SESSION_HANDOFF_20251006.md`

### Files Modified
1. `DBCTool.V2/Cli/CompareAreaV2Command.cs`
   - Added Map.dbc metadata generation at end of Run() method
   
2. `AlphaWDTAnalysisTool/AlphaWdtAnalyzer.Core/Export/AdtExportPipeline.cs`
   - Added MapIdResolver? parameter to ResolveMapIdFromDbc()
   - Updated function body to use resolver first, fallback to LK Map.dbc
   - Added MapIdResolver loading in ExportSingle() (lines 122-132)
   - Updated one call site (line 448) ← needs verification
   - **INCOMPLETE**: Line 238 still needs manual update

3. `WoWRollback/rebuild-and-regenerate.ps1`
   - Fixed Get-DBCToolCrosswalkDir to return version-specific path
   - Added WMO-only map detection (partial - needs completion)

### Files That Need Changes (Next Session)
1. `AlphaWDTAnalysisTool/AlphaWdtAnalyzer.Core/Export/AdtExportPipeline.cs` (line 238 manual fix)
2. `WoWRollback/rebuild-and-regenerate.ps1` (complete WMO-only skip logic)

---

## Build Status

### DBCTool.V2
```
Status: ✅ Builds successfully
Last build: Release configuration
No errors, no warnings
Output: dbctool_out/0.5.3/maps.json created successfully
```

### AlphaWDTAnalysisTool
```
Status: ❌ Compilation error (expected)
Error: Line 238 - wrong number of arguments to ResolveMapIdFromDbc
Fix required: Add mapIdResolver parameter
```

### WoWRollback
```
Status: ⚠️ Runs with issues
- Processes some maps successfully (DeadminesInstance works)
- Crashes on WMO-only maps (MonasteryInstances)
- Shadowfang: 0 AreaIDs patched (MapID resolution issue due to compilation error)
```

---

## Environment Setup for Next Session

### Prerequisites
1. Visual Studio Code or IDE with C# support
2. .NET 9.0 SDK
3. PowerShell 7+
4. Test data extracted:
   - `test_data/0.5.3.3368/` with DBFilesClient and World/Maps

### Before Starting
```powershell
# 1. Navigate to project root
cd i:\parp-tools\pm4next-branch\parp-tools\gillijimproject_refactor

# 2. Verify git submodules (WoWDBDefs, alpha-core, playermap-flask)
git submodule status

# 3. Read the planning documents
code docs/AlphaWDTAnalysis_Plans/README.md
code docs/AlphaWDTAnalysis_Plans/001_output_normalization_and_stabilization.md

# 4. Review current state
Get-Content docs/AlphaWDTAnalysis_Plans/SESSION_HANDOFF_20251006.md
```

---

## Recommended Next Steps (In Order)

### Step 1: Fix Compilation Error (5 minutes)
**File**: `AlphaWDTAnalysisTool/AlphaWdtAnalyzer.Core/Export/AdtExportPipeline.cs`  
**Line**: 238  

```csharp
// Change this line:
int currentMapId = ResolveMapIdFromDbc(wdt.MapName, opts.DbctoolLkDir, opts.Verbose);

// To this:
int currentMapId = ResolveMapIdFromDbc(wdt.MapName, mapIdResolver, opts.DbctoolLkDir, opts.Verbose);
```

### Step 2: Verify Build (2 minutes)
```powershell
cd AlphaWDTAnalysisTool
dotnet build --configuration Release
# Should complete without errors
```

### Step 3: Test DeadminesInstance (10 minutes)
```powershell
cd ..\WoWRollback
.\rebuild-and-regenerate.ps1 `
  -Maps @("DeadminesInstance") `
  -Versions @("0.5.3.3368") `
  -AlphaRoot ..\test_data\ `
  -RefreshCache `
  -Verbose
```

**Success criteria**:
- [ ] Builds complete
- [ ] MapIdResolver loaded message appears
- [ ] MapId resolved: DeadminesInstance -> 36
- [ ] Crosswalk CSV loaded for map36
- [ ] AreaIDs patched > 0
- [ ] Viewer files generated

### Step 4: Test Shadowfang (10 minutes)
```powershell
.\rebuild-and-regenerate.ps1 `
  -Maps @("Shadowfang") `
  -Versions @("0.5.3.3368") `
  -AlphaRoot ..\test_data\ `
  -RefreshCache `
  -Verbose
```

**Success criteria**:
- [ ] MapId resolved: Shadowfang -> 33
- [ ] Crosswalk CSV loaded for map33
- [ ] AreaIDs patched > 0 (this is the key test!)

### Step 5: Add WMO-Only Map Skip (30 minutes)
**File**: `WoWRollback/rebuild-and-regenerate.ps1`  
**Function**: `Ensure-CachedMap`

Add graceful skip logic for maps with 0 tiles:
```powershell
if ($expectedTiles -eq 0) {
    Write-Host "  [cache] Skipping $Version/$Map - WMO-only map (no ADT tiles)" -ForegroundColor Yellow
    return $false
}
```

### Step 6: Test WMO-Only Map (5 minutes)
```powershell
.\rebuild-and-regenerate.ps1 `
  -Maps @("MonasteryInstances") `
  -Versions @("0.5.3.3368") `
  -AlphaRoot ..\test_data\
```

**Success criteria**:
- [ ] No crash
- [ ] Clear skip message
- [ ] Script continues (doesn't abort)

### Step 7: Begin Phase 1B (If time permits)
Start implementing new output structure from plan document.

---

## Known Gotchas

1. **Editing Ban**: The AI assistant is banned from editing `AdtExportPipeline.cs`. Manual edits required.

2. **maps.json Location**: Currently in `dbctool_out/0.5.3/maps.json`. MapIdResolver expects it there or in versioned subdirectory.

3. **Crosswalk CSV Naming**: Must be `Area_patch_crosswalk_map{ID}_0.5.3_to_335.csv` where {ID} matches MapID from maps.json.

4. **WMO-Only Detection**: Currently uses tile count from WDT. If 0, it's WMO-only. Need to check this happens early enough to prevent AlphaWdtAnalyzer from running.

5. **Cache Invalidation**: Current logic is unclear. May reuse stale ADTs even when DBCTool outputs change. This is a Phase 1C concern.

---

## Questions for Next Session

1. **Viewer Status**: What broke the Viewer? Need to investigate viewer generation code.

2. **ExportBatch**: Did line 448 update actually work? Need to verify both ExportSingle and ExportBatch paths.

3. **MapIdResolver Fallback**: Should we warn loudly if maps.json doesn't exist? Currently falls back to LK Map.dbc silently.

4. **Output Structure Timing**: Should we implement Phase 1B (new output structure) before or after fixing all the bugs?

---

## Session Statistics

- **Duration**: ~4 hours
- **Files Created**: 5
- **Files Modified**: 3
- **Bugs Fixed**: 1 (Get-DBCToolCrosswalkDir)
- **New Features**: 2 (MapDbcReader, MapIdResolver)
- **Bugs Introduced**: 1 (compilation error from incomplete integration)
- **Planning Docs**: 3

---

## Final State Checklist

- [x] Planning documents created
- [x] MapDbcReader implemented and working
- [x] MapIdResolver implemented
- [x] maps.json generated successfully
- [ ] AdtExportPipeline compilation fix (NEXT SESSION)
- [ ] WMO-only map skip (NEXT SESSION)
- [ ] Shadowfang AreaID patching verified (NEXT SESSION)
- [ ] Viewer fixed (FUTURE SESSION)
- [ ] Output structure normalized (FUTURE SESSION)

---

**Handoff Complete** - Ready for next session to continue with Step 1 (compilation fix).

**Good luck!** 🚀
