# Critical Bug: Missing Terrain Overlays & Shadow Maps 🐛

**Priority**: CRITICAL - Blocks viewer functionality for most maps

---

## 🚨 Problems Identified

### Problem 1: Missing terrain_complete Files
**Symptom**: Only Azeroth and Kalimdor have terrain overlays; other maps (instances, battlegrounds) don't.

**Root Cause**: `GenerateTerrainOverlays()` silently skips maps without CSV files:

```csharp
// ViewerReportWriter.cs line 461
if (!File.Exists(terrainCsvPath))
{
    // CSV not found - terrain overlays were not extracted, skip silently
    return;  // ← SILENT FAILURE!
}
```

**Impact**: 
- Instances (Deadmines, Wailing Caverns, etc.) have no overlays
- Battlegrounds (Warsong Gulch, etc.) have no overlays
- Viewer loads but shows no terrain data

---

### Problem 2: Shadow Maps Don't Work
**Symptom**: Shadow map layer in viewer doesn't display anything.

**Root Cause** (likely):
1. Shadow CSV files may not be generated for all maps
2. Path resolution issues
3. Missing error logging

**Need to investigate**:
- Are shadow CSVs being created?
- Are shadow overlays being generated?
- Are paths correct in viewer?

---

## 🔍 Investigation Steps

### Step 1: Check What CSVs Exist
```powershell
# Check rollback_outputs structure
Get-ChildItem -Recurse rollback_outputs/0.5.3.3368/csv/ | 
    Where-Object { $_.Name -like "*_mcnk_*" } |
    Select-Object FullName
```

Expected structure:
```
rollback_outputs/
└── 0.5.3.3368/
    └── csv/
        ├── Azeroth/
        │   ├── Azeroth_mcnk_terrain.csv      ✅
        │   └── Azeroth_mcnk_shadow.csv       ✅
        ├── Kalimdor/
        │   ├── Kalimdor_mcnk_terrain.csv     ✅
        │   └── Kalimdor_mcnk_shadow.csv      ✅
        ├── DeadminesInstance/
        │   ├── DeadminesInstance_mcnk_terrain.csv  ❌ MISSING?
        │   └── DeadminesInstance_mcnk_shadow.csv   ❌ MISSING?
        └── WailingCavernsInstance/
            └── ...                            ❌ MISSING?
```

---

### Step 2: Check AlphaWDTAnalysisTool Output
**Question**: Does AlphaWDTAnalysisTool generate CSVs for all maps or only Azeroth/Kalimdor?

Check tool source:
```csharp
// AlphaWDTAnalysisTool - where is CSV generation?
// Is it hardcoded to only process certain maps?
```

---

### Step 3: Verify ViewerReportWriter Logic
Currently:
```csharp
foreach (var mapGroup in maps)  // ← All maps from placements
{
    // ...
    GenerateTerrainOverlays(...);  // ← Called for each map
}
```

**Issue**: If CSV doesn't exist, silently skips. Should log warning!

---

## 🔧 Fixes Required

### Fix 1: Add Logging for Missing CSVs
```csharp
// ViewerReportWriter.cs
if (!File.Exists(terrainCsvPath))
{
    Console.WriteLine($"[WARN] No terrain CSV for {mapName}, skipping overlays: {terrainCsvPath}");
    Console.WriteLine($"[INFO] To generate terrain data, run AlphaWDTAnalysisTool for this map");
    return;
}
```

### Fix 2: Verify CSV Generation in AlphaWDTAnalysisTool
**Check**: Is CSV generation map-specific or universal?

If hardcoded to Azeroth/Kalimdor:
```csharp
// WRONG:
if (mapName == "Azeroth" || mapName == "Kalimdor")
{
    GenerateTerrainCsv();
}

// RIGHT:
GenerateTerrainCsv();  // For ALL maps
```

### Fix 3: Shadow Map Investigation
Check `McnkShadowOverlayBuilder.cs`:
```csharp
public static void BuildOverlaysForMap(...)
{
    // Does this check for CSV existence?
    // Does it log errors?
    // Are paths correct?
}
```

Add logging:
```csharp
var shadowCsvPath = Path.Combine(csvDir, $"{mapName}_mcnk_shadow.csv");

if (!File.Exists(shadowCsvPath))
{
    Console.WriteLine($"[INFO] No shadow data for {mapName}, skipping shadow overlays");
    return;
}

Console.WriteLine($"[INFO] Generating shadow overlays for {mapName} from {shadowCsvPath}");
// ... generate overlays ...
Console.WriteLine($"[INFO] Generated {count} shadow overlay tiles for {mapName}");
```

---

## 🎯 Action Plan

### Priority 1: Identify Missing Data (TODAY)
- [ ] Run comparison on test dataset
- [ ] Check what CSV files actually exist
- [ ] List which maps have terrain CSVs
- [ ] List which maps have shadow CSVs

### Priority 2: Fix AlphaWDTAnalysisTool (IF NEEDED)
- [ ] Check if CSV generation is map-filtered
- [ ] Remove any hardcoded map filters
- [ ] Ensure all maps generate CSVs

### Priority 3: Add Logging
- [ ] Update `ViewerReportWriter.GenerateTerrainOverlays()` with warnings
- [ ] Update `McnkShadowOverlayBuilder.BuildOverlaysForMap()` with logging
- [ ] Update `McnkTerrainOverlayBuilder.BuildOverlaysForMap()` with logging

### Priority 4: Verify Shadow Maps
- [ ] Check shadow CSV format
- [ ] Verify shadow overlay generation
- [ ] Test shadow layer in viewer
- [ ] Check browser console for errors

---

## 📊 Diagnostic Commands

### Check Current State
```powershell
# List all maps with placements
Get-ChildItem rollback_outputs/0.5.3.3368/placements/*.csv | 
    ForEach-Object { $_.BaseName -replace '_placements$' }

# List all maps with terrain CSVs
Get-ChildItem rollback_outputs/0.5.3.3368/csv/*/\_mcnk_terrain.csv |
    ForEach-Object { (Split-Path (Split-Path $_)) | Split-Path -Leaf }

# Compare: which maps have placements but no terrain CSV?
```

### Test Overlay Generation
```csharp
// In WoWRollback
wowrollback compare-versions \
  --alpha-root ../test_data \
  --versions 0.5.3.3368 \
  --maps DeadminesInstance \
  --viewer-report

// Check output:
// - Does it generate terrain CSV?
// - Does it generate overlays?
// - Are there any warnings?
```

---

## 🔍 Expected Behavior After Fix

### Console Output Should Show:
```
[INFO] Processing DeadminesInstance...
[INFO] Generating terrain CSV for DeadminesInstance...
[INFO] Generated terrain CSV: DeadminesInstance_mcnk_terrain.csv (256 chunks)
[INFO] Generating shadow CSV for DeadminesInstance...
[INFO] Generated shadow CSV: DeadminesInstance_mcnk_shadow.csv (256 chunks)
[INFO] Generating terrain overlays for DeadminesInstance...
[INFO] Loaded 256 terrain chunks from DeadminesInstance_mcnk_terrain.csv
[INFO] Built 1 terrain overlay tiles for DeadminesInstance (0.5.3.3368)
[INFO] Generating shadow overlays for DeadminesInstance...
[INFO] Built 1 shadow overlay tiles for DeadminesInstance (0.5.3.3368)
```

### File Structure Should Have:
```
rollback_outputs/0.5.3.3368/
├── csv/
│   └── DeadminesInstance/
│       ├── DeadminesInstance_mcnk_terrain.csv    ✅
│       └── DeadminesInstance_mcnk_shadow.csv     ✅
└── viewer/
    └── overlays/
        └── 0.5.3.3368/
            └── DeadminesInstance/
                ├── terrain_complete/
                │   └── tile_32_18.json         ✅
                └── shadow_maps/
                    └── shadow_32_18.json       ✅
```

---

## 🚀 Quick Fix PR

### Files to Update:
1. `WoWRollback.Core/Services/Viewer/ViewerReportWriter.cs`
   - Add warning for missing terrain CSV
   - Add info logging for success

2. `WoWRollback.Core/Services/Viewer/McnkShadowOverlayBuilder.cs`
   - Add existence check
   - Add logging

3. `WoWRollback.Core/Services/Viewer/McnkTerrainOverlayBuilder.cs`
   - Add logging

4. `AlphaWDTAnalysisTool` (if needed)
   - Remove any map filters
   - Ensure universal CSV generation

---

**This blocks the AreaTable fix! Fix this first, then proceed with 01_CRITICAL_AreaTable_Fix.md** 🚨
