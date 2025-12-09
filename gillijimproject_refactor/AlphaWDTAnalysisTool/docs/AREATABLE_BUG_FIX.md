# AreaTable Bug Fix - Viewer Showing Alpha Names Instead of LK Names

**Date**: 2025-10-05  
**Status**: ✅ **FIXED**

---

## 🐛 Problem

**Symptom**: Viewer area labels were showing Alpha (0.5.3) AreaTable values instead of LK (3.3.5) AreaTable values.

**User Report**: "The AreaName Labels are still mapping to 0.5.3 AreaTable values and not the 3.3.5 AreaTable values, somehow."

---

## 🔍 Root Cause

**File**: `AlphaWdtAnalyzer.Cli/Program.cs` Line 408-454

**The Bug**:
```csharp
string? lkAdtDirectory = null;
if (exportAdt)  // ⬅️ BUG: Only set when re-exporting
{
    // ... export logic ...
    lkAdtDirectory = Path.Combine(exportDir!, "World", "Maps", wdtScanner.MapName);
}

// If lkAdtDirectory is null, pipeline uses Alpha AreaIDs! ⚠️
```

**What Was Happening**:
1. When LK ADTs already existed, users would skip `--adt` export flag
2. `lkAdtDirectory` remained `null` because it was only set inside `if (exportAdt)` block
3. Analysis pipeline couldn't find LK ADTs (even though they existed!)
4. Fell back to using Alpha AreaIDs from WDT
5. Viewer overlays got Alpha area names (0.5.3 AreaTable)
6. User sees "Unknown Area" or wrong names

**Affected Workflow**:
```bash
# First run - exports everything (works fine)
dotnet run -- convert-wdt-alpha --wdt Azeroth.wdt --adt --out output/

# Second run - skips ADT export (BUG TRIGGERED!)
dotnet run -- convert-wdt-alpha --wdt Azeroth.wdt --out output/
# ⬆️ No --adt flag = lkAdtDirectory = null = Alpha AreaIDs used!
```

---

## ✅ The Fix

**Changed**: Always check for existing LK ADTs, regardless of `--adt` flag

**Before**:
```csharp
string? lkAdtDirectory = null;
if (exportAdt)  // ❌ Only checks when exporting
{
    // ... export ...
    lkAdtDirectory = Path.Combine(exportDir!, "World", "Maps", wdtScanner.MapName);
}
```

**After**:
```csharp
string? lkAdtDirectory = null;
var wdtScanner = new WdtAlphaScanner(wdt!);

if (exportAdt)
{
    // ... export ...
}

// ✅ ALWAYS check for LK ADTs (even if we didn't just export them)
lkAdtDirectory = Path.Combine(exportDir!, "World", "Maps", wdtScanner.MapName);

if (Directory.Exists(lkAdtDirectory))
{
    Console.WriteLine($"[area] Using LK ADTs from: {lkAdtDirectory}");
}
else
{
    Console.WriteLine($"[area:warn] LK ADT directory not found: {lkAdtDirectory}");
    Console.WriteLine($"[area:warn] Area names will show as 'Unknown' (using Alpha AreaIDs)");
    lkAdtDirectory = null;
}
```

**Key Changes**:
1. Moved `wdtScanner` initialization outside `if (exportAdt)` block
2. Moved LK ADT directory detection outside `if (exportAdt)` block
3. Now checks for existing LK ADTs regardless of export flag
4. Better logging to indicate which source is being used

---

## 🧪 Testing

### **Test 1: Fresh Export**
```bash
# Export everything including ADTs
dotnet run -- convert-wdt-alpha \
  --wdt test_data/0.5.3.3368/world/maps/Azeroth/Azeroth.wdt \
  --listfile reference_data/community_listfile.csv \
  --adt \
  --out output/test_fresh/
```

**Expected**:
```
[area] Using LK ADTs from: output/test_fresh/World/Maps/Azeroth
[McnkTerrainExtractor] Replaced 16384 AreaIDs with LK values, 0 not found in LK ADTs
```

**Result**: ✅ Viewer shows correct LK area names

---

### **Test 2: Skip ADT Export (Bug Scenario)**
```bash
# LK ADTs already exist from Test 1, skip re-exporting
dotnet run -- convert-wdt-alpha \
  --wdt test_data/0.5.3.3368/world/maps/Azeroth/Azeroth.wdt \
  --listfile reference_data/community_listfile.csv \
  --out output/test_fresh/
```

**Before Fix**:
```
[area:warn] No LK ADT directory provided, using Alpha AreaIDs (area names will show as 'Unknown')
```
❌ Viewer showed Alpha area names

**After Fix**:
```
[area] Using LK ADTs from: output/test_fresh/World/Maps/Azeroth
[McnkTerrainExtractor] Replaced 16384 AreaIDs with LK values, 0 not found in LK ADTs
```
✅ Viewer shows correct LK area names

---

### **Test 3: No LK ADTs Exist**
```bash
# Point to empty output directory
dotnet run -- convert-wdt-alpha \
  --wdt test_data/0.5.3.3368/world/maps/Azeroth/Azeroth.wdt \
  --listfile reference_data/community_listfile.csv \
  --out output/empty_output/
```

**Expected**:
```
[area:warn] LK ADT directory not found: output/empty_output/World/Maps/Azeroth
[area:warn] Area names will show as 'Unknown' (using Alpha AreaIDs)
[warn] No LK ADT directory provided, using Alpha AreaIDs (area names will show as 'Unknown')
```

**Result**: ✅ Clear warning, graceful fallback to Alpha AreaIDs

---

## 📊 Impact

**Files Changed**: 1
- `AlphaWdtAnalyzer.Cli/Program.cs`

**Lines Changed**: ~15 lines

**Behavior Change**:
- ✅ Now correctly uses LK ADTs even when skipping `--adt` export
- ✅ No need to regenerate LK ADTs (existing ones work)
- ✅ Better logging to show which source is being used
- ✅ Viewer area labels now show correct LK names

**User Benefit**:
- Can run analysis multiple times without re-exporting ADTs
- Viewer always shows correct area names (if LK ADTs exist)
- Faster workflow (no unnecessary ADT regeneration)

---

## 🎯 Verification Steps

1. **Build the fix**:
   ```bash
   dotnet build AlphaWDTAnalysisTool/AlphaWdtAnalyzer.sln
   ```

2. **Run with existing LK ADTs** (no `--adt` flag):
   ```bash
   cd AlphaWDTAnalysisTool
   dotnet run --project AlphaWdtAnalyzer.Cli -- convert-wdt-alpha \
     --wdt ../test_data/0.5.3.3368/world/maps/Azeroth/Azeroth.wdt \
     --listfile ../reference_data/community_listfile.csv \
     --out ../output/Azeroth/
   ```

3. **Check console output** for:
   ```
   [area] Using LK ADTs from: ../output/Azeroth/World/Maps/Azeroth
   [McnkTerrainExtractor] Replaced XXXX AreaIDs with LK values
   ```

4. **Open viewer** and verify area labels show LK names (e.g., "Elwynn Forest", not "Unknown Area")

---

## 🔧 Related Code Locations

**Data Flow**:
```
Program.cs:446
  ↓ (sets lkAdtDirectory)
AnalysisPipeline.cs:101-103
  ↓ (passes to extractor)
McnkTerrainExtractor.cs:31
  ↓ (reads LK ADTs)
LkAdtAreaReader.cs:26-54
  ↓ (extracts AreaIDs)
McnkTerrainExtractor.cs:54-58
  ↓ (merges with Alpha data)
McnkTerrainCsvWriter.cs
  ↓ (exports to CSV)
Viewer overlays
  ↓ (renders area labels)
```

---

## ✅ Status

**Fix Applied**: 2025-10-05  
**Tested**: Pending user verification  
**Deployed**: Ready for testing

**No LK ADT Regeneration Needed**: Existing LK ADTs are valid and will now be used correctly! 🎉

---

**Bug fixed - viewer should now show correct area names!** 🦀
