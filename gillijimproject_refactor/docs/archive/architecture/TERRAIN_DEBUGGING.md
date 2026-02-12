# Terrain Overlay Generation - Debugging Guide

## Current Issue

Browser shows 404 errors for terrain_complete overlay files:
```
GET /overlays/0.5.3.3368/Azeroth/terrain_complete/tile_r32_c38.json HTTP/1.1" 404
```

## Investigation Results

### ✅ What's Working

1. **AlphaWdtAnalyzer has terrain extraction code** - `McnkTerrainExtractor.cs` exists
2. **rebuild-and-regenerate.ps1 passes `--extract-mcnk-terrain` flag** - Line 204
3. **Converted ADTs are being generated** - `cached_maps/{version}/World/Maps/{map}/` contains .adt files
4. **Path fix was applied** - Script now looks in correct subdirectory for CSV

### ❌ What's NOT Working

1. **Terrain CSV files are NOT being created**
   - Expected: `cached_maps/analysis/{version}/{map}/csv/{map}/{map}_mcnk_terrain.csv`
   - Actual: Directory doesn't exist

2. **AlphaWdtAnalyzer may be failing silently**
   - Exit code 0 (success) but no terrain CSV produced
   - Could be hitting an exception that's being caught

---

## Root Cause Analysis

### Hypothesis 1: AnalysisPipeline Not Running
**Likely Cause**: When `--export-adt` is specified, AlphaWdtAnalyzer runs TWO separate pipelines:
1. `AnalysisPipeline.Run()` - Extracts terrain CSV (should happen)
2. `AdtExportPipeline.ExportSingle()` - Converts ADTs to LK format (is happening)

**Problem**: Pipeline #1 might not be running, or failing silently.

### Hypothesis 2: Path Mismatch in AnalysisPipeline
**Code Reference**: `AlphaWdtAnalyzer.Core/AnalysisPipeline.cs` line 94
```csharp
var terrainCsvPath = Path.Combine(csvDir, wdt.MapName, $"{wdt.MapName}_mcnk_terrain.csv");
```

**Expected Output**: `{outDir}/csv/{MapName}/{MapName}_mcnk_terrain.csv`
**Actual**: File never created

### Hypothesis 3: Exception Being Swallowed
**Code Reference**: `AnalysisPipeline.cs` lines 91-96
```csharp
if (opts.ExtractMcnkTerrain)
{
    var terrainEntries = McnkTerrainExtractor.ExtractTerrain(wdt);
    var terrainCsvPath = Path.Combine(csvDir, wdt.MapName, $"{wdt.MapName}_mcnk_terrain.csv");
    McnkTerrainCsvWriter.WriteCsv(terrainEntries, terrainCsvPath);
}
```

No try/catch, so exceptions should propagate. But parent code might be catching.

---

## Debugging Added

### rebuild-and-regenerate.ps1 (Lines 210-224)

Added diagnostic output:
```powershell
Write-Host "  [debug] Running AlphaWdtAnalyzer with --extract-mcnk-terrain"
Write-Host "  [debug] Analysis output dir: $analysisDir"
& dotnet @toolArgs

# Verify terrain CSV was created
$expectedTerrainCsv = Join-Path (Join-Path (Join-Path $analysisDir 'csv') $Map) ($Map + '_mcnk_terrain.csv')
if (Test-Path $expectedTerrainCsv) {
    Write-Host "  [debug] ✓ Terrain CSV created: $expectedTerrainCsv"
} else {
    Write-Host "  [warn] ✗ Terrain CSV NOT created at: $expectedTerrainCsv"
    Write-Host "  [warn] Check AlphaWdtAnalyzer output above for errors"
}
```

---

## Testing Steps

### 1. Clean Previous Runs
```powershell
cd WoWRollback
Remove-Item cached_maps -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item rollback_outputs -Recurse -Force -ErrorAction SilentlyContinue
```

### 2. Run with Debugging
```powershell
.\rebuild-and-regenerate.ps1 `
  -Maps @("DeadminesInstance") `
  -Versions @("0.5.3.3368") `
  -AlphaRoot ..\test_data\
```

Use DeadminesInstance (small map) for faster testing.

### 3. Expected Debug Output

**If working**:
```
[cache] Building LK ADTs for 0.5.3.3368/DeadminesInstance
[debug] Running AlphaWdtAnalyzer with --extract-mcnk-terrain
[debug] Analysis output dir: I:\...\cached_maps\analysis\0.5.3.3368\DeadminesInstance
<AlphaWdtAnalyzer output>
[debug] ✓ Terrain CSV created: I:\...\cached_maps\analysis\0.5.3.3368\DeadminesInstance\csv\DeadminesInstance\DeadminesInstance_mcnk_terrain.csv
[cache] Copied terrain CSV to rollback_outputs
```

**If failing**:
```
[cache] Building LK ADTs for 0.5.3.3368/DeadminesInstance
[debug] Running AlphaWdtAnalyzer with --extract-mcnk-terrain
[debug] Analysis output dir: I:\...\cached_maps\analysis\0.5.3.3368\DeadminesInstance
<AlphaWdtAnalyzer output>
[warn] ✗ Terrain CSV NOT created at: I:\...\cached_maps\analysis\0.5.3.3368\DeadminesInstance\csv\DeadminesInstance\DeadminesInstance_mcnk_terrain.csv
[warn] Check AlphaWdtAnalyzer output above for errors
```

### 4. Check AlphaWdtAnalyzer Output

Look for errors in the console output between the two debug lines. Specifically:
- Exception messages
- "Failed to extract terrain" or similar
- Path-related errors
- Missing WDT/ADT file errors

---

## Additional Debugging (If Needed)

### Option 1: Run AlphaWdtAnalyzer Directly

```powershell
cd ..\AlphaWDTAnalysisTool

dotnet run --project AlphaWdtAnalyzer.Cli --configuration Release -- `
  --input ..\test_data\0.5.3\tree\World\Maps\DeadminesInstance\DeadminesInstance.wdt `
  --listfile ..\_Reference_Data\community-listfile.txt `
  --lk-listfile ..\_Reference_Data\lk-listfile.txt `
  --out test_terrain_output `
  --extract-mcnk-terrain `
  --no-web
```

**Check**: Does `test_terrain_output/csv/DeadminesInstance/DeadminesInstance_mcnk_terrain.csv` exist?

### Option 2: Add Logging to AlphaWdtAnalyzer

Edit `AlphaWdtAnalyzer.Core/AnalysisPipeline.cs` line 91:
```csharp
// Extract MCNK terrain data if requested
if (opts.ExtractMcnkTerrain)
{
    Console.WriteLine($"[terrain] Extracting MCNK terrain for {wdt.MapName}");
    var terrainEntries = McnkTerrainExtractor.ExtractTerrain(wdt);
    Console.WriteLine($"[terrain] Extracted {terrainEntries.Count} terrain entries");
    
    var terrainCsvPath = Path.Combine(csvDir, wdt.MapName, $"{wdt.MapName}_mcnk_terrain.csv");
    Console.WriteLine($"[terrain] Writing to: {terrainCsvPath}");
    
    McnkTerrainCsvWriter.WriteCsv(terrainEntries, terrainCsvPath);
    Console.WriteLine($"[terrain] ✓ Terrain CSV written successfully");
}
```

Rebuild and retest.

### Option 3: Check if WDT is Being Loaded

The issue might be that `wdt` object in AnalysisPipeline is null or invalid. Check Program.cs to see how WDT is passed to AnalysisPipeline.

---

## Possible Fixes

### Fix 1: Ensure AnalysisPipeline Runs First

In `AlphaWdtAnalyzer.Cli/Program.cs`, verify that `AnalysisPipeline.Run()` is called BEFORE `AdtExportPipeline.ExportSingle()` when `--export-adt` is used.

### Fix 2: Create Directory Before Writing

`McnkTerrainCsvWriter.WriteCsv()` might not create parent directories. Update it to:
```csharp
public static void WriteCsv(List<McnkTerrainEntry> entries, string path)
{
    var dir = Path.GetDirectoryName(path);
    if (!string.IsNullOrEmpty(dir))
    {
        Directory.CreateDirectory(dir);
    }
    
    // ... existing CSV writing code
}
```

### Fix 3: Verify WDT Has Terrain Data

Add null/empty check in `McnkTerrainExtractor.ExtractTerrain()`:
```csharp
public static List<McnkTerrainEntry> ExtractTerrain(WdtAlphaScanner wdt)
{
    if (wdt == null || wdt.AdtPaths == null || wdt.AdtPaths.Count == 0)
    {
        Console.WriteLine("[warn] No ADT paths found in WDT, skipping terrain extraction");
        return new List<McnkTerrainEntry>();
    }
    
    // ... existing extraction code
}
```

---

## Summary

**Current Status**: ❌ Terrain CSV files not being generated  
**Debugging Added**: ✅ rebuild-and-regenerate.ps1 now reports CSV creation status  
**Next Step**: Run rebuild script and check debug output

**Commands to Run**:
```powershell
cd WoWRollback
Remove-Item cached_maps, rollback_outputs -Recurse -Force -ErrorAction SilentlyContinue

.\rebuild-and-regenerate.ps1 `
  -Maps @("DeadminesInstance") `
  -Versions @("0.5.3.3368") `
  -AlphaRoot ..\test_data\
```

**What to Look For**:
1. Does it say "✓ Terrain CSV created" or "✗ Terrain CSV NOT created"?
2. Are there any errors in AlphaWdtAnalyzer output?
3. Does the expected path actually exist after the script completes?

---

## Related Files

- `AlphaWdtAnalyzer.Core/AnalysisPipeline.cs` - Terrain extraction orchestration
- `AlphaWdtAnalyzer.Core/Terrain/McnkTerrainExtractor.cs` - MCNK parsing
- `AlphaWdtAnalyzer.Core/Terrain/McnkTerrainCsvWriter.cs` - CSV writing
- `AlphaWdtAnalyzer.Cli/Program.cs` - CLI entry point
- `rebuild-and-regenerate.ps1` - Build orchestration

Ready for user testing! 🔍
