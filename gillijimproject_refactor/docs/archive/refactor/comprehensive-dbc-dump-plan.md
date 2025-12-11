# Comprehensive DBC Dump Plan (2025-10-08)

## Problem Statement

**Current State**:
- Only AreaTable.dbc extracted
- Output format: CSV
- Missing critical DBCs: Map.dbc, ItemDisplayInfo.dbc, Spell.dbc, etc.
- Can't resolve map names → folder mappings for viewer

**Required State**:
- ALL DBCs extracted
- Output format: JSON (exploratory, keep everything)
- Map.dbc specifically needed for map name resolution
- Future: Filter down once requirements are known

---

## Solution: Add Universal DBC Dumper

### Step 1: Create UniversalDbcDumper in DbcModule

**New File**: `WoWRollback.DbcModule/UniversalDbcDumper.cs`

```csharp
using System.Text.Json;
using DBCD;
using DBCD.IO;

namespace WoWRollback.DbcModule;

/// <summary>
/// Dumps ALL DBC files from a directory to JSON format for exploratory analysis.
/// </summary>
public sealed class UniversalDbcDumper
{
    private readonly string _dbdDir;
    private readonly string _locale;

    public UniversalDbcDumper(string dbdDir, string locale = "enUS")
    {
        _dbdDir = dbdDir ?? throw new ArgumentNullException(nameof(dbdDir));
        _locale = locale;
    }

    /// <summary>
    /// Dumps all .dbc files from source directory to JSON.
    /// </summary>
    public DumpAllDbcsResult DumpAll(
        string buildVersion,
        string sourceDbcDir,
        string outputDir)
    {
        try
        {
            if (!Directory.Exists(sourceDbcDir))
            {
                return new DumpAllDbcsResult(
                    Success: false,
                    ErrorMessage: $"Source directory not found: {sourceDbcDir}",
                    DumpedFiles: Array.Empty<string>());
            }

            Directory.CreateDirectory(outputDir);

            var dbcFiles = Directory.GetFiles(sourceDbcDir, "*.dbc", SearchOption.TopDirectoryOnly);
            var dumpedFiles = new List<string>();
            var errors = new List<string>();

            var dbcd = new DBCD.DBCD(new DBCProvider(sourceDbcDir), new DBDProvider(_dbdDir));

            foreach (var dbcPath in dbcFiles)
            {
                var dbcName = Path.GetFileNameWithoutExtension(dbcPath);
                
                try
                {
                    // Load DBC using DBCD
                    var storage = dbcd.Load(dbcName, buildVersion, _locale);
                    
                    // Convert to list of dictionaries for JSON serialization
                    var records = new List<Dictionary<string, object>>();
                    
                    foreach (var row in storage.Values)
                    {
                        var record = new Dictionary<string, object>();
                        
                        // Use reflection to get all fields
                        var type = row.GetType();
                        foreach (var prop in type.GetProperties())
                        {
                            var value = prop.GetValue(row);
                            record[prop.Name] = value ?? DBNull.Value;
                        }
                        
                        records.Add(record);
                    }

                    // Write to JSON
                    var jsonPath = Path.Combine(outputDir, $"{dbcName}_{buildVersion.Replace('.', '_')}.json");
                    var options = new JsonSerializerOptions 
                    { 
                        WriteIndented = true,
                        DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.Never
                    };
                    
                    File.WriteAllText(jsonPath, JsonSerializer.Serialize(new
                    {
                        dbc = dbcName,
                        build = buildVersion,
                        recordCount = records.Count,
                        records = records
                    }, options));

                    dumpedFiles.Add(jsonPath);
                }
                catch (Exception ex)
                {
                    errors.Add($"{dbcName}: {ex.Message}");
                    // Continue with other DBCs even if one fails
                }
            }

            if (dumpedFiles.Count == 0)
            {
                return new DumpAllDbcsResult(
                    Success: false,
                    ErrorMessage: $"No DBCs successfully dumped. Errors: {string.Join("; ", errors)}",
                    DumpedFiles: Array.Empty<string>());
            }

            var errorSummary = errors.Count > 0 
                ? $" ({errors.Count} files failed: {string.Join(", ", errors.Take(5))})"
                : string.Empty;

            return new DumpAllDbcsResult(
                Success: true,
                ErrorMessage: errors.Count > 0 ? errorSummary : null,
                DumpedFiles: dumpedFiles.ToArray());
        }
        catch (Exception ex)
        {
            return new DumpAllDbcsResult(
                Success: false,
                ErrorMessage: $"DBC dump failed: {ex.Message}",
                DumpedFiles: Array.Empty<string>());
        }
    }
}

public sealed record DumpAllDbcsResult(
    bool Success,
    string? ErrorMessage,
    IReadOnlyList<string> DumpedFiles);
```

### Step 2: Add Method to DbcOrchestrator

**File**: `WoWRollback.DbcModule/DbcOrchestrator.cs`

Add after `GenerateCrosswalks`:

```csharp
/// <summary>
/// Dumps ALL DBC files to JSON for comprehensive data access.
/// </summary>
public DumpAllDbcsResult DumpAllDbcs(
    string buildVersion,
    string sourceDbcDir,
    string outputDir)
{
    var dumper = new UniversalDbcDumper(_dbdDir, _locale);
    return dumper.DumpAll(buildVersion, sourceDbcDir, outputDir);
}
```

### Step 3: Update DbcStageRunner

**File**: `WoWRollback.Orchestrator/DbcStageRunner.cs`

Replace lines 88-113 with:

```csharp
// Dump ALL DBCs to JSON (comprehensive exploratory dump)
var jsonDumpDir = Path.Combine(dbcVersionDir, "json");
var dumpAllResult = orchestrator.DumpAllDbcs(alias, sourceDir, jsonDumpDir);

if (!dumpAllResult.Success)
{
    success = false;
    error = dumpAllResult.ErrorMessage ?? "Comprehensive DBC dump failed";
}
else
{
    ConsoleLogger.Success($"  ✓ Dumped {dumpAllResult.DumpedFiles.Count} DBCs to JSON");
    
    // Legacy: Also dump AreaTable to CSV for crosswalk compatibility
    var areaTableDumpResult = orchestrator.DumpAreaTables(
        srcAlias: alias,
        srcDbcDir: sourceDir,
        tgtDbcDir: lkDbcDir,
        outDir: dbcVersionDir);

    if (!areaTableDumpResult.Success)
    {
        success = false;
        error = areaTableDumpResult.ErrorMessage ?? "AreaTable dump failed";
    }
    else
    {
        // Copy CSVs to expected location
        var rawSource = Path.Combine(dbcVersionDir, alias, "raw");
        if (Directory.Exists(rawSource))
        {
            foreach (var file in Directory.EnumerateFiles(rawSource, "*.csv"))
            {
                var fileName = Path.GetFileName(file);
                File.Copy(file, Path.Combine(sharedDbcDir, fileName), overwrite: true);
            }
        }
    }
}
```

---

## Output Structure

```
parp_out/session_XXXXXX/
├── 01_dbcs/
│   └── 0.5.3/
│       ├── raw/                   # Legacy CSVs (for crosswalks)
│       │   ├── AreaTable_0_5_3.csv
│       │   └── AreaTable_3_3_5.csv
│       └── json/                  # NEW: Complete JSON dumps
│           ├── AreaTable_0_5_3.json
│           ├── Map_0_5_3.json     ← Critical for map name resolution
│           ├── ItemDisplayInfo_0_5_3.json
│           ├── Spell_0_5_3.json
│           └── ... (ALL DBCs)
```

---

## Map.dbc Example Output

```json
{
  "dbc": "Map",
  "build": "0.5.3",
  "recordCount": 5,
  "records": [
    {
      "ID": 0,
      "Directory": "Azeroth",
      "InstanceType": 0,
      "Flags": 0,
      "MapName_Lang": "Eastern Kingdoms",
      "AreaTableID": 0,
      ...
    },
    {
      "ID": 1,
      "Directory": "Kalimdor",
      "InstanceType": 0,
      "Flags": 0,
      "MapName_Lang": "Kalimdor",
      "AreaTableID": 0,
      ...
    },
    {
      "ID": 33,
      "Directory": "Shadowfang",
      "InstanceType": 1,
      "Flags": 0,
      "MapName_Lang": "Shadowfang Keep",
      "AreaTableID": 0,
      ...
    }
  ]
}
```

**Usage in viewer**:
```javascript
// Load Map.dbc JSON
const mapData = await fetch('01_dbcs/0.5.3/json/Map_0_5_3.json');
const maps = await mapData.json();

// Resolve "Shadowfang" → ID 33 → "Shadowfang Keep"
const shadowfangRecord = maps.records.find(r => r.Directory === "Shadowfang");
console.log(shadowfangRecord.MapName_Lang); // "Shadowfang Keep"
```

---

## Benefits

1. **Complete Data Access** ✅
   - All DBCs available in JSON
   - No need to re-decode later
   - Can explore what data is actually needed

2. **Map Name Resolution** ✅
   - Map.dbc provides Directory → MapName mapping
   - Viewer can show proper map names
   - Fixes the fundamental missing piece

3. **Future-Proof** ✅
   - Keep everything now
   - Filter down once requirements known
   - JSON is human-readable for exploration

4. **Backward Compatible** ✅
   - Still generates CSV for crosswalks
   - Existing pipeline continues to work
   - Additive change, not breaking

---

## Implementation Order

1. **Create UniversalDbcDumper.cs** (30 min)
2. **Add DumpAllDbcs to DbcOrchestrator** (5 min)
3. **Update DbcStageRunner** (15 min)
4. **Test with sample data** (15 min)
5. **Verify JSON outputs** (10 min)

**Total**: ~1 hour 15 minutes

---

## Testing

```powershell
# Run pipeline with comprehensive DBC dump
dotnet run --project WoWRollback.Orchestrator -- \
  --maps Shadowfang \
  --versions 0.5.3 \
  --alpha-root ..\test_data \
  --lk-dbc-dir ..\test_data\3.3.5\tree\DBFilesClient

# Verify outputs
ls parp_out\session_*\01_dbcs\0.5.3\json\*.json
# Should show: AreaTable_0_5_3.json, Map_0_5_3.json, etc.

# Check Map.dbc
cat parp_out\session_*\01_dbcs\0.5.3\json\Map_0_5_3.json | jq '.records[] | {ID, Directory, MapName_Lang}'
```

---

## Next Steps After This

Once DBCs are dumped:
1. Fix AnalysisOrchestrator path bug (analysisIndex location)
2. Add minimap PNG generation (ViewerStageRunner)
3. Fix index.json format (viewer expectations)

**Status**: Ready to implement - comprehensive DBC dump first, then fix viewer pipeline
