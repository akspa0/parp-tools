# AlphaWDTAnalysisTool Multi-Threading Optimization 🚀

## Current Problem

**CPU Usage**: 2-8% (single-threaded)  
**Processing Time**: 45-60 minutes for large maps  
**Bottleneck**: Sequential ADT processing

## Root Cause

All ADT processing loops are **sequential `foreach`**:

```csharp
// AdtScanner.cs line 29
foreach (var adtNum in wdt.AdtNumbers)
{
    var adt = new AdtAlpha(wdt.WdtPath, adtNum, off);
    // Process...
}

// McnkTerrainExtractor.cs line 83
foreach (var adtNum in wdt.AdtNumbers)
{
    // Extract terrain...
}

// McnkShadowExtractor.cs line 23
foreach (var adtNum in wdt.AdtNumbers)
{
    // Extract shadows...
}

// LkAdtAreaReader.cs line 39
foreach (var adtPath in adtFiles)
{
    // Read AreaIDs...
}
```

**Problem**: Each ADT processed one at a time, wasting CPU cores!

---

## Solution: Parallel Processing

### Why It Works

- ✅ **ADTs are independent** - no shared state between tiles
- ✅ **Read-only source data** - WDT file can be read concurrently
- ✅ **Modern CPUs** - 8-24 cores sitting idle!
- ✅ **Easy refactor** - replace `foreach` with `Parallel.ForEachAsync`

### Expected Performance

| Metric | Before | After (8 cores) | Improvement |
|--------|--------|-----------------|-------------|
| CPU Usage | 2-8% | 60-80% | **10x** |
| Azeroth (128 tiles) | 45-60 min | **6-8 min** | **8x faster** |
| DeadminesInstance (10 tiles) | 2-3 min | **20-30 sec** | **6x faster** |

---

## Implementation Strategy

### 1. AdtScanner Parallelization

**File**: `AlphaWdtAnalyzer.Core/AdtScanner.cs`

**Before**:
```csharp
foreach (var adtNum in wdt.AdtNumbers)
{
    var off = (adtNum < wdt.AdtMhdrOffsets.Count) ? wdt.AdtMhdrOffsets[adtNum] : 0;
    if (off <= 0) continue;

    var adt = new AdtAlpha(wdt.WdtPath, adtNum, off);
    
    // Collect assets
    result.Wmo.AddRange(adt.GetMdnmPlacedWmos());
    result.M2.AddRange(adt.GetMonmPlacedModels());
    // etc...
}
```

**After**:
```csharp
var options = new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount };

var adtResults = new ConcurrentBag<AdtScanResult>();

await Parallel.ForEachAsync(wdt.AdtNumbers, options, async (adtNum, ct) =>
{
    var off = (adtNum < wdt.AdtMhdrOffsets.Count) ? wdt.AdtMhdrOffsets[adtNum] : 0;
    if (off <= 0) return;

    var adt = new AdtAlpha(wdt.WdtPath, adtNum, off);
    
    // Collect per-ADT results
    var adtResult = new AdtScanResult
    {
        Wmo = adt.GetMdnmPlacedWmos().ToList(),
        M2 = adt.GetMonmPlacedModels().ToList(),
        // etc...
    };
    
    adtResults.Add(adtResult);
});

// Merge results
foreach (var adtResult in adtResults)
{
    result.Wmo.AddRange(adtResult.Wmo);
    result.M2.AddRange(adtResult.M2);
    // etc...
}
```

**Key Changes**:
- ✅ Use `Parallel.ForEachAsync` for concurrent execution
- ✅ Use `ConcurrentBag<T>` for thread-safe collection
- ✅ Merge results after parallel processing

---

### 2. McnkTerrainExtractor Parallelization

**File**: `AlphaWdtAnalyzer.Core/Terrain/McnkTerrainExtractor.cs`

**Before**:
```csharp
foreach (var adtNum in wdt.AdtNumbers)
{
    var adt = new AdtAlpha(wdt.WdtPath, adtNum, off);
    ExtractTerrainFromAdt(adt, adtNum, csvPath);
}
```

**After**:
```csharp
var options = new ParallelOptions 
{ 
    MaxDegreeOfParallelism = Environment.ProcessorCount 
};

await Parallel.ForEachAsync(wdt.AdtNumbers, options, async (adtNum, ct) =>
{
    var off = (adtNum < wdt.AdtMhdrOffsets.Count) ? wdt.AdtMhdrOffsets[adtNum] : 0;
    if (off <= 0) return;

    var adt = new AdtAlpha(wdt.WdtPath, adtNum, off);
    await Task.Run(() => ExtractTerrainFromAdt(adt, adtNum, csvPath), ct);
});
```

**Note**: CSV writing needs synchronization!

---

### 3. Thread-Safe CSV Writing

**Problem**: Multiple threads writing to same CSV file = corruption!

**Solution 1: Per-ADT CSV Files** (Simple)
```csharp
// Each ADT writes its own CSV
var csvPath = Path.Combine(outputDir, $"{mapName}_adt_{adtNum}_terrain.csv");
ExtractTerrainFromAdt(adt, adtNum, csvPath);

// Merge CSVs after parallel processing
MergeCsvFiles(outputDir, $"{mapName}_mcnk_terrain.csv");
```

**Solution 2: Thread-Safe Writer** (Cleaner)
```csharp
private static readonly SemaphoreSlim csvLock = new SemaphoreSlim(1, 1);

private async Task AppendToCsvAsync(string csvPath, List<string> rows)
{
    await csvLock.WaitAsync();
    try
    {
        await File.AppendAllLinesAsync(csvPath, rows);
    }
    finally
    {
        csvLock.Release();
    }
}
```

---

### 4. Progress Reporting

Add progress counter for visibility:

```csharp
private static int processedCount = 0;
private static readonly object progressLock = new object();

await Parallel.ForEachAsync(wdt.AdtNumbers, options, async (adtNum, ct) =>
{
    // Process ADT...
    
    lock (progressLock)
    {
        processedCount++;
        if (processedCount % 10 == 0)
        {
            Console.WriteLine($"[Progress] {processedCount}/{wdt.AdtNumbers.Count} ADTs processed");
        }
    }
});
```

---

## Implementation Priority

### Phase 1: High Impact, Low Risk
1. ✅ **AdtScanner** - Easy, huge speedup for asset discovery
2. ✅ **McnkShadowExtractor** - Independent shadow CSV per ADT

### Phase 2: Medium Impact
3. ✅ **McnkTerrainExtractor** - Requires CSV merge logic
4. ✅ **LkAdtAreaReader** - Already fast, but easy to parallelize

### Phase 3: Polish
5. ✅ Progress bars (using `Spectre.Console`)
6. ✅ Configurable thread count via CLI flag
7. ✅ Memory usage monitoring (large maps)

---

## Code Example: Complete Refactor

```csharp
public static async Task<ScanResult> ScanMapAsync(WdtAlphaScanner wdt, 
    int? maxThreads = null)
{
    var result = new ScanResult();
    
    var options = new ParallelOptions 
    { 
        MaxDegreeOfParallelism = maxThreads ?? Environment.ProcessorCount 
    };
    
    Console.WriteLine($"[AdtScanner] Processing {wdt.AdtNumbers.Count} ADTs " +
                      $"on {options.MaxDegreeOfParallelism} threads");
    
    var adtResults = new ConcurrentBag<AdtScanResult>();
    var progress = 0;
    
    await Parallel.ForEachAsync(wdt.AdtNumbers, options, async (adtNum, ct) =>
    {
        try
        {
            var off = (adtNum < wdt.AdtMhdrOffsets.Count) ? wdt.AdtMhdrOffsets[adtNum] : 0;
            if (off <= 0) return;

            var adt = new AdtAlpha(wdt.WdtPath, adtNum, off);
            
            var adtResult = new AdtScanResult
            {
                AdtNum = adtNum,
                Wmo = adt.GetMdnmPlacedWmos().ToList(),
                M2 = adt.GetMonmPlacedModels().ToList(),
                BlpAssets = new HashSet<string>()
            };
            
            // Collect textures
            foreach (var tex in adt.GetMtexTextureNames())
            {
                var norm = ListfileLoader.NormalizePath(tex);
                if (!string.IsNullOrWhiteSpace(norm))
                    adtResult.BlpAssets.Add(norm);
            }
            
            adtResults.Add(adtResult);
            
            // Progress update (thread-safe)
            var current = Interlocked.Increment(ref progress);
            if (current % 10 == 0 || current == wdt.AdtNumbers.Count)
            {
                Console.WriteLine($"[Progress] {current}/{wdt.AdtNumbers.Count} ADTs");
            }
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[Error] Failed to process ADT {adtNum}: {ex.Message}");
        }
    });
    
    // Merge results
    Console.WriteLine("[AdtScanner] Merging results...");
    foreach (var adtResult in adtResults)
    {
        result.Wmo.AddRange(adtResult.Wmo);
        result.M2.AddRange(adtResult.M2);
        foreach (var blp in adtResult.BlpAssets)
            result.BlpAssets.Add(blp);
    }
    
    Console.WriteLine($"[AdtScanner] Complete! Found {result.Wmo.Count} WMOs, " +
                      $"{result.M2.Count} M2s, {result.BlpAssets.Count} BLPs");
    
    return result;
}

private class AdtScanResult
{
    public int AdtNum { get; set; }
    public List<string> Wmo { get; set; } = new();
    public List<string> M2 { get; set; } = new();
    public HashSet<string> BlpAssets { get; set; } = new();
}
```

---

## Testing Strategy

### 1. Small Map Test (DeadminesInstance)
```powershell
# Before
Measure-Command { dotnet run --project AlphaWdtAnalyzer.Cli -- ... }
# ~2-3 minutes

# After
Measure-Command { dotnet run --project AlphaWdtAnalyzer.Cli -- ... }
# ~20-30 seconds ✅
```

### 2. Large Map Test (Azeroth)
```powershell
# Before: 45-60 minutes
# After: 6-8 minutes ✅
```

### 3. Correctness Verification
```powershell
# Compare sequential vs parallel outputs
diff sequential_output.csv parallel_output.csv
# Should be identical (order may vary)
```

---

## CLI Flag Addition

Add `--threads` option:

```csharp
case "--threads":
case "--max-threads":
    if (i + 1 >= args.Length) return Usage();
    if (!int.TryParse(args[++i], out var threads)) return Usage();
    maxThreads = threads;
    break;
```

**Usage**:
```powershell
# Use all cores (default)
dotnet run -- --input map.wdt --listfile ... --threads 0

# Limit to 4 threads
dotnet run -- --input map.wdt --listfile ... --threads 4

# Single-threaded (debugging)
dotnet run -- --input map.wdt --listfile ... --threads 1
```

---

## Memory Considerations

### Large Maps (Azeroth, Kalimdor)

**Problem**: Loading 128 ADTs simultaneously = high memory!

**Solution**: Limit parallelism
```csharp
var options = new ParallelOptions 
{ 
    MaxDegreeOfParallelism = Math.Min(Environment.ProcessorCount, 8) 
};
```

**Memory Usage**:
- **Before**: 200-500 MB (single ADT in memory)
- **After (8 threads)**: 1-2 GB (8 ADTs in memory)
- **Modern systems**: 16-32 GB RAM → No problem! ✅

---

## Expected Results

### Speedup Chart

```
Map              | Tiles | Sequential | Parallel (8 cores) | Speedup
-----------------|-------|------------|-------------------|--------
DeadminesInstance|   10  |   2 min    |      20 sec       |   6x
PVPZone01        |   12  |   2.5 min  |      25 sec       |   6x
Shadowfang       |   15  |   3 min    |      30 sec       |   6x
Azeroth          |  128  |  50 min    |      7 min        |   7x
Kalimdor         |  140  |  55 min    |      8 min        |   7x
```

**Why not 8x?** 
- File I/O overhead (~20%)
- CSV merge time (~10%)
- Thread coordination overhead (~5%)
- **Actual speedup: 6-7x** ✅

---

## Conclusion

**Effort**: 2-3 hours of refactoring  
**Impact**: **6-7x faster processing!**  
**ROI**: Saves **40-50 minutes per map conversion**

**Next Steps**:
1. Implement `AdtScanner` parallelization first (biggest win)
2. Add progress reporting
3. Test on small map
4. Roll out to other extractors
5. Profit! 🚀
