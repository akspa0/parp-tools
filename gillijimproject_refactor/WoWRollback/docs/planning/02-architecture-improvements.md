# WoWRollback Architecture Improvements Plan

**Created:** 2025-10-05  
**Status:** Planning  
**Priority:** High

---

## 🎯 **Overview**

Three major architectural improvements needed:
1. **Shadow Overlay Format**: PNG instead of JSON (4x size reduction, browser-native)
2. **Lazy Loading**: Efficient overlay loading/unloading
3. **C# Orchestration**: Replace PowerShell with WoWRollback.JobManager

---

## 📊 **Issue 1: Shadow Overlay Format**

### **Current State (Bad)**
```
Shadow data flow:
MCSH (512 bytes) → Base64 (680 bytes) → JSON (2.5MB per tile!)
                                          ↑ Insane!
```

**Problems:**
- 2.5MB JSON per tile = ~1.7GB for Azeroth alone
- JSON parsing overhead
- Not browser-native (requires decode + canvas render)
- No lazy loading benefits

### **Proposed State (Good)**
```
Shadow data flow:
MCSH (512 bytes) → 256-color PNG (< 50KB per tile)
                    ↑ Browser-native, lazy-loadable!
```

**Benefits:**
- ~50x size reduction (2.5MB → 50KB)
- Browser decodes natively (< img > tag)
- Lazy loading automatic
- No JSON parsing

### **Implementation Plan**

#### **Step 1: PNG Generation in AlphaWDTAnalysisTool**
```csharp
// New: AlphaWdtAnalyzer.Core/Terrain/McnkShadowPngWriter.cs
public static class McnkShadowPngWriter
{
    public static void WriteAsPng(
        List<McnkShadowEntry> shadows,
        string outputDirectory,
        string mapName,
        int tileRow,
        int tileCol)
    {
        // 1. Group shadows by tile
        var tileShadows = shadows
            .Where(s => s.TileRow == tileRow && s.TileCol == tileCol)
            .ToList();
        
        // 2. Create 256-color grayscale PNG (16 chunks × 16 chunks)
        //    Each chunk is 64×64 pixels
        //    Total image: 1024×1024 pixels
        using var bitmap = new Bitmap(1024, 1024, PixelFormat.Format8bppIndexed);
        
        // 3. Set grayscale palette
        var palette = bitmap.Palette;
        for (int i = 0; i < 256; i++)
        {
            palette.Entries[i] = Color.FromArgb(i, i, i);
        }
        bitmap.Palette = palette;
        
        // 4. Write shadow data chunk-by-chunk
        foreach (var shadow in tileShadows)
        {
            if (!shadow.HasShadow) continue;
            
            var shadowBytes = Convert.FromBase64String(shadow.ShadowBitmapBase64);
            WriteShadowToImage(bitmap, shadow.ChunkRow, shadow.ChunkCol, shadowBytes);
        }
        
        // 5. Save as PNG
        var filename = $"{mapName}_{tileRow}_{tileCol}_shadow.png";
        var outputPath = Path.Combine(outputDirectory, filename);
        bitmap.Save(outputPath, ImageFormat.Png);
    }
}
```

#### **Step 2: Update WoWRollback.Core Overlay Generation**
```csharp
// WoWRollback.Core/Services/OverlayBuilder.cs
public class ShadowOverlayBuilder
{
    public void BuildShadowOverlay(...)
    {
        // OLD: Generate JSON with Base64 shadow data
        // NEW: Generate PNG tiles
        
        foreach (var tile in tiles)
        {
            McnkShadowPngWriter.WriteAsPng(
                shadows, 
                outputDir, 
                mapName, 
                tile.Row, 
                tile.Col
            );
        }
        
        // Generate minimal JSON index (just coordinates)
        var index = new {
            format = "png",
            tiles = tiles.Select(t => new {
                row = t.Row,
                col = t.Col,
                url = $"shadows/{mapName}_{t.Row}_{t.Col}_shadow.png"
            })
        };
    }
}
```

#### **Step 3: Update Viewer to Load PNGs**
```javascript
// WoWRollback.Viewer/assets/js/overlays/shadowOverlay.js
class ShadowOverlay {
    constructor(map) {
        this.tiles = new Map(); // Lazy-loaded tile cache
        this.loadedTiles = new Set();
    }
    
    async loadTile(row, col) {
        if (this.loadedTiles.has(`${row}_${col}`)) return;
        
        const url = `/overlays/${version}/${map}/shadows/${map}_${row}_${col}_shadow.png`;
        const img = new Image();
        img.src = url;
        
        img.onload = () => {
            this.tiles.set(`${row}_${col}`, img);
            this.loadedTiles.add(`${row}_${col}`);
            this.renderTile(row, col, img);
        };
    }
    
    unloadTile(row, col) {
        const key = `${row}_${col}`;
        this.tiles.delete(key);
        this.loadedTiles.delete(key);
    }
}
```

---

## 📊 **Issue 2: Lazy Loading / Unloading**

### **Current State**
- All overlay JSON files loaded upfront
- No unloading when off-screen
- Memory usage grows unbounded

### **Proposed State**
- Viewport-based tile loading
- Automatic unloading for off-screen tiles
- LRU cache with configurable size

### **Implementation**
```javascript
// WoWRollback.Viewer/assets/js/core/TileLoader.js
class TileLoader {
    constructor(maxCacheSizeMB = 100) {
        this.cache = new Map();
        this.lruQueue = [];
        this.maxCacheSizeMB = maxCacheSizeMB;
        this.currentCacheSizeMB = 0;
    }
    
    onViewportChange(viewport) {
        const visibleTiles = this.calculateVisibleTiles(viewport);
        const tilesToLoad = visibleTiles.filter(t => !this.cache.has(t.key));
        const tilesToUnload = [...this.cache.keys()].filter(k => 
            !visibleTiles.some(t => t.key === k)
        );
        
        // Unload off-screen tiles
        tilesToUnload.forEach(key => this.unload(key));
        
        // Load new tiles
        tilesToLoad.forEach(tile => this.load(tile));
    }
    
    load(tile) {
        if (this.currentCacheSizeMB > this.maxCacheSizeMB) {
            this.evictLRU();
        }
        // ... load tile
    }
    
    evictLRU() {
        const oldestKey = this.lruQueue.shift();
        const tile = this.cache.get(oldestKey);
        this.currentCacheSizeMB -= tile.sizeMB;
        this.cache.delete(oldestKey);
    }
}
```

---

## 📊 **Issue 3: C# Orchestration (WoWRollback.JobManager)**

### **Why Replace PowerShell?**
- ❌ Not all users have PowerShell
- ❌ Cross-platform concerns
- ❌ Hard to debug/test
- ❌ No type safety
- ❌ Growing complexity

### **Proposed: WoWRollback.JobManager**

#### **Project Structure**
```
WoWRollback.JobManager/
├─ Core/
│  ├─ IJob.cs                    # Job interface
│  ├─ JobContext.cs              # Execution context
│  ├─ JobOrchestrator.cs         # Manages job pipeline
│  └─ JobResult.cs               # Result/error reporting
├─ Jobs/
│  ├─ BuildProjectsJob.cs        # Step 1: dotnet build
│  ├─ CacheMapJob.cs             # Step 2: LK ADT caching
│  ├─ ExtractCsvsJob.cs          # CSV extraction
│  ├─ CompareVersionsJob.cs      # Step 3: compare-versions
│  ├─ GenerateOverlaysJob.cs     # Step 4: overlay generation
│  └─ StartServerJob.cs          # Step 5: HTTP server
├─ Services/
│  ├─ ProcessRunner.cs           # Execute external tools
│  ├─ FileWatcher.cs             # Detect cache changes
│  └─ Logger.cs                  # Structured logging
└─ Program.cs                    # CLI entry point
```

#### **Job Interface**
```csharp
public interface IJob
{
    string Name { get; }
    Task<JobResult> ExecuteAsync(JobContext context, CancellationToken ct);
}

public class JobContext
{
    public ILogger Logger { get; }
    public Dictionary<string, object> Data { get; } // Shared state
    public JobOptions Options { get; }
}

public class JobResult
{
    public bool Success { get; }
    public string? Error { get; }
    public Dictionary<string, object> Outputs { get; } // Pass data to next job
}
```

#### **Orchestrator**
```csharp
public class JobOrchestrator
{
    private readonly List<IJob> _jobs = new();
    
    public async Task RunAsync(JobContext context)
    {
        foreach (var job in _jobs)
        {
            context.Logger.LogInformation($"[{job.Name}] Starting...");
            
            var result = await job.ExecuteAsync(context, CancellationToken.None);
            
            if (!result.Success)
            {
                context.Logger.LogError($"[{job.Name}] FAILED: {result.Error}");
                throw new JobFailedException(job.Name, result.Error);
            }
            
            // Merge outputs into context for next job
            foreach (var (key, value) in result.Outputs)
            {
                context.Data[key] = value;
            }
            
            context.Logger.LogInformation($"[{job.Name}] ✓ Complete");
        }
    }
}
```

#### **CLI Usage**
```bash
# Full rebuild
WoWRollback.JobManager run \
  --maps Azeroth,Kalimdor \
  --versions 0.5.3.3368,0.5.5.3494 \
  --alpha-root ../test_data \
  --full-rebuild

# Refresh analysis only
WoWRollback.JobManager run \
  --maps Azeroth \
  --refresh-analysis \
  --refresh-overlays \
  --serve

# List available jobs
WoWRollback.JobManager jobs

# Run specific job
WoWRollback.JobManager run-job \
  --job ExtractCsvs \
  --map Azeroth \
  --version 0.5.3.3368
```

#### **Migration Path**
1. **Phase 1**: Create WoWRollback.JobManager with core jobs
2. **Phase 2**: Keep PowerShell script as wrapper (calls JobManager)
3. **Phase 3**: Update docs to recommend JobManager
4. **Phase 4**: Deprecate PowerShell script

---

## 📅 **Implementation Priority**

### **High Priority (Do First)**
1. ✅ Fix LK ADT parsing (DONE!)
2. 🔧 PNG shadow overlays (1-2 hours)
3. 🔧 Lazy loading infrastructure (2-3 hours)

### **Medium Priority (After MVP)**
4. 🔧 WoWRollback.JobManager core (4-6 hours)
5. 🔧 Port key jobs to C# (6-8 hours)

### **Low Priority (Polish)**
6. 🔧 Complete PowerShell deprecation
7. 🔧 Advanced caching strategies

---

## 🧪 **Testing Strategy**

### **Shadow PNG Tests**
- Generate PNG from known MCSH data
- Compare with existing JSON visualization
- Measure size reduction
- Browser compatibility (Chrome, Firefox, Safari)

### **Lazy Loading Tests**
- Memory usage before/after
- Load performance (time to first paint)
- Unload verification (memory freed)

### **JobManager Tests**
- Unit tests per job
- Integration tests for pipeline
- Error handling (job failures)
- Cancellation support

---

## 📝 **Questions to Answer**

1. **PNG Generation**: Use System.Drawing or ImageSharp?
   - System.Drawing: Windows-only (GDI+)
   - ImageSharp: Cross-platform, modern
   - **Recommendation**: ImageSharp

2. **Lazy Loading**: Client-side or server-side logic?
   - Client: Better UX, no server changes
   - Server: Could pre-generate tile bundles
   - **Recommendation**: Client-side

3. **JobManager**: CLI-only or also library?
   - CLI: Easy to use
   - Library: Can embed in other tools
   - **Recommendation**: Both (CLI wraps library)

---

## 🦀 **Next Steps**

**Immediate (Today):**
1. ✅ Fix LK ADT parsing build issues
2. 📝 Create this planning doc
3. ⏸️ Wait for user feedback on priorities

**Short-term (This Week):**
1. Implement PNG shadow generation
2. Add lazy loading to viewer
3. Prototype WoWRollback.JobManager

**Long-term (This Month):**
1. Complete JobManager migration
2. Deprecate PowerShell script
3. Documentation updates

---

**Decision Needed:** Which issue should we tackle first?
- **A**: PNG shadows (immediate size/performance win)
- **B**: JobManager (better long-term architecture)
- **C**: Both in parallel (ambitious!)
