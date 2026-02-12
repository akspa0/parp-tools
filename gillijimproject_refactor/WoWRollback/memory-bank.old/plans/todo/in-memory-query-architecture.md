# In-Memory Query Architecture Plan

## Philosophy

**Core Insight:** WoW files are already optimized databases. Don't extract to CSV - query directly!

> "World of Warcraft is just a database exploration tool with tightly packed data structures, at the end of the day. We should probably try to utilize the ingenuity of the original files and not extract data to csv, but harness the data directly, in-memory."

**Why map editors keep ADT format:**
- Already optimized for spatial queries
- Tightly packed, efficient binary format
- Designed for the exact use case we need
- No need to reinvent the wheel

---

## Current Problem

### What We Do Now (Inefficient):
```
WDT/ADT files → Extract ALL data → CSV files → Load into viewer → Query
                    ↓
            Gigabytes of text files
            Slow, redundant, wasteful
```

**Issues:**
- ❌ Extract data we might never use
- ❌ Giant CSV files (hundreds of MB)
- ❌ Slow to generate
- ❌ Slow to load
- ❌ Redundant storage (binary + text)
- ❌ Hard to update (regenerate entire CSV)

### What We Should Do (Efficient):
```
WDT/ADT files → Index (JSON) → Query on-demand → Viewer
                    ↓
            Tiny index files (KB)
            Fast, efficient, scalable
```

**Benefits:**
- ✅ Only decode what's needed
- ✅ Tiny index files (offsets, metadata)
- ✅ Fast startup
- ✅ Low memory usage
- ✅ Single source of truth (binary files)
- ✅ Easy to update (just reindex)

---

## Architecture Design

### 1. Index Files (JSON)

**Purpose:** Fast lookup without parsing entire file.

**Structure:**
```json
{
  "map": "Kalimdor",
  "version": "0.6.0.3592",
  "tiles": [
    {
      "x": 39,
      "y": 27,
      "file": "World/Maps/Kalimdor/Kalimdor_39_27.adt",
      "offset": 123456,
      "size": 789012,
      "chunks": [
        {
          "index": 0,
          "offset": 123500,
          "hasWMO": true,
          "hasDoodads": true,
          "uniqueIDs": [1234, 5678]
        }
      ],
      "bounds": {
        "minX": -5000.0,
        "maxX": -4466.67,
        "minZ": 3000.0,
        "maxZ": 3533.33
      }
    }
  ]
}
```

**Index per map:** `Kalimdor_index.json` (~100KB vs 50MB CSV)

### 2. Query API

**Interface:**
```csharp
public interface IAdtQueryService
{
    // Spatial queries
    IEnumerable<AdtChunk> GetChunksInBounds(string map, BoundingBox bounds);
    IEnumerable<AdtTile> GetTilesInBounds(string map, BoundingBox bounds);
    
    // Asset queries
    IEnumerable<WmoPlacement> GetWmosInTile(string map, int tileX, int tileY);
    IEnumerable<DoodadPlacement> GetDoodadsInTile(string map, int tileX, int tileY);
    
    // UniqueID queries
    IEnumerable<AssetPlacement> GetAssetsByUniqueId(string map, int uniqueId);
    IEnumerable<int> GetUniqueIdsInTile(string map, int tileX, int tileY);
    
    // Lazy loading
    AdtTileData LoadTile(string map, int tileX, int tileY);
    AdtChunkData LoadChunk(string map, int tileX, int tileY, int chunkIndex);
}
```

**Usage:**
```csharp
// Old way (CSV):
var csv = File.ReadAllLines("Kalimdor_placements.csv");
var wmos = csv.Where(line => line.Contains("WMO")).ToList();
// → Loads entire 50MB file, parses all lines

// New way (Query):
var wmos = queryService.GetWmosInTile("Kalimdor", 39, 27);
// → Loads only tile 39,27 from binary, returns WMOs
```

### 3. Viewer Integration

**Zoom-based loading:**
```csharp
// Viewer zoom level determines detail
public class ViewerQueryStrategy
{
    public IEnumerable<AdtTile> GetVisibleData(
        string map,
        BoundingBox viewport,
        int zoomLevel)
    {
        if (zoomLevel < 3)
        {
            // Zoomed out: load tile metadata only (from index)
            return GetTileMetadata(map, viewport);
        }
        else if (zoomLevel < 6)
        {
            // Medium zoom: load tile summaries
            return GetTileSummaries(map, viewport);
        }
        else
        {
            // Zoomed in: load full chunk data
            return GetFullTileData(map, viewport);
        }
    }
}
```

**Progressive loading:**
```
Zoom 1-2: Show tile grid (from index)
Zoom 3-5: Show WMO/doodad counts per tile (from index)
Zoom 6-8: Load actual placements (from ADT)
Zoom 9+:  Load full chunk details (from ADT)
```

---

## Implementation Plan

### Phase 1: Index Generation

**Goal:** Create lightweight index files for fast lookups.

**Steps:**
1. Create `AdtIndexBuilder` class
2. Parse WDT/ADT files once
3. Extract metadata (offsets, bounds, counts)
4. Write JSON index per map
5. Cache in `{output}/indices/` directory

**Output:**
```
output/
├── indices/
│   ├── Kalimdor_index.json      (~100KB)
│   ├── Azeroth_index.json       (~100KB)
│   └── ...
└── archives/
    └── 0.6.0.3592/
        └── World/Maps/...       (original files)
```

### Phase 2: Query Service

**Goal:** Implement efficient query API.

**Steps:**
1. Create `AdtQueryService` class
2. Load index on startup (fast)
3. Implement spatial queries
4. Implement asset queries
5. Add caching layer (LRU cache for recently loaded tiles)

**Caching strategy:**
```csharp
public class AdtQueryService
{
    private readonly LruCache<(string map, int x, int y), AdtTileData> _tileCache;
    private readonly Dictionary<string, MapIndex> _indices;
    
    public AdtQueryService(int maxCachedTiles = 50)
    {
        _tileCache = new LruCache<(string, int, int), AdtTileData>(maxCachedTiles);
        _indices = LoadAllIndices(); // Fast: just JSON
    }
    
    public IEnumerable<WmoPlacement> GetWmosInTile(string map, int x, int y)
    {
        // Check cache first
        if (!_tileCache.TryGet((map, x, y), out var tile))
        {
            // Load from binary (only this tile)
            tile = LoadTileFromBinary(map, x, y);
            _tileCache.Add((map, x, y), tile);
        }
        
        return tile.WmoPlacementsments;
    }
}
```

### Phase 3: Viewer Integration

**Goal:** Update viewer to use query API instead of CSV.

**Steps:**
1. Replace CSV loading with query service
2. Implement zoom-based loading
3. Add progressive rendering
4. Add viewport culling

**Before (CSV):**
```csharp
// Load entire CSV (50MB)
var placements = File.ReadAllLines("Kalimdor_placements.csv")
    .Skip(1)
    .Select(ParseCsvLine)
    .ToList();

// Filter to viewport (in memory)
var visible = placements.Where(p => viewport.Contains(p.Position));
```

**After (Query):**
```csharp
// Load only visible tiles (from index + binary)
var visible = queryService.GetWmosInBounds("Kalimdor", viewport);
// → Only loads tiles that intersect viewport
```

### Phase 4: Debug Mode

**Goal:** Keep CSV exports as optional debug output.

**CLI:**
```bash
# Production mode (default): no CSV exports
dotnet run -- analyze-map-adts-mpq --client-path "..." --map Kalimdor
# → Generates index.json only

# Debug mode: export CSVs for inspection
dotnet run -- analyze-map-adts-mpq --client-path "..." --map Kalimdor --debug
# → Generates index.json + CSV files

# Debug specific data
dotnet run -- analyze-map-adts-mpq --client-path "..." --map Kalimdor --debug-wmo
# → Generates index.json + WMO CSV only
```

**Implementation:**
```csharp
public class AnalysisOptions
{
    public bool DebugMode { get; set; } = false;
    public bool DebugWmo { get; set; } = false;
    public bool DebugDoodads { get; set; } = false;
    public bool DebugPlacements { get; set; } = false;
    
    public bool ShouldExportCsv => DebugMode || DebugWmo || DebugDoodads || DebugPlacements;
}
```

---

## Performance Comparison

### Current (CSV-based):

| Operation | Time | Memory | Disk |
|-----------|------|--------|------|
| Initial extraction | 30-60s | 2GB | 50MB CSV |
| Load in viewer | 5-10s | 500MB | - |
| Query visible tiles | 100ms | - | - |
| **Total startup** | **35-70s** | **2.5GB** | **50MB** |

### Proposed (Index-based):

| Operation | Time | Memory | Disk |
|-----------|------|--------|------|
| Initial indexing | 10-20s | 500MB | 100KB JSON |
| Load index | 10ms | 1MB | - |
| Query visible tiles | 5ms | 10MB | - |
| **Total startup** | **10-20s** | **511MB** | **100KB** |

**Improvements:**
- ⚡ **3-5x faster** startup
- 💾 **80% less memory** usage
- 📦 **99.8% smaller** disk usage

---

## Migration Strategy

### Step 1: Build Index System (Week 1)

**Tasks:**
- [ ] Create `AdtIndexBuilder` class
- [ ] Define JSON schema for indices
- [ ] Implement index generation
- [ ] Test with Kalimdor map

**Deliverable:** `Kalimdor_index.json` generated successfully

### Step 2: Implement Query Service (Week 2)

**Tasks:**
- [ ] Create `IAdtQueryService` interface
- [ ] Implement `AdtQueryService` class
- [ ] Add LRU caching
- [ ] Write unit tests

**Deliverable:** Query API working with test cases

### Step 3: Update Viewer (Week 3)

**Tasks:**
- [ ] Replace CSV loading with query service
- [ ] Implement zoom-based loading
- [ ] Add viewport culling
- [ ] Test performance

**Deliverable:** Viewer using query API, faster loading

### Step 4: Add Debug Mode (Week 4)

**Tasks:**
- [ ] Add `--debug` CLI flag
- [ ] Make CSV exports optional
- [ ] Update documentation
- [ ] Test both modes

**Deliverable:** Production mode (no CSV) + debug mode (with CSV)

---

## Index Schema Details

### Map Index (`{map}_index.json`)

```json
{
  "version": "1.0",
  "map": "Kalimdor",
  "clientVersion": "0.6.0.3592",
  "generated": "2025-10-14T02:43:00Z",
  "archivePath": "archives/0.6.0.3592/",
  "statistics": {
    "totalTiles": 256,
    "totalChunks": 4096,
    "totalWmos": 1234,
    "totalDoodads": 5678
  },
  "tiles": [
    {
      "x": 39,
      "y": 27,
      "file": "World/Maps/Kalimdor/Kalimdor_39_27.adt",
      "fileOffset": 0,
      "fileSize": 789012,
      "bounds": {
        "minX": -5000.0,
        "maxX": -4466.67,
        "minY": 0.0,
        "maxY": 100.0,
        "minZ": 3000.0,
        "maxZ": 3533.33
      },
      "chunks": [
        {
          "index": 0,
          "offset": 1234,
          "size": 5678,
          "flags": 0,
          "hasHoles": false,
          "wmoCount": 2,
          "doodadCount": 15,
          "uniqueIds": [1234, 5678, 9012]
        }
      ],
      "wmos": [
        {
          "uniqueId": 1234,
          "nameId": 567,
          "chunkIndex": 0,
          "offset": 2345
        }
      ],
      "doodads": [
        {
          "uniqueId": 5678,
          "nameId": 890,
          "chunkIndex": 0,
          "offset": 3456
        }
      ]
    }
  ]
}
```

### Tile-Specific Index (`{map}_{x}_{y}_index.json`)

**Optional:** For very large tiles, create per-tile indices.

```json
{
  "map": "Kalimdor",
  "tileX": 39,
  "tileY": 27,
  "file": "World/Maps/Kalimdor/Kalimdor_39_27.adt",
  "chunks": [ /* detailed chunk data */ ],
  "assets": [ /* detailed asset data */ ]
}
```

---

## Query Optimization Strategies

### 1. Spatial Indexing

**R-Tree for tile lookup:**
```csharp
public class SpatialIndex
{
    private readonly RTree<AdtTile> _rtree;
    
    public IEnumerable<AdtTile> QueryBounds(BoundingBox bounds)
    {
        return _rtree.Search(bounds);
    }
}
```

### 2. UniqueID Indexing

**Hash map for fast uniqueID lookup:**
```csharp
public class UniqueIdIndex
{
    private readonly Dictionary<int, List<(string map, int x, int y, int chunk)>> _index;
    
    public IEnumerable<AssetLocation> FindAsset(int uniqueId)
    {
        return _index.TryGetValue(uniqueId, out var locations) 
            ? locations 
            : Enumerable.Empty<AssetLocation>();
    }
}
```

### 3. Lazy Loading

**Load tiles on-demand:**
```csharp
public class LazyTileLoader
{
    private readonly Dictionary<(string, int, int), Lazy<AdtTileData>> _loaders;
    
    public AdtTileData GetTile(string map, int x, int y)
    {
        var key = (map, x, y);
        if (!_loaders.ContainsKey(key))
        {
            _loaders[key] = new Lazy<AdtTileData>(() => LoadFromBinary(map, x, y));
        }
        return _loaders[key].Value;
    }
}
```

---

## Debug Mode Features

### CSV Exports (Optional)

**When `--debug` is enabled:**
```
output/
├── indices/
│   └── Kalimdor_index.json      (always generated)
└── debug/
    ├── Kalimdor_placements.csv  (debug only)
    ├── Kalimdor_wmos.csv        (debug only)
    └── Kalimdor_doodads.csv     (debug only)
```

**Selective debug:**
```bash
# Debug WMO placements only
dotnet run -- analyze-map-adts-mpq --map Kalimdor --debug-wmo
# → Generates index.json + wmos.csv

# Debug everything
dotnet run -- analyze-map-adts-mpq --map Kalimdor --debug
# → Generates index.json + all CSVs
```

### Debug Viewer

**Show query performance:**
```
Viewer Debug Panel:
- Tiles loaded: 12
- Tiles cached: 8
- Query time: 5ms
- Memory usage: 45MB
- Cache hit rate: 87%
```

---

## Benefits Summary

### Performance
- ⚡ **3-5x faster** startup (10-20s vs 35-70s)
- 💾 **80% less memory** (500MB vs 2.5GB)
- 📦 **99.8% smaller** disk usage (100KB vs 50MB)
- 🚀 **5ms queries** vs 100ms CSV parsing

### Maintainability
- ✅ Single source of truth (binary files)
- ✅ Easy to update (just reindex)
- ✅ No CSV parsing bugs
- ✅ Simpler codebase

### Scalability
- ✅ Handles 1000+ tiles easily
- ✅ Low memory footprint
- ✅ Progressive loading
- ✅ Viewport culling

### Developer Experience
- ✅ Query API is intuitive
- ✅ Debug mode for troubleshooting
- ✅ Fast iteration (no CSV regeneration)
- ✅ Type-safe queries

---

## Next Steps

1. **Review this plan** - Ensure it aligns with project goals
2. **Prototype index generation** - Test with one map
3. **Benchmark performance** - Verify 3-5x speedup claims
4. **Implement query service** - Build core API
5. **Update viewer** - Switch from CSV to queries
6. **Add debug mode** - Keep CSV as optional output
7. **Document API** - Write usage guide

---

## Open Questions

1. **Index versioning:** How to handle index format changes?
2. **Cross-map queries:** Should we support querying multiple maps at once?
3. **Index compression:** Should we gzip the JSON indices?
4. **Incremental updates:** How to update index when files change?
5. **Distributed loading:** Should we support loading from remote archives?

---

## Conclusion

**The shift from CSV extraction to in-memory queries is a natural evolution:**

1. ✅ We've proven the data structures are reliable
2. ✅ We understand the binary formats well
3. ✅ We have working decoders
4. ✅ We know what data we need

**Now it's time to:**
- Stop extracting everything to text
- Start querying on-demand from binary
- Keep debug mode for troubleshooting
- Build a scalable, performant system

**This aligns perfectly with the WoW file format philosophy:**
> "It's already optimized perfectly for the use case."

Let's use it as intended! 🎯
