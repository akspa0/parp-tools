# Data Storage Strategy: CSV vs SQLite vs JSON

## Current Architecture Analysis

### Data Flow (Current)

```
┌─────────────────────────────────────┐
│  Extraction (One-Time)              │
│  AlphaWDTAnalysisTool               │
├─────────────────────────────────────┤
│  ADT Binary Files                   │
│         ↓                           │
│  McnkTerrainExtractor               │
│         ↓                           │
│  CSV Files (~128KB per tile)        │
│    - Human-readable                 │
│    - Version-controllable           │
│    - Tool-agnostic                  │
└─────────────────────────────────────┘
            ↓
┌─────────────────────────────────────┐
│  Transformation (Per-Build)         │
│  WoWRollback.Core                   │
├─────────────────────────────────────┤
│  CSV Files                          │
│         ↓                           │
│  OverlayBuilders                    │
│         ↓                           │
│  JSON Files (~50KB per tile)        │
│    - Tile-based structure           │
│    - Browser-native parsing         │
│    - Lazy loading                   │
└─────────────────────────────────────┘
            ↓
┌─────────────────────────────────────┐
│  Visualization (Runtime)            │
│  Viewer (Browser)                   │
├─────────────────────────────────────┤
│  Load visible tiles only            │
│  ~6-16 JSON files at a time         │
│  Parse on-demand                    │
│  Unload out-of-view tiles           │
│    - Fast (< 50ms per tile)         │
│    - Low memory footprint           │
└─────────────────────────────────────┘
```

---

## Performance Analysis

### CSV Performance (Extraction Phase)

**Use Case**: One-time extraction from binary ADT files

**Characteristics**:
- Sequential write (fast)
- No query requirements
- Not read during runtime
- Size: ~128KB per tile × 256 tiles = ~32MB per continent

**Verdict**: ✅ **CSV is PERFECT for this**

**Why**:
- Human-readable for debugging
- Git-friendly (can diff changes between versions)
- Tool-agnostic (any language can read)
- No performance bottleneck (written once, transformed once)

---

### JSON Performance (Viewer Phase)

**Use Case**: Tile-based overlay data loaded on-demand in browser

**Characteristics**:
- Tile-based structure (64×64 grid)
- Lazy loading (only visible tiles)
- ~6-16 tiles visible at typical zoom
- Size: ~50KB per tile (gzipped: ~10KB)

**Benchmark Estimates**:

| Metric | Value | Impact |
|--------|-------|--------|
| JSON parse time | ~5-10ms per tile | ✅ Fast |
| HTTP fetch (gzip) | ~50-100ms per tile | ✅ Acceptable |
| Memory per tile | ~50KB | ✅ Tiny |
| Tiles in view | 6-16 | ✅ Manageable |
| Total load time | ~300-500ms | ✅ Imperceptible |

**Verdict**: ✅ **JSON is EXCELLENT for this**

**Why**:
- Browser-native parsing (highly optimized)
- HTTP caching (tiles cached after first load)
- Gzip compression (50-80% size reduction)
- No query complexity (simple key-value lookups)
- Progressive loading (user sees data as it arrives)

---

## When to Use SQLite?

### ✅ Good Use Cases for SQLite

1. **Cross-Tile Queries**
   ```sql
   -- Find all M2 placements of a specific model across entire continent
   SELECT * FROM placements 
   WHERE model_path LIKE '%Tree%' 
   AND map = 'Azeroth';
   ```

2. **Statistical Analysis**
   ```sql
   -- Count objects by type across all tiles
   SELECT object_type, COUNT(*) 
   FROM placements 
   GROUP BY object_type;
   ```

3. **Complex Filtering**
   ```sql
   -- Find tiles with impassible chunks in Elwynn Forest
   SELECT DISTINCT tile_row, tile_col 
   FROM mcnk_terrain 
   WHERE map = 'Azeroth' 
   AND areaid = 12 
   AND impassible = true;
   ```

4. **Large Dataset Queries**
   - Multiple continents (millions of objects)
   - Need indexes for fast lookups
   - Full-text search requirements

---

### ❌ Bad Use Cases for SQLite (Current Needs)

1. **Tile-Based Visualization** (Already using JSON)
   - No need to query - just load tile N
   - JSON file IS the query result
   - Adding SQLite = extra query step

2. **One-Time Extraction** (Already using CSV)
   - No query requirements
   - Sequential write is fastest
   - CSV is more debuggable

3. **Small Dataset** (Current scope)
   - 256 tiles × 50KB = 12.8MB per continent
   - Browser can handle easily
   - No performance bottleneck

---

## Recommendation: **Keep Current Architecture**

### ✅ Reasons to AVOID SQLite (For Now)

1. **No Performance Problem**
   - Viewer is already fast
   - Tile-based loading is optimal
   - No user complaints about speed

2. **Added Complexity**
   - Need SQLite library in C#
   - Need SQLite library in JavaScript (or HTTP API)
   - Schema migrations
   - Index maintenance

3. **Lost Simplicity**
   - CSV = plaintext, diffable, portable
   - JSON = browser-native, cacheable, simple
   - SQLite = binary, requires tools, less portable

4. **No Query Requirements**
   - Overlays don't need complex queries
   - Tile-based access pattern is simple
   - No cross-tile aggregation needed

5. **Storage Size is Fine**
   - ~12MB per continent (uncompressed)
   - ~3MB per continent (gzipped)
   - Negligible by modern standards

---

## Optimizations (Without SQLite)

If performance becomes an issue, try these first:

### 1. JSON Compression

```javascript
// Enable gzip on web server
// .htaccess or nginx config
AddOutputFilterByType DEFLATE application/json

// Result: 50KB → 10KB per tile
```

### 2. Progressive Loading

```javascript
// Already implemented
async function loadVisibleTiles() {
    const tiles = getVisibleTiles();
    
    // Load in parallel (6-16 tiles)
    await Promise.all(tiles.map(loadTile));
}
```

### 3. Aggressive Caching

```javascript
// Service worker for offline support
self.addEventListener('fetch', (event) => {
    if (event.request.url.includes('/overlays/')) {
        event.respondWith(
            caches.match(event.request)
                .then(cached => cached || fetch(event.request))
        );
    }
});
```

### 4. JSON Minification

```csharp
// Remove whitespace
var options = new JsonSerializerOptions { 
    WriteIndented = false  // 50KB → 35KB
};
```

### 5. Delta Encoding (Version Comparison)

```json
// Instead of full data for each version, store diffs
{
  "base": "0.5.3",
  "0.5.5": {
    "added": [...],
    "removed": [...],
    "modified": [...]
  }
}
```

---

## When to Revisit SQLite

Add SQLite if you encounter:

### Performance Indicators

1. **JSON load time > 1 second** per tile
2. **Memory usage > 100MB** for overlays
3. **User complaints** about sluggishness

### Feature Indicators

1. **Search functionality needed**
   - "Find all caves on Azeroth"
   - "Show me all Defias NPCs"

2. **Statistical dashboards**
   - "How many trees per continent?"
   - "Object density heatmaps"

3. **Cross-version queries**
   - "Which objects were added in 0.5.5?"
   - "Show removed placements"

4. **Export functionality**
   - "Export all M2 paths to CSV"
   - "Generate report of all WMOs"

---

## Hybrid Approach (Future Option)

If you DO add SQLite later, keep CSV/JSON:

### Phase 1: Extraction (CSV) ✅
```
ADT → CSV (human-readable, version-controlled)
```

### Phase 2A: Transformation (JSON) ✅
```
CSV → JSON (viewer overlays, tile-based)
```

### Phase 2B: Import (SQLite) ⭐ NEW
```
CSV → SQLite (analytics, search, complex queries)
```

### Result: Best of Both Worlds

- **CSV**: Source of truth, human-readable
- **JSON**: Fast tile-based visualization
- **SQLite**: Optional analytics/search layer

```
        CSV (source)
       ↙   ↘
    JSON   SQLite
   (view) (query)
```

---

## File Size Estimates

### Per Continent (64×64 tiles)

| Layer | Size (CSV) | Size (JSON) | Size (Gzip) |
|-------|-----------|-------------|-------------|
| MCNK Terrain | ~32MB | ~12MB | ~3MB |
| Shadow Maps | ~128MB | ~50MB (PNG) | ~30MB |
| Placements | ~15MB | ~8MB | ~2MB |
| AreaID | ~2MB | ~1MB | ~200KB |
| **Total** | **~177MB** | **~71MB** | **~35MB** |

### Reality Check

- **1 continent = 35MB gzipped**
- **2 continents = 70MB**
- **CDN/browser cache** = loaded once
- **Modern internet** = ~10 seconds download
- **Subsequent loads** = instant (cached)

**Verdict**: Size is NOT a problem.

---

## SQLite vs JSON Performance (Hypothetical)

### Scenario: Load Single Tile

| Operation | JSON | SQLite | Winner |
|-----------|------|--------|--------|
| HTTP Fetch | 50ms | N/A | - |
| Parse | 5ms | 10ms (query) | JSON |
| Memory | 50KB | 50KB + DB overhead | JSON |
| Caching | Browser cache | App cache | JSON |
| Offline | Service worker | Built-in | Tie |
| **Total** | **55ms** | **10ms*** | SQLite* |

*SQLite assumes database is already loaded in memory

### Scenario: Load 16 Tiles (Full Viewport)

| Operation | JSON | SQLite | Winner |
|-----------|------|--------|--------|
| Parallel fetch | 16 × 50ms = 100ms | 1 query = 10ms | SQLite |
| Parse | 16 × 5ms = 20ms | Included | SQLite |
| Bandwidth | 16 × 10KB = 160KB | 0KB | SQLite |
| **Total** | **120ms** | **10ms*** | SQLite* |

*But requires 71MB database preloaded

### Reality Check

**JSON**: 120ms perceived (with caching: 20ms)
**SQLite**: 10ms query (but 71MB upfront load)

**Winner**: JSON (progressive loading beats upfront cost)

---

## Decision Matrix

| Factor | CSV | JSON | SQLite | Winner |
|--------|-----|------|--------|--------|
| **Extraction** | ✅ Fast | ❌ N/A | ❌ Slow | CSV |
| **Transformation** | ✅ Simple | ✅ Simple | ⚠️ Complex | CSV/JSON |
| **Tile Loading** | ❌ N/A | ✅ Optimal | ⚠️ Overhead | JSON |
| **Cross-Tile Query** | ❌ N/A | ❌ Impossible | ✅ Fast | SQLite |
| **Debugging** | ✅ Perfect | ✅ Good | ⚠️ Tools needed | CSV |
| **Version Control** | ✅ Diffable | ✅ Diffable | ❌ Binary | CSV/JSON |
| **Portability** | ✅ Universal | ✅ Universal | ⚠️ Platform-specific | CSV/JSON |
| **Size** | ⚠️ Large | ✅ Medium | ✅ Compact | SQLite |
| **Complexity** | ✅ Simple | ✅ Simple | ❌ Complex | CSV/JSON |

---

## Final Recommendation

### ✅ **Keep Current Architecture (CSV → JSON)**

**Reasons**:
1. ✅ Viewer is already performant
2. ✅ No real performance bottleneck
3. ✅ Overlays are debug/exploration tools (not search/analytics)
4. ✅ Tile-based loading is optimal for visualization
5. ✅ CSV/JSON maintains simplicity and portability
6. ✅ File sizes are manageable (35MB/continent gzipped)
7. ✅ No complex query requirements
8. ✅ Human-readable formats aid debugging

**Next Steps**:
- Implement overlays with current CSV → JSON pipeline
- Monitor performance during actual usage
- If performance issues arise, try optimizations (gzip, caching)
- Only add SQLite if specific query needs emerge

---

## Future: When to Add SQLite

Add a **parallel SQLite layer** (not replacement) if:

1. ✅ You add search functionality ("find all caves")
2. ✅ You need statistical dashboards
3. ✅ You want cross-tile analytics
4. ✅ Export tools require complex filtering
5. ✅ Users request query capabilities

**Implementation**: Hybrid approach (CSV as source → JSON for viewer + SQLite for queries)

---

## Summary

**Question**: Should we use SQLite for performance?

**Answer**: No, not yet.

**Why**:
- Current approach is fast enough
- No query requirements that need SQLite
- Added complexity outweighs benefits
- CSV/JSON maintains simplicity and interoperability
- Tile-based JSON loading is optimal for visualization

**When to revisit**: 
- If you add search/analytics features
- If users complain about performance (unlikely)
- If you need cross-tile queries

**For now**: Keep it simple. Build the overlays with CSV → JSON pipeline. Add SQLite later only if you discover a concrete need.
