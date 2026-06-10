# Critical Data Loss Issues - Root Cause Analysis

## Issue 1: Terrain Overlays NOT Generated ❌

### Root Cause
**Path mismatch** between terrain CSV writer and reader:

```csharp
// AdtTerrainExtractor writes to:
Path.Combine(outputDir, $"{mapName}_terrain.csv")
// → analysis_output/development_terrain.csv

// OverlayGenerator.GenerateTerrainOverlaysFromCsv reads from:
Path.Combine(adtOutputDir, "csv", "maps", mapName, "terrain.csv")
// → analysis_output/csv/maps/development/terrain.csv  ❌ DOESN'T EXIST!
```

### Fix
**File:** `WoWRollback.AnalysisModule\OverlayGenerator.cs`
**Line 235:**

```csharp
// BEFORE (WRONG):
var terrainCsv = Path.Combine(adtOutputDir, "csv", "maps", mapName, "terrain.csv");

// AFTER (CORRECT):
var terrainCsv = Path.Combine(adtOutputDir, $"{mapName}_terrain.csv");
```

---

## Issue 2: Duplicate UniqueIDs from Cross-Tile Objects ❌

### Root Cause
Objects near tile boundaries get saved in MULTIPLE tiles:
- Object at position (533.2, 100.0, 800.0)
- Sits on boundary between tiles X=1, Y=2
- Gets written to BOTH tiles → **duplicate UniqueID**

### Current Behavior
```
Tile 1_2:  UniqueID=12345, pos=(533.2, 100.0, 800.0)
Tile 1_3:  UniqueID=12345, pos=(533.2, 100.0, 800.0)  ← DUPLICATE!
```

### Solution: WoW Coordinate-Based Tile Assignment

**File:** `WoWRollback.AnalysisModule\AdtPlacementsExtractor.cs`

Add method to determine PRIMARY tile based on WoW coordinates:

```csharp
/// <summary>
/// Determines the PRIMARY tile an object belongs to based on WoW world coordinates.
/// WoW coordinate system: Origin at map center, tiles 533.33 yards apart.
/// </summary>
private (int tileX, int tileY) GetPrimaryTileForPosition(float worldX, float worldY, float worldZ)
{
    // WoW world bounds: -17066.0 to +17066.0 (total 34132.0 yards)
    // 64 tiles × 533.33333 yards/tile = 34133.33 yards
    const float MAP_SIZE = 34133.33f;
    const float TILE_SIZE = 533.33333f;  // MAP_SIZE / 64
    const float HALF_MAP = MAP_SIZE / 2f;
    
    // Convert world coords to map-relative (0 to MAP_SIZE)
    float mapRelativeX = worldX + HALF_MAP;  // X: -17066 → +17066 becomes 0 → 34133
    float mapRelativeY = worldY + HALF_MAP;  // Y: -17066 → +17066 becomes 0 → 34133
    
    // Calculate tile indices (0-63)
    int tileX = (int)Math.Floor(mapRelativeX / TILE_SIZE);
    int tileY = (int)Math.Floor(mapRelativeY / TILE_SIZE);
    
    // Clamp to valid range
    tileX = Math.Clamp(tileX, 0, 63);
    tileY = Math.Clamp(tileY, 0, 63);
    
    return (tileX, tileY);
}

/// <summary>
/// Determines if a position is within a tile's bounds (for validation).
/// </summary>
private bool IsPositionInTile(float worldX, float worldY, int tileX, int tileY)
{
    const float MAP_SIZE = 34133.33f;
    const float TILE_SIZE = 533.33333f;
    const float HALF_MAP = MAP_SIZE / 2f;
    
    float mapRelativeX = worldX + HALF_MAP;
    float mapRelativeY = worldY + HALF_MAP;
    
    float tileMinX = tileX * TILE_SIZE;
    float tileMaxX = (tileX + 1) * TILE_SIZE;
    float tileMinY = tileY * TILE_SIZE;
    float tileMaxY = (tileY + 1) * TILE_SIZE;
    
    return mapRelativeX >= tileMinX && mapRelativeX < tileMaxX &&
           mapRelativeY >= tileMinY && mapRelativeY < tileMaxY;
}
```

### Usage in Extraction
**Modify placement export logic:**

```csharp
// When extracting placements from ADT:
var calculatedTile = GetPrimaryTileForPosition(worldX, worldY, worldZ);

// ONLY export if this is the primary tile
if (calculatedTile.tileX == currentTileX && calculatedTile.tileY == currentTileY)
{
    // Export placement
    csv.AppendLine($"{mapName},{currentTileX},{currentTileY},{kind},{assetPath},{uniqueId},...");
}
else
{
    // Skip - belongs to different tile
    Console.WriteLine($"[Dedup] Skipping UID={uniqueId} in tile ({currentTileX},{currentTileY}), " +
                     $"belongs to ({calculatedTile.tileX},{calculatedTile.tileY})");
}
```

---

## Issue 3: Cluster Object Details Missing ❌

### Root Cause
**Cluster JSON does NOT include object list!**

Current cluster JSON:
```json
{
  "clusterId": 0,
  "objectCount": 3,
  "centroid": {...},
  "bounds": {...}
  // ❌ NO "objects" array!
}
```

### Fix
**File:** `WoWRollback.Core\Services\Viewer\ClusterOverlayBuilder.cs`

Enhance cluster export to include object details:

```csharp
// When building cluster overlays, include full object data:
{
  "clusterId": 0,
  "objectCount": 3,
  "centroid": {...},
  "bounds": {...},
  "uniqueIdRange": { "min": 49415, "max": 49559 },
  "objects": [
    {
      "uniqueId": 49415,
      "fileName": "AERIEPEAKSFIR02.M2",
      "assetPath": "WORLD\\LORDAERON\\...",
      "position": { "x": 555.45, "y": 57.8, "z": 721.78 },
      "scale": 1.0,
      "kind": "M2"
    },
    ...
  ],
  "assetDistribution": {
    "AERIEPEAKSFIR02.M2": 3
  }
}
```

---

## Issue 4: Object Markers Missing After Load ❌

### Root Cause
**OverlayBuilder filters placement data!**

Line 80 in `OverlayBuilder.cs` logs:
```
[OverlayBuilder] BuildOverlay for development tile (60,55): Received 28720 total entries
[OverlayBuilder] After filter: 1 entries for tile (60,55)
```

**28,720 → 1 = 99.996% data loss!**

### Investigation Needed
Check `OverlayBuilder.cs` line 80+ for filtering logic that's removing valid placements.

Likely culprits:
1. Tile coordinate mismatch (row/col swap)
2. World coordinate validation too strict
3. UniqueID=0 filter removing valid objects
4. Duplicate detection removing ALL duplicates instead of keeping one

---

## Fix Priority

### Phase 1: Critical Path Fixes (IMMEDIATE)
1. ✅ **Fix terrain CSV path** - 1 line change
2. ✅ **Add WoW coordinate-based tile assignment** - Prevent duplicates
3. ✅ **Debug OverlayBuilder filtering** - Find why 99.996% of data is lost

### Phase 2: Enhancement (NEXT)
4. ✅ **Add object details to clusters** - Enrich cluster JSON

### Phase 3: Validation (AFTER)
5. ✅ **Add regression test** - Verify no data loss
6. ✅ **Add coordinate validation** - Ensure WoW system used throughout

---

## Testing Checklist

After fixes:
- [ ] `terrain_complete/` directory contains JSON files (not empty)
- [ ] Cluster popups show object lists
- [ ] No duplicate UniqueIDs in CSV
- [ ] Object markers appear on map for all placements
- [ ] Tile boundaries match WoW coordinate system
- [ ] Console logs show reasonable filter ratios (not 99.996% loss)

---

## Key Insight: Coordinate System Consistency

**CRITICAL:** Use WoW coordinates EVERYWHERE:
- Origin: Map center (0, 0)
- Range: -17066 to +17066 (34133 yards total)
- Tile size: 533.33 yards
- Tile count: 64×64
- Tile (0,0) starts at (-17066, -17066)

**NEVER mix with Leaflet coordinates** in backend C# code!

Leaflet transformation happens ONLY in viewer JS (pixelToLatLng, tileBounds, etc.)
