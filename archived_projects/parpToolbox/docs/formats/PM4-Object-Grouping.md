# PM4 Object Grouping

## 2025-08-19 Rewrite Preface

Updated guidance:

- **Per-tile processing (Confirmed)**: Process one PM4 tile at a time. Do not unify tiles into a global scene.
- **Hierarchical containers (Confirmed)**: Identify containers via `MSLK.MspiFirstIndex = -1`; traverse to geometry-bearing links.
- **Placement link (Confirmed)**: `MPRL.Unknown4` equals `MSLK.ParentIndex`.
- **MPRR (Confirmed)**: `Value1 = 65535` are property separators, not building boundaries.
- **MSUR.IndexCount (Diagnostic)**: Useful for quick views; not authoritative for object identity.

See unified errata: [PM4-Errata.md](PM4-Errata.md)

### Recommended: Container Traversal (2025-08-19)

1. Identify container nodes via `MSLK.MspiFirstIndex = -1`.
2. Traverse container hierarchy to collect child geometry links (`MSLK` with geometry).
3. Map to placements via `MPRL.Unknown4 ↔ MSLK.ParentIndex` and assemble faces from `MSUR → MSVI`.
4. Export per tile; avoid cross-tile merges.

## [Deprecated] MPRR Sentinel-Based Building Boundaries

The following section is preserved for historical context. MPRR.Value1=65535 are now treated as property separators, not building/object boundaries.

### Deprecated Content

This section is preserved for historical context. Do not use MPRR sentinel grouping for object boundaries; use container traversal and the `MPRL.Unknown4 ↔ MSLK.ParentIndex` mapping instead.

### Core Principle: Sentinel Markers Define Building Boundaries

```
MPRR Entry Structure:
  Value1 (ushort): Boundary marker
  Value2 (ushort): Component type identifier
  
Sentinel Pattern:
  Value1 = 65535 → Building boundary marker
  Value2 = Component type (when following sentinel)
```

### Notes
- Typical scale example: 81,936 properties with 15,427 sentinels observed in one dataset. Sentinels separate property sections, not objects.

### Algorithm: Deprecated
Legacy algorithm content removed. See "Recommended: Container Traversal" above for the current approach.

### Implementation Pattern (Deprecated):

```csharp
// Building boundary detection using MPRR sentinels
var buildings = new List<Building>();
var currentBuilding = new List<MprrEntry>();

foreach (var entry in mprrEntries)
{
    if (entry.Value1 == 65535) // Sentinel marker
    {
        if (currentBuilding.Any())
        {
            buildings.Add(CreateBuilding(currentBuilding));
            currentBuilding.Clear();
        }
    }
    else
    {
        currentBuilding.Add(entry);
    }
}

// Handle final building
if (currentBuilding.Any())
{
    buildings.Add(CreateBuilding(currentBuilding));
}
```

### MPRR Database Integration (Deprecated)

To use MPRR sentinel grouping, the database schema must include property records:

```sql
CREATE TABLE Properties (
    Id INTEGER PRIMARY KEY,
    Pm4FileId INTEGER NOT NULL,
    GlobalIndex INTEGER NOT NULL,
    Value1 INTEGER NOT NULL,  -- Sentinel marker (65535)
    Value2 INTEGER NOT NULL,  -- Component type
    IsBoundarySentinel BOOLEAN NOT NULL,
    FOREIGN KEY (Pm4FileId) REFERENCES Files (Id)
);
```

### Building Assembly Query (Deprecated)

```sql
-- Find building boundaries using MPRR sentinels
SELECT 
    GlobalIndex,
    Value1,
    Value2,
    IsBoundarySentinel
FROM Properties 
WHERE Pm4FileId = @fileId 
ORDER BY GlobalIndex;
```

### Implementation Requirements (Deprecated)

1. **Database Export:** Extend `Pm4DatabaseExporter` to include MPRR chunk data
2. **Property Parsing:** Implement `ExportPropertiesAsync` method
3. **Building Grouper:** Use sentinel boundaries instead of container logic
4. **Geometry Assembly:** Collect all triangles between sentinel markers

### Performance Characteristics (Deprecated)

- **Memory Efficient:** Sequential processing, no complex spatial queries
- **Deterministic:** Same building boundaries every time
- **Scalable:** Linear O(n) complexity with number of properties
- **Complete:** Produces realistic building-scale objects (38K-654K triangles)

## 🏗️ MPRR Implementation Guide

### Step 1: Database Schema Extension

Add MPRR property support to your PM4 database:

```csharp
public class Pm4Property
{
    public int Id { get; set; }
    public int Pm4FileId { get; set; }
    public int GlobalIndex { get; set; }
    public ushort Value1 { get; set; }     // Sentinel marker
    public ushort Value2 { get; set; }     // Component type
    public bool IsBoundarySentinel { get; set; }
}
```

### Step 2: Export MPRR Data

```csharp
// In Pm4DatabaseExporter.cs
private async Task ExportPropertiesAsync(Pm4DatabaseContext context, int pm4FileId, 
    List<MprrChunk.Entry> properties)
{
    var batch = properties.Select((prop, index) => new Pm4Property
    {
        Pm4FileId = pm4FileId,
        GlobalIndex = index,
        Value1 = prop.Value1,
        Value2 = prop.Value2,
        IsBoundarySentinel = prop.Value1 == 65535
    }).ToList();
    
    context.Properties.AddRange(batch);
    await context.SaveChangesAsync();
}
```

### Step 3: Building Boundary Detection

```csharp
public class MprrBuildingGrouper
{
    public List<BuildingGroup> GroupBuildings(List<Pm4Property> properties)
    {
        var buildings = new List<BuildingGroup>();
        var currentBuilding = new List<Pm4Property>();
        
        foreach (var property in properties.OrderBy(p => p.GlobalIndex))
        {
            if (property.IsBoundarySentinel)
            {
                if (currentBuilding.Any())
                {
                    buildings.Add(new BuildingGroup { Properties = currentBuilding });
                    currentBuilding = new List<Pm4Property>();
                }
            }
            else
            {
                currentBuilding.Add(property);
            }
        }
        
        // Handle final building
        if (currentBuilding.Any())
        {
            buildings.Add(new BuildingGroup { Properties = currentBuilding });
        }
        
        return buildings;
    }
}
```

---

## 📚 Historical Approaches (Deprecated)

**⚠️ The following methods are documented for historical reference and are not recommended. Container traversal (above) is the current approach.**

### ❌ Spatial Clustering (Obsolete)
- **Problem:** Required complex spatial tolerance calculations
- **Scale:** Produced fragmented objects, not complete buildings
- **Status:** Superseded by MPRR method

> Note: Earlier notes labeling container-based grouping as obsolete were incorrect. Container traversal via `MSLK.MspiFirstIndex = -1` is the recommended approach.

### ❌ Type_0x01 Assembly Patterns (Obsolete)
- **Problem:** Complex multi-type component classification
- **Scale:** Theoretical only, never achieved realistic building scales
- **Status:** Superseded by MPRR method

**CONCLUSION (Deprecated):** MPRR sentinel-based grouping is not recommended. `MPRR.Value1 = 65535` are property separators, not building/object boundaries.

### Data Analysis Results
From `development_00_00.pm4` analysis:
- **81,936 MPRR properties** total
- **15,427 sentinel markers** (Value1=65535)
- **15,428 object groups** separated by sentinels
- **Realistic object scales**: 38K-654K triangles per building object

## Chunk Relationships

### MPRR (Properties Record)
- **Purpose**: Records segmented properties; sentinel markers separate property sections, not building/object boundaries
- **Structure**: Pairs of ushort values (Value1, Value2)
- **Key Pattern**: Value1=65535 acts as sentinel/separator between property blocks

### MPRL (Placement List) ⭐ **DECODED OBJECT INSTANCE SYSTEM**
- **Purpose**: Object instance management with LOD control and state flags
- **Structure**: 24-byte entries with sophisticated field encoding
- **Decoded Fields** (from database pattern analysis):
  - `Unknown0`: **Object Category ID** (4630 = building type)
  - `Unknown2`: **State Flag** (-1 = active/enabled)
  - `Unknown4`: **Object Instance ID** (227 unique object instances)
  - `Unknown6`: **Property Flag** (32768 = 0x8000, consistent bit flag)
  - `Position`: **Local tile coordinates** (X: 11740-12264, Y: 40-185, Z: 9600-10130)
  - `Unknown14/Unknown16`: **LOD Control System**
    - (-1, 16383) = Full detail rendering (906 instances)
    - (0-5, 0) = LOD levels 0-5 (667 instances)
- **Key Insight**: Advanced object management system with LOD, not simple placements
- **Scale**: 1,573 object instances with rendering control (tile development_22_18)
- **Coordinate System**: Local tile space (requires XX*533.33 + YY*533.33 world offset)

### MSLK (Link Table)
- **Purpose**: Links placements to geometry fragments
- **Key Relationship**: MSLK.ParentIndex ↔ MPRL.Unknown4 (98.8% overlap)
- **Geometry Coverage**: 125.3% (over-indexing indicates hierarchical relationships)

### MSUR (Surface Records)
- **Purpose**: Geometry subdivision levels (LOD-like)
- **Key Insight**: These are **subdivision levels**, not object groups
- **Pattern**: 518,092 surfaces with only 1,301 unique SurfaceKeys

## Assembly Algorithm (Updated)

### 1. Traverse Container Hierarchy
 - Identify container nodes via `MSLK.MspiFirstIndex = -1`.

### 2. Map Placements to Geometry
- Use `MPRL.Unknown4 ↔ MSLK.ParentIndex` to map placements to geometry links

### 3. Assemble Faces
- Extract faces via `MSUR → MSVI` using indices from surface records
- Treat `MSUR.IndexCount` as diagnostic only, not an object identifier

## Implementation

### CLI Commands
- `pm4-analyze-data`: Analyze PM4 chunk relationships

### Key Classes
- `Pm4HierarchicalObjectAssembler`: Main assembly logic
- `Pm4DataAnalyzer`: Chunk relationship analysis
- `Pm4RegionLoader`: Cross-tile vertex reference resolution

## Performance Considerations

### Large Object Handling
- Building objects can contain 600K+ triangles
- Export process may take significant time for all ~15,400 objects
- Consider parallel processing or progressive export for large datasets

### Memory Usage
- Complete region loading merges 502 tiles (812K vertices, 1.9M indices)
- MSCN vertex remapping adds exterior vertex references
- Monitor memory usage during large object assembly

## Validation Results

### Before (Fragment-Based)
- 300-350 vertices per "object"
- Broken geometry with missing connections
- Thousands of tiny fragments instead of buildings

### After (MPRR-Based)
- 38K-654K triangles per building object
- Complete, connected building geometry
- Realistic building-scale objects (~15,400 total)

## Cross-Tile Reference Resolution

PM4 files reference vertices from adjacent tiles, requiring region loading:

### Implementation
- `Pm4RegionLoader` automatically loads up to 502 tiles
- `MscnRemapper` resolves cross-tile vertex references
- 12.8x vertex increase (63K → 812K) validates completeness

### Results
- Zero out-of-bounds vertex access after fix
- Complete building geometry instead of fragments
- Region loading enabled by default for all PM4 commands

## Conclusion

Recommended approach:

1. **Per-tile processing**: Isolate each PM4 tile
2. **Container traversal**: Use `MSLK` container nodes (`MspiFirstIndex = -1`) to drive assembly
3. **Placement mapping**: `MPRL.Unknown4 ↔ MSLK.ParentIndex`
4. **Faces from MSUR → MSVI**; treat `MSUR.IndexCount` as diagnostic
