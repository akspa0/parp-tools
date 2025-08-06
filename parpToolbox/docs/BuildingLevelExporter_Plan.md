# Building-Level Exporter Implementation Plan

## Project Overview

**Goal**: Implement a CLI tool to export complete PM4 buildings as unified OBJ files using the **MPRR sentinel-based building boundary system**.

**Command**: `export-buildings <scene.db> [outDir]`

## 🎯 BREAKTHROUGH: MPRR Sentinel-Based Building Boundaries

**DEFINITIVE DISCOVERY**: The MPRR chunk contains the true building boundary system using sentinel values that produce complete building-scale objects.

### Core Principle

**Building Assembly Rule**: MPRR entries with `Value1 = 65535` act as sentinel markers that define building boundaries. All properties between consecutive sentinels belong to a single building object.

### Verified Scale
- **~15,427 sentinel markers** create **~15,428 building objects**
- **Building scale: 38,000-654,000 triangles** per building (realistic architecture)
- **Linear O(n) complexity** - simple and efficient implementation
- **Deterministic results** - same boundaries every time

## Implementation Requirements

### 1. Core Components

#### 1.1 BuildingLevelExporter Class
**File**: `src/PM4Rebuilder/BuildingLevelExporter.cs`

**Key Methods**:
- `ExportAllBuildings(string dbPath, string outputDir)` - Main export orchestrator
- `GetBuildingDefinitions(SqliteConnection conn)` - Query MPRR sentinel boundaries
- `ExportBuilding(SqliteConnection conn, BuildingDefinition building, string outputDir)` - Export single building
- `ClassifyBuilding(BuildingDefinition building)` - Categorize by triangle count and complexity
- `GenerateBuildingIndex(List<BuildingExportResult> results, string outputDir)` - Create summary CSV

#### 1.2 Data Structures
```csharp
public class BuildingDefinition
{
    public int BuildingId { get; set; }
    public int StartPropertyIndex { get; set; }
    public int EndPropertyIndex { get; set; }
    public List<Pm4Property> Properties { get; set; }
    public int EstimatedTriangleCount { get; set; }
    public BuildingClassification Classification { get; set; }
}

public class Pm4Property
{
    public int GlobalIndex { get; set; }
    public ushort Value1 { get; set; }     // 65535 = sentinel marker
    public ushort Value2 { get; set; }     // Component type
    public bool IsBoundarySentinel { get; set; }
}

public enum BuildingClassification
{
    Fragment,    // < 1K triangles
    Detail,      // 1K-10K triangles  
    Object,      // 10K-38K triangles
    Building,    // 38K-654K triangles (target scale)
    Complex      // > 654K triangles
}
```

### 2. Database Queries

#### 2.1 MPRR Sentinel Discovery Query
```sql
SELECT 
    p.GlobalIndex,
    p.Value1,
    p.Value2,
    p.IsBoundarySentinel
FROM Properties p
WHERE p.Pm4FileId = @fileId
ORDER BY p.GlobalIndex;
```

#### 2.2 Building Boundary Detection
```sql
-- Get sentinel positions to define building ranges
SELECT 
    ROW_NUMBER() OVER (ORDER BY GlobalIndex) as BuildingId,
    GlobalIndex as SentinelIndex,
    LEAD(GlobalIndex, 1, (SELECT MAX(GlobalIndex) + 1 FROM Properties WHERE Pm4FileId = @fileId)) 
        OVER (ORDER BY GlobalIndex) as NextSentinelIndex
FROM Properties 
WHERE Pm4FileId = @fileId AND IsBoundarySentinel = 1
ORDER BY GlobalIndex;
```

#### 2.3 Building Geometry Assembly Query
```sql
-- Get all triangles for a building range defined by MPRR sentinels
SELECT 
    t.VertexA,
    t.VertexB, 
    t.VertexC,
    v1.X as V1_X, v1.Y as V1_Y, v1.Z as V1_Z,
    v2.X as V2_X, v2.Y as V2_Y, v2.Z as V2_Z,
    v3.X as V3_X, v3.Y as V3_Y, v3.Z as V3_Z
FROM Properties p
JOIN Links l ON (l.GlobalIndex BETWEEN p.StartPropertyIndex AND p.EndPropertyIndex)
JOIN Surfaces s ON s.GlobalIndex = l.ReferenceIndex
JOIN Triangles t ON (t.GlobalIndex BETWEEN s.MsviFirstIndex AND s.MsviFirstIndex + s.IndexCount - 1)
JOIN Vertices v1 ON v1.GlobalIndex = t.VertexA
JOIN Vertices v2 ON v2.GlobalIndex = t.VertexB  
JOIN Vertices v3 ON v3.GlobalIndex = t.VertexC
WHERE p.BuildingId = @buildingId AND p.Pm4FileId = @fileId;
```

### 3. MPRR Export Logic

#### 3.1 Building Classification Algorithm
```csharp
private BuildingClassification ClassifyBuilding(BuildingDefinition building)
{
    var triangleCount = building.EstimatedTriangleCount;
    
    return triangleCount switch
    {
        < 1000 => BuildingClassification.Fragment,
        < 10000 => BuildingClassification.Detail,
        < 38000 => BuildingClassification.Object,
        <= 654000 => BuildingClassification.Building,  // Target scale
        _ => BuildingClassification.Complex
    };
}
```

#### 3.2 MPRR Building Export Process
1. **Query MPRR sentinels** to identify building boundaries
2. **Create building ranges** between consecutive sentinel markers  
3. **Extract all geometry** within each building range
4. **Classify buildings** by triangle count and complexity
5. **Export unified OBJ files** with complete building geometry

#### 3.3 OBJ File Generation
- **Naming Pattern**: `building_{BuildingId}_triangles_{count}.obj`
  - Example: `building_001_triangles_45672.obj` (Complete building)
  - Example: `building_156_triangles_234891.obj` (Large complex)
- **Geometry Assembly**: Direct triangle export from MPRR-defined ranges
- **Coordinate System**: Maintain original PM4 coordinates (no transformations)

#### 3.3 Building Index CSV
**Columns**: `ParentIndex,Classification,TypeComposition,ComponentCount,GeometryLinks,VertexCount,FaceCount,ObjFileName`

**Example Rows**:
```
11,WallRoof,"1:10+3:6",16,16,1247,834,building_0011_types_1-3.obj
37,WallRoof,"1:10+3:6",16,16,1183,798,building_0037_types_1-3.obj
5206,Platform,"2:25",25,25,4832,3216,building_5206_types_2.obj
```

### 4. CLI Integration

#### 4.1 Program.cs Updates
```csharp
else if (args[0].Equals("export-buildings", StringComparison.OrdinalIgnoreCase))
{
    if (args.Length < 2)
    {
        Console.WriteLine("ERROR: export-buildings command requires <scene.db> argument.");
        return 1;
    }
    
    string dbPath = args[1];
    string outDir = args.Length >= 3 ? args[2] : 
        Path.Combine(Directory.GetCurrentDirectory(), "project_output", DateTime.Now.ToString("yyyyMMdd_HHmmss"));
    
    Directory.CreateDirectory(outDir);
    int exitCode = BuildingLevelExporter.ExportAllBuildings(dbPath, outDir);
    return exitCode;
}
```

#### 4.2 Usage String Update
```
PM4Rebuilder Commands:
  export-subcomponents <scene.db> [outDir] - Export all sub-components as individual OBJ files
  export-buildings <scene.db> [outDir]     - Export complete buildings as unified OBJ files
  analyze-building-groups <scene.db> [outDir] - Analyze building aggregation patterns
```

### 5. Validation Strategy

#### 5.1 Sample Buildings for Testing
- **ParentIndex 11**: Wall+Roof (Type 1:10 + Type 3:6) - Expected: house-like structure
- **ParentIndex 37**: Wall+Roof (Type 1:10 + Type 3:6) - Expected: similar house structure  
- **ParentIndex 5206**: Platform (Type 2:25) - Expected: large floor/platform
- **ParentIndex 6**: Container+Roof (Type 0:2 + Type 3:2) - Expected: small roofed structure

#### 5.2 Validation Checklist
- [ ] Buildings export as single unified OBJ files
- [ ] Type composition matches expected patterns (walls+roofs, floors, etc.)
- [ ] Geometry assembly produces recognizable building shapes
- [ ] Building index accurately summarizes all exports
- [ ] No missing or duplicate geometry compared to sub-component exports
- [ ] Performance acceptable for full dataset (8,488 sub-components → ~hundreds of buildings)

### 6. Error Handling & Edge Cases

#### 6.1 Data Quality Issues
- **Incomplete buildings**: Flag buildings with suspicious patterns (e.g., roofs without walls)
- **Missing geometry**: Handle Links with `MspiFirstIndex = -1` (containers)
- **Database corruption**: Graceful handling of malformed JSON fields

#### 6.2 Export Issues  
- **Large buildings**: Memory management for buildings with 100+ components
- **Duplicate vertices**: Vertex deduplication across components within same building
- **File system**: Handle invalid characters in building type lists for filenames

### 7. Future Enhancements

#### 7.1 Advanced Classification
- **Architectural analysis**: Detect specific building types (houses, towers, bridges)
- **Complexity scoring**: Rate buildings by component diversity and size
- **Material consistency**: Validate material usage patterns within buildings

#### 7.2 District-Level Analysis
- **Building clustering**: Group buildings into neighborhoods/districts
- **Spatial relationships**: When X,Y,Z coordinates become available
- **Infrastructure**: Detect roads, utilities connecting buildings

## Implementation Timeline

1. **Phase 1**: Core BuildingLevelExporter class and data structures
2. **Phase 2**: Database queries and building discovery logic  
3. **Phase 3**: OBJ export and file generation
4. **Phase 4**: CLI integration and testing
5. **Phase 5**: Validation with sample buildings and documentation

## Success Criteria

- ✅ CLI command `export-buildings` works end-to-end
- ✅ Sample buildings (11, 37, 5206) export as recognizable structures
- ✅ Building classification correctly identifies wall+roof vs platform vs complex buildings
- ✅ Export performance handles full PM4 database efficiently
- ✅ Building index provides accurate summary statistics
- ✅ Documentation updated with usage examples and results

---

**Note**: This plan builds directly on the Type_0x01 discovery and existing `BulkSubComponentExporter` and `BuildingAggregationAnalyzer` implementations. The building-level exporter represents the natural evolution from individual polygon export to complete building reconstruction.
