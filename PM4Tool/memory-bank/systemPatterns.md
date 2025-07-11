# System Patterns: Service-Oriented Architecture for PM4 Processing

> 🔗 Definitive linkage spec: see `mslk_linkage_reference.md` for how MSLK connects to other chunks.
> 📄 Quick cheat-sheets live in `memory-bank/overview/`.

**Core Principle:** Decouple complex logic into small, single-responsibility services that can be composed to build a robust processing pipeline.

**Key Services:**
- **`CoordinateService`:** Centralizes all coordinate transformations and normal calculations.
- **`Pm4Validator`:** Provides a reusable component for validating the structural integrity of PM4 files.
- **`RenderMeshBuilder`:** Constructs render-ready geometry from PM4 data.
- **`Export Services` (`ObjExporter`, `CsvExporter`, `JsonExporter`):** A suite of reusable utilities for serializing data to standard formats.
- **`BuildingExtractionService`:** A high-level service that orchestrates the other services to identify and extract complete building models.

## Current Architecture: Individual Object Analysis ✅

### Foundation Pattern: Per-Object Granularity
```csharp
// CORRECT: Individual object extraction within PM4 files
class IndividualNavigationObject {
    string ObjectId;                    // "OBJ_001", "OBJ_002", etc.
    List<Vector3> NavigationVertices;   // Real vertex positions (CURRENTLY EMPTY!)
    List<Face> NavigationTriangles;     // Real triangle data (CURRENTLY EMPTY!)
    BoundingBox3D NavigationBounds;     // Real bounds (CURRENTLY FAKE!)
}

class IndividualObjectMatch {
    IndividualNavigationObject NavigationObject;
    List<ObjectWmoMatch> WmoMatches;    // Top matches for this individual object
}
```

### Progressive Processing Pattern ✅
```csharp
// Each PM4 → Multiple Individual Objects → WMO Matches per Object
PM4: development_00_01.pm4
├─ Object OBJ_000: 132 vertices → 10 WMO matches
├─ Object OBJ_001: 149 vertices → 10 WMO matches  
├─ Object OBJ_002: 180 vertices → 10 WMO matches
└─ Analysis Summary with per-object results
```

## CRITICAL FLAW: Fake Implementation Pattern ❌

### Current Broken Pattern:
```csharp
// FAKE PM4 PARSING - COMPLETELY WRONG!
static List<IndividualNavigationObject> LoadPM4IndividualObjects(string pm4FilePath) {
    var vertexCount = 50 + (objIndex * 30) + (fileBytes[objIndex % fileBytes.Length] % 200); // RANDOM!
    NavigationVertices = new List<Vector3>(),     // EMPTY! No real data!
    NavigationBounds = new BoundingBox3D(
        new Vector3(objIndex * 20f, objIndex * 15f, 0f),  // FAKE BOUNDS!
        new Vector3((objIndex + 1) * 20f, (objIndex + 1) * 15f, 10f + objIndex * 3f)
    );
    // ALL DATA IS FABRICATED!
}
```

### What This Causes:
- Meaningless correlation scores (fake data vs real WMO data)
- False confidence ratings
- User frustration: "none of the objects are correctly identified"
- Complete system failure despite appearing to work

## Required Pattern: Real Geometric Correlation

### 1. Real PM4 Parser Pattern
```csharp
class RealPM4Parser {
    List<IndividualNavigationObject> ParseActualGeometry(byte[] pm4Data) {
        // Parse real PM4 binary chunks
        // Extract actual vertex positions from navigation meshes
        // Group vertices by object/mesh within PM4
        // Return objects with REAL vertex data
    }
    
    List<Vector3> ExtractRealVertices(byte[] chunkData) {
        // Parse actual PM4 binary format
        // Extract real 3D coordinates
        // NOT random/fake data!
    }
}
```

### 2. Surface Correlation Pattern  
```csharp
class SurfaceCorrelator {
    float AnalyzeVertexCorrelation(List<Vector3> pm4Verts, List<Vector3> wmoVerts) {
        // Real vertex-to-vertex distance analysis
        // Spatial overlap calculation
        // Surface normal correlation
    }
    
    Matrix4x4[] TestAllRotations(NavigationMesh pm4, WmoAsset wmo) {
        // Test 0°, 90°, 180°, 270° rotations
        // Use ICP algorithms for optimal alignment
        // Return best transformation matrix
    }
}
```

### 3. geometry3Sharp Integration Pattern
```csharp
using g3;

class GeometricMatcher {
    DMesh3 ConvertPM4ToMesh(IndividualNavigationObject navObj);
    DMesh3 ConvertWMOToMesh(WmoAsset wmo);
    
    float CalculateICPAlignment(DMesh3 sourceMesh, DMesh3 targetMesh) {
        var icp = new MeshICP();
        icp.SetSource(sourceMesh);
        icp.SetTarget(targetMesh);
        return icp.Solve(); // Real geometric alignment score
    }
}
```

## User's "Negative Mold" Theory Integration

### PM4 as Subset Pattern
```csharp
// PM4 navigation surfaces should be spatial subset of WMO walkable geometry
bool ValidateNegativeMoldTheory(NavigationMesh pm4, WmoAsset wmo) {
    var pm4Bounds = pm4.CalculateActualBounds();
    var wmoBounds = wmo.BoundingBox;
    
    // PM4 should fit INSIDE WMO bounds
    return wmoBounds.Contains(pm4Bounds) && 
           CalculateSpatialOverlap(pm4, wmo) > 0.7f;
}
```

## Multi-Pass Progressive Architecture (KEEP) ✅

### Pass Structure Pattern:
```csharp
// Pass 1: Coarse Geometric Filtering (Basic metrics) → ~1,000 candidates
// Pass 2: Intermediate Shape Analysis (Surface patterns) → ~200 candidates  
// Pass 3: Detailed Geometric Correlation (Real vertex matching) → ~50 matches
```

### Output Pattern (KEEP):
```
PM4: development_00_01.pm4/
├─ pass1_coarse.txt        // Basic filtering results
├─ pass2_intermediate.txt  // Shape analysis results
├─ pass3_detailed.txt      // Final geometric correlation
└─ analysis_summary.txt    // Complete per-object breakdown
```

## File Processing Pattern (KEEP) ✅

### Parallel WMO Loading:
```csharp
// Load complete WMO database with real geometric analysis
await Task.Run(() => {
    Parallel.ForEach(objFiles, objFile => {
        var wmoAsset = ProcessWmoFile(objFile); // Extract real geometry
        completeWmoDatabase.Add(wmoAsset);
    });
});
```

### Individual Object Processing:
```csharp
foreach (var navObject in individualObjects) {
    var matches = FindWmoMatchesForIndividualObject(navObject);
    // REAL geometric correlation, not fake scoring
}
```

## Critical Fix Required

### Replace Fake Implementation:
1. **Remove all fake data generation**
2. **Implement real PM4 binary parsing**
3. **Use geometry3Sharp for surface correlation**
4. **Add rotation analysis (0°, 90°, 180°, 270°)**
5. **Generate meaningful confidence scores**

### Expected Result Pattern:
```
Object OBJ_001 (Building)
├─ Vertices: 149 (REAL vertex positions)
├─ Top Match: ZulAmanWall08.obj (87.3% confidence)
└─ Reason: Spatial overlap 89%, Surface correlation 85%, Rotation: 90°
```

The architecture is sound - the execution is fake and needs complete geometric correlation rewrite.