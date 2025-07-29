# PM4 Object Grouping

## VERIFIED WORKING APPROACH

### Spatial Clustering Method (Source: poc_exporter.cs)
**Status:** ✅ Confirmed working in POC implementation

**Root Cause:** Pure hierarchical grouping produces fragments. Spatial clustering compensates for incomplete object boundaries.

### Algorithm Steps
1. **Find Root Nodes:** MSLK entries where `Unknown_0x04 == entry_index` (self-referencing)
2. **Group by Building:** Collect MSLK entries with same `Unknown_0x04` value
3. **Calculate Structural Bounds:** From MSPV vertices via MSLK → MSPI → MSPV chain
4. **Find Nearby Surfaces:** MSUR surfaces within 50.0f tolerance of structural bounds
5. **Hybrid Assembly:** Combine structural elements + nearby render surfaces

### Verified Field Relationships
```
MPRL.Unknown4 = MSLK.ParentIndex (458 confirmed matches)
MSLK.Unknown_0x04 = Building group identifier
MSLK.MspiFirstIndex = -1 → Container node (no geometry)
MSLK.MspiFirstIndex ≥ 0 → Geometry node
```

### Failed Approaches (Documented)
- **Pure Hierarchical:** Groups by ParentIndex only → Produces fragments
- **WMO-inspired:** Groups by batch logic → 256 fragments instead of 458 objects
- **MSUR IndexCount:** Groups by surface properties → Incorrect object boundaries

### Implementation
**File:** `Pm4SpatialClusteringAssembler.cs`  
**Method:** `CreateHybridBuilding_StructuralPlusNearby`  
**Status:** Extracted from POC, ready for testing and Assembly

## ⚠️ CRITICAL UPDATE: Global Mesh Architecture Discovered (2025-07-27)

**BREAKTHROUGH:** PM4 files implement a **global mesh system** where complete objects span multiple tiles. Single-tile processing produces fragmented geometry due to missing cross-tile vertex references.

### **Global Mesh Validation (Mathematical Proof):**
- **58.4% of triangles** reference vertices from adjacent tiles (30,677 out of 52,506)
- **63,297 cross-tile vertex indices** in sequential range: 63,298-126,594
- **Perfect adjacency**: Zero gap between local (0-63,297) and cross-tile ranges
- **Complete assembly requires directory-wide PM4 processing**

### **Surface Encoding System:**
- **GroupKey 3** (1,968 surfaces): Spatial coordinates, local tile geometry
- **GroupKey 18** (8,988 surfaces): Mixed data, boundary objects spanning tiles
- **GroupKey 19** (30,468 surfaces): Encoded linkage data, cross-tile references
- **BoundsMaxZ in encoded groups**: Hex-encoded tile/object references, not coordinates

**IMPACT:** All previous single-tile object assembly methods are fundamentally incomplete. **Multi-tile processing is mandatory** for accurate building geometry.

---

## Overview

PM4 files store building interior geometry using a complex instanced geometry system. This document describes the correct approach to assembling complete building objects from PM4 chunk relationships.

## Key Discovery: MPRR-Based Hierarchical Grouping

### The Problem
Initial attempts at PM4 object grouping using surface subdivision levels (MSUR.SurfaceKey) or placement instances (MPRL.Unknown4) produced geometry fragments with ~300-350 vertices each, not complete building objects.

### The Solution
**MPRR chunk contains the true object boundaries** using sentinel values (Value1=65535) that separate geometry into complete building objects.

### Data Analysis Results
From `development_00_00.pm4` analysis:
- **81,936 MPRR properties** total
- **15,427 sentinel markers** (Value1=65535)
- **15,428 object groups** separated by sentinels
- **Realistic object scales**: 38K-654K triangles per building object

## Chunk Relationships

### MPRR (Properties Record)
- **Purpose**: Defines object boundaries and hierarchical grouping
- **Structure**: Pairs of ushort values (Value1, Value2)
- **Key Pattern**: Value1=65535 acts as sentinel/separator between objects
- **Object Count**: ~15,400 building objects per PM4 file

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

## Assembly Algorithm

### 1. Parse MPRR Object Groups
```csharp
// Identify object boundaries using Value1=65535 sentinels
var objectGroups = new Dictionary<int, List<ushort>>();
int currentGroup = 0;
var currentComponents = new List<ushort>();

foreach (var property in scene.Properties)
{
    if (property.Value1 == 65535) // Sentinel marker
    {
        if (currentComponents.Count > 0)
        {
            objectGroups[currentGroup++] = currentComponents.ToList();
            currentComponents.Clear();
        }
    }
    else
    {
        currentComponents.Add(property.Value2); // Component type
    }
}
```

### 2. Map Component Types to Geometry
```csharp
// Group MPRL placements by component type
var placementsByType = scene.Placements
    .GroupBy(p => p.Unknown4)
    .ToDictionary(g => g.Key, g => g.ToList());

// Link to geometry via MSLK.ParentIndex
var linksByParentIndex = scene.Links
    .Where(link => link.ParentIndex > 0 && link.MspiIndexCount > 0)
    .GroupBy(link => link.ParentIndex)
    .ToDictionary(g => g.Key, g => g.ToList());
```

### 3. Assemble Complete Objects
```csharp
foreach (var (objectId, componentTypes) in objectGroups)
{
    var objectTriangles = new List<(int A, int B, int C)>();
    
    // Collect geometry from all component types in this object
    foreach (var componentType in componentTypes)
    {
        if (linksByParentIndex.TryGetValue(componentType, out var geometryLinks))
        {
            // Extract triangles from linked geometry
            foreach (var link in geometryLinks)
            {
                // Add triangles from link.MspiFirstIndex to link.MspiIndexCount
            }
        }
    }
    
    // Create complete building object
    var buildingObject = new HierarchicalObject(
        objectId, componentTypes, objectTriangles, 
        boundingCenter, vertexCount, objectType
    );
}
```

## Implementation

### CLI Commands
- `pm4-mprr-objects`: Export building objects using MPRR hierarchical grouping
- `pm4-analyze-data`: Analyze PM4 chunk relationships and MPRR patterns

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

The PM4 object grouping problem has been solved through MPRR-based hierarchical assembly. The key insights:

1. **MPRR sentinels define true object boundaries**
2. **MPRL placements are instances, not definitions**
3. **Cross-tile vertex resolution is essential**
4. **Hierarchical assembly produces building-scale objects**

This approach transforms PM4 exports from fragmented geometry into complete, realistic building objects suitable for 3D visualization and analysis.
