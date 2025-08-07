# PM4 OBJ Export Implementation Plan

## Current Status Assessment
- Our tool's refactoring removed key functionality: previously achieved per-object OBJ exports
- Current implementation produces multiple fragmented files per object group
- MSUR raw fields alone don't produce coherent single-object outputs

## Complete PM4 OBJ Export Pipeline

### 1. Data Loading & Preparation
- Load PM4 data structures (MPRL, MPRR, MSVI, MSUR, MSLK chunks)
- Process vertex and index buffers
- Build initial reference maps and connectivity graphs
- Load adjacent tiles if implementing global loading

### 2. Initial Geometric Analysis
- Calculate bounding volumes for primitive groups
- Analyze vertex connectivity patterns
- Identify potential object boundaries based on geometric discontinuities
- Build spatial relationships between surface groups

### 3. Multi-level Grouping Strategy

#### Field-Based Classification (Current Focus)
- Group by MSUR raw fields (most promising approach)
- Utilize FlagsOrUnknown_0x00 for primary categorization
- Use Unknown_0x02 for secondary subdivision

#### Structural Analysis
- Analyze MPRL → MSLK → MSUR chains for hierarchical relationships
- Map parent-child relationships between object components
- Identify sentinel values (65535) in MPRR as section delimiters

#### Spatial Merging
- Calculate overlapping bounding boxes
- Apply proximity-based merging with configurable thresholds
- Use adjacency graphs to identify connected components
- Implement hierarchical clustering for progressive object assembly

### 4. Mesh Optimization & Assembly
- Deduplicate vertices within and across groups
- Apply vertex welding with configurable tolerance
- Optimize index buffers for merged objects
- Generate coherent UV mapping for texture coordinates

### 5. OBJ/MTL Generation
- Create hierarchical object structures with proper grouping
- Generate materials based on available texture information
- Write optimized OBJ format with efficient face encoding
- Include metadata for group relationships

### 6. Post-processing & Validation
- Validate mesh integrity and topology
- Detect and repair potential mesh errors
- Generate statistics and quality metrics
- Produce visualization aids for object boundaries

## Technical Implementation Details

### Bounding Volume Calculation
```csharp
// Pseudocode for bounding box calculation
private static (Vector3 min, Vector3 max) CalculateBoundingBox(List<Vector3> vertices)
{
    Vector3 min = new Vector3(float.MaxValue);
    Vector3 max = new Vector3(float.MinValue);
    foreach (var vertex in vertices)
    {
        min = Vector3.Min(min, vertex);
        max = Vector3.Max(max, vertex);
    }
    return (min, max);
}
```

### Spatial Proximity Analysis
```csharp
// Pseudocode for bounding box overlap calculation
private static float CalculateOverlapPercentage(BoundingBox a, BoundingBox b)
{
    // Calculate intersection volume
    Vector3 overlapMin = Vector3.Max(a.Min, b.Min);
    Vector3 overlapMax = Vector3.Min(a.Max, b.Max);
    
    if (overlapMax.X < overlapMin.X || overlapMax.Y < overlapMin.Y || overlapMax.Z < overlapMin.Z)
        return 0f; // No overlap
        
    float overlapVolume = (overlapMax.X - overlapMin.X) * 
                          (overlapMax.Y - overlapMin.Y) * 
                          (overlapMax.Z - overlapMin.Z);
                          
    float aVolume = (a.Max.X - a.Min.X) * (a.Max.Y - a.Min.Y) * (a.Max.Z - a.Min.Z);
    float bVolume = (b.Max.X - b.Min.X) * (b.Max.Y - b.Min.Y) * (b.Max.Z - b.Min.Z);
    
    // Return overlap as percentage of smaller volume
    return overlapVolume / Math.Min(aVolume, bVolume);
}
```

### Hierarchical Clustering Algorithm
```csharp
// Pseudocode for hierarchical clustering of object groups
private static List<ObjectGroup> ClusterObjectGroups(List<ObjectGroup> initialGroups, float overlapThreshold)
{
    var clusters = new List<ObjectGroup>(initialGroups);
    bool merged;
    
    do {
        merged = false;
        
        for (int i = 0; i < clusters.Count; i++)
        {
            for (int j = i + 1; j < clusters.Count; j++)
            {
                float overlap = CalculateOverlapPercentage(clusters[i].BoundingBox, clusters[j].BoundingBox);
                
                if (overlap > overlapThreshold)
                {
                    // Merge groups i and j
                    clusters[i] = MergeGroups(clusters[i], clusters[j]);
                    clusters.RemoveAt(j);
                    merged = true;
                    break;
                }
            }
            
            if (merged) break;
        }
    } while (merged);
    
    return clusters;
}
```

## Implementation Roadmap

### Phase 1: Restore & Enhance Basic Functionality
1. **Implement comprehensive bounding volume calculation** for all geometry
2. **Develop spatial proximity analyzer** with configurable thresholds
3. **Create hierarchical clustering algorithm** for object assembly
4. **Restore and enhance CLI options** with new spatial merging capabilities
5. **Implement progress reporting** with detailed statistics

### Phase 2: Advanced Object Assembly
1. **Analyze MPRL → MSLK → MSUR chains** for hierarchical relationships
2. **Implement parent-child relationship mapping** between components
3. **Develop adjacency graphs** for connected component analysis
4. **Add configurable merging thresholds** based on object types
5. **Create visualization tools** for object boundaries

### Phase 3: Optimization & Global Integration
1. **Implement global tile loading** to resolve cross-boundary references
2. **Develop unified vertex/index pool** across tiles
3. **Create advanced mesh optimization** techniques
4. **Implement UV mapping preservation** during merges
5. **Add metadata export** for debugging and analysis

## Success Criteria
- Single coherent OBJ file per logical object (building, terrain section, etc.)
- Significantly reduced file count compared to current implementation
- Maintained geometric integrity and topology during merges
- Clear hierarchical organization in exported files
- Documented spatial relationships between object components
- Comprehensive progress reporting and statistics
