# Product Context: Real PM4-WMO Geometric Correlation System

> For concise technical docs, check `memory-bank/overview/`.

## Project Vision & Purpose

### Core Problem Statement
**Challenge**: Match individual navigation meshes within PM4 files to their corresponding WMO (World Model Object) visual geometry files through real geometric correlation analysis.

**Current Crisis**: The implementation generates fake data instead of performing real geometric analysis, making all results meaningless and causing justified user frustration.

### User's Revolutionary "Negative Mold" Theory
> "PM4s are like a negative mold of the actual data in the ADT"

**Key Insight**: PM4 files contain navigation-relevant surfaces (walkable areas) while WMO files contain complete visual geometry. The PM4 navigation meshes should spatially correlate with specific walkable surfaces within the corresponding WMO models.

### User's Core Requirements (FROM ORIGINAL SPECIFICATION)

1. **Individual Object Analysis**: Each PM4 contains multiple navigation objects that must be analyzed separately
2. **Real Geometric Correlation**: Surface-to-surface analysis using actual vertex positions, not bounding box comparisons  
3. **Rotation Handling**: Test 0°, 90°, 180°, 270° orientations for optimal geometric alignment
4. **Progressive Refinement**: Multi-pass filtering to eliminate false positives through increasingly detailed analysis
5. **Meaningful Results**: Show which specific WMO models correspond to each individual navigation object

### What Doesn't Work (User Feedback)
- ❌ **Bounding box comparison**: "comparing just bounding boxes is going to cause us to have a lot of false-positives"
- ❌ **Volume/complexity scoring**: "volume and complexity fit will never find the real matches"  
- ❌ **Filename matching**: "how the fuck a filename matching thing will work, when there is no filename information in the pm4 for any object?!"
- ❌ **Fake data generation**: "why would we spend time building bullshit nonsense all this time?!"

## Product Requirements

### Functional Requirements

#### Core Geometric Analysis ⚠️ **CRITICAL**
1. **Real PM4 Parsing**: Extract actual navigation vertices from PM4 binary data (not fake random numbers)
2. **Surface-to-Surface Correlation**: Compare actual geometric surfaces between PM4 navigation meshes and WMO geometry
3. **Iterative Closest Point (ICP) Analysis**: Use proven geometric algorithms for surface alignment scoring
4. **Multi-Rotation Testing**: Analyze geometric correlation at 0°, 90°, 180°, 270° rotations
5. **Spatial Overlap Assessment**: Verify PM4 navigation bounds fit within WMO walkable surface areas

#### Individual Object Processing ✅ **WORKING**
1. **Per-Object Granularity**: Analyze each navigation object within a PM4 file separately  
2. **Object Classification**: Categorize objects by complexity (Simple Structure, Building, Complex Building, etc.)
3. **Top Match Analysis**: Show top 10 WMO matches per individual navigation object
4. **Confidence Scoring**: Generate meaningful confidence based on real geometric correlation

#### Progressive Refinement System ✅ **ARCHITECTURE WORKS**
1. **Pass 1 - Coarse Filtering**: Basic geometric metrics to reduce ~10,000 WMOs to ~1,000 candidates
2. **Pass 2 - Shape Analysis**: Surface pattern analysis to narrow to ~200 candidates  
3. **Pass 3 - Detailed Correlation**: Precise geometric matching for final ~50 matches
4. **Output Generation**: Separate analysis files for each pass plus comprehensive summary

### Non-Functional Requirements

#### Performance Requirements
- **Database Loading**: Load and index 1000+ WMO files in parallel within reasonable time
- **Progressive Processing**: Handle multiple PM4 files concurrently without memory issues
- **Real-time Feedback**: Console progress reporting during long-running geometric analysis
- **Scalability**: Support analysis of hundreds of PM4 files in batch processing mode

#### Quality Requirements  
- **Geometric Accuracy**: All correlation scores must be based on real geometric analysis
- **No Fake Data**: Zero tolerance for simulated/random data generation
- **Meaningful Results**: Match confidence scores must reflect actual geometric similarity
- **Rotation Invariance**: System must find matches regardless of object orientation

#### Integration Requirements
- **WoWToolbox.Core Integration**: Leverage existing real PM4 binary parsing capabilities
- **geometry3Sharp Library**: Use professional geometric algorithms for surface correlation
- **File Format Compatibility**: Output results in human-readable text format for analysis

## User Experience Goals

### Primary User Workflow
```
1. User runs system against PM4 test dataset
2. System loads complete WMO database and builds spatial index  
3. For each PM4 file:
   a. Extract individual navigation objects using real binary parsing
   b. Find WMO matches through progressive geometric correlation
   c. Generate per-object analysis with meaningful confidence scores
4. User reviews results showing which WMO models match each navigation object
```

### Expected Output Quality
```
PM4: development_00_01.pm4

OBJECT OBJ_001 (Building Navigation Mesh)
├─ Vertices: 149 (REAL positions from PM4 binary)
├─ Triangles: 49 (REAL navigation triangles)  
├─ Spatial Bounds: Calculated from actual vertex positions
├─ Top Match: ZulAmanWall08.obj (87.3% confidence)
├─ Geometric Analysis:
│   ├─ Surface Overlap: 89% (ICP alignment score)
│   ├─ Optimal Rotation: 90° clockwise for best alignment
│   ├─ Vertex Correlation: 23 PM4 vertices within 0.5 units of WMO surfaces
│   └─ Negative Mold Validation: PM4 bounds fit within WMO walkable areas
└─ Match Reason: Strong geometric correlation with optimal rotation alignment
```

### Success Criteria
1. **Real Matches Found**: System identifies correct WMO models for navigation objects
2. **Meaningful Confidence**: High confidence scores correlate with visually obvious matches
3. **Rotation Detection**: System finds matches regardless of object orientation  
4. **False Positive Elimination**: Progressive refinement removes irrelevant WMO candidates
5. **User Validation**: Results make intuitive sense and can be verified visually

## Technical Product Architecture

### Core Components

#### Real Geometric Engine ⚠️ **NEEDS IMPLEMENTATION**
```csharp
class GeometricCorrelationEngine {
    // Convert PM4 navigation mesh to geometry3Sharp format
    DMesh3 ConvertPM4ToMesh(IndividualNavigationObject navObj);
    
    // Test geometric alignment with rotation variants
    RotationResult FindOptimalAlignment(DMesh3 pm4Mesh, DMesh3 wmoMesh);
    
    // Calculate surface-to-surface correlation score
    float AnalyzeRealSurfaceCorrelation(IndividualNavigationObject pm4Obj, WmoAsset wmo);
}
```

#### Progressive Analysis Pipeline ✅ **ARCHITECTURE EXISTS**
```csharp
// Multi-pass refinement system
Pass1: CoarseGeometricFilter    → ~1,000 candidates
Pass2: IntermediateShapeAnalyzer → ~200 candidates  
Pass3: DetailedGeometricCorrelator → ~50 final matches
```

#### Individual Object Processor ✅ **WORKING**
```csharp
class IndividualObjectProcessor {
    // Extract multiple navigation objects from single PM4
    List<IndividualNavigationObject> LoadPM4IndividualObjects(string pm4FilePath);
    
    // Find WMO matches for specific navigation object
    List<ObjectWmoMatch> FindWmoMatchesForIndividualObject(IndividualNavigationObject navObj);
}
```

## Product Validation

### Current Status Assessment
- ✅ **Architecture Design**: Individual object processing structure is sound
- ✅ **Output Format**: Per-PM4 analysis files with object breakdown working
- ✅ **WMO Database**: Parallel loading and indexing of 1000+ files working
- ❌ **Geometric Correlation**: Currently generates fake data instead of real analysis
- ❌ **PM4 Parsing**: Uses random data generation instead of binary parsing
- ❌ **Confidence Scores**: All scores are meaningless due to fake data

### Critical Implementation Gap
**The entire correlation system is built on fake data generation**, making all results meaningless despite appearing to work. This is the core reason for user frustration and system failure.

### User Frustration Context
The user has been extremely patient through multiple iterations of fake implementations and specifically called out the need for:
- Real geometric correlation (not bounding box comparison)
- Actual PM4 vertex data (not random generation)  
- Surface-to-surface analysis (not volume/complexity scoring)
- Rotation handling (for optimal alignment)
- Meaningful results (not fake confidence scores)

## Product Success Definition

### Immediate Success (Phase 1)
- **Real PM4 Data**: System extracts actual navigation vertices from PM4 binary format
- **Basic Correlation**: geometry3Sharp integration provides real geometric correlation scores
- **Meaningful Output**: Match confidence reflects actual geometric similarity

### Short-term Success (Phase 2)  
- **Rotation Analysis**: System finds optimal alignment through multi-rotation testing
- **Surface Correlation**: ICP algorithms provide accurate surface-to-surface matching
- **Progressive Refinement**: Multi-pass system effectively eliminates false positives

### Long-term Success (Phase 3)
- **Production Ready**: System handles hundreds of PM4 files with consistent quality
- **User Validation**: Results correlate with visually obvious geometric matches
- **Negative Mold Theory**: PM4 navigation surfaces correctly correlate with WMO walkable areas

The product vision is clear and the architecture is sound. The implementation needs complete replacement of fake data generation with real geometric correlation algorithms. 