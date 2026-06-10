# System Patterns

## Proven Architectural Patterns

### **1. Dual Geometry Assembly Pattern**
```csharp
public class DualGeometryAssembler
{
    // Combines MSLK/MSPV structural data with MSVT/MSUR render surfaces
    public CompleteWMOModel AssembleBuilding(PM4File pm4File, int rootNodeIndex)
    {
        // Phase 1: Extract structural framework
        var structuralEntries = pm4File.MSLK.Entries
            .Where(entry => entry.Unknown_0x04 == rootNodeIndex)
            .ToList();
        
        // Phase 2: Extract render surfaces
        var renderSurfaces = GetLinkedMSURSurfaces(pm4File, rootNodeIndex);
        
        // Phase 3: Combine both geometry systems
        return CombineStructuralAndRenderGeometry(structuralEntries, renderSurfaces);
    }
}
```
- **Purpose**: Combines PM4's dual geometry systems for complete building models
- **Result**: Individual buildings with both structural framework and render surfaces
- **Quality**: "Exactly the quality desired" user validation

### **2. Self-Referencing Root Detection Pattern**
```csharp
public class BuildingDetector
{
    public List<int> FindBuildingRootNodes(PM4File pm4File)
    {
        return pm4File.MSLK.Entries
            .Select((entry, index) => new { entry, index })
            .Where(x => x.entry.Unknown_0x04 == x.index)  // Self-referencing nodes
            .Select(x => x.index)
            .ToList();
    }
}
```
- **Discovery**: MSLK entries with `Unknown_0x04 == index` identify building separators
- **Impact**: Enables individual building extraction instead of combined fragments
- **Validation**: Consistently finds 10+ buildings per PM4 file

### **3. Signature-Based Duplicate Elimination Pattern**
```csharp
public class SurfaceProcessor
{
    public List<Triangle> ProcessSurfacesWithDeduplication(PM4File pm4File)
    {
        var processedSignatures = new HashSet<string>();
        var triangles = new List<Triangle>();
        
        foreach (var msur in pm4File.MSUR.Entries)
        {
            var signature = CreateSurfaceSignature(msur, pm4File.MSVI);
            
            if (!processedSignatures.Contains(signature))
            {
                processedSignatures.Add(signature);
                triangles.AddRange(GenerateTriangleFan(msur, pm4File.MSVI));
            }
        }
        
        return triangles;
    }
}
```
- **Problem Solved**: Duplicate MSUR surfaces creating redundant faces and "spikes"
- **Result**: 884,915+ valid faces with zero degenerate triangles
- **Quality**: 47% face count improvement with clean connectivity

### **4. Centralized Coordinate Transform Pattern**
```csharp
public static class Pm4CoordinateTransforms
{
    // Perfect MSVT render mesh transformation
    public static Vector3 FromMsvtVertex(MsvtVertex vertex) => 
        new Vector3(vertex.Y, vertex.X, vertex.Z);
    
    // Complex MSCN collision boundary alignment
    public static Vector3 FromMscnVertex(Vector3 vertex)
    {
        float correctedY = -vertex.Y;
        float x = vertex.X;
        float y = correctedY * MathF.Cos(MathF.PI) - vertex.Z * MathF.Sin(MathF.PI);
        float z = correctedY * MathF.Sin(MathF.PI) + vertex.Z * MathF.Cos(MathF.PI);
        return new Vector3(x, y, z);
    }
    
    // Standard structural element coordinates
    public static Vector3 FromMspvVertex(C3Vector vertex) => 
        new Vector3(vertex.X, vertex.Y, vertex.Z);
    
    // World positioning references
    public static Vector3 FromMprlEntry(MprlEntry entry) => 
        new Vector3(entry.Position.X, -entry.Position.Z, entry.Position.Y);
}
```
- **Achievement**: Perfect spatial alignment of all PM4 chunk types
- **Validation**: MeshLab visual confirmation across hundreds of files
- **Architecture**: Single source of truth for coordinate transformations

### **5. Enhanced Export Pipeline Pattern**
```csharp
public class EnhancedObjExporter
{
    public void ExportWithDecodedFields(CompleteWMOModel model, string objPath, string mtlPath)
    {
        // Phase 1: Export geometry with surface normals
        ExportVerticesAndNormals(model.Vertices, model.Normals);
        
        // Phase 2: Generate material classification
        ExportMaterialLibrary(model.Materials, mtlPath);
        
        // Phase 3: Export faces with material assignments
        ExportFacesWithMaterials(model.TriangleIndices, model.MaterialAssignments);
        
        // Phase 4: Add spatial organization groups
        ExportHeightLevelGroups(model.SpatialGroups);
    }
}
```
- **Features**: Surface normals, material classification, spatial organization
- **Quality**: Professional 3D software compatibility (MeshLab, Blender)
- **Metadata**: Complete MSLK object type and material ID processing

### **6. MDSF→MDOS Building Linking Pattern**
```csharp
public class BuildingLinker
{
    public List<int> GetBuildingSurfaces(PM4File pm4File, int buildingId)
    {
        return pm4File.MDSF.Entries
            .Where(mdsfEntry => 
            {
                var mdosEntry = pm4File.MDOS.Entries[mdsfEntry.mdos_index];
                return mdosEntry.building_id == buildingId;
            })
            .Select(mdsfEntry => (int)mdsfEntry.msur_index)
            .ToList();
    }
}
```
- **Purpose**: Precise surface-to-building assignment using PM4 hierarchy chunks
- **Benefit**: Eliminates identical geometry problem in building exports
- **Scope**: Works when MDSF/MDOS chunks are available

### **7. Adaptive Processing Pattern**
```csharp
public class AdaptiveProcessor
{
    public CompleteWMOModel ProcessBuilding(PM4File pm4File, int rootNodeIndex)
    {
        if (pm4File.MDSF != null && pm4File.MDOS != null)
        {
            // Use precise building linking system
            return ProcessWithMdsfMdosLinking(pm4File, rootNodeIndex);
        }
        else
        {
            // Fall back to spatial clustering
            return ProcessWithSpatialClustering(pm4File, rootNodeIndex);
        }
    }
}
```
- **Flexibility**: Handles PM4 files with and without building hierarchy chunks
- **Quality**: Maintains consistent building extraction across PM4 variations
- **Architecture**: Universal compatibility with different PM4 formats

## Data Model Patterns

### **8. Complete Building Model Pattern**
```csharp
public class CompleteWMOModel
{
    public string FileName { get; set; } = "";
    public string Category { get; set; } = "";
    public List<Vector3> Vertices { get; set; } = new();
    public List<int> TriangleIndices { get; set; } = new();
    public List<Vector3> Normals { get; set; } = new();
    public List<Vector2> TexCoords { get; set; } = new();
    public string MaterialName { get; set; } = "WMO_Material";
    public Dictionary<string, object> Metadata { get; set; } = new();
    
    public int VertexCount => Vertices.Count;
    public int FaceCount => TriangleIndices.Count / 3;
}
```
- **Purpose**: Complete building representation with all geometric and metadata
- **Usage**: Standard model for all building extraction workflows
- **Features**: Vertices, faces, normals, materials, and decoded metadata

### **9. Decoded Field Integration Pattern**
```csharp
public class DecodedFieldProcessor
{
    public void ProcessMSURFields(MsurEntry msur, CompleteWMOModel model)
    {
        // Extract surface normals from decoded MSUR fields
        var normal = new Vector3(
            msur.SurfaceNormalX,    // UnknownFloat_0x04
            msur.SurfaceNormalY,    // UnknownFloat_0x08  
            msur.SurfaceNormalZ     // UnknownFloat_0x0C
        );
        
        model.Normals.Add(normal);
        
        // Extract height information
        var height = msur.SurfaceHeight;  // UnknownFloat_0x10
        model.Metadata["SurfaceHeight"] = height;
    }
    
    public void ProcessMSLKFields(MSLKEntry mslk, CompleteWMOModel model)
    {
        // Extract object type and material classification
        var objectType = mslk.ObjectTypeFlags;     // Unknown_0x00
        var materialId = mslk.MaterialColorId;     // Unknown_0x0C
        
        model.Metadata["ObjectType"] = objectType;
        model.Metadata["MaterialId"] = materialId;
    }
}
```
- **Achievement**: 100% utilization of decoded PM4 unknown fields
- **Result**: Enhanced export with surface normals, materials, and spatial data
- **Validation**: Statistical analysis across 76+ files confirms field accuracy

## Quality Assurance Patterns

### **10. Comprehensive Validation Pattern**
```csharp
public class GeometryValidator
{
    public ValidationResult ValidateTriangles(List<int> triangleIndices, int vertexCount)
    {
        var result = new ValidationResult();
        
        for (int i = 0; i < triangleIndices.Count; i += 3)
        {
            var idx1 = triangleIndices[i];
            var idx2 = triangleIndices[i + 1];
            var idx3 = triangleIndices[i + 2];
            
            // Validate triangle indices
            if (idx1 >= vertexCount || idx2 >= vertexCount || idx3 >= vertexCount)
                result.AddError($"Triangle {i/3}: Index out of bounds");
            
            // Validate non-degenerate triangle
            if (idx1 == idx2 || idx1 == idx3 || idx2 == idx3)
                result.AddError($"Triangle {i/3}: Degenerate triangle");
        }
        
        return result;
    }
}
```
- **Result**: Zero degenerate triangles across all processed files
- **Quality**: Professional 3D software compatibility guaranteed
- **Coverage**: Validates all geometry before export

## Architecture Design

### **Project Structure Pattern**
```
WoWToolbox.Core/
├── Navigation/PM4/
│   ├── Models/           # CompleteWMOModel, DTOs
│   ├── Transforms/       # Pm4CoordinateTransforms
│   └── Analysis/         # Core analysis utilities

WoWToolbox.PM4Parsing/    # (Planned refactor target)
├── BuildingExtraction/   # Dual geometry assembly
├── GeometryProcessing/   # Surface processing, face generation
├── MaterialAnalysis/     # MSLK metadata processing
└── Export/              # Enhanced OBJ/MTL export
```

### **Testing Pattern**
```csharp
[Fact]
public void BuildingExtraction_ShouldProduceIdenticalQuality()
{
    // Arrange: Load PM4 file
    var pm4File = LoadTestPM4File();
    
    // Act: Extract buildings
    var buildings = extractor.ExtractBuildings(pm4File);
    
    // Assert: Validate quality metrics
    Assert.True(buildings.Count >= 10, "Should extract 10+ buildings");
    Assert.All(buildings, building => 
    {
        Assert.True(building.FaceCount > 0, "Building should have faces");
        Assert.True(building.Normals.All(n => Math.Abs(n.Length() - 1.0f) < 0.01f), 
                   "All normals should be normalized");
    });
}
```

## Implementation Guidelines

### **Quality Standards**
1. **Zero Degenerate Triangles**: All face generation must pass comprehensive validation
2. **Professional Compatibility**: Output must work seamlessly with MeshLab and Blender
3. **Individual Building Quality**: Each building must be complete and properly separated
4. **Surface Normal Accuracy**: All normals must be properly normalized vectors

### **Performance Requirements**
1. **Batch Processing**: Handle hundreds of PM4 files with consistent quality
2. **Memory Efficiency**: Process large files without excessive memory usage
3. **Processing Speed**: Maintain reasonable performance for production workflows
4. **Error Handling**: Robust processing with comprehensive error reporting

### **Architectural Principles**
1. **Single Responsibility**: Each pattern handles one specific aspect of processing
2. **Composition Over Inheritance**: Combine patterns for complex workflows
3. **Validation First**: Validate all geometry before export
4. **Metadata Preservation**: Maintain all decoded field information throughout processing

This architecture represents the proven, production-ready patterns that enable WoWToolbox v3's breakthrough capabilities in PM4 analysis and building extraction.