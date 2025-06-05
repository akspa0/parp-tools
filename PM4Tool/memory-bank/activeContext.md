# Project Vision & Immediate Technical Goal (2025-01-15)

## Vision
- Build tools that inspire others to explore, understand, and preserve digital history, especially game worlds.
- Use technical skill to liberate hidden or lost data, making it accessible and reusable for future creators and historians.

## Immediate Technical Goal
- Use PM4 files (complex, ADT-focused navigation/model data) to infer and reconstruct WMO (World Model Object) placements in 3D space.
- Match PM4 mesh components to WMO meshes, deduce which model is where, and generate placement data for ADT chunks.
- Output reconstructed placement data as YAML for now, with the intent to build a chunk creator/exporter later.

---

# Mode: PLAN

# Active Context: PM4 3D Structural Analysis Tool Enhancement

## Current Status: ✅ **3D Viewer Successfully Built & Tested**

### **Major Achievement: Working PM4 3D Viewer**
**Tool**: `src/WoWToolbox.PM4Viewer/` (WPF + HelixToolkit.Wpf + .NET 9.0)

#### **✅ Working Features:**
- **3D Visualization**: Successfully displays PM4 geometry with proper coordinate transformations
  - **Blue dots**: MSVT render vertices with `(Y, X, Z)` transform
  - **Red dots**: MSCN collision points with Y-axis correction + 180° rotation  
  - **Green dots**: MSPV structure vertices with direct `(X, Y, Z)` coordinates
  - **Ground plane reference**: Grid lines for spatial orientation
  - **Camera controls**: Mouse navigation (rotate, pan, zoom)
- **File Loading**: PM4 file browser and loading functionality
- **Chunk Visualization**: Real-time display of vertex counts and chunk presence
- **Interactive Controls**: Toggle visibility of different geometry types
- **Export Capability**: Analysis report export to text files

#### **⚠️ Needs Enhancement:**
- **Analysis Logic**: Structure analysis features not fully functional
- **Button Interactions**: Menu items and analysis controls need proper implementation
- **Error Handling**: Analysis pipeline requires debugging and stabilization
- **Pattern Recognition**: Unknown field analysis needs refinement
- **Hierarchical Visualization**: Node relationship display needs development

## Current Investigation Focus

### **PM4 Structural Mysteries to Explore**
Using the 3D viewer as our exploration platform:

1. **Node-Based Hierarchies**: 
   - **Unknown_0x04 as internal indices** rather than group identifiers
   - **Padding as metadata**: Non-zero bytes between chunks may contain node information
   - **MSLK hierarchical patterns**: Group analysis via Unknown_0x04 values

2. **Missing Connectivity Data**:
   - **MSRN/MPRR chunks**: Potential additional structural information
   - **Cross-chunk relationships**: Hidden references between geometry types
   - **Material/metadata fields**: Unknown_0x0C and other mysterious values

3. **Enhanced 3D Analysis**:
   - **Pattern visualization**: Display discovered hierarchical connections in 3D space
   - **Interactive field exploration**: Click vertices to inspect unknown field values
   - **Comparative analysis**: Multi-file pattern validation

## Next Steps (Immediate Priority)

### **Phase 1: Fix Analysis Pipeline** 🔧
1. **Debug analysis functionality**: Identify why structural analysis isn't populating UI
2. **Enhance error handling**: Make analysis failures visible and actionable
3. **Implement missing button logic**: Connect menu items to actual functionality
4. **Test field pattern recognition**: Validate Unknown_0x04 grouping theories

### **Phase 2: Enhanced Structural Investigation** 🔍
1. **Padding investigation tool**: Analyze non-zero padding for hidden metadata
2. **Interactive field inspector**: Click geometry to see unknown field values
3. **Hierarchical visualization**: Display node connections in 3D space
4. **Pattern correlation engine**: Compare structures across multiple PM4 files

### **Phase 3: Advanced Analysis Features** 🚀
1. **MSRN/MPRR investigation**: Decode additional navigation chunks
2. **Cross-reference validation**: Verify structural theories across dataset
3. **WMO preparation pipeline**: Bridge PM4 analysis to WMO reconstruction
4. **Batch analysis capabilities**: Process entire development dataset

## Technical Architecture Insights

### **PM4 3D Viewer Design Pattern**
```
WPF MainWindow → ViewModel (MVVM) → StructuralAnalyzer → PM4File
     ↓                ↓                    ↓              ↓
HelixViewport3D → ObservableCollections → Analysis → Chunk Data
```

### **Analysis Pipeline (Needs Debugging)**
```
PM4File → StructuralAnalyzer → {
  - PaddingAnalysis: Detect non-zero padding metadata
  - UnknownFieldAnalysis: Pattern recognition in mysterious fields  
  - HierarchicalRelationships: Parent-child chunk connections
  - NodeStructureAnalysis: MSLK grouping via Unknown_0x04
}
```

### **Key Technical Discoveries**
- **Coordinate transformation mastery**: Fixed "polar opposite corners" with proper PM4 transforms
- **Face generation understanding**: MSUR triangle fans vs linear triangles
- **MSCN design pattern**: Point cloud collision (not mesh) confirmed
- **MSLK master controller**: Links all geometry types via MSPI indices

## Strategic Value

The **PM4 3D Structural Analysis Tool** represents a breakthrough in PM4 understanding because:

1. **Visual-guided investigation**: 3D context makes structural patterns obvious
2. **Interactive exploration**: Real-time analysis while viewing geometry
3. **Pattern validation**: Cross-reference discoveries across multiple files
4. **Hidden data detection**: Systematic investigation of unknown fields and padding
5. **Foundation for WMO reconstruction**: Direct bridge to placement data generation

The tool transforms PM4 investigation from **blind data analysis** to **guided 3D exploration**, enabling discovery of hierarchical patterns that would be invisible in text-based analysis.

## Success Metrics

### **Short-term (This Session)**
- ✅ Analysis pipeline functional and populating UI
- ✅ Button interactions working as expected  
- ✅ Unknown field patterns clearly displayed
- ✅ Error handling robust and informative

### **Medium-term (Next Sessions)**
- ✅ Hierarchical node connections visualized in 3D
- ✅ Padding metadata patterns discovered and documented
- ✅ Cross-file structural validation completed
- ✅ Enhanced export formats (WMO preparation data)

### **Long-term (Project Goals)**
- ✅ Complete PM4 → WMO reconstruction pipeline
- ✅ Unknown field mysteries decoded and documented
- ✅ Digital preservation system for WoW map data
