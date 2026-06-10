# Active Context: MSCN-WMO Spatial Correlation & Large-Scale Analysis

## PM4NextExporter — Current Focus (2025-08-10)

- **Composite-hierarchy is default** assembly path; instance variant removed per simplification.
- **Assembler tagging**: `CompositeHierarchyAssembler` sets `AssembledObject.Meta["tileId"|"tileX"|"tileY"]` using dominant tile derived from surfaces→indices→tile offsets.
- **Per-tile OBJ export**: baseline implemented; currently merges by tile. Refactor in progress to output one OBJ per tile composed of distinct assembled objects (preserve boundaries).
- **Coordinate projection**: `--project-local` implemented; optional export-time normalization (centroid translation) for both per-object and per-tile paths.
- **MSCN controls**: `--no-remap` (optional skip), `--export-mscn-obj` (point OBJ) available for minimal, reversible visualization.

### Immediate Next Steps

- Finish refactor of `PerTileObjExporter` to group/export assembled objects per tile, using `AssembledObject.Meta.tileId`.
- Keep knobs minimal; no new diagnostics unless explicitly requested.

**MAJOR BREAKTHROUGH ACHIEVED**: Successfully implemented robust MSCN anchor to WMO geometry correlation with batch processing capabilities for large-scale World of Warcraft map data analysis.

## 🎯 **CURRENT FOCUS: MSCN Vertex Extraction & Diagnostics Fix**

### ✅ **BREAKTHROUGH ACHIEVEMENTS (August 2025)**

#### 🔧 **Fixed Critical Correlation Bug**
- **Problem**: Correlation analysis was finding object-level matches but reporting 0% correlation
- **Root Cause**: Object-level clustering logic wasn't properly creating vertex-level `SpatialMatch` objects
- **Solution**: Implemented proper vertex-to-vertex matching within overlapping object clusters
- **Result**: Accurate correlation percentages, match counts, and visualization export

#### 🎨 **Working Visualization Confirmed**
- **Visual Proof**: MSCN anchors properly align with WMO geometry in 3D visualization (MeshLab)
- **Test Case**: Stormwind Harbor canal geometry shows clear spatial correlation
- **Export Format**: OBJ files with matched vertex pairs and connection lines
- **Coordinate Systems**: Successfully normalized MSCN world coordinates to WMO local space

#### 🚀 **Batch Processing System**
- **Command**: `batch-mscn-wmo-correlation` for large-scale analysis
- **Parallel Processing**: Multi-threaded with configurable parallelism
- **Intelligent Filtering**: Skip obviously unrelated file pairs for performance
- **Comprehensive Reporting**: JSON + text reports with statistics and top correlations
- **Scalability**: Ready for processing thousands of PM4/WMO combinations

### 🔬 **TECHNICAL IMPLEMENTATIONS**

#### **Spatial Correlation Engine**
- **Spatial Hash Grids**: O(1) nearest neighbor lookup for large datasets
- **Multi-tile MSCN Aggregation**: Load anchors from 3x3 tile grids for complete coverage
- **Coordinate Normalization**: Transform MSCN world coordinates to WMO local space
- **Distance Tolerance**: Configurable matching tolerance for collision-to-geometry correlation

#### **Performance Optimizations**
- **Memory Efficient**: Handles large datasets without excessive memory usage
- **Semaphore Control**: Prevents resource exhaustion during parallel processing
- **Early Termination**: Skip processing for obviously unrelated file pairs
- **Progress Tracking**: Real-time feedback during batch operations

#### **Visualization & Analysis**
- **Match Visualization**: Export OBJ files showing vertex pairs and connection lines
- **Statistical Analysis**: Distance metrics, correlation percentages, match distributions
- **Debug Tolerance Testing**: Automatic testing with multiple tolerance values
- **Comprehensive Reports**: Detailed analysis with top matches and spatial bounds

## 📊 **PROVEN RESULTS**

### **Visual Confirmation**
- MSCN anchors spatially align with WMO canal geometry in Stormwind Harbor
- Yellow dots (MSCN) properly positioned relative to wireframe WMO structure
- Correlation percentages now accurately reflect spatial relationships

### **Scalable Architecture**
- Batch processing system ready for enterprise-scale analysis
- Configurable parallelism for different hardware capabilities
- Intelligent filtering reduces unnecessary processing

### **Technical Validation**
- Fixed correlation calculation bug that was causing 0% reports
- Proper vertex-level matching within object-level clusters
- Accurate distance calculations and statistical analysis

## 🎯 **IMMEDIATE NEXT STEPS**

### **Phase 1: Enhanced File Discovery**
- **CASC Integration**: Add wow.tools.local file discovery for game files
- **Geographic Filtering**: Intelligent PM4-to-WMO matching by region/coordinates
- **Metadata Extraction**: Parse tile coordinates and region information from filenames

### **Phase 2: Machine Learning Enhancement**
- **Pattern Recognition**: ML algorithms for large-scale correlation analysis
- **Automated Classification**: Identify correlation patterns across different regions
- **Predictive Matching**: Suggest likely PM4-WMO pairs before processing

### **Phase 3: Database & Persistence**
- **Result Storage**: Database for persistent correlation results
- **Query Interface**: Search and filter correlation data
- **Historical Analysis**: Track correlation changes over time

## 🔮 **LONG-TERM VISION**

### **Automated Map Analysis Platform**
- **Full WoW World Coverage**: Process entire game world for spatial relationships
- **Quality Assurance**: Validate game asset spatial consistency
- **Research Platform**: Enable academic research on virtual world design
- **Developer Tools**: Assist game developers with spatial validation

### **Advanced Analytics**
- **Spatial Pattern Discovery**: Identify common architectural patterns
- **Asset Relationship Mapping**: Build comprehensive spatial relationship database
- **Anomaly Detection**: Identify spatial inconsistencies or errors
- **Performance Optimization**: Guide game engine optimization based on spatial analysis

## 🏆 **PROJECT STATUS: BREAKTHROUGH PHASE COMPLETE**

- **Core correlation system**: ✅ COMPLETE and WORKING
- **Batch processing**: ✅ COMPLETE and TESTED
- **Visualization export**: ✅ COMPLETE with OBJ output
- **Performance optimization**: ✅ COMPLETE with spatial hashing
- **Bug fixes**: ✅ COMPLETE - correlation calculation fixed
- **Documentation**: 🔄 IN PROGRESS - updating memory bank

**Ready for production use and large-scale analysis deployment.**

## Update (2025-08-18): Current Focus – PM4 MSCN Analyzer

We are initiating a dedicated analyzer tool to investigate MSCN relations to MSLK/MSUR and the role of MPRR as range/linkage metadata.

Planned separate console app: `src/PM4MscnAnalyzer/`

Planned commands:
- `mscn-dump` — export MSCN points CSV (region-aware)
- `object-centroids` — group by `MSLK.ParentIndex_0x04` and emit centroids/AABBs
- `mprr-dump` — raw MPRR fields (for schema discovery)
- `mscn-associate` — kNN-based MSCN↔object association with thresholds and optional MPRR gating
- `glb-overlay` (optional) — visual overlay of MSCN markers and centroid links

Outputs will be written under `project_output/mscn_analysis/<session>/` with CSVs and optional `overlay.glb`.

Next steps:
1. Scaffold project and implement `mscn-dump`, `object-centroids`.
2. Implement `mprr-dump`, `mscn-associate`.
3. Optional GLB overlay and write `docs/Diagnostics_MSCN.md`.
