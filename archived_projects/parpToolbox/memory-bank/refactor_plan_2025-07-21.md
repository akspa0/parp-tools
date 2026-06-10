# PM4/PD4 Tooling Refactor Plan
## Created: 2025-07-21 | Updated: 2025-07-21

## Problem Statement

The PM4 object grouping investigation has created significant **tool fragmentation and feature creep**:

### Current Tool Chaos (8+ Overlapping Exporters)
1. `Pm4Adapter` - Core PM4 loader
2. `Pm4RegionLoader` - Cross-tile loading 
3. `Pm4MprlObjectGrouper` - MPRL-based grouping
4. `Pm4HierarchicalObjectAssembler` - MPRR-based grouping
5. `Pm4OptimizedObjectExporter` - Performance optimization
6. `Pm4TileBasedExporter` - Tile-based processing
7. `Pm4RawGeometryExporter` - Raw geometry output
8. `Pm4SurfaceGroupExporter` - Surface group export

### CLI Command Explosion (12+ Commands)
- `pm4`, `pm4-region`, `pm4-export-scene`
- `pm4-mprl-objects`, `pm4-mprr-objects`, `pm4-mprr-objects-fast`
- `pm4-tile-objects`, `pm4-raw-geometry`, `pm4-buildings`
- `pm4-analyze-data`, `pm4-test-chunks`, `pm4-analyze-indices`

**Result**: Confusing, unmaintainable, redundant tooling that obscures our discoveries.

## Key Discoveries to Preserve

### Cross-Tile Reference Resolution
- MSCN remapping working correctly
- Region loading produces 12.8x more vertices (502 tiles merged)
- Resolves out-of-bounds vertex access issues

### PM4 Contains Building Objects
- 3D visualization confirms discrete building clusters exist
- Raw geometry shows scattered building point clouds (not terrain)
- Surface Groups (MSUR.SurfaceGroupKey) appear to be correct object boundaries

### Previous Grouping Attempts Failed
- MPRL-based grouping produced fragments
- MPRR-based grouping produced subdivisions/containers
- Need to focus on Surface Group = Building Object hypothesis

## Detailed Refactoring Plan

### Phase 1: Analysis & Documentation Integration (3 days)

#### 1.1 Documentation & Code Alignment
- Document MSUR field meanings from MSUR_FIELDS.md into class comments
- Add XML documentation to chunk classes based on PM4-Chunk-Reference.md
- Apply correct field aliases based on documentation (e.g., `parent_index` for MSLK.Unknown4)
- Update model classes with correct semantic names for fields

#### 1.2 Core Component Preservation
- Identify and document critical components that must be preserved:
  - Cross-tile vertex reference resolution (MSCN remapping)
  - MPRR-based hierarchical object assembly
  - Global tile loading system
  - Surface group boundary identification
- Create descriptive comments for these components

### Phase 2: Exporter Consolidation (4 days)

#### 2.1 Create Enhanced Unified Exporter
- Create new `Pm4Exporter.cs` class with:
  - Configurable export strategies via an enum
  - Option flags for cross-tile loading, façade filtering, etc.
  - Built-in progress reporting and validation

#### 2.2 Exporter Strategy Implementation
- Implement export strategies as protected methods within unified exporter:
  - `ExportRawGeometry()` (from `Pm4RawGeometryExporter`)
  - `ExportMprrHierarchicalObjects()` (from `Pm4HierarchicalObjectAssembler`)
  - `ExportMsurSurfaceGroups()` (from `Pm4SurfaceGroupExporter`)
  - `ExportCompleteScene()` (from `Pm4SceneExporter`)

#### 2.3 Export Helper Methods
- Move common helper methods to the unified exporter:
  - Vertex transformation
  - Index validation
  - Object grouping
  - OBJ file formatting

#### 2.4 Legacy Exporter Deprecation
- Deprecate redundant exporters:
  - `Pm4MprlObjectGrouper.cs` → delete
  - `Pm4HierarchicalObjectAssembler.cs` → consolidate
  - `Pm4OptimizedObjectExporter.cs` → consolidate
  - `Pm4TileBasedExporter.cs` → consolidate
  - `Pm4RawGeometryExporter.cs` → consolidate
  - `Pm4SurfaceGroupExporter.cs` → consolidate
  - `Pm4SceneExporter.cs` → consolidate

### Phase 3: Analyzer Consolidation (3 days)

#### 3.1 Create Unified Analyzer
- Create new `Pm4Analyzer.cs` class with:
  - Configurable analysis types
  - Report format options (console, CSV, detailed)
  - Performance benchmarking

#### 3.2 Analysis Method Implementation
- Implement analysis methods from existing analyzers:
  - `AnalyzeChunkRelationships()` (from `Pm4DataAnalyzer`)
  - `AnalyzeIndexPatterns()` (from `Pm4IndexPatternAnalyzer`)
  - `AnalyzeUnknownFields()` (from `Pm4UnknownFieldAnalyzer`)
  - `TestGroupingStrategies()` (from `Pm4GroupingTester`)

#### 3.3 Legacy Analyzer Deprecation
- Deprecate redundant analyzers:
  - `Pm4DataAnalyzer.cs` → consolidate
  - `Pm4ChunkCombinationTester.cs` → consolidate
  - `Pm4IndexPatternAnalyzer.cs` → consolidate
  - `Pm4UnknownFieldAnalyzer.cs` → consolidate
  - `Pm4BulkDumper.cs` → consolidate
  - `Pm4CsvDumper.cs` → consolidate
  - `Pm4GroupingTester.cs` → consolidate

### Phase 4: CLI Interface Simplification (2 days)

#### 4.1 Command Structure Redesign
- Reduce commands to 3 main commands:
  1. `pm4-analyze`: All analysis capabilities
  2. `pm4-export`: All export options
  3. `pm4-test`: Validation and testing

#### 4.2 Command-Line Argument Updates
- Update `Program.cs` to handle new command structure
- Create option parsers for each main command
- Add comprehensive help documentation

#### 4.3 Legacy Command Removal
- Remove legacy commands:
  - `pm4-mprl-objects`, `pm4-mprr-objects`, etc.
  - `pm4-analyze-data`, `pm4-test-chunks`, etc.

### Phase 5: Test Suite Consolidation (3 days)

#### 5.1 Test Audit
- Analyze existing test projects and identify:
  - Tests to keep
  - Tests to consolidate
  - Tests to remove

#### 5.2 Test Suite Implementation
- Create new focused test suites:
  - `Pm4ChunkTests`: Basic chunk loading/parsing
  - `Pm4ExporterTests`: Export validation
  - `Pm4AnalyzerTests`: Analysis validation
  - `RegionLoadingTests`: Cross-tile behavior

#### 5.3 Legacy Test Cleanup
- Remove or update legacy tests that are no longer relevant

### Phase 6: Documentation & Validation (2 days)

#### 6.1 Update Memory Bank
- Update all memory bank files with new architecture
- Document key decisions and implementation details

#### 6.2 End-to-End Validation
- Verify all capabilities preserved through refactoring
- Run full test suite
- Validate PM4 exports against known good files

## Implementation Strategy

### Step 1: Start with Core Component Enhancement (Days 1-2)
- Update chunk models with proper documentation and aliases
- Enhance `Pm4Adapter` with cross-tile functionality
- Document critical discoveries directly in code comments

### Step 2: Implement Unified Exporter (Days 3-5)
- Create `Pm4Exporter` class with strategy pattern
- Move core functionality from existing exporters
- Add configuration options for all export modes

### Step 3: Implement Unified Analyzer (Days 6-7)
- Create `Pm4Analyzer` class with analysis types
- Move core functionality from existing analyzers
- Implement reporting system

### Step 4: Update CLI Interface (Days 8-9)
- Update `Program.cs` with new command structure
- Implement option parsing for new commands
- Update help documentation

### Step 5: Clean Up and Test (Days 10-12)
- Remove redundant classes and files
- Implement consolidated test suites
- Run full validation

### Step 6: Documentation (Days 13-14)
- Update memory bank with final architecture
- Create technical specification documents for new classes

## Success Criteria

### Code Quality
- Single enhanced Pm4Adapter with all capabilities
- Remove 6+ redundant exporters
- 3 clean CLI commands instead of 12+
- Comprehensive test suite with real data

### Knowledge Preservation
- Cross-tile reference resolution integrated
- Surface group = building object logic preserved
- All chunk relationship discoveries documented
- Performance considerations maintained

### Maintainability
- Clear, documented APIs
- Minimal, focused codebase
- Easy to extend for future discoveries
- Clean separation of concerns

### 2025-07-21 Immediate CLI Simplification Kickoff

We are pulling **Phase 4: CLI Interface Simplification** forward to begin immediately.  The first deliverable will be:

1. Introduce a lightweight `CliCommands` namespace with three handlers:
   - `AnalyzeCommand`
   - `ExportCommand`
   - `TestCommand`
2. Extract the current `pm4-analyze` implementation from `Program.cs` into `AnalyzeCommand`.  `Program.cs` will serve only as a router.
3. Keep legacy commands, but mark them deprecated and delegate internally to the new handlers with a warning banner.
4. Once extraction compiles and passes smoke-tests, remove the bulky logic blocks from `Program.cs`.

This accelerated start reduces immediate complexity while still aligning with the long-term refactor roadmap.

---

## Risk Mitigation

- **Git safety**: All changes are version controlled, can revert if needed
- **Incremental approach**: Refactor in small, testable steps
- **Preserve working functionality**: Don't break existing capabilities
- **Document decisions**: Keep clear record of what was removed and why
