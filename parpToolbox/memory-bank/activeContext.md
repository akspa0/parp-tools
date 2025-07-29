# Active Context

## Current Work
**Objective:** Implement and test PM4 spatial clustering assembler for building-scale object extraction

**Status:** Spatial clustering assembler implemented and CLI command registered

## Verified Working Approach
**Source:** poc_exporter.cs lines 7402-7683 (`ExportBuildings_UsingMslkRootNodesWithSpatialClustering` and `CreateHybridBuilding_StructuralPlusNearby`)

### Working Algorithm
1. **Root Node Identification:** Find MSLK entries where `ParentIndex == entry_index` (self-referencing)
2. **Structural Grouping:** Group MSLK entries by `ParentIndex` value matching root nodes
3. **Bounds Calculation:** Calculate bounding box from MSPV vertices via MSLK → MSPI → MSPV chain
4. **Spatial Clustering:** Find MSUR surfaces within expanded bounds (50.0f tolerance)
5. **Hybrid Assembly:** Combine MSPV structural elements + nearby MSUR render surfaces

### Verified Data Flow
- **MSLK.ParentIndex** → Group identifier for building objects
- **MSLK.MspiFirstIndex/MspiIndexCount** → Indices into MSPI array
- **MSPI.Indices** → Point to MSPV vertices for structural geometry
- **MSUR.MsviFirstIndex/IndexCount** → Indices into MSVI array for render triangles
- **MSVI.Indices** → Point to MSVT vertices for render geometry

### Coordinate Transformations (Verified)
- **MSPV:** `(vertex.X, vertex.Y, vertex.Z)` (direct)
- **MSVT:** `(vertex.Position.Y, vertex.Position.X, vertex.Position.Z)` (Y-X swap)

## Implementation Status
**Created:** `Pm4SpatialClusteringAssembler.cs` - Direct port of working POC logic
**CLI Command:** `Pm4ExportSpatialClusteringCommand.cs` - Registered in Program.cs
**Command Registration:** `pm4-export-spatial-clustering` command available in CLI

## Key Insight
Spatial clustering was added to compensate for incomplete hierarchical grouping in PM4 data. Pure hierarchical approaches fail to produce complete building objects.

## Current Work Focus
- **Test spatial clustering implementation** with real PM4 data
- **Validate root node detection** and building-scale object assembly
- **Verify cross-tile reference resolution** through region loading
- **Analyze output quality** and compare with expected building count

## Recent Changes
- **Implemented Pm4SpatialClusteringAssembler** with complete building assembly logic
- **Created CLI command** for spatial clustering export with proper argument parsing
- **Registered command** in Program.cs with support for both `--output=value` and `--output value` formats
- **Integrated region loading** to resolve cross-tile vertex references
- **Added detailed logging** for debugging and progress tracking

## PM4 Architecture Insights
- **Global Tile System Confirmed:** PM4 files reference vertices from adjacent tiles, requiring unified region loading
- **Surface "bounds" fields:** Are encoded linkage containers, not spatial bounds
- **Object Assembly:** Requires linkage-based decoding rather than spatial clustering
- **Data Loss:** ~64% of vertex data missing without global tile loading

## Next Steps
1. **Test CLI command** with real PM4 data
2. **Validate building count** against expected 458 buildings
3. **Analyze output geometry** for completeness and correctness
4. **Compare with reference implementation** outputs
5. **Document findings** and update progress accordingly

## Next Steps
- **Start a new session.**
- **Fix the Build:** The absolute first priority is to fix `Pm4PerObjectExporter.cs` by completely replacing its content with a clean, correct implementation.
- **Run Regression Test:** Once the build is fixed, run the regression test to validate the output of the new exporter.
- **Generate Golden Hash:** If the output is correct, generate a SHA256 hash of the largest object file and update the regression test to use it.

## Key Discoveries

### MAJOR BREAKTHROUGH: Bounds Encoding Discovery
- **Surface "bounds" fields are encoded linkage containers, NOT spatial bounds**
- **BoundsMaxZ = 3285479936** → Tile/chunk reference ID (consistent across entries)
- **BoundsMaxX/Y** → Cross-tile vertex indices (explains out-of-bounds access)
- **BoundsMinX/Y/Z** → Direction vectors or normalized coordinates
- **BoundsCenterX/Y/Z** → Duplicated geometry indices (MsviFirstIndex, IndexCount, GroupKey)
- **Field overloading confirmed** - PM4 uses single fields to pack multiple data types

### Object Grouping Reality
- **PM4 files DO contain building objects** - 3D visualization confirms discrete building clusters
- **Surface Groups (MSUR.SurfaceGroupKey) appear to be correct object boundaries** - not terrain subdivisions
- **All previous "object grouping" attempts (MPRL, MPRR) produced fragments/subdivisions** - not complete objects
- **Raw geometry export shows scattered building point clouds** - confirming objects exist but need proper grouping
- **Spatial clustering was fundamentally wrong** - based on non-existent bounds data

### Cross-Tile Reference Success
- Cross-tile vertex reference resolution working correctly (12.8x data increase, 502 tiles merged)
- Region loading successfully resolves out-of-bounds vertex access issues
- MSCN remapping functional and validated

### Critical Problem: Tool Fragmentation
- **8+ different PM4 exporters created** with overlapping/redundant functionality
- **Feature creep has made analysis confusing and unmaintainable**
- **Need immediate refactor** to consolidate knowledge into core library

## Immediate Priority: Refactor Plan

### Phase 1: Consolidate Core Library
- Enhance `Pm4Adapter` with all new knowledge (cross-tile, surface groups, etc.)
- Remove redundant exporters (keep only essential functionality)
- Create single `Pm4Analyzer` for comprehensive analysis

### Phase 2: Clean CLI Interface
- Simplify to 3 commands: `pm4-analyze`, `pm4-export`, `pm4-test`
- Remove fragmented commands (pm4-mprl-objects, pm4-mprr-objects, etc.)

### Phase 3: Test Suite
- Build proper unit/integration tests using real data
- Remove old/broken tests
- Establish clean foundation for future work

## Next Steps

### (2025-07-21 04:24) - CLI Simplification Progress
- Extracted `AnalyzeCommand` from `Program.cs`; `pm4-analyze` now routes through this handler.
- Removed bulky inline analysis logic from main entry to reduce complexity.
- Build remains green; upcoming work will extract `ExportCommand` and `TestCommand` and deprecate legacy aliases.

Execute refactor plan to consolidate tooling and integrate all discoveries into maintainable core library.

## Phases & Tasks

### Phase 1: Project Foundation (Complete)
- [x] Create a new solution `parpToolbox.sln` in the root directory.
- [x] Create a new .NET 9.0 console project `parpToolbox` in the `src` folder.
- [x] Add the existing `wow.tools.local` project to the solution.
- [x] Add a project reference from `parpToolbox` to `wow.tools.local`.

### Phase 2: WMO & OBJ Export (Complete)
- [x] Implement manual command-line argument parsing.
- [x] Create `LocalFileProvider` to enable file system access for `wow.tools.local`.
- [x] Fix WMO group file loading in `WMOReader` to prevent infinite recursion.
- [x] Implement `ProjectOutput` utility for clean, timestamped directory creation.
- [x] Implement `ObjExporter` service to convert WMO models to OBJ and MTL files.
- [x] Integrate all components into `Program.cs` to create a complete export pipeline.

### Phase 3: PM4 / PD4 Integration (In Progress)
- [ ] **Debug CLI:** Resolve issue where arguments are not received when using `dotnet run`.
- [x] Scaffold `Formats/PM4` and `Formats/PD4` modules inside `parpToolbox`.
- [x] Implement `WowToolsLocalWmoLoader` exposing high-level `WmoGroup` objects.
- [ ] Port essential models and readers from legacy `WoWToolbox.Core.v2` (read-only) into new namespaces.
- [x] Move shared PM4/PD4 chunks to `Formats/P4/Chunks/Common`.
- [x] Update namespaces of moved chunks to `ParpToolbox.Formats.P4.Chunks.Common`.
- [x] Adapt `Pm4Adapter` to new namespace and scaffold `Pd4Adapter`.
- [ ] Implement full `Pm4Adapter` / `Pd4Adapter` behaviour (PD4 chunk audit, geometry export).
- [x] Extend CLI parser with `pm4` / `pd4` commands and route outputs via `ProjectOutput`.
- [x] Update OBJ exporter to output vertices only (omit faces) for initial validation.
- [ ] Port legacy `Pm4BatchTool` research utilities into new `Pm4ResearchTool` project.
- [ ] Create integration tests loading real PM4/PD4 data under `test_data/`, verifying OBJ vertex counts.

---

## Current Status

### (2025-07-21 00:31) - Cross-Tile Reference Fix SUCCESSFUL ✅
- **MASSIVE DATA LOSS RESOLVED**: Cross-tile vertex reference system now fully functional
- **Validation Results** (development_00_00.pm4):
  - **Before**: ~63,298 vertices from single tile with 110,988 out-of-bounds accesses (~64% data loss)
  - **After**: 812,648 vertices from 502 merged tiles (12.8x increase in data coverage)
  - **Region Loading**: Successfully merged 502 tiles automatically
  - **MSCN Remapping**: Processed 9,990 exterior vertices with 0 cross-tile references needing remapping
  - **Complete Scene**: 1,930,146 indices, 518,092 surfaces, 1,273,335 links
- **Key Insights**:
  - **High/Low Pair Encoding**: Unknown fields likely encode 32-bit indices as two 16-bit values
  - **Tile Boundary References**: Vertex indices cross tile boundaries requiring global mesh loading
  - **(0,0,0) Anchor Points**: Not mysterious anchors but invalid vertex data from out-of-bounds access
- **Tools Created**:
  - **Pm4IndexPatternAnalyzer**: Analyzes index patterns, high/low pairs, missing data
  - **Enhanced vertex validation**: Skip triangles with invalid indices, prevent (0,0,0) artifacts
- **Previous PM4 Discovery**: Individual objects identified by **MSUR SurfaceGroupKey**, not IndexCount
- **Chunk Relationship Analysis**:
  - **MPRL.Unknown4 = MSLK.ParentIndex** (458 confirmed matches) - links placements to geometry
  - **MSLK entries with MspiFirstIndex = -1** are container/grouping nodes (no geometry)
  - **MPRR.Value1 = 65535** are property separators (15,427 sentinel values)
  - **MPRL.Unknown6 = 32768** consistently (likely type flag)
- **Object Assembly Flow**:
  1. **MPRL** defines object placements (positions + type IDs)
  2. **MSLK** links placements to geometry via ParentIndex → MPRL.Unknown4
  3. **MPRR** provides segmented properties between sentinel markers
  4. **MSUR** defines surface geometry with **IndexCount as object identifier**
- **Implementation Status**:
  - ✅ `Pm4MsurObjectAssembler` created using MSUR IndexCount grouping
  - ✅ `Pm4SceneExporter` for complete building interior export
  - ✅ Coordinate system fix (X-axis inversion) applied
  - ✅ CSV analysis pipeline for chunk relationship validation

### (2025-07-14 22:51) - PM4 Export Issues
- PM4 export produced **825** groups (expected ~10–20). Grouping algorithm still incorrect.
- MSUR chunk loader rewritten to 32-byte authoritative spec; alignment confirmed.
- **Root Cause**: Complex multi-object relationships in PM4 not properly understood
- **Solution Path**: Port legacy `MsurObjectExporter` grouping algorithm
- **Critical**: Ensure PD4 export stability while fixing PM4 logic

### WMO Export Status
WMO → OBJ export has been fully validated: façade planes are correctly filtered by default, and group/file naming matches in-game names. The command-line pipeline works reliably when arguments are passed via an executable build; the `dotnet run --` quirk is still under investigation.

### Next Priority Tasks
1. **Integrate ParentIndex Mapping:** Build dictionary MSLK.ParentIndex → geometry nodes and attach to MPRL placements.
2. **Enhance Object Assembly:** Combine ParentIndex-linked surfaces into composite PlacedObjects with correct transforms.
3. **Validate Full Link Coverage:** Expect 100 % `LinksToMSLK=True` rows in `mprl_detailed.csv`.
4. **Global Tile Loader** (unchanged): implement unified 64×64 tile loading.
5. **High/Low Pair Decoding** (unchanged): verify 32-bit index encoding.

1. **Port `MsurObjectExporter` grouping routine** from legacy codebase
2. **Implement proper surface range matching** via MSLK `ReferenceIndex`
3. **Validate PM4 group counts** against real data (target: 10-20 groups)
4. **Maintain PD4 export stability** during PM4 fixes
5. **Update memory bank with current format understanding  

### (2025-07-19 20:53) - Grouping Still Incorrect & Unknown Field Correlation Needed

### (2025-07-19 22:20) - SurfaceGroupKey Hierarchy & Grouping Tester
- **Grouping Tester Implemented (`pm4-test-grouping`)**: Exports geometry grouped by `MSUR.SurfaceGroupKey` to visualize hierarchy.
- **Findings**:
  - Values above 19 represent progressively smaller subdivisions (e.g., 24 ≈ per-surface / polygon).
  - Value 19 aligns with WMO object groups (building-level grouping).
  - Lower values (<19) appear to be higher-level spatial containers.
- **Implications**:
  - `SurfaceGroupKey` must be interpreted as a hierarchy (subdivision → object → container) rather than a flat identifier.
  - `MSUR.IndexCount` grouping alone remains insufficient; composite or hierarchical keys are required for correct assembly.
  - Next: prototype composite grouping using `(ParentIndex, SurfaceGroupKey, IndexCount)` and validate geometry alignment.
- MSUR.IndexCount grouping reduced objects to **15**, but faces remain mis-assigned and geometry still fragmented.
- Evidence: OBJ output shows scattered clusters; warnings reveal ~480 k missing vertices (indices beyond global pool).
- Hypothesis: True object key likely involves multiple fields (e.g., high/low bytes of `surface_key`, `reference_index`, or padding fields).
- Action Plan:
  1. Dump **all** fields (including padding/flags) from MSUR, MSLK, MPRL, MPRR.
  2. Auto-correlate field values to surface counts, triangle counts, and invalid-index percentages.
  3. Prototype `pm4-test-grouping` command to iterate candidate keys and measure object/face alignment.
  4. Update assembler/exporters after key confirmed.
- All unknown fields are treated as potentially meaningful; no data considered filler.
