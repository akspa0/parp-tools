# Active Context: Building the new parpToolbox Project

## Guiding Principles
- **Start Fresh:** All new development occurs in the `parpToolbox` project. The legacy `WoWToolbox` project is for reference only.
- **Dependency, Not Fork:** `wow.tools.local` is a strict library dependency; we will never modify its code directly.
- **Real Data Testing:** All tests must use real game data to ensure accuracy.
- **Clean Output:** All generated files must be written to the timestamped `project_output` directory.

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

### (2025-07-19 03:33) - Critical Data Loss Discovery & Index Pattern Analysis
- **MASSIVE DATA LOSS IDENTIFIED**: 110,988 out-of-bounds vertex accesses (~64% data loss)
- **Root Cause**: PM4 files are part of global tile system - individual tiles reference vertices from adjacent tiles
- **Evidence**:
  - Available vertices: 63,298 (indices 0-63297)
  - Maximum vertex index accessed: 126,595
  - Missing ~63,000 vertices from adjacent/related tiles
  - Sequential out-of-bounds patterns: 63298, 63299, 63300...
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
5. **Update memory bank** with current format understanding 
