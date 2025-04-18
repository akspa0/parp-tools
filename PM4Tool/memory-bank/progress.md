# Project Vision & Immediate Technical Goal (2024-07-21)

## Vision
- Inspire others to explore, understand, and preserve digital history, especially game worlds.
- Use technical skill to liberate hidden or lost data, making it accessible and reusable for future creators and historians.

## Immediate Technical Goal
- Use PM4 files to infer and reconstruct WMO placements in 3D space for ADT chunks.
- Match PM4 mesh components to WMO meshes, deduce model placements, and generate placement data.
- Output reconstructed placement data as YAML for now, with plans for a chunk creator/exporter in the future.

---

# Progress

## PM4 Mesh Extraction Pipeline

### What Works
* MSUR surfaces are clustered and grouped by (unk00, unk01) and unk00, with OBJ exports for WMO group/root geometry.
* Visual validation confirms strong correspondence to in-game assets, with some missing geometry (likely in MSCN/MSLK).
* Core PM4/PD4 loading, chunk definitions, and OBJ export logic are stable.
* MSCN chunk is now confirmed to be an array of Vector3 (float) values, not int32/C3Vectori.
* All MSCN vectors have been exported as OBJ points for visualization.

### Next Steps
* Visually analyze OBJ output of MSCN points.
* Attempt to correlate MSCN points with other mesh data (e.g., MSUR, MSVT).
* Continue research into the semantic meaning and usage of MSCN data.
* Refine grouping logic as more is learned.
* Automate mapping of unk00/unk01 to WMO filenames.

### Known Issues
* Some geometry is still missing, likely defined in MSCN/MSLK.
* Doodad property decoding and MPRR index target identification remain open research items.

**Status: Pipeline reconstructs most WMO group geometry; further work needed for full fidelity, MSCN/MSLK research, and automation.**

## WMO Group File Handling Progress (2024-06)

- WMO group file handling is now correct: group files are parsed individually and merged, not concatenated.
- This resolves previous mesh extraction failures due to invalid concatenation of root and group files.
- Loader and tooling now follow the canonical split-group pattern for WMO parsing and analysis.

**Status: WMO group file handling is now correct; further work needed for full fidelity, MSCN/MSLK research, and automation.**

## WMO Group File Parsing (3.3.5 WotLK) Progress (2024-06)

- Refactored `WmoGroupMesh.LoadFromStream` parser in `WoWToolbox.Core`.
- **What Works:**
    - Successfully parses the `MOGP` header.
    - Correctly attempts to read subsequent standard geometry chunks (`MOPY`, `MOVI`, `MOVT`, etc.) sequentially with standard headers, enforcing the strict order required by the format.
    - Gracefully handles group files that lack geometry chunks after `MOGP` (e.g., `ND_IronDwarf_LargeBuilding` groups) by detecting EOF and producing an empty mesh without errors.
    - Resolves previous parsing errors caused by incorrect offset reading or assumptions about sub-chunks within `MOGP`.
- **Tools Updated:**
    - `MSCNExplorer`'s `compare-root` command confirmed to use the updated `WmoGroupMesh.LoadFromStream` via the standard root->group loading pattern.
- **Next Steps:**
    - Test the parser with 3.3.5 group files known to *contain* geometry chunks after `MOGP` to verify that code path.
    - Integrate parser into other tools/workflows as needed.

**Status: 3.3.5 WMO group parsing logic in Core is implemented and handles files with missing trailing geometry chunks correctly. Further testing with geometry-containing files is recommended.**

## Core WMO API Finalization (2024-06)

- **Status:** Core APIs for handling WMO root and group files are finalized for current requirements.
- **Components:**
    - `WmoRootLoader.LoadGroupInfo`: Successfully reads group count from `MOHD` and actual group file names from `MOGN`.
    - `WmoGroupMesh.LoadFromStream`: Handles 3.3.5 group file structure (sequential chunks after `MOGP`) and correctly loads empty meshes for groups lacking geometry chunks.
    - `WmoGroupMesh.MergeMeshes`: Correctly merges geometry from multiple loaded groups.
- **Tooling Integration:**
    - `MSCNExplorer` updated to use `LoadGroupInfo` and implement flexible group file finding (tries internal MOGN name, falls back to numbered convention `*_XXX.wmo`). This allows loading test data where filenames don't match internal WMO names.
- **Next Steps:**
    - Document these Core APIs.
    - Integrate into other tools as needed. 

## PM4 vs. WMO Mesh Comparison (Updated 2024-07-21)

### What Works
-   **WMO Extraction:** Loading WMO root/groups, merging geometry, exporting merged mesh to OBJ (`ND_IronDwarf_LargeBuilding_merged.obj` generated successfully).
-   **PM4 Extraction:** Loading PM4 files, extracting *all* state-0 geometry (using MSUR/MDSF/MDOS links), exporting full state-0 mesh to OBJ (`extracted_pm4_mesh_unfiltered_dev0000.obj` generated successfully).
-   Build environment stable on .NET 9.0.

### What's Left to Build / In Progress
-   **PM4 Component Analysis:** Implement logic to analyze extracted PM4 `MeshData` to find connected components.
-   **PM4 Largest Component Extraction:** Implement logic to identify and extract the largest connected component from the PM4 `MeshData`.
-   **Comparison Logic:** Develop methods to compare the WMO `MeshData` against the largest PM4 component `MeshData`.
-   **Batch Processing Framework:** Design and implement a tool/script to automate the process across multiple PM4 files and their corresponding WMOs.

### Current Status
-   Successfully generated baseline OBJ files for a test WMO and the full state-0 geometry of its associated PM4 tile.
-   Pivoted away from flawed filename-based filtering within PM4 extraction.
-   Current strategy is to analyze connected components *after* extracting the full PM4 mesh. Ready to plan implementation of component analysis.

### Known Issues
-   PM4 extraction currently skips non-triangle faces defined in MSUR chunks (warnings observed). This might affect component analysis accuracy.
-   Potential coordinate system mismatches (inversions) between WMO output and PM4 output need to be accounted for during comparison.
-   The "largest component" hypothesis needs validation across more test cases.

---
*Previous progress notes removed for brevity, focus is now on component analysis.*

## Progress: PM4/WMO Mesh Comparison

### What Works
- **WMO Mesh Extraction:**
    - Loading root WMO (`WmoRootLoader`) and associated group files (`WmoGroupMesh`).
    - Correctly handling 3.3.5 WMO group file structure variations.
    - Extracting vertices, normals, texture coordinates, and face indices from WMO groups.
    - Saving extracted WMO group mesh data to OBJ format (`WmoGroupMesh.SaveToObj`).
    - Basic tests (`WmoGroupMeshTests`) for loading/saving WMO group OBJ files.
- **PM4 Mesh Extraction (Initial Implementation):**
    - `Pm4MeshExtractor` class created in `WoWToolbox.MSCNExplorer`.
    - Logic for reading `MSVT` (vertices), `MSVI` (indices), `MSUR` (faces) chunks.
    - Initial implementation of vertex transformation (`MsvtToWorld_PM4`).
    - Initial implementation of face generation based on `MSUR` and `MSVI`.
    - Basic tests (`Pm4MeshExtractorTests`, now in `WoWToolbox.Tests`) for saving PM4 mesh to OBJ.
- **Build Environment:**
    - All relevant projects (`Core`, `Tests`, `MSCNExplorer`, `MPRRExplorer`, `AnalysisTool`) successfully target `.NET 9.0`.
    - Removed DBCD dependency from `WoWToolbox.Core`.

### What's Left to Build / Verify
- **PM4 Mesh Extraction (Build & Refinement):**
    - Resolve build errors in `Pm4MeshExtractor.cs` related to index types (UInt32 vs. Int32).
    - Verify correctness of `MsvtToWorld_PM4` transformation logic.
    - Verify correctness of face generation logic (indexing into `MSVI`).
    - Successfully generate OBJ file from `Pm4MeshExtractorTests`.
- **Mesh Comparison:**
    - Define comparison metrics (vertex positions, face indices/topology).
    - Implement the comparison algorithm.
    - Integrate comparison into a tool/test framework.
- **Testing & Validation:**
    - Visually compare generated OBJ files from PM4 and WMO sources.
    - Expand test cases for both PM4 and WMO extraction with diverse data.
    - Validate comparison results against known WMO/PM4 pairs.

### Current Status
- **Blocked:** PM4 mesh extraction is blocked by build errors in `Pm4MeshExtractor.cs`.
- WMO mesh extraction functionality is implemented and tested at a basic level.
- Ready to proceed with build error resolution and then OBJ verification.

### Known Issues
- Build errors in `Pm4MeshExtractor.cs` when assigning `UInt32` indices from `MSUR` to `Int32` properties in `Triangle`. 

## What Works
-   **Core WMO Loading:** `WmoRootFile.cs` and `WmoGroupFile.cs` can load basic WMO structure and group data.
-   **Chunk Handling:** Generic chunk reading mechanism is in place (`ChunkReader`).
-   **Basic Mesh Structure:** `MeshGeometry` struct exists in `WoWToolbox.Core`.
-   **WMO Group Mesh Loading:** `WmoGroupMesh.LoadFromStream` implemented in `WoWToolbox.Core`.
-   **WMO Group OBJ Export:** `WmoGroupMesh.SaveToObj` implemented and tested (`WmoGroupMeshTests`).
-   **PM4 File Loading:** `PM4File.cs` can load PM4 file chunks.
-   **`Pm4MeshExtractor` Base:** Initial structure for `Pm4MeshExtractor` exists in `WoWToolbox.MSCNExplorer`.
-   **Framework Alignment:** All projects target `net9.0`.

## What's Left to Build
-   **Resolve Build Errors:** Fix build errors in `WoWToolbox.MSCNExplorer` related to `Pm4MeshExtractor` and type mismatches (`uint` vs `int`).
-   **PM4 Mesh Extraction Logic:** Complete and verify the vertex transformation (`MsvtToWorld_PM4`) and face generation logic within `Pm4MeshExtractor.ExtractMesh`.
-   **PM4 Mesh OBJ Export:** Finalize and test `Pm4MeshExtractor.SaveToObj` (dependent on resolving build errors and completing extraction logic).
-   **Mesh Comparison:** Implement logic to compare `MeshGeometry` objects from PM4 and WMO sources.
-   **Comprehensive Testing:** Add more test cases covering different WMOs and PM4 files, including edge cases.

## Current Status
-   **Blocked:** Progress on PM4 mesh extraction and comparison is blocked by the build errors in the `WoWToolbox.MSCNExplorer` project.
-   WMO group mesh loading and OBJ export are functional and tested.
-   Project structure is refactored (.NET 9.0, tests consolidated, `DBCD` dependency removed from `Core`).

## Known Issues
-   **Build Error (CS0029):** `WoWToolbox.MSCNExplorer` fails to build due to an implicit conversion error (`uint` to `int`) in `Pm4MeshExtractor.cs` when assigning face indices.
-   **`MsvtToWorld_PM4` Verification:** The transformation logic in `Pm4MeshExtractor` needs thorough verification against expected WoW coordinates.
-   **Test Data:** Ensure necessary test files (WMOs, PM4s) are available and correctly referenced in tests.

### What Works
-   **Core WMO Loading:** `WmoRoot.cs`, `WmoGroup.cs`.
-   **Basic mesh extraction structure in `WmoGroupMesh.cs`.
-   **OBJ export for `WmoGroupMesh` (`SaveToObj` method and associated tests in `WmoGroupMeshTests.cs`).
-   **Basic PM4 mesh extraction structure in `Pm4MeshExtractor.cs`.
-   **OBJ export for `Pm4MeshExtractor` (`SaveToObj` method and associated tests moved to `WoWToolbox.Tests`).
-   **All projects target .NET 9.0**.
-   **Testing framework transitioned to xUnit**.

### What's Left to Build / In Progress
-   **Resolve Build Error:** The `WoWToolbox.MSCNExplorer` project does not build due to a type conversion error (CS0029) in `Pm4MeshExtractor.cs` when assigning triangle indices. This blocks further progress on PM4 extraction and comparison.
-   **Verify PM4 Mesh Extraction:** Once the build error is fixed, complete and verify the vertex transformation (`MsvtToWorld_PM4`) and face generation logic in `Pm4MeshExtractor.cs`.
-   **Run Tests:** Execute `Pm4MeshExtractorTests` and `WmoGroupMeshTests` to generate OBJ files.
-   **Visual Comparison:** Visually inspect the OBJ outputs from both extractors.
-   **Implement Programmatic Comparison:** Develop logic to compare `MeshGeometry` objects from PM4 and WMO sources.

### Current Status
-   **Blocked** by build error CS0029 in `Pm4MeshExtractor.cs`.
-   Memory Bank files (`activeContext.md`, `techContext.md`, `progress.md`) are updated.

### Known Issues
-   Build error CS0029: Cannot implicitly convert type 'uint' to 'int' in `Pm4MeshExtractor.cs` (Line ~181-183).
-   The exact transformation logic in `Pm4MeshExtractor.MsvtToWorld_PM4` needs verification against a known correct implementation or sample data.

## Project Progress & Status

**Overall Goal:** Develop tools and libraries within WoWToolbox to compare mesh geometry between PM4/PD4 map tile files and corresponding WMO group files.

### Completed

*   **WMO Root/Group Loading:** Core logic exists to load WMO root (`.wmo`) and group (`_###.wmo`) files, parsing major chunks.
*   **PM4/PD4 Loading:** Core logic exists to load PM4/PD4 files, parsing various chunks including MSVT, MSVI, MSUR, MDOS, MDSF, MSCN, etc.
*   **WMO Group Mesh Extraction (Initial):** `WmoGroupMesh.LoadFromStream` implemented to extract vertices, normals, UVs, and triangles from WMO group chunks.
*   **WMO Group OBJ Export:** `WmoGroupMesh.SaveToObj` implemented to save extracted WMO group mesh to OBJ format.
*   **WMO Group OBJ Export Test:** Test `LoadAndExportWmoGroup_ShouldCreateObjFile` in `WmoGroupMeshTests.cs` verifies OBJ creation.
*   **PM4 Mesh Extraction Refactoring (Initial):**
    *   Created `WoWToolbox.MSCNExplorer.Pm4MeshExtractor` class.
    *   Identified and moved basic vertex transformation logic (`MsvtToWorld_PM4`) from `PM4FileTests.cs`.
    *   Identified and moved face extraction logic (processing MSUR, MSVI, filtering by MDOS/MDSF) from `PM4FileTests.cs`.
*   **Common Mesh Data Structure (`MeshData`):** Created `WoWToolbox.Core.Models.MeshData` (`List<Vector3> Vertices`, `List<int> Indices`) to standardize extracted mesh representation.
*   **`Pm4MeshExtractor` Update:** Refactored `ExtractMesh` method to use and return `MeshData`, removing dependency on `WmoGroupMesh`.
*   **PM4 Mesh Extraction Test:** Created `Pm4MeshExtractorTests.cs` with test `ExtractMesh_ValidPm4_ShouldReturnMeshDataAndSaveObj` that verifies extraction and saves result to OBJ.

### In Progress / Next Steps

1.  **Build & Verify:** Build the solution to confirm recent changes compile.
2.  **Run Tests:** Run `Pm4MeshExtractorTests` and `WmoGroupMeshTests` to generate OBJ files.
3.  **Visual Inspection:** Visually compare the generated OBJ files.
4.  **Refactor `WmoGroupMesh`:** Update `WmoGroupMesh.LoadFromStream` to return `MeshData`.
5.  **Update `WmoGroupMeshTests`:** Update tests to use `MeshData` and the common OBJ saving helper.
6.  **Implement Mesh Comparison Logic:** Develop algorithms and tests to compare the `MeshData` from PM4 and WMO sources.

### Known Issues / Blockers

*   None currently identified after resolving the `RenderableMesh` issue by creating `MeshData`. 

# PM4/ADT UniqueID Correlation Progress (2024-07-21)

### What Works
- Added and executed `AdtServiceTests.CorrelatePm4MeshesWithAdtPlacements_ByUniqueId`.
- The test loads a PM4 file and its corresponding ADT _obj0 file, extracts meshes by uniqueID from the PM4, and placements from the ADT.
- For each uniqueID in the PM4, the test asserts a matching placement exists in the ADT and prints asset path and mesh stats.
- The test passes for the provided sample data, confirming the uniqueID correlation logic is correct and robust.

### Next Steps
- Expand test coverage with additional PM4/ADT pairs and edge cases.
- Integrate this correlation logic into higher-level workflows or tools as needed.
- Maintain and document this pattern as a core part of the analysis pipeline.

### Status
- UniqueID correlation between PM4 mesh groups and ADT placements is now covered by automated tests.
- Codebase is healthy and stable for this feature.

---

# UniqueID Correlation Limitation (2024-07-21)

**NOTE:** UniqueID-based mesh extraction and ADT correlation is ONLY possible for development_00_00.pm4. For all other PM4 files, uniqueID data and ADT correlation are NOT available—only baseline or chunk-based mesh exports are possible. This limitation is fundamental and should guide all future analysis, tests, and tooling. Do NOT attempt to generalize uniqueID grouping or ADT correlation beyond this special case.

---

# Mesh Analysis and Comparison Focus (2024-07-21)

**Note:** Mesh analysis and comparison is now the primary active work area. Previous blockers related to mesh extraction are resolved; the focus is on implementing robust comparison logic and diagnostics.

## What Works
- Mesh extraction from PM4 and WMO files is robust and validated in the test project.

## What's Left to Build
- No dedicated mesh analysis or comparison logic exists yet.
- Need to define what constitutes a mesh match (geometry, shape, bounding box, centroid, etc.).
- Need to implement comparison metrics: vertex/triangle count, bounding box, centroid, surface area, and advanced shape similarity as needed.
- Need to support tolerance for translation, rotation, and scale differences.
- Need to provide detailed diagnostic output for mismatches.

## Next Steps
1. Design a mesh comparison API/interface (input: two MeshData objects, output: result object with match/mismatch, score, diagnostics).
2. Implement basic geometric comparisons (vertex/triangle count, bounding box, centroid).
3. Add advanced shape similarity metrics as needed.
4. Integrate with test project and validate on real data.
5. Document rationale and design in memory bank.

--- 