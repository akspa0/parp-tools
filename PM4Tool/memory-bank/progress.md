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
* Visual inspection now confirms that the MSCN chunk in PM4 files represents the exterior (boundary) vertices for each object.

### Next Steps
* Visually analyze OBJ output of MSCN points.
* Attempt to correlate MSCN points with other mesh data (e.g., MSUR, MSVT).
* Continue research into the semantic meaning and usage of MSCN data.
* Refine grouping logic as more is learned.
* Automate mapping of unk00/unk01 to WMO filenames.
* Update the PM4 exporter to annotate/export MSCN points as "exterior vertices" and validate this in future exports.

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

*   **WMO Root/Group Loading:** Core logic exists to load WMO root (`.wmo`) and group (`

## Mesh+MSCN Boundary Test Progress (2024-07-21)

### What Works
- New test for mesh extraction and MSCN boundary output successfully writes OBJ and diagnostics files for key PM4 files.
- All build errors related to type mismatches (`uint` vs `int`, `MsvtVertex` to `Vector3`) have been resolved.

### Current Issue
- The test process hangs after outputting the mesh and MSCN boundary files. The process does not exit and must be manually cancelled.
- All file streams appear to be closed, and no exceptions are reported, but the test method does not return.

### What's Left to Build / In Progress
- Debug and resolve the test process hang after mesh+MSCN boundary output.
- Once resolved, proceed with mesh comparison and placement inference.

### Known Issues
- Test automation is currently blocked by the process hang after mesh+MSCN boundary output.
- Further investigation is needed to ensure all resources are disposed and the test method completes as expected.

## Batch WMO-to-OBJ Exporter Progress (2025-04-18)

- Successfully implemented and validated a batch WMO-to-OBJ exporter as an xUnit test (`WmoBatchObjExportTests`).
- Recursively processes WMO binaries, exports each to OBJ, and writes to `/output/wmo/` with preserved folder structure.
- Uses caching: skips export if OBJ already exists, making repeated test runs efficient.
- This exporter is now the canonical workflow for generating OBJ caches for mesh comparison and analysis.
- Unblocks robust, repeatable PM4/WMO mesh analysis and placement inference workflows.

## WMO v14 Group Mesh Extraction Progress (2025-04-19)

### What Works
- v14-specific group chunk parser implemented: parses MOVT, MONR, MOTV, MOPY, MOVI, and assembles mesh data into mesh structures.
- OBJ export now works for v14 WMOs if all required subchunks are present and correctly parsed.
- Logging and error handling improved; missing geometry is now traceable to missing or malformed subchunks.

### What's Left to Build / In Progress
- Refine mesh assembly logic for edge cases and additional subchunks (MOBA, MLIQ, etc.).
- Add batch/group export support for multiple groups per WMO.
- Further improve error handling and debug output.
- Test and validate on a wider range of v14 WMOs.

### Known Issues
- Some geometry may still be missing due to incomplete chunk parsing or undocumented subchunk formats.
- Need for robust handling of optional/missing subchunks.
- Further research required for less common subchunks and edge cases.

## Completed Tasks (2024-07-21)
- Removed all liquid handling code and DBCD dependency from the project. Liquid support will be implemented in a separate project.

## New Focus: WMO Texturing Investigation (2024-07-21)
- Current focus is on robust handling of WMO chunk data for texturing, supporting both v14 (mirrormachine) and v17 (wow.export) formats.
- Next steps:
  - Review wow.export for v17 chunk/texturing support.
  - Crosswalk v14 (mirrormachine) knowledge to v17 structures.
  - Enumerate all relevant chunks and string fields.
  - Design a unified data model for texturing.
  - Update parsers and export logic accordingly.
- Goal: Enable full WMO/OBJ+MTL reconstruction with correct texturing for both legacy and modern formats.

## Progress Update (2024-07-21)
- v17 WMO writer skeleton is complete.
- Real serialization for MOMT (materials) and MOGI (group info) is implemented.
- WmoMaterial and WmoGroupInfo classes are mapped to the correct v17 binary layouts.
- Next steps: implement real serialization for MOHD (root header), MOBA (batch info), MOGP (group header), and integrate the writer into the v14→v17 conversion pipeline.
- Deep-dive analysis of chunk mapping and reference implementations is ongoing.
- Validation with wow.export, mirrormachine, and noggit-red is planned.

## Pending Implementation (2024-07-21)
- Unconditional export of MSCN points to all OBJ outputs (combined and per-file) is not yet implemented.
- The plan is to always include MSCN points with the (X, -Y, Z) transform and clear labeling, with no conditional flags.
- This will be retried in a new session.

## Progress Update: PM4 Test/Export Refactor v2

- **What works:** Existing test suite exports all mesh and chunk data, but with much duplicated logic and scattered coordinate conventions.
- **What's next:** Begin modularization:
  1. Implement `Pm4CoordinateTransforms` and migrate all coordinate logic.
  2. Refactor export logic into `ObjExportUtil`.
  3. Standardize validation with `ChunkValidator`.
  4. Update test flows and documentation.
- **Known issues:** High risk of errors when updating transforms or adding new chunk types due to code duplication.
- **Current status:** Refactor plan approved and mapped; implementation to begin.