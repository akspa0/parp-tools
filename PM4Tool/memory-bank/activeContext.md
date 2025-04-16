# Mode: PLAN

# Active Context: PM4/WMO Mesh Comparison (2024-07-21)

### Goal
Implement and test functionality to extract and compare renderable mesh geometry from PM4/PD4 navigation files and corresponding WMO group files.

### Recent Progress
- Finalized WMO APIs (`WmoGroupMesh.cs`) for loading and exporting WMO group geometry to OBJ.
- Finalized PM4 state-0 extraction (`Pm4MeshExtractor.cs`) to export geometry to OBJ.
- Confirmed both WMO and PM4 OBJ files can be generated for test assets (`ND_IronDwarf_LargeBuilding.wmo` and `development_00_00.pm4`).

### Current Focus: PM4 Component Analysis Strategy

**Update (Strategy Pivot):**

*   **Observation:** The extracted PM4 OBJ file (`extracted_pm4_mesh_unfiltered_dev0000.obj`) contains geometry for multiple assets, not just the single WMO being compared against (`ND_IronDwarf_LargeBuilding_merged.obj`). Direct comparison of these full files is not meaningful for identifying correspondence.
*   **Failed Approach:** Attempted to implement filtering in `Pm4MeshExtractor` based on WMO filenames potentially stored in the PM4's `MDBH` chunk. This failed because `MDBH` filenames are unreliable or incomplete, likely requiring data from corresponding ADT files (which are not fully available).
*   **New Hypothesis:** For a given PM4 (like `development_00_00.pm4`), the geometry corresponding to the primary WMO asset for that area (like `ND_IronDwarf_LargeBuilding`) is likely represented by the single largest "island" or "connected component" within the PM4's extracted state-0 mesh.
*   **New Goal:** Implement logic to analyze the full extracted PM4 `MeshData`, identify all connected components, isolate the largest one (by triangle/vertex count), and *then* compare that largest component to the corresponding WMO mesh.

### Next Steps (Component Analysis Plan)

1.  **Revert:** Ensure `Pm4MeshExtractor.cs` and `Pm4MeshExtractorTests.cs` are reverted to the state *before* the flawed filename filtering logic was added. (Should be current state after build error fixes).
2.  **Implement Component Analysis:**
    *   Create a new utility class (`MeshAnalysisUtils.cs`) likely in `WoWToolbox.Core`.
    *   Implement methods `FindConnectedComponents(MeshData)` and `GetLargestComponent(List<MeshData>)` using graph traversal (BFS/DFS).
3.  **Update PM4 Test:** Modify `Pm4MeshExtractorTests.cs` to:
    *   Extract the full mesh from `development_00_00.pm4`.
    *   Use `MeshAnalysisUtils` to find the largest component.
    *   Save the largest component to a new OBJ file (`extracted_pm4_dev0000_largest_component.obj`).
4.  **Run Tests:** Execute WMO and updated PM4 tests.
5.  **Visual Comparison:** Compare the WMO OBJ with the largest PM4 component OBJ.

### Open Questions/Decisions
- Location for `MeshAnalysisUtils`.
- Best metric for "largest" component (triangles vs. vertices).
- Robustness of the "largest component" hypothesis across different PM4/WMO pairs.

### Blockers
- None currently. Ready to plan and implement component analysis.

# WMO Group File Handling Update (2024-06)

## Key Finding
- WMO files are split into a root file and multiple group files (e.g., _000.wmo, _001.wmo, etc.).
- **Do NOT concatenate root and group files for parsing.** The WMO format does not support monolithic concatenation; the root references group files, but does not embed their data.
- The correct approach is:
  1. Parse the root file for group count and metadata.
  2. Parse each group file individually for geometry.
  3. Merge the resulting meshes for analysis or export.
- Loader and tools have been updated to follow this pattern. Previous attempts to concatenate files led to invalid parsing and must be avoided.
- This pattern is critical for all future WMO work and should be documented in all relevant tooling and documentation.

## Recent Changes
-   Reverted explicit `(int)` casts on index assignments in `Pm4MeshExtractor.cs` (lines 181-183) as an attempt to fix CS0029, anticipating the build will still fail.
-   Updated all projects (`Core`, `Common`, `Mpq`, `Tests`, `MSCNExplorer`) to target `.NET 9.0`.
-   Consolidated all tests (including `Pm4MeshExtractorTests`) into the `WoWToolbox.Tests` project.
-   Updated `WmoGroupMeshTests` to use `335_wmo/World/wmo/Dungeon/Ulduar/Ulduar_Raid.wmo`.
-   Updated `memory-bank/progress.md` and `memory-bank/techContext.md`.
