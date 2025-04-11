# Memory Bank Archive

This file contains historical context, completed tasks, and resolved issues from previous development phases of the WoWToolbox project.

## Phase: Initial Analysis Tool & PM4/ADT Refactor (Approx. Q1-Q2 2025)

### Discoveries & Key Changes:
*   **ADT Loading Refactor:** Implemented handling for split ADTs (loading `_obj0.adt` into `TerrainObjectZero`) and non-split ADTs (attempting to load base `.adt` into `BfA.Terrain`, treating as 0 placements if successful but `_obj0` missing).
*   **PM4-Centric Logic:** Refactored `AnalysisTool` to iterate through PM4s, check for corresponding `_obj0.adt`, and perform correlation if both are valid.
*   **Consolidated Output:** Tool now generates a single `analysis_results.yaml` containing results for all processed PM4s (including PM4 IDs, ADT IDs, Correlated Placements, and Errors) instead of multiple files/directories. Removed summary report and individual file outputs.
*   **MVER/MSHD Optional:** Marked `MVER` and `MSHD` as `[ChunkOptional]` in `PM4File.cs` to handle files potentially missing these chunks.
*   **0-Byte Handling:** Added checks to skip 0-byte PM4 files and handle 0-byte `_obj0.adt` files gracefully during correlation.
*   **Pairing Logic Fixed:** Corrected ADT/PM4 filename pairing to account for zero-padding differences.
*   **ADT ID Extraction:** Added separate extraction of `UniqueId`s from `_obj0.adt` (`MDDF`/`MODF`) and included them in the YAML output.
*   **`Program.cs` Updates:** Added `Pm4AnalysisResult` class. Modified main loop, error handling, YAML serialization, 0-byte checks, file pairing logic, and added `ExtractAdtUniqueIds` helper.
*   **`PM4File.cs` Updates:** Marked `MVER` and `MSHD` properties with `[ChunkOptional]`.
*   **`AdtService.cs` Updates:** Simplified to have only one `ExtractPlacements` overload taking `TerrainObjectZero`.

### ADT Parser Implementation Details (Pre-Correlation Focus):
*   Created ADT `Placement` model (`Placement.cs`).
*   Implemented `AdtService` logic to extract `Placement` data using `Warcraft.NET` classes (`Terrain` for base `.adt` containing MMDX/MWMO, `TerrainObjectZero` for `_obj0.adt` containing MDDF/MODF).
*   Integrated `AdtService` into `WoWToolbox.AnalysisTool` using the correct `Warcraft.NET` ADT objects.
*   Added `YamlDotNet` dependency to `WoWToolbox.AnalysisTool`.
*   Implemented comparison logic in `AnalysisTool` to filter ADT placements by PM4 UniqueIDs.
*   Implemented YAML output for correlated placements.
*   Added `FilePath` property to `Placement` model and updated `AdtService` to populate it from `MMDX`/`MWMO` chunks (via `Terrain` object).
*   Implemented listfile loading in `AnalysisTool`.
*   Implemented `FileDataId` lookup in `AdtService` using `NameIdIsFiledataId` flag or reverse path matching (case/slash insensitive) against listfile.
*   Populated obsolete `Name` property from `Path.GetFileName(FilePath)`.
*   Verified `AnalysisTool` runs successfully (on single files) and generates `correlated_placements.yaml` with `FilePath`, `FileDataId`, `UsesFileDataId`, `Name`, and human-readable `FlagNames`.
*   Handled `Warcraft.NET` `Rotator` conversion in `AdtService.ConvertRotation`.
*   Handled `MODFEntry`/`MDDFEntry` `UniqueId` casting in `AdtService`.

## Phase: PD4 Parser Development & OBJ Export (Earlier Phase)

### PD4 Parser:
*   **Core Chunk Loading:** Loaded test files, identified all documented chunks.
*   **Build System:** Project built successfully after removing the unused `WoWToolbox.Validation` project from the solution.
*   **Test Suite:** `LoadPD4Files_ShouldLoadChunks` test created and built.
*   **Separate OBJ Export:** Test updated to export separate `_mspv.obj`, `_msvt.obj`, `_mscn.obj` files (reverted from combined). Also exported `_mslk.obj` and `_mslk_nodes.obj`. Output files generated successfully in `test/WoWToolbox.Tests/bin/Debug/net8.0/output/`.
*   **OBJ Face Generation:** Implemented logic in `PD4FileTests` to generate `f` lines in `_msvt.obj` based on MSUR -> MSVI -> MSVT links.
*   **Visual Geometry Confirmation:** Exported MSPV, MSVT, MSCN vertex data visually confirmed to match source WMO geometry in previous runs.
*   **Coordinate Transforms:** Confirmed MSPV (`X, Y, Z`), MSVT (`offset - v.X`, `offset - v.Y`, `v.Z`), and MSCN (`X, Y, Z`) transforms.
*   **Logging:** Added detailed debug logging to tests, including MSLK.Unk12.
*   **Analysis Tool Execution:** `WoWToolbox.AnalysisTool` ran on logs, parsed and logged MSLK.Unk12. Confirmed "Node Only" / "Geometry Only" groups via `Unk04`. Confirmed Node Anchor Mechanism logic worked.
*   **Chunk Audit:** Confirmed implementation matched documentation.
*   **Resolved Issues:** Fixed issue preventing tests from generating OBJ/log files.
*   **Known Issues (at the time, potentially still relevant):**
    *   MSUR/MSVI structure/purpose needed further investigation.
    *   `PD4File.Serialize` method was not implemented.
    *   MSLK `Unk10` and `Unk12` meaning TBD.
    *   Some PD4 node entries had invalid `Unknown_0x10` links.
    *   Validation Assertions were commented out.

## Other Historical Notes
*   Older PM4 OBJ export and MSLK analysis work might exist in test code but is less relevant to the current `AnalysisTool`. 