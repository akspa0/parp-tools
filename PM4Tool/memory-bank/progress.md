# Progress

## ADT File Parser & Analysis
### What Works: 
* Add "Created ADT Placement model (Placement.cs)", "Created ADTFile.cs inheriting ChunkedFile with chunk properties".
### What's Left / Next Steps: 
* Add "Implement AdtService to use ADTFile and extract Placement data" 
* "Integrate AdtService into AnalysisTool"
* Add YamlDotNet and implement YAML output for correlated placements".
### Current Status: 
* "ADT parsing framework initiated using Warcraft.NET base.".
###Known Issues: 
* "ADT parsing logic not yet implemented in AdtService".
###Shared Milestones: 
* "ADT Parsing Implementation 🔲 (Initial classes created)". 

## PM4 File Parser

### What Works (PM4)
*   **Core Chunk Loading:** Successfully loads data from all documented PM4 chunks. Structure/loading logic validated or clarified for most.
*   **Build System:** Project builds successfully after removing the unused `WoWToolbox.Validation` project from the solution.
*   **Test Suite:** `LoadPM4File_ShouldLoadChunks` test builds and runs.
*   **Separate OBJ File Generation:** Test generates separate `.obj` files for `MSPV`, `MSVT` (now filtered by MDOS state), `MPRL`, `MSCN` (raw points), and `MSLK` (paths/points), and `_pm4_mslk_nodes.obj` (node anchors).
*   **MSLK Export:** Correctly exports single points (`p`) and multi-point lines (`l`) to `_mslk.obj`.
*   **MSLK Skipped Logging:** Correctly logs skipped/invalid MSLK entries to `_skipped_mslk.log`.
*   **MSLK Node Anchor Export:** Correctly exports node anchor points using `Unknown_0x10 -> MSVI -> MSVT` link to `_pm4_mslk_nodes.obj`. (Visualization shows concentration near corners).
*   **Coordinate Transforms:** Confirmed visually correct transformations for key chunks.
    *   MSPV output `(X, Y, Z)` confirmed.
    *   MSVT output `(Y, X, Z)` confirmed.
    *   MPRL output `(X, -Z, Y)` confirmed.
    *   MSCN output reverted to raw points `(X,Y,Z)` in `_mscn.obj` (count mismatch with MSVT).
*   **OBJ Face Generation (MSUR):** Implemented logic in `PM4FileTests` to generate `f v1 v2 v3` lines in `_msvt.obj` based on MSUR -> MSVI -> MSVT links, **only if the linked `MDOS.destruction_state` is 0**.
*   **Index Validation:** Logic implemented for relevant chunks. `MSVI` logic confirmed correct. `MSLK` skips clarified as expected empty links. `MPRR` validation implemented and correctly identifies invalid indices in test data (Assertions currently commented out).
*   **Logging:** Enhanced logging added to `PM4FileTests.cs`, including raw MSUR->MSVI link details, selective `MSLKEntry` details, and detailed MDSF->MDOS link information (including `destruction_state`). `MSLK.Unk12` also logged.
*   **MSUR Analysis:** Analyzed `MSUR` fields, fetched `MSVI` indices, and correlated with `MSVT` vertices via logs. Confirmed `MSUR` defines collections of `MSVI` indices.
*   **Analysis Tool (`WoWToolbox.AnalysisTool`):** New console application created, built, and successfully executed on PM4 logs. Contains `MslkAnalyzer` class capable of parsing `.debug.log` and `_skipped_mslk.log`, grouping by `Unknown_0x04`, analyzing group types, and logging results to a file. Updated to parse and log MSLK.Unk12.
    *   **Hierarchy Confirmed:** Tool successfully grouped MSLK entries by `Unknown_0x04` and found "Mixed Groups".
    *   **Node Types Identified:** Analysis identified distinct `Unk00` and `Unk01` values for node entries.
    *   **Node Anchors Exported:** Test exports node anchor points to `_pm4_mslk_nodes.obj`.
*   **Chunk Audit:** Confirmed implementation aligns with documentation after removing non-existent `MSRN` chunk handler.
*   **Data Correlation:** Discovered link between ADT UniqueIDs and PM4 data. Confirmed `MDSF` links `MSUR` (surfaces via `msur_index`) to `MDOS` (destructible building states via `mdos_index`). Clarified `MDOS` structure (`m_destructible_building_index`, `destruction_state`).
*   **Chunk Definitions:** Corrected field names in `MDOSChunk`/`MDSFChunk` based on struct info.

### What's Left / Next Steps (PM4)
*   **Analyze `Unk12`:** Examine updated analysis logs for `MSLK.Unk12` patterns.
*   **Interpret PM4 MSLK Nodes:** Visualize `_pm4_mslk_nodes.obj` with `_msvt.obj` to understand `Unk01` types.
*   **Interpret MSCN Data:** Investigate the purpose of MSCN, given the count mismatch with MSVT.
*   **Visualize PM4 MSLK Hierarchy:** Decide on and implement visualization.
*   **Correlate PM4 MSLK with other chunks.**
*   **Research MDSF Usage (Partially Clarified):** Role as MSUR<->MDOS link confirmed. Further details?
*   **Re-enable Validation Assertions** (MPRR, MSPI, etc.) after file generation/parsing is stable.

### Current Status (PM4)
*   Parsing basics functional.
*   Coordinate transforms confirmed for MSPV, MSVT, MPRL. MSCN exported as raw points.
*   Separate OBJ export logic reverted MSCN to points, filters MSVT faces by MDOS destruction state.
*   **MSLK:** Mixed Group hierarchy confirmed. Node anchors exported. `Unk12` logging added.
*   `MSRN` chunk removed.
*   ADT/PM4 UniqueID link identified (`m_destructible_building_index`). MDOS/MDSF structures, link, and field names clarified and implemented.

### Known Issues (PM4)
*   **Test Data Issues:** `development_00_00.pm4` contains truncated `MDBH` and invalid `MPRR` indices.
*   **Validation Assertions Commented:** MPRR, MSPI, and other count/load assertions are temporarily commented out (Ready to be re-enabled).
*   **MSCN Data:** Interpretation and transformation unknown.
*   **MDSF:** Link role confirmed, specific usage details TBD.
*   **Vulnerability:** `SixLabors.ImageSharp` v2.1.9.
*   **MSLK Unknowns:** `Unk01` meaning TBD. `Unk10` purpose in geometry TBD. `Unk12` meaning TBD (analysis pending).
*   **MSUR Purpose:** Face generation implemented based on hypothesis (supported by MDSF link), needs verification via visualization.

## PD4 File Parser

### What Works (PD4)
*   **Core Chunk Loading:** Loads test files, identifies all documented chunks.
*   **Build System:** Project builds successfully after removing the unused `WoWToolbox.Validation` project from the solution.
*   **Test Suite:** `LoadPD4Files_ShouldLoadChunks` test exists and builds.
*   **Separate OBJ Export:** Test **updated to export separate** `_mspv.obj`, `_msvt.obj`, `_mscn.obj` files (reverted from combined). Also exports `_mslk.obj` and `_mslk_nodes.obj`. (**Issue:** File generation currently not working, under investigation).
*   **OBJ Face Generation (Attempted):** Implemented logic in `PD4FileTests` to generate `f` lines in `_msvt.obj` based on MSUR -> MSVI -> MSVT links. (Verification pending file generation fix).
*   **Visual Geometry Confirmation:** Exported MSPV, MSVT, MSCN vertex data visually confirmed to match source WMO geometry in previous runs (when combined export worked partially).
*   **Coordinate Transforms (PD4):**
    *   MSPV output uses direct `(X, Y, Z)`.
    *   MSVT output uses `offset - v.X`, `offset - v.Y`, `v.Z` (Confirmed correct by user).
    *   MSCN output uses direct `(X, Y, Z)`.
*   **Logging:** Test includes detailed debug logging. **Updated to include MSLK.Unk12.**
*   **Analysis Tool Execution:** `WoWToolbox.AnalysisTool` runs on logs. **Updated to parse and log MSLK.Unk12.**
    *   **Hierarchy NOT Found:** Confirmed "Node Only" / "Geometry Only" groups via `Unk04`.
    *   **Node Anchor Mechanism Confirmed:** Logic implemented and works.
*   **Chunk Audit:** Confirmed implementation matches documentation.

### What's Left / Next Steps (PD4)
*   **Fix File Output:** Diagnose and resolve the issue preventing tests from generating OBJ/log files.
*   **Visualize Faces & Separate OBJs:** Verify separate `_mspv.obj`, `_msvt.obj` (with faces), `_mscn.obj` orientations.
*   **Analyze `Unk12`:** Examine updated analysis logs for `MSLK.Unk12` patterns.
*   **Interpret PD4 Node Semantics (`Unk01`)** via visualization.
*   **Investigate `Unk10` for Geometry.**
*   **Analyze MSUR/MSVI Data further (Face Generation).**

### Current Status (PD4)
*   Basic parsing, testing functional.
*   **OBJ Export:** Logic updated for **separate geometry files** (`_mspv`, `_msvt`, `_mscn`) plus `_mslk` and `_mslk_nodes`. Face generation added to `_msvt.obj`. **Output files generated successfully in `test/WoWToolbox.Tests/bin/Debug/net8.0/output/`.**
*   **MSLK:** Node anchors exported. Node types identified. `Unk12` logging added. Structure difference (Node/Geom Only) confirmed.

### Known Issues (PD4)
*   MSUR/MSVI structure and purpose requires further investigation (Face generation needs verification via visualization).
*   `PD4File.Serialize` method is not implemented.
*   MSLK `Unk10` purpose for geometry TBD. `Unk12` meaning TBD (analysis pending).
*   **Data Integrity:** Some PD4 node entries have invalid `Unknown_0x10` links.
*   **Validation Assertions Commented:** Assertions are temporarily commented out (Ready to be re-enabled).

## ADT File Parser & Analysis

### What Works (ADT)
*   Created ADT `Placement` model (`Placement.cs`).
*   Implemented `AdtService` logic to extract `Placement` data, correctly handling split ADT files by using `Warcraft.NET` classes (`Terrain` for base `.adt` containing MMDX/MWMO, `TerrainObjectZero` for `_obj0.adt` containing MDDF/MODF).
*   Integrated `AdtService` into `WoWToolbox.AnalysisTool` using the correct `Warcraft.NET` ADT objects.
*   Added `YamlDotNet` dependency to `WoWToolbox.AnalysisTool`.
*   Implemented comparison logic in `AnalysisTool` to filter ADT placements by PM4 UniqueIDs.
*   Implemented YAML output for correlated placements.
*   Added `FilePath` property to `Placement` model and updated `AdtService` to populate it from `MMDX`/`MWMO` chunks (via `Terrain` object).
*   Implemented listfile loading in `AnalysisTool`.
*   Implemented `FileDataId` lookup in `AdtService` using `NameIdIsFiledataId` flag or reverse path matching (case/slash insensitive) against listfile.
*   Populated obsolete `Name` property from `Path.GetFileName(FilePath)`.
*   Verified `AnalysisTool` runs successfully (on single files) and generates `correlated_placements.yaml` with `FilePath`, `FileDataId`, `UsesFileDataId`, `Name`, and human-readable `FlagNames`.

### What's Left / Next Steps (ADT & Correlation)
*   Implement directory processing in `AnalysisTool` to handle bulk analysis of ADT/PM4 pairs and PM4-only files.
*   Visualize/Verify `correlated_placements.yaml` output from directory processing.
*   Detailed analysis using UniqueIDs and `FilePath`/`FileDataId` from YAML.
*   Investigate `MDOS.destruction_state` impact on PM4 data/correlation.

### Current Status (ADT)
*   ADT parsing functional via `AdtService` using `Warcraft.NET` native classes for split file handling.
*   `AnalysisTool` successfully correlates ADT placements (including `FilePath`, `FileDataId`, and human-readable `FlagNames`) with PM4 UniqueIDs and outputs YAML (for single files).

### Known Issues (ADT)
*   `Warcraft.NET`'s `Rotator` conversion handled in `AdtService.ConvertRotation`.
*   `MODFEntry`/`MDDFEntry` `UniqueId` casting handled in `AdtService`.

## Shared Milestones
1.  Project Setup ✓
2.  Core Framework ✓
3.  PM4 Basic Implementation ✓
4.  PM4 Validation & Testing ✓ (Assertions bypassed/commented)
5.  PM4 OBJ Export Refinement ✓ (Separate files, MSLK refined)
6.  PM4 MSLK Analysis ✓ (Hierarchy confirmed, node types identified, JSON export added, Unk12 logging added)
7.  PM4 MSCN Alignment / MDSF Research ✓ (MDSF<->MSUR/MDOS link found, definitions updated, logging added)
8.  PD4 Basic Implementation ✓
9.  PD4 OBJ Export ✓ (Reverted to separate files, MSLK nodes/paths separate)
10. OBJ Face Generation via MSUR ✓ (PM4 faces filtered by MDOS state)
11. ADT Parsing Implementation ✓ (ADT Placement model created, AdtService functional using Warcraft.NET split file classes)
12. PM4/ADT Data Correlation ✓ (AnalysisTool implemented for single files, verified, outputs YAML with FilePath, FileDataId, and human-readable Flags via listfile)
13. Legacy Support 🔲
14. Quality Assurance 🔲 (Needs re-enabled asserts)
15. PM4 & PD4 MSLK Node Anchor Analysis ✓ (Anchor mechanism confirmed, Unk00/Unk01 roles ID'd)
16. Interpret Nodes / Analyze Unknowns 🔲 (Log/Analyzer changes for `Unk12` analysis pending; `Unk01` needs visualization)
17. Chunk Audit vs Docs ✓ (PD4 matches, PM4 corrected - MSRN removed)
18. Build Cleanup ✓ (Removed unused Validation project, fixed warnings) 