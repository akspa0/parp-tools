# Progress

## PM4 File Parser

### What Works (PM4)
*   **Core Chunk Loading:** Successfully loads data from all documented PM4 chunks. Structure/loading logic validated or clarified for most.
*   **Build System:** Project builds successfully (including tests after analysis tool moved).
*   **Test Suite:** `LoadPM4File_ShouldLoadChunks` test builds and runs.
*   **Separate OBJ File Generation:** Test generates separate `.obj` files for `MSPV`, `MSVT`, `MPRL`, `MSCN`, **and `MSLK`**.
*   **MSLK Export:** Correctly exports single points (`p`) and multi-point lines (`l`) to `_mslk.obj`.
*   **MSLK Skipped Logging:** Correctly logs skipped/invalid MSLK entries to `_skipped_mslk.log`.
*   **Coordinate Transforms:** Iteratively refining transforms for OBJ export.
    *   MSPV output `(X,Y,Z)` confirmed visually correct by user.
    *   MSVT output `(Y,X,Z)` confirmed visually correct by user.
    *   MPRL output `(Y,Z,X)` confirmed visually correct by user.
    *   MSCN output `(X,Y,Z)` (raw) alignment needs verification.
*   **Index Validation:** Logic implemented for relevant chunks. `MSVI` logic confirmed correct. `MSLK` skips clarified as expected empty links. `MPRR` validation implemented and correctly identifies invalid indices in test data.
*   **Logging:** Enhanced logging added to `PM4FileTests.cs`, including raw MSUR->MSVI link details and selective `MSLKEntry` details in `summary.log`.
*   **MSUR Analysis:** Analyzed `MSUR` fields, fetched `MSVI` indices, and correlated with `MSVT` vertices via logs. Confirmed `MSUR` defines collections of `MSVI` indices (potentially non-sequential, reused). `Unk02` field may correlate with index count/primitive type. Purpose remains unclear but face generation assumption is invalid.
*   **Analysis Tool (`WoWToolbox.AnalysisTool`):** New console application created, built, and successfully executed on PM4 logs. Contains `MslkAnalyzer` class capable of parsing `.debug.log` and `_skipped_mslk.log`, grouping by `Unknown_0x04`, analyzing group types, and logging results to a file.
    *   **Hierarchy Confirmed:** Tool successfully grouped MSLK entries by `Unknown_0x04` and found "Mixed Groups", confirming `Unknown_0x04` links node entries (`MspiFirstIndex == -1`) to geometry entries (`MspiFirstIndex >= 0`).
    *   **Node Types Identified:** Analysis identified distinct `Unk00` and `Unk01` values for node entries, indicating different node types.

### What's Left / Next Steps (PM4)
*   **Interpret PM4 MSLK Analysis:** Fully understand the meaning of the different `Unk00`/`Unk01` node types identified.
*   **Visualize PM4 MSLK Hierarchy:** Decide on and implement a method (e.g., DOT, JSON, modified OBJ) to visualize the confirmed group structure.
*   **Correlate PM4 MSLK with other chunks:** Investigate links based on analysis results (e.g., using `Unk04`, `Unk10`, node types).
*   **Visualize MSCN Point Cloud:** Load `_mscn.obj` in viewer to check structure/alignment.
*   **Verify/Adjust MSCN Alignment:** Adjust `MSCN` export transform if needed.
*   **Research MDSF Usage:** Investigate the purpose and structure of `MDSF` data and potentially enable processing.
*   **Re-enable MPRR Validation:** Once parsing/visualization is stable, decide how to handle the test file's invalid MPRR data (e.g., keep assertion commented, modify test, get new test file).

### Current Status (PM4)
*   Parsing basics functional, index bounds checks refined (except `MPRR->MPRL` bypassed for test data).
*   Coordinate transforms for MSPV `(X,Y,Z)`, MSVT `(Y,X,Z)`, and MPRL `(Y,Z,X)` confirmed correct. MSCN uses `(X,Y,Z)`. Alignment checks pending for MSCN.
*   Separate OBJ files generated for all key geometric chunks including MSLK.
*   **MSLK:** Geometry export distinguishes points/lines. Skipped entries logged separately. **Mixed Group hierarchy (`Unk04` linking nodes/geometry for specific objects) confirmed via `MslkAnalyzer`. JSON hierarchy export implemented.** Node types identified via `Unk00`/`Unk01`. Next steps involve interpretation and visualization of nodes/hierarchy.
*   MSUR analysis deferred.
*   `MDSF` export disabled.

### Known Issues (PM4)
*   **Test Data Issues:** `development_00_00.pm4` contains:
    *   Truncated `MDBH` chunk.
    *   `MPRR` entries with indices out of bounds for `MPRL`.
*   **MSUR Validation Assertion:** Commented out in `PM4FileTests.cs` as the underlying face generation assumption was removed.
*   **MPRR Validation Assertion:** Currently commented out in `PM4FileTests.cs` due to test data issues.
*   **MSCN Data:** Interpretation and transformation unknown. Needs visual check.
*   **MDSF Usage:** Unknown.
*   **Vulnerability:** `SixLabors.ImageSharp` v2.1.9.
*   **Test Filtering:** `dotnet test --filter` not working reliably in user environment.
*   **MSLK Unknowns:** `Unknown_0x04` confirmed as group/object ID. Distinct node types identified by `Unk00`/`Unk01` (e.g., 0x01, 0x11). **Precise *meaning* of node types TBD.** Purpose of `Unknown_0x10` (likely MSVI index) TBD. Meaning of `Unk12` flag (`0x8000`) TBD.
*   **MSUR Purpose Unknown:** `MSUR` defines collections of `MSVI` indices. Face generation assumption invalid. `Unk02` potentially related to primitive type.

## PD4 File Parser (New Focus)

### What Works (PD4)
*   **Core Chunk Loading:** Successfully loads PD4 test files (`6or_garrison_workshop_v3_snow.pd4`, `_lod1.pd4`) using dedicated `PD4File` class.
*   **Chunk Detection:** Correctly identifies and loads all expected PD4 chunks (MVER, MCRC, MSHD, MSPV, MSPI, MSCN, MSLK, MSVT, MSVI, MSUR).
*   **Test Suite:** `LoadPD4Files_ShouldLoadChunks` test created and passes basic loading/chunk detection.
*   **Separate OBJ File Generation:** Test generates separate `.obj` files for `MSPV`, `MSVT`, `MSCN`, and `MSLK` (paths/points).
*   **Visual Geometry Confirmation:** Exported MSPV, MSVT, MSCN vertex data visually confirmed to match source WMO geometry.
*   **Coordinate Transforms (PD4):**
    *   MSPV output uses direct `(X, Y, Z)` mapping.
    *   MSVT output uses `worldX = offset - v.X`, `worldY = offset - v.Y`, `worldZ = v.Z` (Z scaling removed based on visual confirmation, correcting documentation error).
    *   MSCN output uses direct `(X, Y, Z)` mapping.
*   **Logging:** Test includes detailed debug logging for chunk counts, basic index validation checks, export steps, MSUR/MSVI details, **and MSLK processing/skipped entries in a format compatible with `MslkAnalyzer`**.
*   **Analysis Tool Execution:** `WoWToolbox.AnalysisTool` successfully executed using generated PD4 log files (`6or_garrison_workshop_v3_snow` and `_lod1`).
    *   **Hierarchy NOT Found:** Analysis revealed only "Node Only" or "Geometry Only" groups based on `Unknown_0x04`. No "Mixed Groups" (linking nodes to geometry via `Unk04`) were found, unlike in the PM4 test file.

### What's Left / Next Steps (PD4)
*   **Analyze More PD4 MSLK Nodes:** Examine detailed logs for more "Node Only" groups to confirm `Unk00`/`Unk01` patterns.
*   **Interpret PD4 MSLK Node Semantics:** Determine the meaning/purpose of different `Unk01` values and the overall role of nodes in the single-object context.
*   **Investigate PD4 MSLK `Unk10` Links:** Cross-reference node `Unk10` values with `MSVI` data to understand the link's purpose.
*   Analyze MSLK Unknown Fields (`Unk12`).
*   Analyze MSUR/MSVI Data.
*   Analyze/Visualize Exports (`_mscn.obj`, `_mslk.obj`).
*   Refactor Test Location (Optional).

### Current Status (PD4)
*   Basic parsing, testing, and OBJ export for geometry chunks are functional.
*   **MSLK:** Grouping structure difference (Node/Geom Only vs Mixed) confirmed and likely explained by file scope. Preliminary analysis of node fields (`Unk00`, `Unk01`, `Unk10`) performed based on updated analyzer logs.

### Known Issues (PD4)
*   MSUR/MSVI structure and purpose requires further investigation (similar to PM4).
*   `PD4File.Serialize` method is not implemented.
*   *Note: MSLK.Unk10 likely indexes MSVI, but the meaning of the data at that index is TBD.*
*   **MSLK Structure Difference:** The hierarchical linking pattern found in PM4 MSLK via `Unk04` is absent in the tested PD4 files.

## Shared Milestones
1.  Project Setup ✓
2.  Core Framework ✓
3.  PM4 Basic Implementation ✓
4.  PM4 Validation & Testing ✓ (Issues understood/bypassed)
5.  PM4 OBJ Export Refinement ✓ (MSLK points/lines separated, skipped logged)
6.  PM4 MSLK Analysis ✓ (Hierarchy confirmed, node types identified, JSON export added)
7.  PM4 MSCN Alignment / MDSF Research 🔲
8.  PD4 Basic Implementation ✓
9.  PD4 OBJ Export ✓
10. Legacy Support 🔲
11. Quality Assurance 🔲
12. **PD4 MSLK Analysis ✓** (Structure difference confirmed & contextualized, preliminary node analysis done, detailed logging enabled)
13. **Interpret PD4 MSLK Structure 🔲** (In Progress - Analyzing node details) 