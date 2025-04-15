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

## Phase: PM4 Mesh Extraction & WMO Grouping (2025)

### Key Background & Breakthroughs
- Documented the iterative process of debugging the orientation of the combined_render_mesh_transformed.obj.
- Initial attempts using transformations derived from MPRL (X,-Z,Y) or the individual transformed render mesh (Y,X,-Z variations) resulted in vertical or horizontal inversions when normals were added.
- Confirmed through visual inspection (MeshLab) that applying the (X, Z, Y) transformation to both vertices and normals derived from MSVT/MSCN results in the correct orientation (top is top, left is left) in the combined_render_mesh_transformed.obj.
- Noted that the large size/complexity of the combined OBJ file causes issues loading in Blender, making individual file inspection or viewers like MeshLab necessary for full dataset verification.
- The core geometry data from MSVT/MSCN, combined with MSUR face indices and MSCN normals, now appears to render correctly oriented when the (X, Z, Y) transform is used.
- WoWToolbox.FileDumper tool tested successfully.
- Manual loading logic for MDSFChunk handles empty chunks.
- AnalysisTool PM4/ADT correlation investigation complete.
- Visualization confirms MSLK node entries represent Doodad placements (M2/MDX models).
- MSLK node entries identified as Doodad placements.
- Corrected MSUR->MDSF->MDOS linking implemented.
- Logic added to include unlinked MSUR faces as default geometry (state 0).
- Build and tests successful after render mesh changes.
- Chunk documentation created in docs/pm4_pd4_chunks.md.
- PM4FileTests refactored for batch processing: Iterates directory, skips zero-byte files, handles errors per file, and generates outputs for valid PM4s.
- PM4FileTests Output Changes: Removed MSLK hierarchy JSON output. Added individual transformed render mesh OBJ output (Offset - Coordinate). Added combined transformed render mesh OBJ output (Offset - Coordinate, stitched).
- Tests passing after latest changes.
- Render mesh OBJ generation (original, transformed, combined) verified.
- MSLK node entries identified as Doodad placements, anchor points exported via Unk10 -> MSVI -> MSVT.
- MPRR Structure: Confirmed to be variable-length sequences of ushort values, terminated by 0xFFFF. The value before the terminator is a potential flag. Indices likely do not point to MPRL.
- Unknown fields in MSLK, MSUR, MPRL, MSHD documented with current best guesses/hypotheses.
- Analysis suggests Doodad orientation is likely Quaternion-based and scale is float-based (similar to WMO MODD), potentially encoded in MSLK.Unk00/Unk01/Unk12.
- Link between MSLK.Unk04 (Group ID) and MDBH (filenames) failed in tests for development_22_18.pm4, assumption reverted in code.
- Test code (PM4FileTests) updated to output raw Unk00, Unk01, Unk04, Unk12 values in _pm4_mslk_nodes.obj comments and dedicated _mprr_data.csv file created.
- Blocker: Log file size/access issues prevent deeper automated analysis. Specific MSLK property encoding remains undecoded due to lack of documentation.
- MPRR Visualization: Based on incorrect assumption of paired MPRL indices; needs revisiting based on new structure.
- MPRL->MSLK ID Hypothesis: Deprioritized/Paused due to conflicting data and structural changes.
- Edit Tool Failure: Attempts to modify PM4FileTests.cs to output these combined IDs failed repeatedly.
- Code Fixes: Fixed syntax errors in PM4FileTests.cs related to improper class structure, fixing braces and moving TestDevelopment49_28_WithSpecializedHandling inside the PM4FileTests class.
- Test Data Path Fix: Updated the test path for "development_49_28.pm4" to correctly point to the development directory.
- Coordinate Transformation Fix: Updated the coordinate transformation constants in PM4HighRatioProcessor class: Changed ScaleFactor from 1.0f / 100.0f to 36.0f; Changed CoordinateOffset from 32.0f to 17066.666f to match the constants used elsewhere.
- High-MPRR/MPRL-Ratio File Handling: Specialized processing implemented for PM4 files with high MPRR/MPRL ratios, skipping MPRR links to avoid index out-of-range exceptions while still processing MSVT vertices and MPRL points.
- Build Errors Resolved: Fixed Mprr/MPRR typo in PM4FileTests.cs, resolving build failures.

### What Works (Historical)
- PM4 processing generates multiple OBJ outputs, including a combined render mesh (combined_render_mesh_transformed.obj) containing MSVT/MSUR geometry with MSCN normals. The coordinate transformation (X, Z, Y) for this combined file has been validated as producing the correct visual orientation in MeshLab. Individual file generation (_render_mesh_transformed.obj, _mprl.obj, _mslk.obj, etc.) continues.
- The primary issue with the combined render mesh is now its large size/complexity causing loading problems in tools like Blender, not incorrect geometry orientation. Further investigation into MSLK geometry, MPRR data, or planning a more robust exporter like glTF 2.0 are potential next steps.
- The combined OBJ is too large for easy viewing in some tools. The specific nature of MSLK geometry and MPRR data remains unclear. The development_49_28.pm4 file still requires specialized handling (though the current processing seems to handle it).
- Core PM4/PD4 Loading: PM4File.cs and PD4File.cs load known chunks using the ChunkedFile base.
- Chunk Definitions: C# classes exist for most known PM4/PD4 chunks (MVER, MSHD, MSVT, MSVI, MSUR, MDOS, MDSF, MDBH, MPRL, MPRR, MSPV, MSPI, MSLK, MCRC, MSCN).
- Vertex/Index Export: Test code (PM4FileTests/PD4FileTests) successfully exports raw MSVT vertices (transformed Y,X,Z for OBJ) and MSVI indices.
- Path Node Linking: Logic correctly links MSPV entries to MSVT vertex coordinates via the MSLK/MSPI/MSVI chain.
- Face Generation Logic (MSUR -> MDSF -> MDOS): PM4FileTests correctly uses MDSF to map MSUR surfaces to MDOS states. Filters faces based on MDOS.destruction_state == 0 for linked entries. Includes faces from MSUR entries that LACK an MDSF link, assuming they represent the default state (0).
- OBJ Export (PM4FileTests): Generates _render_mesh.obj (original coordinates). Generates _mspv.obj, _pm4_mslk_nodes.obj, _mprl.obj etc. for structural/Doodad visualization. Generates _render_mesh_transformed.obj per file (using Offset - Coordinate transform). Generates combined_render_mesh_transformed.obj stitching all transformed meshes together.
- Batch Processing (PM4FileTests): The test LoadAndProcessPm4FilesInDirectory_ShouldGenerateOutputs iterates through .pm4 files, skips zero-byte files, and processes valid files, generating individual output sets including transformed OBJs.
- Chunk Documentation: docs/pm4_pd4_chunks.md created and populated with current chunk understanding.
- Unknown Field Documentation: Comments updated in chunk classes (MSLK, MSUR, MPRR, MPRL, MSHD) based on current knowledge and hypotheses.
- Doodad Node Identification: MSLK nodes (MspiFirstIndex == -1) confirmed as Doodad placements, anchor points (Unk10 -> MSVI -> MSVT) exported.
- Doodad Node Raw Data Export: PM4FileTests now outputs raw values for Unk00, Unk01, Unk04(Grp), Unk10, Unk12 in _pm4_mslk_nodes.obj comments.
- MPRR Link Visualization: Test code (PM4FileTests) generates _mprr_links.obj with vertices and lines connecting MPRL points referenced by MPRR pairs.
- High-MPRR/MPRL-Ratio File Handling: PM4HighRatioProcessor class successfully handles PM4 files with high MPRR/MPRL ratios by: Skipping problematic MPRR links to avoid index out-of-range exceptions; Successfully processing MSVT vertices and MPRL points; Generating OBJ models with correct coordinate transformations; Using appropriate scale factor (36.0f) and coordinate offset (17066.666f).
- Coordinate Transformation: Constants used for coordinate transformations have been standardized: ScaleFactor = 36.0f for scaling coordinates; CoordinateOffset = 17066.666f for offsetting X and Y coordinates; Ensures coordinates fall within the expected game coordinate range (+/- 17066.66).

### Shared Milestones (Historical)
- Project Setup ✓
- Core Framework ✓
- PM4 Basic Implementation ✓
- PM4 Validation & Testing ✓ (Batch processing added, Some asserts bypassed, Build errors resolved)
- PM4 OBJ Export Refinement ✓ (Geometry assembly, Transformed outputs)
- PM4 MSLK Analysis ✓ (Hierarchy, Node Types, Doodad Anchors ID'd)
- PM4 MSCN/MDSF Research ✓ (MDSF link implemented, MSCN analysis paused)
- PD4 Basic Implementation ✓
- PD4 OBJ Export ✓
- OBJ Face Generation via MSUR ✓
- ADT Parsing Implementation ✓
- New Tool: File Dumper (WoWToolbox.FileDumper) ✓
- PM4/ADT Data Correlation ✓
- New Documentation: Chunk Guide (docs/pm4_pd4_chunks.md) ✓
- Assemble Render Geometry (MSVT/MSVI/MSUR) ✓ (Visually verified)
- Document Unknown Fields ✓
- Export Raw Doodad Node Data ✓
- Export Raw MPRR Data ✓
- Analyze MPRR Structure ✓ (Paired indices, sentinel identified, visualization enabled)
- Analyze MPRR Structure (Revised) ✓ (Confirmed 0xFFFF-terminated sequences, flag before terminator, indices likely not MPRL)
- Handle High-MPRR/MPRL-Ratio Files ✓ (Specialized processor implemented)
- Standardize Coordinate Transformations ✓ (Scale and offset constants aligned)
- Resolve Build Errors ✓ (Fixed Mprr/MPRR typo in PM4FileTests.cs)
- Decode Doodad Data (MSLK/MDBH) 🚧 (Blocked - Needs manual analysis/research)
- Assemble Structure Geometry (MSPV/MSLK paths) 🔲
- MPRR Index Target Identification ⏳ (Sequence structure known, indices target TBD)
- Other Unknown Field Decoding 🔲
- Legacy Support 🔲
- Quality Assurance 🔲 (Needs re-enabled asserts)
- Interpret Nodes / Analyze Unknowns  (Blocked for Doodad properties, MSLK.Unk10 confirmed as anchor index)
- Build Cleanup ✓

### Known Issues (Historical)
- Log File Access: Debug/Output files (.debug.log, _pm4_mslk_nodes.obj, _mprr_data.csv) can become too large to read with available tools, hindering automated analysis.
- Doodad Decoding Incomplete: Rotation, scale, and model ID linkage for MSLK Doodads are not implemented.
- MPRR Partially Unknown: Structure (pairs of MPRL indices, sentinel identified; later revised to 0xFFFF-terminated sequences) understood, visualization available (though based on old assumption), but exact functional purpose and index targets TBD.
- AnalysisTool Termination (De-prioritized): Exits after processing only the first file in directory mode.
- Null Reference Warnings (WoWToolbox.AnalysisTool): Three CS8625/CS8600 warnings exist in MprrHypothesisAnalyzer.cs.
- Test Data Issues (development_00_00.pm4): Contains truncated MDBH and invalid MPRR indices.
- Zero-Byte Test Files: Several .pm4 files skipped by batch test.
- Missing Chunks in Some PM4s: Some non-zero-byte PM4 files cause Chunk "MSLK" not found errors (handled by try-catch).
- Validation Assertions Commented: Need re-enabling for QA.
- Interpretation/Use of MSCN, some MSLK Unk* fields TBD.
- Vulnerability: SixLabors.ImageSharp (dependency).
- Edit Tool Unreliability: Failed to apply code changes to PM4FileTests.cs when attempting to add combined ID output to OBJ comments.

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