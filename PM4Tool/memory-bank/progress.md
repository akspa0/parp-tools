# Progress

*(See `memory-bank/archive.md` for historical details on previous features like PD4 parsing/export)*

## PM4 File Parser & Geometry Assembly (`WoWToolbox.Core` / `WoWToolbox.Tests`)

### What Works
*   **Core PM4/PD4 Loading:** `PM4File.cs` and `PD4File.cs` load known chunks using the `ChunkedFile` base.
*   **Chunk Definitions:** C# classes exist for most known PM4/PD4 chunks (MVER, MSHD, MSVT, MSVI, MSUR, MDOS, MDSF, MDBH, MPRL, MPRR, MSPV, MSPI, MSLK, MCRC, MSCN).
*   **Vertex/Index Export:** Test code (`PM4FileTests`/`PD4FileTests`) successfully exports raw `MSVT` vertices (transformed Y,X,Z for OBJ) and `MSVI` indices.
*   **Path Node Linking:** Logic correctly links `MSPV` entries to `MSVT` vertex coordinates via the `MSLK`/`MSPI`/`MSVI` chain.
*   **Face Generation Logic (MSUR -> MDSF -> MDOS):**
    *   `PM4FileTests` correctly uses `MDSF` to map `MSUR` surfaces to `MDOS` states.
    *   Filters faces based on `MDOS.destruction_state == 0` for linked entries.
    *   Includes faces from `MSUR` entries that LACK an `MDSF` link, assuming they represent the default state (0).
*   **OBJ Export (`PM4FileTests`):**
    *   Generates `_render_mesh.obj` (original coordinates).
    *   Generates `_mspv.obj`, `_pm4_mslk_nodes.obj`, `_mprl.obj` etc. for structural/Doodad visualization.
    *   Generates `_render_mesh_transformed.obj` per file (using `Offset - Coordinate` transform).
    *   Generates `combined_render_mesh_transformed.obj` stitching all transformed meshes together.
*   **Batch Processing (`PM4FileTests`):** The test `LoadAndProcessPm4FilesInDirectory_ShouldGenerateOutputs` iterates through `.pm4` files, skips zero-byte files, and processes valid files, generating individual output sets including transformed OBJs.
*   **Chunk Documentation:** `docs/pm4_pd4_chunks.md` created and populated with current chunk understanding.
*   **Unknown Field Documentation:** Comments updated in chunk classes (MSLK, MSUR, MPRR, MPRL, MSHD) based on current knowledge and hypotheses.
*   **Doodad Node Identification:** MSLK nodes (`MspiFirstIndex == -1`) confirmed as Doodad placements, anchor points (`Unk10` -> MSVI -> MSVT) exported.
*   **Doodad Node Raw Data Export:** `PM4FileTests` now outputs raw values for `Unk00`, `Unk01`, `Unk04`(Grp), `Unk10`, `Unk12` in `_pm4_mslk_nodes.obj` comments.
*   **MPRR Link Visualization:** Test code (`PM4FileTests`) generates `_mprr_links.obj` with vertices and lines connecting MPRL points referenced by MPRR pairs.
*   **High-MPRR/MPRL-Ratio File Handling:** `PM4HighRatioProcessor` class successfully handles PM4 files with high MPRR/MPRL ratios by:
    *   Skipping problematic MPRR links to avoid index out-of-range exceptions
    *   Successfully processing MSVT vertices and MPRL points
    *   Generating OBJ models with correct coordinate transformations
    *   Using appropriate scale factor (36.0f) and coordinate offset (17066.666f)
*   **Coordinate Transformation:** Constants used for coordinate transformations have been standardized:
    *   `ScaleFactor = 36.0f` for scaling coordinates
    *   `CoordinateOffset = 17066.666f` for offsetting X and Y coordinates
    *   Ensures coordinates fall within the expected game coordinate range (+/- 17066.66)

### What's Left / Next Steps
*   **Doodad Property Decoding (Blocked):**
    *   Determine how `MSLK` entries link to specific M2 models (connection to `MDBH` via `Unk04` is unclear/failed in tests).
    *   Decode `MSLK` unknown fields (`Unk00`, `Unk01`, `Unk12`) to extract Doodad rotation (quaternion?) and scale.
    *   Requires manual analysis of generated `_pm4_mslk_nodes.obj` / `_mprr_data.csv` files and potentially `.debug.log` files (if accessible) or external research/reverse engineering due to lack of documentation and log access issues.
*   **MPRR Decoding:** Purpose partially clarified - links pairs of MPRL indices, uses 0xFFFF sentinel and index 0 as potential hub. Visualization of links is now available (`_mprr_links.obj`). A hypothesis exists that combined `MPRLUnk` fields (`04`/`06`, `14`/`16`) form 32-bit IDs linking to other structures (e.g., `MSLK`), requiring manual verification.
*   **Other Unknowns:** Decode remaining unknown fields in `MSUR`, `MPRL`, `MSHD`.
*   **WMO Geometry Parsing:** Begin parsing WMO file formats.
*   **QA:** Re-enable commented assertions.
*   **AnalysisTool (Low Priority):** Resolve directory processing termination bug.

## File Structure Dumping (`WoWToolbox.FileDumper`)

### What Works
*   **Tool Created & Functional:** Dumps PM4 and `_obj0.adt` data to individual YAML files.

### What's Left / Next Steps
*   **Enhance with New Data:** Add newly decoded PM4/ADT structures as they become available in `WoWToolbox.Core`.
*   **Testing/Verification:** Ongoing review of YAML output.

## Overall Status
*   Core focus was on verifying render geometry (complete) and decoding Doodads.
*   **Doodad decoding is currently blocked** pending manual analysis/research.
*   Parser correctly handles PM4 structure, generates geometry/face data, and outputs raw MSLK node data.
*   Chunk documentation updated with current understanding.
*   `WoWToolbox.FileDumper` is stable.
*   **PM4FileTests code structure** issues have been fixed, ensuring proper class organization and eliminating syntax errors.
*   **High-MPRR/MPRL-ratio files** can now be processed without exceptions.

### Known Issues
*   **Log File Access:** Debug/Output files (`.debug.log`, `_pm4_mslk_nodes.obj`, `_mprr_data.csv`) can become too large to read with available tools, hindering automated analysis.
*   **Doodad Decoding Incomplete:** Rotation, scale, and model ID linkage for MSLK Doodads are not implemented.
*   **MPRR Partially Unknown:** Structure (pairs of MPRL indices, sentinel) understood, visualization available, but exact functional purpose and potential links (ID hypothesis) TBD.
*   **AnalysisTool Termination (De-prioritized):** Exits after processing only the first file in directory mode.
*   **Test Data Issues (`development_00_00.pm4`):** Contains truncated `MDBH` and invalid `MPRR` indices.
*   **Zero-Byte Test Files:** Several `.pm4` files skipped by batch test.
*   **Missing Chunks in Some PM4s:** Some non-zero-byte PM4 files cause `Chunk "MSLK" not found` errors (handled by `try-catch`).
*   **Validation Assertions Commented:** Need re-enabling for QA.
*   **Interpretation/Use of MSCN, some MSLK `Unk*` fields TBD.**
*   **Vulnerability:** `SixLabors.ImageSharp` (dependency).
*   **Edit Tool Unreliability:** Failed to apply code changes to `PM4FileTests.cs` when attempting to add combined ID output to OBJ comments.

## Shared Milestones

*   Project Setup ✓
*   Core Framework ✓
*   PM4 Basic Implementation ✓
*   PM4 Validation & Testing ✓ *(Batch processing added, Some asserts bypassed)*
*   PM4 OBJ Export Refinement ✓ *(Geometry assembly, Transformed outputs)*
*   PM4 MSLK Analysis ✓ *(Hierarchy, Node Types, Doodad Anchors ID'd)*
*   PM4 MSCN/MDSF Research ✓ *(MDSF link implemented, MSCN analysis paused)*
*   PD4 Basic Implementation ✓
*   PD4 OBJ Export ✓
*   OBJ Face Generation via MSUR ✓
*   ADT Parsing Implementation ✓
*   **New Tool:** File Dumper (`WoWToolbox.FileDumper`) ✓
*   PM4/ADT Data Correlation ✓
*   **New Documentation:** Chunk Guide (`docs/pm4_pd4_chunks.md`) ✓
*   **Assemble Render Geometry (MSVT/MSVI/MSUR)** ✓ *(Visually verified)*
*   **Document Unknown Fields** ✓
*   **Export Raw Doodad Node Data** ✓
*   **Export Raw MPRR Data** ✓
*   **Analyze MPRR Structure** ✓ (Paired indices, sentinel identified, visualization enabled)
*   **Handle High-MPRR/MPRL-Ratio Files** ✓ (Specialized processor implemented)
*   **Standardize Coordinate Transformations** ✓ (Scale and offset constants aligned)
*   **Decode Doodad Data (MSLK/MDBH)** 🚧 *(Blocked - Needs manual analysis/research)*
*   Assemble Structure Geometry (MSPV/MSLK paths) 🔲
*   MPRR Decoding ⏳ (Structure known, visualization available, purpose/links TBD, ID hypothesis pending manual verification)
*   Other Unknown Field Decoding 🔲
*   Legacy Support 🔲
*   Quality Assurance 🔲 *(Needs re-enabled asserts)*
*   Interpret Nodes / Analyze Unknowns 🚧 *(Blocked for Doodads)*
*   Build Cleanup ✓ 