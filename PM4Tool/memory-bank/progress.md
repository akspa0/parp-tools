# Progress

*(See `memory-bank/archive.md` for historical details on previous features like PD4 parsing/export)*

## PM4 File Parser & Geometry Assembly (`WoWToolbox.Core` / `WoWToolbox.Tests`)

### What Works
*   **Core PM4/PD4 Loading:** `PM4File.cs` and `PD4File.cs` load known chunks using the `ChunkedFile` base.
*   **Chunk Definitions:** C# classes exist for most known PM4/PD4 chunks (MVER, MSHD, MSVT, MSVI, MSUR, MDOS, MDSF, MDBH, MPRL, MPRR, MSPV, MSPI, MSLK, MCRC, MSCN).
*   **Vertex/Index Export:** Test code (`PM4FileTests`/`PD4FileTests`) successfully exports raw `MSVT` vertices (transformed Y,X,Z for OBJ) and `MSVI` indices.
*   **Path Node Linking:** Logic correctly links `MSPV` entries to `MSVT` vertex coordinates via the `MSLK`/`MSPI`/`MSVI` chain.
*   **Face Generation Logic (MSUR -> MDSF -> MDOS):**
    *   `PM4FileTests` now correctly uses `MDSF` to map `MSUR` surfaces to `MDOS` states.
    *   Filters faces based on `MDOS.destruction_state == 0` for linked entries.
    *   **Includes faces from `MSUR` entries that LACK an `MDSF` link**, assuming they represent the default state (0).
*   **OBJ Export:** Tests generate `_render_mesh.obj` containing filtered faces (including unlinked ones) and `_mspv.obj`/`_pm4_mslk_nodes.obj` for structural/Doodad visualization.
*   **Chunk Documentation:** `docs/pm4_pd4_chunks.md` created and populated with current chunk understanding.

### What's Left / Next Steps
*   **Visual Verification:** Verify the latest `_render_mesh.obj` in MeshLab to confirm the face generation logic yields the expected geometry.
*   **Doodad Assembly:**
    *   Decode `MSLK` unknown fields (`Unk00`, `Unk01`, `Unk04`, `Unk12`) to extract Doodad properties (Model ID, rotation, scale).
    *   Investigate the link between `MSLK` and `MDBH` for specific model identification.
*   **WMO Geometry Parsing:** Begin parsing WMO file formats.
*   **QA:** Re-enable commented assertions in tests and fix any bypassed test code.
*   **AnalysisTool (Low Priority):** Resolve directory processing termination bug if needed.

## File Structure Dumping (`WoWToolbox.FileDumper`)

### What Works
*   **Tool Created & Functional:** Dumps PM4 and `_obj0.adt` data to individual YAML files.

### What's Left / Next Steps
*   **Enhance with New Data:** Add newly decoded PM4/ADT structures as they become available in `WoWToolbox.Core`.
*   **Testing/Verification:** Ongoing review of YAML output.

## Overall Status
*   Core focus is on **verifying render geometry assembly** and then moving to **Doodad decoding**.
*   `PM4FileTests` correctly implements `MSUR` -> `MDSF` -> `MDOS` linking and handles unlinked faces.
*   Chunk documentation created.
*   `WoWToolbox.FileDumper` is stable for current data structures.
*   `AnalysisTool` remains de-prioritized.

### Known Issues
*   **AnalysisTool Termination (De-prioritized):** Exits after processing only the first file in directory mode.
*   **Test Data Issues (`development_00_00.pm4`):** Contains truncated `MDBH` and invalid `MPRR` indices (currently skipped).
*   **Validation Assertions Commented:** Need re-enabling for QA.
*   **Interpretation/Use of MSCN, some MSLK `Unk*` fields TBD.**
*   **Vulnerability:** `SixLabors.ImageSharp` (dependency).

## Shared Milestones

*   Project Setup ✓
*   Core Framework ✓
*   PM4 Basic Implementation ✓
*   PM4 Validation & Testing ✓ *(Assertions bypassed)*
*   PM4 OBJ Export Refinement ✓ *(Geometry assembly focus)*
*   PM4 MSLK Analysis ✓ *(Hierarchy, Node Types, Doodad Anchors ID'd)*
*   PM4 MSCN/MDSF Research ✓ *(MDSF link implemented, MSCN analysis paused)*
*   PD4 Basic Implementation ✓
*   PD4 OBJ Export ✓ *(Separate files, geometry assembly pending)*
*   OBJ Face Generation via MSUR ✓ *(Current logic implemented in tests)*
*   ADT Parsing Implementation ✓ *(Via Warcraft.NET)*
*   **New Tool:** File Dumper (`WoWToolbox.FileDumper`) ✓
*   PM4/ADT Data Correlation ✓
*   **New Documentation:** Chunk Guide (`docs/pm4_pd4_chunks.md`) ✓
*   **Assemble Render Geometry (MSVT/MSVI/MSUR)** ⏳ *(Logic implemented, needs visual verification)*
*   **Decode Doodad Data (MSLK/MDBH)** 🔲 *(Next major step after verification)*
*   Assemble Structure Geometry (MSPV/MSLK paths) 🔲
*   Legacy Support 🔲
*   Quality Assurance 🔲 *(Needs re-enabled asserts)*
*   Interpret Nodes / Analyze Unknowns 🔲 *(Ongoing for Doodads)*
*   Build Cleanup ✓ 