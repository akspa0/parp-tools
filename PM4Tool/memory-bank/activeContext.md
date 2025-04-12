# Mode: PLAN

# Active Context: Verify Geometry & Decode Doodads

**Goal:** Accurately parse PM4/PD4 files, **assemble the core render geometry (`MSVT`/`MSVI`/`MSUR`) into usable OBJ models**, understand the structure and **decode Doodad placements** from `MSLK` and related chunks (like `MDBH`), and provide file dumping tools.

**Current Focus:** **Doodad Placement Analysis & MPRR Link Interpretation.** We've identified MSLK node entries as Doodad placements and are exporting their anchor points. The current sub-focus is trying to understand how the specific Doodad model, rotation, and scale are encoded within the `MSLKEntry`'s unknown fields (`Unk00`, `Unk01`, `Unk04`, `Unk12`). Simultaneously, we are investigating the `MPRR` chunk, which links pairs of `MPRL` points. A new hypothesis suggests pairs of unknown fields in `MPRL` (`MPRLUnk04`/`MPRLUnk06`, `MPRLUnk14`/`MPRLUnk16`) might form combined 32-bit IDs linking to other structures like `MSLK` nodes.

**Key Background (See archive.md for full history):**
*   `WoWToolbox.FileDumper` tool tested successfully.
*   Manual loading logic for `MDSFChunk` handles empty chunks.
*   `AnalysisTool` PM4/ADT correlation investigation complete.
*   **NEW Insight:** Visualization confirms `MSLK` node entries represent **Doodad placements** (M2/MDX models).
*   **Issue (De-prioritized):** `AnalysisTool` directory processing bug.
*   `MSLK` node entries identified as Doodad placements.
*   Corrected `MSUR`->`MDSF`->`MDOS` linking implemented.
*   Logic added to include unlinked `MSUR` faces as default geometry (state 0).
*   Build and tests successful after render mesh changes.
*   Chunk documentation created in `docs/pm4_pd4_chunks.md`.
*   **`PM4FileTests` refactored for batch processing:** Iterates directory, skips zero-byte files, handles errors per file, and generates outputs for valid PM4s.
*   **`PM4FileTests` Output Changes:**
    *   Removed MSLK hierarchy JSON output.
    *   Added individual transformed render mesh OBJ output (`Offset - Coordinate`).
    *   Added combined transformed render mesh OBJ output (`Offset - Coordinate`, stitched).
*   **Tests passing after latest changes.**
*   Render mesh OBJ generation (original, transformed, combined) verified.
*   `MSLK` node entries identified as Doodad placements, anchor points exported via `Unk10` -> `MSVI` -> `MSVT`.
*   `MPRR` chunk purpose remains unknown, but analysis suggests it links pairs of `MPRL` indices, using `0xFFFF` as a sentinel and potentially index `0` as a common reference/hub. Validation logic commented out.
*   Unknown fields in `MSLK`, `MSUR`, `MPRL`, `MSHD` documented with current best guesses/hypotheses.
*   Analysis suggests Doodad orientation is likely Quaternion-based and scale is float-based (similar to WMO `MODD`), potentially encoded in `MSLK.Unk00`/`Unk01`/`Unk12`.
*   Link between `MSLK.Unk04` (Group ID) and `MDBH` (filenames) failed in tests for `development_22_18.pm4`, assumption reverted in code.
*   Test code (`PM4FileTests`) updated to output raw `Unk00`, `Unk01`, `Unk04`, `Unk12` values in `_pm4_mslk_nodes.obj` comments and dedicated `_mprr_data.csv` file created.
*   **Blocker:** Log file size/access issues prevent deeper automated analysis. Specific MSLK property encoding remains undecoded due to lack of documentation.
*   **MPRR Visualization:** Regenerated `_mprr_links.obj` successfully, now including lines connecting MPRL points referenced by MPRR pairs.
*   **MPRR->MSLK ID Hypothesis:** Proposed that pairs of `MPRLUnk` fields might form combined 32-bit IDs linking MPRR/MPRL points to MSLK nodes.
*   **Edit Tool Failure:** Attempts to modify `PM4FileTests.cs` to output these combined IDs failed repeatedly.
*   **Code Fixes:** Fixed syntax errors in `PM4FileTests.cs` related to improper class structure, fixing braces and moving `TestDevelopment49_28_WithSpecializedHandling` inside the `PM4FileTests` class.
*   **Test Data Path Fix:** Updated the test path for "development_49_28.pm4" to correctly point to the development directory.
*   **Coordinate Transformation Fix:** Updated the coordinate transformation constants in `PM4HighRatioProcessor` class:
    *   Changed `ScaleFactor` from `1.0f / 100.0f` to `36.0f`
    *   Changed `CoordinateOffset` from `32.0f` to `17066.666f` to match the constants used elsewhere
*   **High-MPRR/MPRL-Ratio File Handling:** Specialized processing implemented for PM4 files with high MPRR/MPRL ratios, skipping MPRR links to avoid index out-of-range exceptions while still processing MSVT vertices and MPRL points.

**Next Steps:**
1.  **Manual Analysis / External Research (User Task):**
    *   Analyze `_pm4_mslk_nodes.obj` files for patterns in unknown fields relative to position/grouping.
    *   Attempt manual correlation of `MSLK.Unk04` with `MDBH` data from debug logs (if accessible).
    *   Research `MSLK` Doodad property encoding (quaternion/scale packing) in other parsers or forums.
    *   **NEW:** Manually analyze `_mprr_referenced_points.obj` comments. Calculate potential combined 32-bit IDs (e.g., `(MPRLUnk04 << 16) | MPRLUnk06`, `(((ushort)MPRLUnk14) << 16) | MPRLUnk16`). Cross-reference these calculated IDs with data from `MSLK` nodes (e.g., `Grp_Unk04` from `_mslk_doodad_data.csv` or data in `_pm4_mslk_nodes.obj`).
2.  **(If Information Found)** Implement decoding logic for rotation/scale/ID in `MSLKEntry.cs` and/or MPRL->MSLK linking logic, and update test output.
3.  **(If Blocked)** Consider pausing Doodad property decoding and MPRR->MSLK linking, and moving to WMO geometry parsing or other tasks.
4.  **Additional Testing:** Continue testing the updated coordinate transformation to ensure proper scaling of coordinates for all PM4 files, validating that the X, Y, Z coordinates fall within the expected game coordinate range (+/- 17066.66).

**--- (Context Updated: Doodad property decoding (rotation/scale/ID) is blocked. MPRR structure known, visualization enabled, hypothesis about MPRL->MSLK ID links needs manual verification due to tool failures. Code structure and syntax issues in PM4FileTests.cs have been fixed. Coordinate transformation constants have been updated to provide correct scaling and offset.) ---**
