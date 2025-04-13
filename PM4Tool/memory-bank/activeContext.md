# Mode: PLAN

# Active Context: Verify Geometry & Decode Doodads

**Goal:** Accurately parse PM4/PD4 files, **assemble the core render geometry (`MSVT`/`MSVI`/`MSUR`) into usable OBJ models**, understand the structure and **decode Doodad placements** from `MSLK` and related chunks (like `MDBH`), and provide file dumping tools.

**Current Focus:** **Doodad Placement Analysis & MPRR Structure/Index Target.** We've identified MSLK node entries as Doodad placements (anchored via `MSLK.Unk10` -> MSVI -> MSVT). Focus remains on decoding `MSLK` properties (`Unk00`, `Unk01`, `Unk04`, `Unk12`). For `MPRR`, the focus shifts from interpreting pairs to understanding the **variable-length sequence structure (terminated by 0xFFFF)** and determining **what the indices within these sequences point to** (likely *not* MPRL).

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
*   **MPRR Structure:** Confirmed to be variable-length sequences of `ushort` values, terminated by `0xFFFF`. The value before the terminator is a potential flag. Indices likely *do not* point to MPRL.
*   Unknown fields in `MSLK`, `MSUR`, `MPRL`, `MSHD` documented with current best guesses/hypotheses.
*   Analysis suggests Doodad orientation is likely Quaternion-based and scale is float-based (similar to WMO `MODD`), potentially encoded in `MSLK.Unk00`/`Unk01`/`Unk12`.
*   Link between `MSLK.Unk04` (Group ID) and `MDBH` (filenames) failed in tests for `development_22_18.pm4`, assumption reverted in code.
*   Test code (`PM4FileTests`) updated to output raw `Unk00`, `Unk01`, `Unk04`, `Unk12` values in `_pm4_mslk_nodes.obj` comments and dedicated `_mprr_data.csv` file created.
*   **Blocker:** Log file size/access issues prevent deeper automated analysis. Specific MSLK property encoding remains undecoded due to lack of documentation.
*   **MPRR Visualization:** Based on incorrect assumption of paired MPRL indices; needs revisiting based on new structure.
*   **MPRL->MSLK ID Hypothesis:** Deprioritized/Paused due to conflicting data and structural changes.
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
    *   **NEW:** Analyze `_mprr_analysis.csv` to understand the flag values and index ranges within the MPRR sequences.
    *   **NEW:** Research what the MPRR indices might point to (if not MPRL).
2.  **(If Information Found)** Implement decoding logic for rotation/scale/ID in `MSLKEntry.cs` and/or MPRL->MSLK linking logic, and update test output.
3.  **(If Blocked on Doodads)** Consider refactoring `MPRRChunk.cs` in the core library to match the sequence structure, even if index target is unknown.
4.  **Refine Analysis Tool:** Add MPRR Index vs MPRL Count check to the analysis tool.
5.  **Additional Testing:** Continue testing coordinate transformations.

**--- (Context Updated: Doodad property decoding blocked. MPRR structure confirmed as 0xFFFF-terminated sequences; indices likely *not* MPRL. MPRL->MSLK ID hypothesis paused. `MSLK.Unk10` confirmed as node anchor index (not constant 0xFFFF). Several MSLK/MPRL fields confirmed constant.) ---**
