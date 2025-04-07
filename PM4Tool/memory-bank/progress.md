# Progress: PM4 File Parser

## What Works
*   **Core Chunk Loading:** Successfully loads data from various PM4 chunks including `MVER`, `MSLK`, `MSPI`, `MSPV`, `MSVT`, `MSVI`, `MSUR`, `MPRL`, and `MPRR`.
*   **MPRL Data Reading:** Correctly reads `MPRL` chunk data using `C3Vectori`.
*   **Build System:** Project now builds successfully after resolving complex issues involving duplicate files, case sensitivity, and `.csproj` configuration.
*   **Test Suite:** The primary test (`LoadPM4File_ShouldLoadChunks`) runs and passes. Test console output is still not reliably captured in redirected logs.
*   **OBJ File Generation:** The test successfully generates an `.obj` file (`output_development_00_00.obj`) containing vertices from `MSPV`, `MSVT`, and `MPRL`, and faces from `MSVI`. MSVT vertices use reverted `C3Vectori`-based loading. OBJ file size is large (~1GB), suggesting face data is being written.
*   **MPRR Index Validation:** The validation check `MPRR?.ValidateIndices` passes in the test run.

## What's Left
*   **Visual Verification:** The `.obj` file loads but contains errors ("Identical vertex indices found") due to degenerate faces and lacks correct 3D geometry (Z-axis issues). Needs fixing and visual inspection.
*   **Degenerate Face Handling:** Implement check in `.obj` export to skip faces with duplicate vertex indices.
*   **Coordinate Transformation Refinement:** Re-evaluate `MSVT` and `MPRL` coordinate transformations (Y/X swap, offset, Z scaling) after fixing degenerate faces to achieve correct 3D visualization. Revisit `MSVT` data type (`C3Vectori` vs `float`).
*   **Dependency Vulnerability:** Address the known high-severity vulnerability in `SixLabors.ImageSharp` v2.1.9.
*   **Test Logging:** Improve debugging workflow by ensuring `Console.WriteLine` output from the core library is reliably captured during test runs.
*   **MPRR Chunk Investigation:** Although index validation passes, the purpose and structure of `MPRR` data need further investigation if relevant geometry is still missing.

## Current Status
Development focus shifted temporarily to resolving major build system failures. With builds now working, focus returns to correcting the `.obj` export. The immediate issue is degenerate faces causing errors in viewers. After fixing that, the primary goal is achieving correct 3D visualization by refining coordinate transformations, potentially revisiting the `MSVT` data type interpretation. The project is currently blocked, pending the fix for degenerate faces.

## Known Issues
*   **OBJ Degenerate Faces:** `.obj` export writes faces with duplicate indices (e.g., `f 10 11 10`), causing errors in viewers.
*   **OBJ Visualization:** Exported `.obj` file does not display correct 3D geometry (Z-axis appears flat or incorrect).
*   **Vulnerability:** `SixLabors.ImageSharp` v2.1.9 has GHSA-2cmq-823j-5qj8.
*   **Test Logging:** `Console.WriteLine` calls within the `WoWToolbox.Core` library are not consistently captured in redirected output (`> file.log`) when running `dotnet test`.
*   **MPRR Index Usage:** Purpose/usage of `MPRR` chunk data remains unclear, even if index validation passes.

## Milestones
1.  Project Setup ✓
2.  Core Framework ✓
3.  PM4 Basic Implementation ✓
4.  PM4 Validation & Testing ✓
5.  PM4 Extended Geometry Implementation ✓ (Partial: `MSVT`, `MSVI`, `MSUR`, `MPRL`, `MPRR` loaded)
6.  **.obj Export Basic Implementation** ✓ (Vertices and faces written, but with errors)
7.  **.obj Export Visualization Correction** 🔲 (Fix degenerate faces, Fix Z-axis)
8.  Legacy Support 🔲
9.  Quality Assurance 🔲 