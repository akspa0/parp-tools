# Active Context: OBJ Export Debugging (Degenerate Faces & Z-Axis)

**Goal:** Correctly parse PM4 geometry chunks and export a valid `.obj` file that accurately represents the 3D geometry for visualization.

**Current Focus:** 

1.  **Review Core Files:** I'll mentally review `projectbrief.md`, `productContext.md`, `systemPatterns.md`, and `techContext.md`. Based on our debugging journey, these likely remain accurate regarding the project's overall goals and technical foundation. No updates anticipated here.
2.  **Update `activeContext.md`:**
    *   **Goal:** Remains: Correctly parse PM4 geometry chunks and export a valid `.obj` file accurately representing the 3D geometry.
    *   **Current Focus:** **Debugging index calculation errors** in line (`l`) generation (from `MSLK`/`MSPI`/`MSPV` and `MPRR`/`MPRL`) as the most likely cause of the persistent "sky spike" visual artifact. Vertex transformation issues are now considered secondary suspects for the spike itself.
    *   **Transformation Attempts Summary:**
        *   Unified MSVT Transform (Offset-Y, Offset-X, Z/36 for `MSVT`, `MSPV`, `MPRL`): Builds, runs, spike persists.
        *   MSVT Transform Only (Raw X,Y,Z for `MSPV`, `MPRL`): Builds, runs, spike persists, coordinate scales diverge wildly.
        *   MSVT Transform + Z-Scaling (Raw X,Y + Z/36 for `MSPV`, `MPRL`): Builds, runs, spike persists, coordinate scales diverge significantly.
    *   **Current Hypothesis:** An incorrect 1-based index is being calculated and used in an `l` statement, referencing either an incorrect vertex pool (e.g., using an MSPV-based index on MSVT vertices) or an index outside the bounds of the intended vertex pool.
    *   **Next Steps (Debugging Plan):**
        *   Temporarily revert `MSPV`/`MPRL` vertex export back to the full MSVT transform (Offset-Y, Offset-X, Z/36) solely to maintain a consistent coordinate space for easier index debugging.
        *   Implement detailed debug logging within the `MSLK`/`MSPI` and `MPRR` loops in `PM4FileTests.cs`:
            *   Log the raw indices read from the file (`mspiValue_MspvIndex`, `mprrEntry.Unknown_0x00`, `mprrEntry.Unknown_0x02`).
            *   Log the calculated final 1-based OBJ vertex indices being written to `l` statements.
            *   Log the world coordinates of the start and end vertices for each generated `l` statement.
        *   Run tests and meticulously analyze the `.debug.log` output for index calculation errors (e.g., negative indices, indices exceeding vertex counts) or lines with extreme coordinate values/lengths.
3.  **Update `progress.md`:**
    *   **What works:** PM4 chunk loading framework, test setup, basic OBJ vertex/normal export. Code builds successfully with various transformation strategies applied. Optional `MSRN` chunk handling. Degenerate face skipping.
    *   **What's left:** **Resolving the "sky spike" artifact** by fixing the underlying cause (suspected indexing error). Determining the definitively correct coordinate transformations/scaling for `MSPV` and `MPRL` vertices relative to `MSVT`. Verifying face (`f`) and line (`l`) index calculations are robust.
    *   **Current status:** Ready to implement the debugging plan detailed in `activeContext.md`: temporarily revert vertex transforms, add extensive index/coordinate logging to line generation loops, and prepare for detailed log analysis.
    *   **Known issues:** Persistent "sky spike" artifact, likely due to faulty index calculation in line generation. Uncertainty about the correct coordinate transformation for non-MSVT vertices (`MSPV`, `MPRL`). Potential for off-by-one errors or incorrect pool referencing in complex index calculations involving `base*VertexIndex` offsets. Potential mismatch between `MSVT` vertex count and `MSCN` normal count impacting `f v//vn` faces (lower priority until spike is fixed).

