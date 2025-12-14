# Active Context

## Current Focus: PM4 Object Injection & MSCN Analysis (Dec 13, 2025)

### Status Summary
We have successfully implemented the **WMO Rotation MVP** and identified the next critical data source (**MSCN**).

1.  **WMO Rotation (Applied)**:
    - **Analysis**: `analyze-pm4` extracts candidates, matches against WMOs, and calculates rotation (Pitch/Yaw/Roll).
    - **Conversion**: `convert-matches-to-modf` handles coordinate transforms (Placement -> World) and sanitization (no `-0`).
    - **Injection**: `inject-modf` applies patching.
    - **Blocker**: Local validation is blocked because `matches.csv` (Tile 37) and `test_data` (Tiles 56-63) do not overlap.

2.  **MSCN Analysis (Next)**:
    - **Goal**: Decode the unknown `MSCN` chunk (~16MB).
    - **Hypothesis**: Contains detailed object scene nodes, likely **M2 placements** and precise WMO boundaries.
    - **Importance**: Critical for solving "Link Objects" (CK24=0) and achieving 100% match rate.

### Recent Changes
- **Refactored**: `WmoPathfindingExtractor` to handle sub-objects (Groups).
- **Implemented**: `RunConvertMatchesToModf` command to bridge analysis and patching.
- **Fixed**: Rotation sanitization (removed `-0` values).
- **Identified**: Coordinate system mismatch between PM4 (Placement) and ADT (World).
- **Updated**: `PM4-Format-Specification.md` with new findings.

### Next Steps
1.  **Research MSCN**: Dump and visualize the `MSCN` chunk structure.
2.  **Decode M2s**: Extract M2 placement data from MSCN.
3.  **Integrate**: Use MSCN data to refine WMO matching and place M2 objects.
4.  **Fallback (Procedural)**: Generate "fake" WMOs from CK24 OBJ data for visual verification in Noggit if matching fails.
