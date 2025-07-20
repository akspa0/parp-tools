# Progress for parpToolbox

## What Works
- **Project Scaffolding & Build Health:** The solution and project structure are stable and build correctly.
- **WMO Loading:** The tool loads complex WMO files (root + groups) via `LocalFileProvider`.
- **OBJ Exporting:** WMO → OBJ/MTL verified. PM4/PD4 exporters now reliably emit point-cloud OBJ (vertices only) to avoid viewer crashes.
- **Output Management:** `ProjectOutput` creates timestamped sub-directories under `project_output` for all generated assets.
- **CLI Parsing:** Manual argument parser covers `wmo`, `pm4`, and `pd4` commands.
- **PM4 Chunk Relationship Analysis:** Complete understanding of MPRL ↔ MSLK ↔ MSUR relationships with CSV validation.
- **PM4 Object Assembly:** Working exporters with SurfaceGroupKey-based grouping:
  - `Pm4SceneExporter`: Complete building interior as unified OBJ
  - `Pm4MsurObjectAssembler`: Objects grouped by MSUR SurfaceGroupKey with MPRL transformations
  - Enhanced vertex validation to prevent (0,0,0) artifacts from invalid indices
- **Coordinate System:** X-axis inversion fix applied for proper geometry orientation.
- **Data Loss Detection:** Comprehensive analysis tools to identify missing vertex references and tile boundary issues.

## What's Left to Build
- **CRITICAL: Global Tile Loading System:** Implement unified PM4/PD4 loader that processes entire 64x64 tile regions as single mesh to resolve massive data loss (110,988+ missing vertex references)
- **Index Pattern Investigation:** Complete analysis of high/low pair encodings in unknown fields to discover proper 32-bit index decoding
- **Vertex Reference Mapping:** Implement cross-tile vertex index mapping and boundary handling
- **PM4 Validation:** Test global tile loading with real multi-tile regions
- **PD4 Support:** Port PM4 insights and global loading to PD4 format
- **Output Path Standardization:** Ensure all exports go to unified `project_output` location
- **Legacy Comparison:** Compare new global mesh exports with legacy Core.v2 outputs
- **Test Suite:** Integration tests for PM4/PD4 and regression tests for WMO export

## Current Status
- **WMO Export Complete.** Group naming is correct and facade planes filtered. Users can generate clean OBJs per group.
- **CRITICAL DATA LOSS DISCOVERED.** 110,988 out-of-bounds vertex accesses (~64% data loss) due to cross-tile vertex references.
- **PM4 Global Tile System Confirmed.** Individual PM4 files reference vertices from adjacent tiles, requiring unified region loading.
- **Root Cause Analysis Complete.** (0,0,0) vertices are invalid data artifacts, not mysterious anchors. High/low pair encoding likely used for 32-bit indices.
- **Enhanced Object Assembly.** SurfaceGroupKey-based grouping with MPRL transformations and improved vertex validation implemented.
- **Investigation Tools Ready.** Pm4IndexPatternAnalyzer created to systematically analyze missing data patterns and high/low pair encodings.
- **Next Priority: Global Tile Loading.** Must implement unified region loader to access missing ~63,000 vertices for complete object reconstruction.

## Recent Updates (2025-07-19)
### SurfaceGroupKey Hierarchy & Grouping Tester (22:20)
- Implemented `pm4-test-grouping` command which groups and exports geometry by `MSUR.SurfaceGroupKey`.
- Visual inspection confirms:
  - Group 19 ≈ WMO/group-level objects.
  - Groups 20–23 are sub-objects; 24 is near per-surface granularity.
  - Values <19 appear to be larger spatial containers.
- Conclusion: `SurfaceGroupKey` represents a hierarchy, not a flat object ID.
- Action: continue exploring composite keys `(ParentIndex, SurfaceGroupKey, IndexCount)` and validate face alignment.

### Grouping still incorrect (20:53)
- MSUR.IndexCount grouping yields 15 objects but faces still mis-assigned; geometry remains scattered.
- OBJ warnings show ~480 k references to missing vertices > global vertex pool.
- Hypothesis: true object key combines multiple unknown fields (e.g., high/low bytes of `surface_key`, `reference_index`, padding).
- Action: expand UnknownFieldAnalyzer to dump **all** raw fields and auto-correlate; prototype `pm4-test-grouping` to iterate key expressions.

### MSCN Cross-Tile Remap Started
- Implemented `MscnRemapper` to append MSCN vertices and rewrite indices.
- Placeholder per-tile logic in place—needs full region loader integration.
- Next: create `Pm4RegionLoader` to merge 64×64 tiles and call remapper.
- **BREAKTHROUGH: Massive Data Loss Root Cause Identified**
  - Discovered 110,988 out-of-bounds vertex accesses (~64% data loss) in PM4 exports
  - Confirmed PM4 files are part of global tile system with cross-tile vertex references
  - Available: 63,298 vertices (0-63297), Accessed: up to 126,595 → Missing ~63,000 vertices
- **Critical Insights Gained**
  - (0,0,0) vertices are invalid data artifacts from out-of-bounds access, not mysterious anchors
  - High/low pair encoding pattern identified in unknown fields (likely 32-bit indices as two 16-bit values)
  - Sequential out-of-bounds patterns suggest adjacent tile vertex pools
- **Enhanced PM4 Object Assembly**
  - Refactored to use MSUR.SurfaceGroupKey instead of IndexCount for object grouping
  - Implemented MPRL transformation logic with position offset application
  - Added comprehensive vertex validation to prevent invalid triangles and (0,0,0) artifacts
  - Created Pm4IndexPatternAnalyzer for systematic investigation of missing data patterns
- **Tools and Infrastructure**
  - Enhanced Pm4MsurObjectAssembler with transform-aware vertex extraction
  - Added detailed logging and debugging for vertex index validation
  - Created automated chunk combination testing system (Pm4ChunkCombinationTester)
  - Implemented comprehensive CSV export for chunk cross-reference analysis

## Recent Updates (2025-07-14)
- Shared P4 chunk reader set created; identical PM4/PD4 chunks moved to `Formats/P4/Chunks/Common` with namespace `ParpToolbox.Formats.P4.Chunks.Common`.
- `Pm4Adapter` updated and `Pd4Adapter` scaffolded to use shared chunks.
- Implemented `FourCc` helper and refactored `MSPI` reader to correctly detect 16- vs 32-bit indices, preventing invalid faces.
- OBJ exporter changed to output vertices only (no faces) for initial PD4 validation; Meshlab now opens files without crashing.
- CLI enhanced with `pm4` / `pd4` commands; build passes after refactor.
- Memory bank `activeContext` tasks updated; port of legacy `Pm4BatchTool` planned for next session.

## Recent Updates (2025-07-14 16:30)
- Added bounds checks in `Pm4Adapter` when building faces to prevent invalid index ranges.
- Implemented defensive vertex index validation in `Pm4GroupObjExporter` (skip out-of-range indices, remap checks).
- CLI `--exportchunks` now functional; tool exports ~2.4k MSUR groups without crash (pending validation).

## Recent Updates (2025-07-14 22:51)
- Rewrote MSUR loader to 32-byte spec, fixing structure misalignment.
- Ran PM4 export; received **825** OBJ groups instead of expected 10-20.
- Conclusion: grouping logic still wrong; must port `MsurObjectExporter` grouping routine.
- Next step: replicate reference grouping by surface ranges matching MSLK `ReferenceIndex`, validate counts with real data.

## Known Issues
- **`dotnet run` Argument Parsing:** When using `dotnet run`, arguments passed after `--` are not being received by the application. The immediate priority is to diagnose and fix this issue to enable proper testing and use of the tool.
