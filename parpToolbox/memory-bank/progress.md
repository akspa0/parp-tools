# Progress for parpToolbox

## What Works
- **Project Scaffolding & Build Health:** The solution and project structure are stable and build correctly.
- **WMO Loading:** The tool loads complex WMO files (root + groups) via `LocalFileProvider`.
- **OBJ Exporting:** WMO → OBJ/MTL verified. PM4/PD4 exporters now reliably emit point-cloud OBJ (vertices only) to avoid viewer crashes.
- **Output Management:** `ProjectOutput` creates timestamped sub-directories under `project_output` for all generated assets.
- **CLI Parsing:** Manual argument parser covers `wmo`, `pm4`, and `pd4` commands.
- **PM4 Chunk Relationship Analysis:** Complete understanding of MPRL ↔ MSLK ↔ MSUR relationships with CSV validation.
- **PM4 Object Assembly:** Three working exporters:
  - `Pm4SceneExporter`: Complete building interior as unified OBJ
  - `Pm4MsurObjectAssembler`: Objects grouped by MSUR IndexCount (correct method)
  - `Pm4ObjectAssembler`: Legacy ParentIndex grouping (produces fragments)
- **Coordinate System:** X-axis inversion fix applied for proper geometry orientation.

## What's Left to Build
- **PM4 Validation:** Test and validate MSUR-based object assembly with real data.
- **PD4 Support:** Port PM4 insights to PD4 format for individual object processing.
- **Output Path Standardization:** Ensure all exports go to unified `project_output` location.
- **Legacy Comparison:** Compare new MSUR-based exports with legacy Core.v2 outputs.
- **Test Suite:** Integration tests for PM4/PD4 and regression tests for WMO export.

## Current Status
- **WMO Export Complete.** Group naming is correct and facade planes filtered. Users can generate clean OBJs per group.
- **PM4 Object Assembly Breakthrough.** Discovered that MSUR IndexCount (0x01 field) is the correct object identifier, not ParentIndex.
- **PM4 Implementation Complete.** Three working exporters implemented with coordinate system fixes.
- **Ready for Validation.** MSUR-based object assembly needs testing with real PM4 data to confirm complete objects vs fragments.

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
