# Progress for parpToolbox

## What Works
- **Project Scaffolding & Build Health:** The solution and project structure are stable and build correctly.
- **WMO Loading:** The tool loads complex WMO files (root + groups) via `LocalFileProvider`.
- **OBJ Exporting:** WMO → OBJ/MTL verified. PM4/PD4 exporters now reliably emit point-cloud OBJ (vertices only) to avoid viewer crashes.
- **Output Management:** `ProjectOutput` creates timestamped sub-directories under `project_output` for all generated assets.
- **CLI Parsing:** Manual argument parser covers `wmo`, `pm4`, and `pd4` commands.

## What's Left to Build
- **PM4/PD4 Support:** Interior geometry loaders, adapters, and exporters.
- **CLI Enhancements:** `pm4` and `pd4` commands with parity to `wmo` flags.
- **Test Suite:** Integration tests for PM4/PD4 and regression tests for WMO export.

## Current Status
- **WMO Export Complete.** Group naming is correct and facade planes filtered. Users can generate clean OBJs per group.
- **dotnet run Quirk.** Argument parsing works in compiled exe; nuance with `dotnet run --` still noted but low priority.
- **PM4/PD4 Phase Started.** Loader interface scaffolded; implementation to follow in next session.

## Recent Updates (2025-07-14)
- Shared P4 chunk reader set created; identical PM4/PD4 chunks moved to `Formats/P4/Chunks/Common` with namespace `ParpToolbox.Formats.P4.Chunks.Common`.
- `Pm4Adapter` updated and `Pd4Adapter` scaffolded to use shared chunks.
- Implemented `FourCc` helper and refactored `MSPI` reader to correctly detect 16- vs 32-bit indices, preventing invalid faces.
- OBJ exporter changed to output vertices only (no faces) for initial PD4 validation; Meshlab now opens files without crashing.
- CLI enhanced with `pm4` / `pd4` commands; build passes after refactor.
- Memory bank `activeContext` tasks updated; port of legacy `Pm4BatchTool` planned for next session.

## Known Issues
- **`dotnet run` Argument Parsing:** When using `dotnet run`, arguments passed after `--` are not being received by the application. The immediate priority is to diagnose and fix this issue to enable proper testing and use of the tool.
