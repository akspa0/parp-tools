# Progress for parpToolbox

## What Works
- **Project Scaffolding & Build Health:** The solution and project structure are stable and build correctly.
- **WMO Loading:** The tool can successfully load complex WMO files, including root and all external group files, using a `LocalFileProvider`.
- **OBJ/MTL Exporting:** A dedicated `ObjExporter` service can convert the loaded WMO geometry and materials into `.obj` and `.mtl` files.
- **Output Management:** The `ProjectOutput` utility correctly creates timestamped directories inside a `project_output` folder for all generated files, ensuring source data is never modified.
- **CLI Parsing:** A simple, manual command-line argument parser has been implemented.

## What's Left to Build
- **PM4/PD4 Support:** Interior geometry loaders, adapters, and exporters.
- **CLI Enhancements:** `pm4` and `pd4` commands with parity to `wmo` flags.
- **Test Suite:** Integration tests for PM4/PD4 and regression tests for WMO export.

## Current Status
- **WMO Export Complete.** Group naming is correct and facade planes filtered. Users can generate clean OBJs per group.
- **dotnet run Quirk.** Argument parsing works in compiled exe; nuance with `dotnet run --` still noted but low priority.
- **PM4/PD4 Phase Started.** Loader interface scaffolded; implementation to follow in next session.

## Known Issues
- **`dotnet run` Argument Parsing:** When using `dotnet run`, arguments passed after `--` are not being received by the application. The immediate priority is to diagnose and fix this issue to enable proper testing and use of the tool.
