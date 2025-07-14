# Progress for parpToolbox

## What Works
- **Project Scaffolding & Build Health:** The solution and project structure are stable and build correctly.
- **WMO Loading:** The tool can successfully load complex WMO files, including root and all external group files, using a `LocalFileProvider`.
- **OBJ/MTL Exporting:** A dedicated `ObjExporter` service can convert the loaded WMO geometry and materials into `.obj` and `.mtl` files.
- **Output Management:** The `ProjectOutput` utility correctly creates timestamped directories inside a `project_output` folder for all generated files, ensuring source data is never modified.
- **CLI Parsing:** A simple, manual command-line argument parser has been implemented.

## What's Left to Build
- **PM4/PD4 Support:** Core logic for processing these formats needs to be implemented.
- **Test Suite:** A comprehensive test suite using real data needs to be built to validate all functionality.

## Current Status
- **Debugging CLI.** The WMO-to-OBJ export pipeline is functionally complete. However, we are currently blocked by an issue where command-line arguments are not being correctly interpreted by the application when launched via `dotnet run`. 

## Known Issues
- **`dotnet run` Argument Parsing:** When using `dotnet run`, arguments passed after `--` are not being received by the application. The immediate priority is to diagnose and fix this issue to enable proper testing and use of the tool.
