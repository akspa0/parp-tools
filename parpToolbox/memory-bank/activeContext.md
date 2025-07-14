# Active Context: Building the new parpToolbox Project

## Guiding Principles
- **Start Fresh:** All new development occurs in the `parpToolbox` project. The legacy `WoWToolbox` project is for reference only.
- **Dependency, Not Fork:** `wow.tools.local` is a strict library dependency; we will never modify its code directly.
- **Real Data Testing:** All tests must use real game data to ensure accuracy.
- **Clean Output:** All generated files must be written to the timestamped `project_output` directory.

## Phases & Tasks

### Phase 1: Project Foundation (Complete)
- [x] Create a new solution `parpToolbox.sln` in the root directory.
- [x] Create a new .NET 9.0 console project `parpToolbox` in the `src` folder.
- [x] Add the existing `wow.tools.local` project to the solution.
- [x] Add a project reference from `parpToolbox` to `wow.tools.local`.

### Phase 2: WMO & OBJ Export (Complete)
- [x] Implement manual command-line argument parsing.
- [x] Create `LocalFileProvider` to enable file system access for `wow.tools.local`.
- [x] Fix WMO group file loading in `WMOReader` to prevent infinite recursion.
- [x] Implement `ProjectOutput` utility for clean, timestamped directory creation.
- [x] Implement `ObjExporter` service to convert WMO models to OBJ and MTL files.
- [x] Integrate all components into `Program.cs` to create a complete export pipeline.

### Phase 3: Core Implementation (In Progress)
- [ ] **Debug CLI:** Resolve issue where command-line arguments are not being correctly passed or parsed when using `dotnet run`.
- [ ] Implement core logic for PM4 and PD4 file handling.
- [ ] Set up an initial test suite for the new project.

---

## Current Status
The full pipeline for loading WMO files and exporting them to OBJ format is implemented. However, a persistent issue is preventing the command-line arguments from being correctly processed when the application is launched with `dotnet run`. Diagnostic code has been added to `Program.cs` to log the received arguments at startup. The immediate next step is to analyze this output and resolve the argument parsing issue.
