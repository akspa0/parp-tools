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

### Phase 3: PM4 / PD4 Integration (In Progress)
- [ ] **Debug CLI:** Resolve issue where command-line arguments are not being correctly passed or parsed when using `dotnet run`.
- [x] Scaffold `Formats/PM4` and `Formats/PD4` modules inside `parpToolbox`.
- [x] Implement `WowToolsLocalWmoLoader` exposing high-level `WmoGroup` objects for clean geometry access.
- [ ] Port essential models and readers from legacy `WoWToolbox.Core.v2` (read-only) into new namespaces.
- [ ] Implement `Pm4Adapter` / `Pd4Adapter` that leverage `wow.tools.local` for WMO geometry and low-level IO.
- [ ] Extend CLI parser with `--pm4` / `--pd4` commands and route outputs via `ProjectOutput`.
- [ ] Create integration tests loading real PM4/PD4 data under `test_data/`, verifying OBJ export counts.

---

## Current Status
WMO → OBJ export has been fully validated: façade planes are correctly filtered by default, and group/file naming matches in-game names. The command-line pipeline works reliably when arguments are passed via an executable build; the `dotnet run --` quirk is still under investigation.

Focus now shifts to Phase 3 (PM4 / PD4 interior geometry support).  `IPm4Loader` has been scaffolded and the next tasks are:
1. Port Core.v2 PM4/PD4 readers into clean adapters.
2. Extend CLI with `pm4` / `pd4` commands mirroring existing flags.
3. Re-use `ObjExporter` for quick geometry validation.
4. Maintain the same clean output discipline under `project_output/`. 
