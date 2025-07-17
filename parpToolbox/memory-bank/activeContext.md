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
- [ ] **Debug CLI:** Resolve issue where arguments are not received when using `dotnet run`.
- [x] Scaffold `Formats/PM4` and `Formats/PD4` modules inside `parpToolbox`.
- [x] Implement `WowToolsLocalWmoLoader` exposing high-level `WmoGroup` objects.
- [ ] Port essential models and readers from legacy `WoWToolbox.Core.v2` (read-only) into new namespaces.
- [x] Move shared PM4/PD4 chunks to `Formats/P4/Chunks/Common`.
- [x] Update namespaces of moved chunks to `ParpToolbox.Formats.P4.Chunks.Common`.
- [x] Adapt `Pm4Adapter` to new namespace and scaffold `Pd4Adapter`.
- [ ] Implement full `Pm4Adapter` / `Pd4Adapter` behaviour (PD4 chunk audit, geometry export).
- [x] Extend CLI parser with `pm4` / `pd4` commands and route outputs via `ProjectOutput`.
- [x] Update OBJ exporter to output vertices only (omit faces) for initial validation.
- [ ] Port legacy `Pm4BatchTool` research utilities into new `Pm4ResearchTool` project.
- [ ] Create integration tests loading real PM4/PD4 data under `test_data/`, verifying OBJ vertex counts.

---

## Current Status

### (2025-07-14 22:51)
- PM4 export produced **825** groups (expected ~10–20). Grouping algorithm still incorrect.
- MSUR chunk loader rewritten to 32-byte authoritative spec; alignment confirmed.
- Hypothesis: grouping must mimic `MsurObjectExporter` logic – surface ranges matched by `ReferenceIndex` clusters.
- Next session: port reference grouping routine and verify output counts.

### (2025-07-14 21:16)
- Initial attempt to map MSUR → MSLK via low-word `LinkIdRaw` succeeded for a subset of surfaces but left many unmapped.
- Added containment fallback, but OBJ exporter now writes invalid geometry (Meshlab crashes) and still outputs just one OBJ file.
- Hypothesis: 32-bit `Unknown_0x04` group key is correct, but our surface-to-group resolution and/or vertex-remap in `Pm4GroupObjExporter` is corrupting face indices.
- Next session: audit face-index remapping, verify OBJ validity, ensure one OBJ per `SurfaceGroup`.

WMO → OBJ export has been fully validated: façade planes are correctly filtered by default, and group/file naming matches in-game names. The command-line pipeline works reliably when arguments are passed via an executable build; the `dotnet run --` quirk is still under investigation.

Focus now shifts to Phase 3 (PM4 / PD4 interior geometry support).  `IPm4Loader` has been scaffolded and the next tasks are:
1. Port Core.v2 PM4/PD4 readers into clean adapters.
2. Extend CLI with `pm4` / `pd4` commands mirroring existing flags.
3. Re-use `ObjExporter` for quick geometry validation.
4. Maintain the same clean output discipline under `project_output/`. 
