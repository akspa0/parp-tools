# wow-viewer PM4 Library

## When To Use

- The task is inside `wow-viewer/src/core/WowViewer.Core.PM4`.
- You are porting a PM4 reader, analyzer, report, or placement-math seam.
- You are adding or updating `WowViewer.Tool.Inspect` PM4 verbs such as `pm4 inspect`, `pm4 audit`, `pm4 audit-directory`, `pm4 linkage`, `pm4 mscn`, `pm4 unknowns`, or `pm4 export-json`.
- You are extracting a narrow PM4 helper from `gillijimproject_refactor/src/MdxViewer/Terrain/WorldScene.cs`.
- You need to add or adjust PM4 regression coverage in `wow-viewer/tests/WowViewer.Core.PM4.Tests`.

## Read First

1. `gillijimproject_refactor/memory-bank/activeContext.md`
2. `gillijimproject_refactor/memory-bank/progress.md`
3. `gillijimproject_refactor/plans/wow_viewer_pm4_library_plan_2026-03-25.md`
4. `wow-viewer/README.md`
5. `AGENTS.md`

## Current Source Of Truth

- `WowViewer.Core.PM4` is the canonical implementation target for new PM4 work.
- `Pm4Research` is still the library seed and a major extraction input for `WowViewer.Core.PM4`.
- `MdxViewer`, `PM4Tool`, `parpToolbox`, and `WoWRollback.PM4Module` are reference or extraction inputs, not the default source of truth.

## Procedure

1. Confirm the exact seam.
2. Inspect the active library boundary first.
3. Extract the smallest reusable slice.
4. Keep research seams explicit.
5. Validate concretely.
6. Update shared continuity files.

## High-Value Files

- `wow-viewer/src/core/WowViewer.Core.PM4/Models/Pm4PlacementContracts.cs`
- `wow-viewer/src/core/WowViewer.Core.PM4/Services/Pm4PlacementMath.cs`
- `wow-viewer/src/core/WowViewer.Core.PM4/Research/*`
- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs`
- `wow-viewer/tests/WowViewer.Core.PM4.Tests/Pm4ResearchIntegrationTests.cs`
- `gillijimproject_refactor/src/MdxViewer/Terrain/WorldScene.cs`

## Validation Commands

- Build: `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`
- Tests: `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`
- Placement-focused tests: `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug --filter PlacementMath`
- Inspect corpus: `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- pm4 <verb> --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development`

## Guardrails

- Do not claim viewer runtime PM4 signoff from `wow-viewer` builds, tests, or active-viewer compile success.
- Do not broaden active-viewer integration when the task is library-first unless the user explicitly asks for it.
- Do not treat `MdxViewer` behavior as authoritative just because it exists; re-home or replace the behavior in `Core.PM4` with explicit validation.
- Do not flatten exploratory PM4 semantics into stable contracts without saying the evidence level changed.
