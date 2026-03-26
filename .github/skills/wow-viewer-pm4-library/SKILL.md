---
name: wow-viewer-pm4-library
description: 'Use when implementing or reviewing a wow-viewer Core.PM4 slice such as pm4 inspect, pm4 audit, pm4 audit-directory, pm4 linkage, pm4 mscn, pm4 unknowns, pm4 export-json, PM4 regression tests, or a narrow shared-solver extraction from MdxViewer. Includes the current source-of-truth rule, validation commands, and update checklist.'
argument-hint: 'Describe the PM4 slice, analyzer, contract, or solver seam you want to change'
user-invocable: true
---

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
5. `.github/copilot-instructions.md`

## Current Source Of Truth

- `MdxViewer` is still the runtime PM4 reference implementation.
- `Pm4Research` is still the library seed for `WowViewer.Core.PM4`.
- `PM4Tool`, `parpToolbox`, and `WoWRollback.PM4Module` are supporting evidence, not the default source of truth.

## Procedure

1. Confirm the exact seam.
   Decide whether the task is a reader or report slice, a placement-contract or solver slice, a test slice, or a consumer-wiring slice.

2. Inspect the active library boundary first.
   Start in `wow-viewer/src/core/WowViewer.Core.PM4`, then check `wow-viewer/tests/WowViewer.Core.PM4.Tests`, `wow-viewer/tools/inspect/WowViewer.Tool.Inspect`, and only then inspect `MdxViewer` if the slice depends on runtime behavior.

3. Extract the smallest reusable slice.
   Prefer small, typed contracts and single-responsibility helpers over broad viewer rewrites.

4. Keep research seams explicit.
   If the semantics are still exploratory, keep them in `Research` or label them as research in contracts, reports, or notes.

5. Validate concretely.
   Add or update the smallest high-value test in `wow-viewer/tests/WowViewer.Core.PM4.Tests`. If the slice changes analyzer or report output, also run the relevant `pm4` inspect command on the fixed development dataset.

6. Update shared continuity files.
   Sync `wow-viewer/README.md`, the relevant memory-bank file, and `gillijimproject_refactor/plans/wow_viewer_pm4_library_plan_2026-03-25.md` when the active workflow, commands, or migration boundary changes.

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
- Do not flatten exploratory PM4 semantics into stable contracts without saying the evidence level changed.
- Do not replace a narrow extractable seam with a broad rewrite just because the old `WorldScene` code is messy.