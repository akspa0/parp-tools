# wow-viewer Shared I/O Library

## When To Use

- The task is inside `wow-viewer/src/core/WowViewer.Core` or `wow-viewer/src/core/WowViewer.Core.IO`.
- You are adding or updating shared file detection, chunk reading, top-level format summaries, or non-PM4 format contracts.
- The next slice is specifically about ADT root files, split ADT companions such as `_tex0.adt`, `_obj0.adt`, or `_lod.adt`, WDT summaries, WMO top-level reads, or early BLP or DBC or DB2 classification seams.
- You are adding or updating `WowViewer.Tool.Inspect` non-PM4 verbs such as `map inspect`.
- You are adding or updating `WowViewer.Tool.Converter` non-PM4 commands that should sit on shared library seams instead of tool-local parsing.
- You need to validate a shared-format slice against archive-backed client roots or decide how WoWArchive-mounted data should be staged before broader scans.

## Read First

1. `gillijimproject_refactor/memory-bank/activeContext.md`
2. `gillijimproject_refactor/memory-bank/progress.md`
3. `gillijimproject_refactor/memory-bank/data-paths.md`
4. `gillijimproject_refactor/plans/wow_viewer_shared_io_library_plan_2026-03-26.md`
5. `wow-viewer/README.md`
6. `AGENTS.md`

## Current Source Of Truth

- `wow-viewer/src/core/WowViewer.Core` and `wow-viewer/src/core/WowViewer.Core.IO` are now the intended home for shared non-PM4 format ownership.
- Current `gillijimproject_refactor` code remains the runtime and compatibility reference where behavior is still being ported.
- `WowViewer.Tool.Inspect` and `WowViewer.Tool.Converter` should be thin consumers of shared detection or reader seams, not owners of duplicate heuristics.

## Procedure

1. Confirm the exact seam.
2. Inspect the active shared boundary first.
3. Extract the smallest reusable slice.
4. Keep evidence levels explicit.
5. Stage archive-backed clients before wide real-data scans.
6. Validate concretely.
7. Update shared continuity files.

## High-Value Files

- `wow-viewer/src/core/WowViewer.Core/Chunks/*`
- `wow-viewer/src/core/WowViewer.Core/Files/*`
- `wow-viewer/src/core/WowViewer.Core/Maps/*`
- `wow-viewer/src/core/WowViewer.Core.IO/Chunked/*`
- `wow-viewer/src/core/WowViewer.Core.IO/Files/*`
- `wow-viewer/src/core/WowViewer.Core.IO/Maps/*`
- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs`
- `wow-viewer/tools/converter/WowViewer.Tool.Converter/Program.cs`

## Validation Commands

- Build: `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`
- Tests: `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`
- Inspect map summary: `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- map inspect --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development.wdt`
- Converter detect: `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -- detect --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development_0_0_tex0.adt`

## Guardrails

- Do not duplicate file-family heuristics in inspect or converter once a shared seam exists in `Core` or `Core.IO`.
- Do not describe classification or top-level summary work as full format parsing or writing.
- Do not route non-PM4 shared-format work back into PM4 prompts or PM4 plans.
- Keep readable FourCC handling in memory and reverse only at I/O boundaries.
- Do not treat a mounted WoWArchive path as the preferred working root for heavy validation when a staged local copy is practical.
