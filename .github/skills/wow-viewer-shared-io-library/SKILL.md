---
name: wow-viewer-shared-io-library
description: 'Use when implementing or reviewing a wow-viewer Core or Core.IO shared format slice such as ADT root, _tex0.adt, _obj0.adt, _lod.adt, WDT, WMO, BLP, DBC, DB2, file detection, chunk readers, map inspect, converter detect, or shared-format regression tests. Includes the current source-of-truth rule, validation commands, and update checklist.'
argument-hint: 'Describe the shared I/O slice, file family, detector, reader, or tool consumer you want to change'
user-invocable: true
---

# wow-viewer Shared I/O Library

## When To Use

- The task is inside `wow-viewer/src/core/WowViewer.Core` or `wow-viewer/src/core/WowViewer.Core.IO`.
- You are adding or updating shared file detection, chunk reading, top-level format summaries, or non-PM4 format contracts.
- The next slice is specifically about ADT root files, split ADT companions such as `_tex0.adt`, `_obj0.adt`, or `_lod.adt`, WDT summaries, WMO top-level reads, or early BLP or DBC/DB2 classification seams.
- You are adding or updating `WowViewer.Tool.Inspect` non-PM4 verbs such as `map inspect`.
- You are adding or updating `WowViewer.Tool.Converter` non-PM4 commands that should sit on shared library seams instead of tool-local parsing.
- You need to add or adjust regression coverage in `wow-viewer/tests/WowViewer.Core.Tests`.

## Read First

1. `gillijimproject_refactor/memory-bank/activeContext.md`
2. `gillijimproject_refactor/memory-bank/progress.md`
3. `gillijimproject_refactor/plans/wow_viewer_shared_io_library_plan_2026-03-26.md`
4. `wow-viewer/README.md`
5. `.github/copilot-instructions.md`

## Current Source Of Truth

- `wow-viewer/src/core/WowViewer.Core` and `wow-viewer/src/core/WowViewer.Core.IO` are now the intended home for shared non-PM4 format ownership.
- Current `gillijimproject_refactor` code remains the runtime and compatibility reference where behavior is still being ported.
- `WowViewer.Tool.Inspect` and `WowViewer.Tool.Converter` should be thin consumers of shared detection or reader seams, not owners of duplicate heuristics.

## Procedure

1. Confirm the exact seam.
   Decide whether the task is a detector slice, a chunk-reader slice, a file-summary slice, a tool-consumer slice, or a regression slice.

2. Inspect the active shared boundary first.
   Start in `wow-viewer/src/core/WowViewer.Core` and `wow-viewer/src/core/WowViewer.Core.IO`, then check `wow-viewer/tests/WowViewer.Core.Tests`, `wow-viewer/tools/inspect`, and `wow-viewer/tools/converter`.

3. Extract the smallest reusable slice.
   Prefer typed contracts and shared helper seams over tool-local parsing or one-off heuristics.

4. Keep evidence levels explicit.
   Say clearly whether the slice proves detection, top-level summary, or deeper payload parsing. Do not blur those together.

5. Validate concretely.
   Add or update the smallest high-value test in `wow-viewer/tests/WowViewer.Core.Tests`. If the slice changes inspect or converter output, also run the relevant command on the fixed development dataset.

6. Update shared continuity files.
   Sync `wow-viewer/README.md`, the relevant memory-bank file, and `gillijimproject_refactor/plans/wow_viewer_shared_io_library_plan_2026-03-26.md` when the active workflow, commands, or migration boundary changes.

## High-Value Files

- `wow-viewer/src/core/WowViewer.Core/Chunks/*`
- `wow-viewer/src/core/WowViewer.Core/Files/*`
- `wow-viewer/src/core/WowViewer.Core/Maps/*`
- `wow-viewer/src/core/WowViewer.Core.IO/Chunked/*`
- `wow-viewer/src/core/WowViewer.Core.IO/Files/*`
- `wow-viewer/src/core/WowViewer.Core.IO/Maps/*`
- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs`
- `wow-viewer/tools/converter/WowViewer.Tool.Converter/Program.cs`
- `wow-viewer/tests/WowViewer.Core.Tests/*`

## Validation Commands

- Build: `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`
- Tests: `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`
- Inspect map summary: `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- map inspect --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development.wdt`
- Converter detect: `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -- detect --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development_0_0_tex0.adt`

## Guardrails

- Do not duplicate file-family heuristics in inspect or converter once a shared seam exists in `Core` or `Core.IO`.
- Do not describe classification or top-level summary work as full format parsing or writing.
- Do not route non-PM4 shared-format work back into PM4 prompts or PM4 plans.
- Do not flatten Alpha and standard terrain handling together when the format boundary is real.
- Keep readable FourCC handling in memory and reverse only at I/O boundaries.