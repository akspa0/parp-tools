# wow-viewer Shared I/O Library Plan

This document locks the current non-PM4 shared-format direction for `wow-viewer` after the first shared map-summary and cross-family detection slices landed on Mar 26, 2026.

## Source-Of-Truth Rule

- `wow-viewer/src/core/WowViewer.Core` and `wow-viewer/src/core/WowViewer.Core.IO` are the intended home for first-party non-PM4 file-format ownership.
- `gillijimproject_refactor` remains the current runtime and compatibility reference while deeper format behavior is still being ported.
- `WowViewer.Tool.Inspect` and `WowViewer.Tool.Converter` should consume shared detection or reader seams instead of owning parallel heuristics.

## Goal

Create one first-party shared format stack in `wow-viewer` that:

- owns reusable FourCC, chunk, detection, and summary contracts
- grows into canonical readers or writers for map, object, model, texture, and table families
- feeds inspect and converter surfaces without duplicate parsing logic
- keeps top-level detection, summary, deep parsing, and writing boundaries explicit instead of pretending they are already the same level of proof

## Current Landed Shared Boundary

### Core contracts

- `WowViewer.Core/Chunks`
  - `FourCC`
  - `ChunkHeader`
- `WowViewer.Core/Maps`
  - `MapChunkIds`
  - `MapFileKind`
  - `MapChunkLocation`
  - `MapFileSummary`
- `WowViewer.Core/Files`
  - `WowFileKind`
  - `WowFileDetection`

### Core.IO seams

- `WowViewer.Core.IO/Chunked`
  - `ChunkHeaderReader`
  - `ChunkedFileReader`
- `WowViewer.Core.IO/Maps`
  - `MapFileSummaryReader`
- `WowViewer.Core.IO/Files`
  - `WowFileDetector`

### Tool consumers

- `WowViewer.Tool.Inspect`
  - `map inspect --input <file.wdt|file.adt>`
- `WowViewer.Tool.Converter`
  - `detect --input <file>`

### Regression floor

- `wow-viewer/tests/WowViewer.Core.Tests`
  - FourCC and chunk-header boundary tests
  - synthetic WDT and ADT summary tests
  - real-data `development.wdt` and `development_0_0.adt` summary tests
  - synthetic WDT detection tests
  - real-data WDT, ADT, split ADT, and PM4 detection tests

## Current Fixed-Dataset Proof

- `development.wdt` -> `Wdt`, version `18`
- `development_0_0.adt` -> `Adt`, version `18`
- `development_0_0_tex0.adt` -> `AdtTex`, version `18`
- `development_0_0_obj0.adt` -> `AdtObj`, version `18`
- `development_00_00.pm4` -> `Pm4`, version `12304`

## Current Boundaries

- shared detection is real
- top-level WDT or ADT chunk summary is real
- this does not yet prove:
  - deep WDT semantic parsing
  - deep ADT root or split payload parsing
  - WMO, M2, BLP, DBC, or DB2 payload parsing
  - any write path or round-trip support
  - runtime cutover inside the active viewer

## Immediate Next Slices

1. deepen shared ADT root and split-file top-level summaries
2. add shared WDT semantic summary beyond raw chunk order
3. add first shared WMO or model-family detection or top-level summary slice
4. keep inspect and converter as thin consumers of shared seams instead of adding direct parsing in tool entrypoints

## Failure Modes To Avoid

- letting inspect and converter rebuild their own file-family heuristics instead of consuming `WowFileDetector`
- describing shared detection as if it were canonical payload parsing
- drifting back into PM4-only workflow assets when the active task is non-PM4 shared I/O
- keeping multiple first-party parser roots alive once a shared `wow-viewer` seam exists

## Copilot Continuity Surface

- future sessions should treat these `.github` assets as the canonical shared workflow for the active non-PM4 shared-I/O slice:
  - `.github/skills/wow-viewer-shared-io-library/SKILL.md`
  - `.github/skills/wow-viewer-migration-continuation/SKILL.md`
  - `.github/prompts/wow-viewer-shared-io-implementation.prompt.md`
  - `.github/prompts/wow-viewer-tool-suite-plan-set.prompt.md`
  - `.github/prompts/wow-viewer-shared-io-library-plan.prompt.md`
- use the shared-I/O implementation prompt when the ask is an actual `Core` or `Core.IO` slice, inspect verb, converter command, or regression update
- use the broader shared-I/O plan prompt when the ask is format ownership, source-root consolidation, read/write authority, or migration ordering
- whenever a new `wow-viewer` skill or implementation prompt is added, update `.github/copilot-instructions.md`, `wow-viewer/README.md`, this plan, and the memory-bank continuity files in the same slice so the new route is discoverable without manual recovery

## Validation Status

- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `32` tests after the shared detector slice landed
- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- map inspect --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development.wdt` passed on Mar 26, 2026
- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -- detect --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development_00_00.pm4` passed on Mar 26, 2026
- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -- detect --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development_0_0_tex0.adt` passed on Mar 26, 2026
- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -- detect --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development_0_0_obj0.adt` passed on Mar 26, 2026