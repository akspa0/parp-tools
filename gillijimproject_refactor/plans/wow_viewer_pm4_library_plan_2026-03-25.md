# wow-viewer PM4 Library Plan

This document locks the PM4 migration direction for wow-viewer after the Mar 25, 2026 correction from the user.

## Source-Of-Truth Rule

- the active MdxViewer implementation is the de facto reference implementation for PM4 runtime behavior
- Pm4Research should be ported as the future PM4 library family because PM4 semantics are still under active research
- parpToolbox, PM4Tool, WoWRollback.PM4Module, and older PM4 helpers remain supporting evidence, not the top-level source of PM4 truth

This means wow-viewer should not wait for a mythical final PM4 winner before giving PM4 a real library home.

## Goal

Create one first-party PM4 library family in wow-viewer that:

- preserves the current MdxViewer PM4 behavior that already works in runtime
- absorbs Pm4Research.Core as the ongoing research and decode backbone
- exposes PM4 services to the viewer, inspect CLI, and converter CLI
- keeps unstable research seams clearly marked instead of mixing them into stable contracts silently

## Proposed Project Boundary

### Shipping project

- WowViewer.Core.PM4
  - PM4 chunk models and decode contracts
  - grouping and object identity contracts
  - coordinate and transform services
  - correlation and report generation contracts
  - stable exports needed by the viewer and tools

### Research namespace or sub-area inside the same project family

- WowViewer.Core.PM4.Research
  - hypothesis scanners
  - exploratory linkage analysis
  - unstable field interpretations
  - promotion path for reports not yet considered stable

The point is not to hide research. The point is to keep research in the same PM4 family instead of leaving it marooned in parp-tools.

## Reference Hierarchy

### Runtime reference

- current MdxViewer PM4 behavior
- current MdxViewer PM4 selection and overlay contracts
- current MdxViewer PM4 transform and grouping behavior that has already been validated enough to rely on as the active viewer path

### Library seed

- Pm4Research.Core
- Pm4Research.Cli report families that can become inspect verbs

### Supporting evidence only

- WoWRollback.PM4Module
- PM4Tool
- parpToolbox
- archived PM4 helper tools

## What Ports First

### Immediate first slice

- chunk models and typed decode contracts from Pm4Research.Core
- object identity, grouping, and transform contracts already proven in current MdxViewer behavior
- report and validation seams needed for inspect pm4 verbs

Current status in the workspace:

- first slice partially landed in `wow-viewer/src/core/WowViewer.Core.PM4`
- ported elements:
  - typed PM4 chunk models
  - research document container
  - binary PM4 reader
  - exploration snapshot builder
- analyzer or inspect follow-up now also landed:
  - single-file PM4 analysis report types
  - single-file PM4 analyzer
  - decode audit report types
  - decode audit and corpus-audit analyzers
  - `WowViewer.Tool.Inspect` verbs for `pm4 inspect` and `pm4 export-json`
  - `WowViewer.Tool.Inspect` verbs for `pm4 audit` and `pm4 audit-directory`
- first viewer-facing runtime contract slice now also landed:
  - shared `Pm4AxisConvention`, `Pm4CoordinateMode`, and `Pm4PlanarTransform` contracts
  - shared `Pm4CoordinateService`
  - shared `Pm4PlacementContract` candidate set
- current research note worth preserving:
  - CK24 low-16 object values, read as integers, appear to be plausible `UniqueID` candidates on the development map, but that remains a hypothesis until correlated against real placed-object data
- not ported yet:
  - broader analyzer and reporting families beyond decode audit and single-file inspect
  - MdxViewer-facing reconstruction and transform solvers beyond the first placement-contract seam
  - broader audit, corpus-scan, hypothesis, and linkage report families

### Second slice

- the current MdxViewer PM4 workspace should switch from app-owned internals to Core.PM4-backed services without changing visible behavior
- CLI verbs from Pm4Research.Cli should move into Tool.Inspect on top of the same Core.PM4 services

### Third slice

- WoWRollback.PM4Module restore pipeline should consume Core.PM4 instead of its own isolated PM4 contract
- selected correlation or export logic from PM4Tool and parpToolbox can be promoted if still useful

## What Does Not Port First

- every historical PM4 executable identity
- every experimental report from parpToolbox or PM4Tool
- old batch or exporter wrappers that only exist because PM4 logic was fragmented before

## PM4 Consumers In wow-viewer

| Consumer | Uses Core.PM4 for | Notes |
| --- | --- | --- |
| WowViewer.App | PM4 workspace, alignment, correlation, selection, metadata, export hooks | current MdxViewer behavior is the target baseline |
| WowViewer.Tool.Inspect | pm4 inspect, validate, export-json, scan-hypotheses, scan-linkage | unstable verbs should be labeled experimental |
| WowViewer.Tool.Converter | pm4 restore and related conversion jobs | restore pipeline becomes a consumer, not the PM4 truth owner |

## Migration Sequence

1. create WowViewer.Core.PM4 as a first-class project on day one
2. port Pm4Research.Core into it
3. codify current MdxViewer PM4 behavior as the runtime reference contract
4. move the app PM4 workspace onto Core.PM4 without intended behavior changes
5. port Pm4Research.Cli verbs into Tool.Inspect over the same library
6. move WoWRollback.PM4Module restore logic behind the same PM4 services
7. selectively promote only proven algorithms from PM4Tool or parpToolbox

## Validation Rules

- if Core.PM4 changes the active runtime behavior, say so explicitly
- if a PM4 interpretation is still exploratory, label it clearly as research or experimental
- do not claim PM4 correctness based only on decoding or export success
- keep using the fixed real datasets and runtime validation guidance already captured in pm4_support_plan.md and the memory bank

## Failure Modes To Avoid

- letting the viewer remain the only place where working PM4 behavior exists
- porting Pm4Research without tying it to the current MdxViewer behavior, which would risk replacing working runtime behavior with cleaner but less proven abstractions
- importing whole PM4Tool or parpToolbox app splits into wow-viewer instead of lifting just the useful algorithms

## Bottom Line

- PM4 is not a later optional migration item anymore
- wow-viewer should start life with a real Core.PM4 project
- that project should be seeded from Pm4Research but anchored to current MdxViewer behavior as the runtime reference

## Validation Status

- planning plus first code-port slice
- `wow-viewer/src/core/WowViewer.Core.PM4` now contains the first research-seeded PM4 reader layer ported from `Pm4Research.Core`
- that PM4 project now also contains the first single-file analyzer or report layer, the first decode-audit path, and the first extracted runtime placement-contract seam from current MdxViewer behavior
- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 25, 2026 after this port
- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- pm4 inspect --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development_00_00.pm4` passed on Mar 25, 2026 and produced a real summary report
- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- pm4 audit --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development_00_00.pm4` and `pm4 audit-directory --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development` are now part of the active Core.PM4 research surface
- this is still not runtime viewer integration or PM4 correctness signoff