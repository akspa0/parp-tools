# wow-viewer PM4 Library Plan

This document locks the PM4 migration direction for wow-viewer after the Mar 25, 2026 correction from the user.

## Mar 26, 2026 - Source-Of-Truth Reset

- `WowViewer.Core.PM4` is now the canonical implementation target and intended source of truth for new PM4 work in `wow-viewer`.
- `MdxViewer` is no longer the default runtime reference for PM4 library design. Use it only as a historical, extraction, or consumer-compatibility input when a task explicitly needs that comparison.
- `Pm4Research` remains the main library seed and research input.
- Default validation for PM4 work is `wow-viewer` build or test plus the relevant inspect command against the fixed development dataset.
- `MdxViewer` compile validation is now optional and should be run only when a slice intentionally changes consumer compatibility or when the user explicitly asks for it.
- If older sections below still describe `MdxViewer` as the default runtime reference, treat this reset section as the newer rule.

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
  - linkage report types
  - corpus linkage analyzer
  - MSCN relationship report types
  - corpus MSCN analyzer
  - unknowns report types
  - corpus unknowns analyzer
  - `WowViewer.Tool.Inspect` verbs for `pm4 inspect` and `pm4 export-json`
  - `WowViewer.Tool.Inspect` verbs for `pm4 audit` and `pm4 audit-directory`
  - `WowViewer.Tool.Inspect` verb for `pm4 linkage`
  - `WowViewer.Tool.Inspect` verb for `pm4 mscn`
  - `WowViewer.Tool.Inspect` verb for `pm4 unknowns`
- first viewer-facing runtime contract slice now also landed:
  - shared `Pm4AxisConvention`, `Pm4CoordinateMode`, and `Pm4PlanarTransform` contracts
  - shared `Pm4CoordinateService`
  - shared `Pm4PlacementContract` candidate set
  - shared `Pm4PlacementMath` helper layer ported from the current `WorldScene` range-based axis/tile-local/world-placement math
  - first normal-based axis scoring and detection helpers from current `WorldScene`
  - first planar-transform resolver from current `WorldScene`, including MPRL centroid, footprint, and yaw candidate scoring
  - first world-yaw correction solver from current `WorldScene`, including signed basis fallback against MPRL heading evidence
  - first world-space surface centroid helper from current `WorldScene`, keeping shared pivot computation on surface geometry under the chosen PM4 basis and planar transform
  - first world-space yaw-application helper layer from current `WorldScene`, keeping pivot rotation and corrected world-position conversion in shared PM4 math while leaving renderer-space mapping outside the library
  - first reusable `Pm4PlacementSolution` contract plus `ResolvePlacementSolution(...)` entry point, returning typed transform, pivot, and yaw-correction results in one library-owned object
  - first typed coordinate-mode resolution seam, returning the chosen tile-local or world-space interpretation, chosen planar transform, both scores, and whether the fallback path was used
- first PM4 test slice now also landed:
  - `tests/WowViewer.Core.PM4.Tests`
  - real-data assertions for `development_00_00.pm4`
  - real-data corpus-audit assertions for the fixed development PM4 directory
  - real-data linkage assertions for the fixed development PM4 directory
  - placement-helper assertions for `development_00_00.pm4`, including normal-based axis scoring
  - planar-transform resolver assertions for `development_00_00.pm4` and a synthetic world-space quarter-turn case
  - coordinate-mode resolver assertions for `development_00_00.pm4`, a synthetic world-space case, and the missing-evidence fallback path
  - synthetic world-yaw correction assertion for the signed-basis fallback layer
  - synthetic world-space centroid assertion for the tile-local centroid helper layer
  - synthetic pivot-rotation and corrected world-position assertions for the world-space yaw helper layer
  - synthetic end-to-end placement-solution assertions for typed transform, pivot, and yaw-correction resolution
  - real-data MSCN assertions for the fixed development PM4 directory
  - real-data unknowns assertions for the fixed development PM4 directory
- first active viewer consumer slice now also landed:
  - `gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj` now references `wow-viewer/src/core/WowViewer.Core.PM4`
  - `WorldScene.ResolvePlanarTransform(...)` now delegates to shared `Core.PM4`
  - `WorldScene.TryComputeWorldYawCorrectionRadians(...)` now delegates to shared `Core.PM4`
  - `WorldScene.ComputeSurfaceWorldCentroid(...)` now delegates to shared `Core.PM4`
- current research note worth preserving:
  - CK24 low-16 object values, read as integers, appear to be plausible `UniqueID` candidates on the development map, but that remains a hypothesis until correlated against real placed-object data
- not ported yet:
  - broader analyzer and reporting families beyond decode audit, linkage, MSCN, unknowns, and single-file inspect
  - MdxViewer-facing reconstruction and transform solvers beyond the first placement-contract and placement-math seam, normal-based axis scoring, planar-transform resolution, world-yaw correction, world-space centroid computation, the world-space yaw helper layer, the new typed placement-solution layer, and the first narrow consumer hook-ups
  - broader hypothesis and structure-confidence report families

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

## Copilot Continuity Surface

- future sessions should treat these `.github` assets as the canonical shared workflow for the active `wow-viewer` PM4 slice:
  - `.github/skills/wow-viewer-pm4-library/SKILL.md`
  - `.github/skills/wow-viewer-migration-continuation/SKILL.md`
  - `.github/prompts/wow-viewer-pm4-library-implementation.prompt.md`
  - `.github/prompts/wow-viewer-tool-suite-plan-set.prompt.md`
- use the PM4 library prompt when the ask is an actual `Core.PM4` slice, inspect verb, regression update, or narrow solver extraction
- use the broader tool-suite routing prompt when the ask is repo-shape, tool inventory, CLI or GUI parity, or migration sequencing
- keep `.github/copilot-instructions.md`, the memory bank, and this plan in sync when the active `wow-viewer` workflow changes so future chats do not restart from stale bootstrap assumptions

## Failure Modes To Avoid

- letting the viewer remain the only place where working PM4 behavior exists
- porting Pm4Research without tying it to the current MdxViewer behavior, which would risk replacing working runtime behavior with cleaner but less proven abstractions
- importing whole PM4Tool or parpToolbox app splits into wow-viewer instead of lifting just the useful algorithms

## Bottom Line

- PM4 is not a later optional migration item anymore
- wow-viewer should start life with a real Core.PM4 project
- that project should be seeded from Pm4Research but anchored to current MdxViewer behavior as the runtime reference
- PM4 is still only one part of the intended library stack; `WowViewer.Core`, `WowViewer.Core.IO`, bootstrap dependency wiring, and later runtime/data-source cutover still need explicit follow-through or the repo will drift away from the original migration plan
- first non-PM4 follow-through is now starting to exist:
  - `WowViewer.Core` has a first shared WDT or ADT map-summary contract family
  - `WowViewer.Core.IO` has a first shared WDT or ADT top-level reader
  - `WowViewer.Tool.Inspect` now consumes that shared reader through `map inspect --input <file.wdt|file.adt>`
  - `WowViewer.Core` and `WowViewer.Core.IO` now also have a first shared cross-family detector used by both inspect and converter surfaces
  - this is still only a top-level summary slice, not broader map-format parity

## Validation Status

- planning plus first code-port slice
- `wow-viewer/src/core/WowViewer.Core.PM4` now contains the first research-seeded PM4 reader layer ported from `Pm4Research.Core`
- that PM4 project now also contains the first single-file analyzer or report layer, the first decode-audit path, and the first extracted runtime placement-contract seam from current MdxViewer behavior
- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 25, 2026 after this port
- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- pm4 inspect --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development_00_00.pm4` passed on Mar 25, 2026 and produced a real summary report
- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- pm4 audit --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development_00_00.pm4` and `pm4 audit-directory --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development` are now part of the active Core.PM4 research surface
- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- pm4 linkage --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development` passed on Mar 25, 2026 and reported `150` mismatch-bearing files, `58` bad-MDOS files, and `4553` total ref-index mismatches in the fixed corpus
- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- pm4 mscn --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development` passed on Mar 25, 2026 and reported `309` MSCN-bearing files, `1,342,410` MSCN points, and much stronger raw-than-swapped MSCN bounds overlap in the fixed corpus
- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- pm4 unknowns --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development` passed on Mar 26, 2026 and reported `1,273,335` sentinel-pattern `MSLK.LinkId` values, `598,882` active `MSLK` path windows, and the same `4,553` unresolved `MSLK.RefIndex -> MSUR` misses already visible in linkage
- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `18` PM4 tests, including the first coordinate-mode resolver regression slice on top of the planar-transform resolver, world-yaw correction, world-space centroid, world-space yaw helper, and typed placement-solution regression slices
- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug -p:OutDir=i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer-pm4-corepm4-hookup/` passed on Mar 26, 2026 and confirmed the active viewer can compile against the shared `Core.PM4` solver slice
- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug -p:OutDir=i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer-pm4-centroid-hookup/` passed on Mar 26, 2026 and confirmed the active viewer can also compile after moving world-space centroid computation into shared `Core.PM4`
- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.PM4.Tests/WowViewer.Core.PM4.Tests.csproj -c Debug` passed again on Mar 26, 2026 with `18` PM4 tests
- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug --filter PlacementMath` passed again on Mar 26, 2026 with `11` placement-focused tests
- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed again on Mar 26, 2026 with `32` total tests across PM4 and shared-I/O suites
- this is still not runtime viewer integration or PM4 correctness signoff

## Fresh-Chat Next Slice

- The clean next PM4 implementation slice is direct library completion in `Core.PM4`, not another default `MdxViewer` hookup.
- Best next seam:
  - continue re-homing remaining placement, grouping, transform, and correlation ownership into `WowViewer.Core.PM4` so the library stops depending on the old viewer as its implicit authority
- Why this is the right next step:
  - it moves the real implementation boundary into the library instead of preserving a permanent split with the old viewer
  - it keeps validation centered on `wow-viewer` builds, tests, and inspect commands
  - it matches the explicit user direction that the new repo should own the implementation rather than orbit the old app
- Still out of scope after that slice:
  - viewer runtime signoff
  - treating old `MdxViewer` parity as the default success condition
  - closing the remaining exploratory field semantics without evidence