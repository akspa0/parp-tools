# wow-viewer Minimap Generation Plan

## Status

- status: active
- intent: turn ad hoc large-area aerial capture into deterministic minimap generation with a real `wow-viewer` CLI path and explicit runtime extraction out of `MdxViewer`
- current proof floor:
  - `MdxViewer` now supports larger fog distance and larger detailed ADT residency for wide aerial captures
  - `MdxViewer` now supports per-map world-object path-family filters for `WMO` and `MDX` families
  - `wow-viewer/src/core/WowViewer.Core.Runtime/World/ObjectPathFilterEntry.cs` is now the first shared extraction seam for this work
  - there is still no deterministic one-PNG-per-ADT export queue yet
  - there is still no `wow-viewer` CLI minimap command yet

## Why This Plan Exists

- the user wants high-resolution minimap harvesting to become a first-class workflow instead of a manual screenshot process
- the output must be deterministic enough that each produced image corresponds to one ADT tile and can be regenerated later
- object suppression for capture must be path-family-based, not hand-toggled per instance, so whole families of WMOs or doodads can be removed reproducibly
- this work is also a forcing function for the long-requested `WorldScene` split, because minimap generation should not stay permanently owned by one giant viewer host file

## Already-Landed Prerequisites

### Compatibility-host capture prerequisites

- `gillijimproject_refactor/src/MdxViewer/ViewerApp.cs` now supports longer-range terrain far planes tied to fog instead of the old `5000` cap
- `gillijimproject_refactor/src/MdxViewer/Terrain/TerrainManager.cs` now supports larger detailed ADT budgets for aerial capture setups
- `gillijimproject_refactor/src/MdxViewer/Terrain/WorldScene.cs` now uses larger world-object distance caps so terrain and world-object visibility no longer diverge around the old fog ceiling

### Filter prerequisites

- `gillijimproject_refactor/src/MdxViewer/ViewerApp.cs` now persists object path-family filters per map
- `gillijimproject_refactor/src/MdxViewer/ViewerApp_Sidebars.cs` now exposes manual filter editing plus selected-object quick-add buttons
- `wow-viewer/src/core/WowViewer.Core.Runtime/World/ObjectPathFilterEntry.cs` now owns the shared normalization and family-aware prefix matching logic

## Ordered Workstreams

### Workstream 01 - Deterministic One-PNG-Per-ADT Capture Queue

- status: open
- target problem:
  - the viewer can take captures, but there is still no deterministic tile queue or one-output-per-ADT contract
- likely file scope:
  - `gillijimproject_refactor/src/MdxViewer/ViewerApp_CaptureAutomation.cs`
  - `gillijimproject_refactor/src/MdxViewer/ViewerApp.cs`
  - `gillijimproject_refactor/src/MdxViewer/ViewerApp_Sidebars.cs`
  - `gillijimproject_refactor/src/MdxViewer/Terrain/WorldScene.cs`
- proof goal:
  - the active viewer can run a repeatable tile job for a chosen map area
  - each output PNG maps to exactly one ADT coordinate in deterministic naming or manifest form
  - the same job can honor the existing path-family filters

### Workstream 02 - wow-viewer CLI Minimap Surface

- status: open
- target problem:
  - minimap generation is still trapped in the active viewer and cannot be run as a first-class `wow-viewer` tool workflow
- likely file scope:
  - `wow-viewer/tools/converter/WowViewer.Tool.Converter/Program.cs`
  - `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs`
  - shared minimap-generation helpers or services under `wow-viewer/src/core/WowViewer.Core` or `wow-viewer/src/core/WowViewer.Core.Runtime`
- proof goal:
  - there is a real `wow-viewer` command surface for minimap jobs
  - the command is thin over shared services and produces deterministic outputs or manifests
  - proof is based on real-data runs against the fixed workspace paths, not only synthetic fixtures

### Workstream 03 - Runtime-Owned Minimap Generation Extraction

- status: open
- target problem:
  - even after viewer proof and a CLI front door exist, tile-job planning and capture ownership cannot stay buried in `ViewerApp` or `WorldScene`
- likely file scope:
  - `gillijimproject_refactor/src/MdxViewer/Terrain/WorldScene.cs`
  - `gillijimproject_refactor/src/MdxViewer/ViewerApp.cs`
  - `gillijimproject_refactor/src/MdxViewer/ViewerApp_CaptureAutomation.cs`
  - new shared runtime or core services in `wow-viewer/src/core/WowViewer.Core.Runtime/World`
  - companion updates in `gillijimproject_refactor/plans/wow_viewer_world_runtime_service_plan_2026-03-31.md`
- proof goal:
  - the next tile-job or framing contract is runtime-owned and reusable by both viewer and CLI
  - `WorldScene` or `ViewerApp` becomes smaller for real in a way that can be pointed to concretely

## Workflow Surface

- route broad minimap-generation asks through `.github/prompts/wow-viewer-minimap-generation-plan-set.prompt.md`
- ordered execution prompts now live in `.github/prompts/wow-viewer-minimap-generation/`
- when the ask is primarily about the broader `WorldScene` split rather than minimap output itself, continue to use `.github/prompts/wow-viewer-world-runtime-plan-set.prompt.md`
- when the ask spans both, use the minimap plan set first and call out the exact world-runtime follow-up seam explicitly

## Validation Rules

- do not call minimap generation complete based only on library tests or synthetic fixtures
- use the fixed real-data paths already recorded in `gillijimproject_refactor/memory-bank/data-paths.md`
- be explicit about proof level:
  - viewer-hosted deterministic queue proof
  - CLI command proof
  - shared runtime extraction proof
  - these are different milestones and should not be conflated
- do not describe build or test passes as viewer runtime signoff

## Immediate Next Slice

- the next slice should be Workstream 01: deterministic one-PNG-per-ADT capture queue in the active viewer host
- that slice should reuse the already-landed path-family filters and treat the `wow-viewer` CLI surface as the follow-up, not as part of the same patch