# wow-viewer World Runtime Service Plan

## Status

- status: active
- intent: split `gillijimproject_refactor/src/MdxViewer/Terrain/WorldScene.cs` into explicit runtime services owned by `wow-viewer`
- current proof floor:
  - renderer telemetry contracts and optimization-hint logic already live in `wow-viewer/src/core/WowViewer.Core.Runtime/World`
  - `MdxViewer` currently consumes that seam successfully

## Why This Plan Exists

- the user wants the long-term world renderer design to move into `wow-viewer`, not stay trapped in `MdxViewer`
- the remaining `WorldScene` file is still too large and mixes asset churn, visible-set collection, pass sequencing, and overlay/debug work
- repeated `.skin` miss churn and failed MDX retry spam are active performance noise and should be reduced before deeper renderer extraction so later measurements stay interpretable

## Ordered Slices

### Slice 01 - Negative Asset Lookup Suppression

- target problem:
  - repeated `.skin` candidate searches
  - repeated failed MDX retry paths in `WorldAssetManager`
  - noisy asset-miss logging that hides other hot paths
- likely files:
  - `gillijimproject_refactor/src/MdxViewer/Terrain/WorldAssetManager.cs`
  - `gillijimproject_refactor/src/MdxViewer/Rendering/WmoRenderer.cs`
  - `gillijimproject_refactor/src/MdxViewer/ViewerApp.cs`
  - optional narrow shared contract in `wow-viewer/src/core/WowViewer.Core.Runtime`
- proof goal:
  - known-missing companion `.skin` and known-failed MDX loads stop redoing the same work every frame or deferred load pass
  - logs report the miss once or in bounded form instead of flooding
  - build validation plus optional fixed-shot capture smoke

### Slice 02 - Visible Set Runtime Extraction

- target problem:
  - `WorldScene` still owns visible WMO/MDX/taxi collection and frame scratch lists
- likely destination:
  - `wow-viewer/src/core/WowViewer.Core.Runtime/World/Visibility/*`
- proof goal:
  - `WorldScene` no longer owns the visible-set contracts or scratch orchestration directly

### Slice 03 - World Pass Service Extraction

- target problem:
  - `WorldScene.Render(...)` still sequences terrain, WMO, MDX, liquids, and overlays inline
- likely destination:
  - `wow-viewer/src/core/WowViewer.Core.Runtime/World/Passes/*`
  - coordinator in `wow-viewer/src/core/WowViewer.Core.Runtime/World/*`
- proof goal:
  - explicit runtime-owned pass services exist for terrain/WMO/MDX/overlay work

### Slice 04 - WorldScene Host Thinning

- target problem:
  - even after service extraction, `WorldScene` can still act like the design owner if delegation remains vague
- proof goal:
  - `WorldScene` becomes a thin host or adapter over runtime services instead of the orchestration owner

### Slice 05 - wow-viewer App Runtime Consumer

- target problem:
  - extracted runtime services still need a first direct consumer in `wow-viewer`
- proof goal:
  - `wow-viewer/src/viewer/WowViewer.App` exercises the extracted runtime seams directly without claiming full production parity

## Prompt Surface

- root router:
  - `.github/prompts/wow-viewer-world-runtime-plan-set.prompt.md`
- ordered prompts:
  - `.github/prompts/wow-viewer-world-runtime/01-negative-asset-lookup-suppression.prompt.md`
  - `.github/prompts/wow-viewer-world-runtime/02-visible-set-runtime-extraction.prompt.md`
  - `.github/prompts/wow-viewer-world-runtime/03-world-pass-service-extraction.prompt.md`
  - `.github/prompts/wow-viewer-world-runtime/04-world-scene-host-thinning.prompt.md`
  - `.github/prompts/wow-viewer-world-runtime/05-wow-viewer-app-runtime-consumer.prompt.md`

## Validation Rules

- default `wow-viewer` proof stays `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` and `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`
- build `gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` only when a slice changes compatibility or current-viewer behavior
- treat capture automation in `src/MdxViewer/ViewerApp_CaptureAutomation.cs` as a useful smoke aid for fixed-shot before/after checks, not as runtime signoff by itself

## Explicit Non-Claims

- this plan does not claim that the renderer can move in one jump
- this plan does not claim that every performance issue is `.skin` related
- this plan does not claim that a wow-viewer build or test pass proves viewer-runtime closure