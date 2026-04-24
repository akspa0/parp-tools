# Viewer Legacy Cutover Boundary

This note records the viewer-app ownership boundary after slices 01 through 08 of the `wow-viewer` cutover plan.

The purpose is narrow: future viewer work should stop drifting back into `gillijimproject_refactor/src/MdxViewer` by default.

## Canonical Owner Now

The default home for new viewer-shell work is `wow-viewer`.

Today that means:

- `wow-viewer/src/viewer/WowViewer.App`
  - desktop window host
  - workspace and session state
  - standalone shell panels and menu layout
  - bounded M2 preview flow
  - bounded world-session bootstrap, one-tile runtime inspection flow, and first GPU terrain preview flow
  - viewer-facing CLI proof commands such as `viewer`, `m2-frame`, `m2-gpu-frame`, `world-bootstrap`, and `world-frame`
- `wow-viewer/src/core/WowViewer.Core.Runtime`
  - canonical runtime contracts for extracted M2 and world seams
- `wow-viewer/src/core/WowViewer.Core` and `wow-viewer/src/core/WowViewer.Core.IO`
  - canonical shared contracts and file or map access seams used by the new app

If the task is about any of these, start in `wow-viewer` first:

- app shell layout
- session or workspace design
- navigator, inspector, status, or selection surfaces
- new runtime-backed standalone viewers
- new world bootstrap or bounded inspection flows
- new shared runtime or I/O contracts consumed by the app shell

## MdxViewer Compatibility-Only Surface

`gillijimproject_refactor/src/MdxViewer` remains a compatibility consumer and proof harness where `wow-viewer` seams still need active old-viewer validation.

That includes file families such as:

- `Rendering/WowViewerM2RuntimeBridge.cs`
- `Rendering/M2Renderer.cs`
- `Rendering/M2RuntimeAnimator.cs`
- `Rendering/WmoRenderer.cs`
- `Terrain/WorldScene.cs`
- `Terrain/WorldAssetManager.cs`
- `ViewerApp.cs` when it is only consuming already-extracted `wow-viewer` seams

Use `MdxViewer` here only for:

- bounded compatibility hotfixes in the active old viewer
- validating that an extracted `wow-viewer` seam can still be consumed by the old viewer
- runtime evidence gathering when `wow-viewer` does not yet own the equivalent behavior

Do not treat successful `MdxViewer` consumption as design ownership returning to the old repo.

## MdxViewer Editor And Archaeology Surface

Several active `MdxViewer` surfaces are still real and useful, but they are legacy editor or archaeology work rather than the forward design owner for the new viewer app.

Treat these as legacy-only unless the user explicitly asks for a bounded hotfix there:

- `ViewerApp_TerrainAnalysis.cs`
- `ViewerApp_Sidebars.cs`
- `ViewerApp_WdlPreview.cs`
- `ViewerApp_Pm4Utilities.cs`
- `ViewerApp_MlTraining.cs`
- `ViewerApp_Investigation.cs`
- `ViewerApp_MinimapAndStatus.cs`
- `ViewerApp_CaptureAutomation.cs`
- `ViewerApp_ClickSelection.cs`
- `ViewerApp_ClientDialogs.cs`
- `Catalog/*`
- `Population/*`
- `Export/Terrain*`
- most of `Terrain/*` while terrain, liquid, and WDL runtime ownership are still being extracted

Those surfaces may continue to exist for:

- terrain editing or restoration
- archaeology and investigation workflows
- WDL preview and minimap-oriented legacy tooling
- PM4 investigation helpers that are not yet replaced by dedicated `wow-viewer` tool or editor surfaces
- existing capture or export workflows that have not been cut over yet

They should not be used as the default place to invent new viewer-app architecture.

## Do Not Add New Work Here

Do not add new long-range viewer design work to `MdxViewer` for:

- new shell panels, workspaces, or menu surfaces
- new session-state contracts
- new navigator, inspector, or selection architecture
- new standalone viewer consumers
- new world-runtime data models that duplicate `wow-viewer` runtime contracts
- new shared file or map readers that should live in `WowViewer.Core` or `WowViewer.Core.IO`
- new CLI proof commands that belong in `WowViewer.App` or a `wow-viewer` tool

If one of those is needed, move the work into `wow-viewer` and keep any `MdxViewer` follow-up strictly compatibility-scoped.

## Validation Language

Use these proof rules:

- a `wow-viewer` build, test pass, or real-data CLI proof is the primary validation for `wow-viewer` viewer work
- an `MdxViewer` build is optional compatibility-compile validation only
- `MdxViewer` runtime screenshots, captures, or behavior checks are evidence inputs for extraction and parity work, not proof that `MdxViewer` is the design owner again
- unless the user explicitly asks for old-viewer work, state `MdxViewer` results as compatibility or archaeology evidence only

## Next Direction

After this boundary pass, follow-up viewer-facing implementation should continue deepening `wow-viewer` runtime or renderer ownership, not drift back into shell expansion in `MdxViewer`.

For world-runtime continuation, route planning through `.github/prompts/wow-viewer-world-runtime-plan-set.prompt.md` and keep `WowViewer.App` as the shell consumer.