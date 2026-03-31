# wow-viewer World Runtime Prompt Set

Ordered prompt set for splitting `gillijimproject_refactor/src/MdxViewer/Terrain/WorldScene.cs` into explicit runtime services owned by `wow-viewer`.

Current assumptions:

- the first reusable telemetry seam already moved into `wow-viewer/src/core/WowViewer.Core.Runtime/World`
- `WorldScene` still owns live render orchestration, visible-set collection, and overlay/debug behavior
- repeated `.skin` lookup churn and failed MDX retry spam are active performance noise and should be treated as the first stabilization slice before deeper pass extraction
- capture automation in `src/MdxViewer/ViewerApp_CaptureAutomation.cs` appears usable enough to support fixed-shot before or after smoke checks when a slice changes live viewer behavior

Run these in order unless a user explicitly asks for one named later slice:

1. `01-negative-asset-lookup-suppression.prompt.md`
2. `02-visible-set-runtime-extraction.prompt.md`
3. `03-world-pass-service-extraction.prompt.md`
4. `04-world-scene-host-thinning.prompt.md`
5. `05-wow-viewer-app-runtime-consumer.prompt.md`

Validation rule:

- default proof for `wow-viewer` slices is `dotnet build` and `dotnet test` in `wow-viewer`
- use `MdxViewer` build only when the slice changes compatibility or current-viewer behavior
- use capture automation or fixed camera shots as smoke evidence only; do not present them as full runtime closure